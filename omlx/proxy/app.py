# SPDX-License-Identifier: Apache-2.0
"""FastAPI application for the MLX-free oMLX proxy."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

from omlx._version import __version__
from omlx.api.anthropic_models import (
    MessagesRequest,
    TokenCountRequest,
    TokenCountResponse,
)
from omlx.api.anthropic_utils import (
    convert_anthropic_to_internal,
    convert_anthropic_tools_to_internal,
    convert_internal_to_anthropic_response,
    create_content_block_start_event,
    create_content_block_stop_event,
    create_error_event,
    create_input_json_delta_event,
    create_message_delta_event,
    create_message_start_event,
    create_message_stop_event,
    create_text_delta_event,
    create_thinking_delta_event,
    map_finish_reason_to_stop_reason,
    request_has_cache_control,
)
from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.server_metrics import ServerMetrics
from omlx.utils.tokenizer import is_gemma4_model

from .admin import configure_admin
from .backend import (
    BackendError,
    OpenAIBackend,
    coalesce_tool_call_deltas,
    filter_tool_calls_by_tools,
    openai_response_to_internal,
    parse_sse_line,
    recover_text_tool_calls,
)
from .config import ProxyConfig
from .metrics import backend_context_limit
from .protocol_markers import Gemma4StreamProcessor, MarkerStripper, strip_markers
from .responses_adapter import (
    non_streaming_responses_response,
    responses_to_chat_body,
    stream_responses_events,
)
from .scaling import scale_token_count
from .stats import (
    inject_include_usage,
    model_from_body,
    record_chat_response,
    record_request,
    stats_path_from_env,
    track_usage_stream,
)

security = HTTPBearer(auto_error=False)


def create_app(
    config: ProxyConfig | None = None,
    backend: OpenAIBackend | None = None,
) -> FastAPI:
    proxy_config = config or ProxyConfig.from_env()
    backend = backend or OpenAIBackend(proxy_config)
    server_metrics = ServerMetrics(stats_path=stats_path_from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.proxy_config = backend.config
        app.state.backend = backend
        yield
        server_metrics.save_alltime()
        await backend.close()

    app = FastAPI(
        title="oMLX Proxy",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.server_metrics = server_metrics
    configure_admin(app, backend, proxy_config)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/admin/dashboard", status_code=302)

    async def verify_proxy_key(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        x_api_key: str | None = Header(default=None),
    ) -> bool:
        expected = backend.config.proxy_api_key
        if not expected:
            return True
        bearer_token = credentials.credentials if credentials else None
        if bearer_token != expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return True

    @app.get("/health")
    async def health(request: Request):
        backend_status: dict[str, Any]
        try:
            models = await backend.get_models(
                _backend_authorization(request, backend.config)
            )
            backend_status = {
                "reachable": True,
                "models": len(models.get("data") or []),
            }
        except Exception as exc:
            backend_status = {
                "reachable": False,
                "error": str(exc),
            }
        default_model = None
        try:
            models_data = await backend.get_models(
                _backend_authorization(request, backend.config)
            )
            models = models_data.get("data") or []
            if models:
                default_model = models[0].get("id")
        except Exception:
            pass
        return {
            "status": "healthy",
            "version": __version__,
            "mode": "proxy",
            "default_model": default_model,
            "backend": backend_status,
        }

    @app.get("/v1/models", dependencies=[Depends(verify_proxy_key)])
    async def list_models(request: Request):
        try:
            data = await backend.get_models(
                _backend_authorization(request, backend.config)
            )
        except Exception as exc:
            raise _backend_http_exception(exc)
        return _enrich_model_list(
            data,
            getattr(request.app.state, "proxy_admin_state", None),
        )

    @app.get("/v1/models/status", dependencies=[Depends(verify_proxy_key)])
    async def list_models_status(request: Request):
        try:
            data = await backend.get_models(
                _backend_authorization(request, backend.config)
            )
        except Exception as exc:
            raise _backend_http_exception(exc)
        models = []
        for item in data.get("data") or []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            models.append(
                {
                    "id": item["id"],
                    "loaded": True,
                    "is_loading": False,
                    "model_type": "llm",
                    "engine_type": "remote",
                    "max_context_window": item.get("max_model_len"),
                    "max_tokens": None,
                    "pinned": False,
                }
            )
        return {"models": models, "mode": "proxy"}

    @app.get("/v1/mcp/tools", dependencies=[Depends(verify_proxy_key)])
    async def list_mcp_tools():
        return {"tools": [], "count": 0}

    @app.api_route(
        "/v1/messages",
        methods=["GET", "HEAD", "OPTIONS"],
        dependencies=[Depends(verify_proxy_key)],
    )
    @app.api_route(
        "/v1/messages/count_tokens",
        methods=["GET", "HEAD", "OPTIONS"],
        dependencies=[Depends(verify_proxy_key)],
    )
    async def anthropic_endpoint_probe(request: Request):
        headers = {"allow": "GET, HEAD, OPTIONS, POST"}
        if request.method in {"HEAD", "OPTIONS"}:
            return Response(status_code=204, headers=headers)
        if request.url.path.endswith("/count_tokens"):
            return JSONResponse(
                {
                    "type": "endpoint",
                    "endpoint": "/v1/messages/count_tokens",
                    "methods": ["POST"],
                },
                headers=headers,
            )
        return JSONResponse(
            {
                "type": "endpoint",
                "endpoint": "/v1/messages",
                "methods": ["POST"],
            },
            headers=headers,
        )

    @app.post("/v1/messages", dependencies=[Depends(verify_proxy_key)])
    async def anthropic_messages(request: Request):
        try:
            payload = await request.json()
            anth_request = MessagesRequest.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        openai_body = anthropic_to_openai_chat_body(anth_request)
        apply_proxy_request_defaults(
            openai_body,
            getattr(request.app.state, "proxy_admin_state", None),
            include_chat_template=True,
            context_limit=await _request_context_limit(backend),
        )
        if anth_request.stream:
            return StreamingResponse(
                _stream_anthropic_response(
                    backend,
                    backend.config,
                    anth_request,
                    openai_body,
                    _backend_authorization(request, backend.config),
                    metrics=server_metrics,
                ),
                media_type="text/event-stream",
            )

        try:
            data = await backend.chat_completion(
                openai_body,
                _backend_authorization(request, backend.config),
            )
            internal = openai_response_to_internal(
                data,
                tools=openai_body.get("tools"),
            )
            # Stats record the real backend token counts, before any
            # Claude Code context scaling is applied to the client reply.
            record_chat_response(
                server_metrics, data, fallback_model=str(anth_request.model)
            )
            active_config = backend.config
            prompt_tokens = scale_token_count(internal.prompt_tokens, active_config)
            completion_tokens = scale_token_count(
                internal.completion_tokens,
                active_config,
            )
            response = convert_internal_to_anthropic_response(
                text=internal.text,
                model=anth_request.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=internal.finish_reason,
                tool_calls=internal.tool_calls,
                thinking=internal.reasoning_content,
                cached_tokens=scale_token_count(internal.cached_tokens, active_config),
                request_uses_cache_control=request_has_cache_control(anth_request),
            )
            return response.model_dump(exclude_none=True)
        except Exception as exc:
            raise _backend_http_exception(exc)

    @app.post("/v1/messages/count_tokens", dependencies=[Depends(verify_proxy_key)])
    async def count_anthropic_tokens(request: Request):
        try:
            payload = await request.json()
            count_request = TokenCountRequest.model_validate(payload)
            messages_request = MessagesRequest(
                model=count_request.model,
                max_tokens=1,
                messages=count_request.messages,
                system=count_request.system,
                tools=count_request.tools,
                tool_choice=count_request.tool_choice,
                thinking=count_request.thinking,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        body = anthropic_to_openai_chat_body(messages_request)
        estimated = estimate_tokens(body.get("messages", []), body.get("tools"))
        return TokenCountResponse(
            input_tokens=scale_token_count(estimated, backend.config),
        )

    @app.post("/v1/responses", dependencies=[Depends(verify_proxy_key)])
    async def responses_endpoint(request: Request):
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        cc_body = responses_to_chat_body(payload)
        apply_proxy_request_defaults(
            cc_body,
            getattr(request.app.state, "proxy_admin_state", None),
            include_chat_template=True,
            context_limit=await _request_context_limit(backend),
        )
        auth = _backend_authorization(request, backend.config)
        resp_id = f"resp_{uuid.uuid4().hex}"
        created_at = int(time.time())
        model = payload.get("model") or cc_body.get("model", "")

        if cc_body.get("stream"):
            return StreamingResponse(
                stream_responses_events(
                    backend,
                    cc_body,
                    auth,
                    resp_id,
                    model,
                    created_at,
                    metrics=server_metrics,
                ),
                media_type="text/event-stream",
            )

        try:
            data = await backend.chat_completion(cc_body, auth)
        except Exception as exc:
            raise _backend_http_exception(exc)
        record_chat_response(server_metrics, data, fallback_model=str(model))
        return non_streaming_responses_response(data, resp_id, created_at)

    @app.api_route(
        "/v1/chat/completions",
        methods=["POST"],
        dependencies=[Depends(verify_proxy_key)],
    )
    @app.api_route(
        "/v1/completions",
        methods=["POST"],
        dependencies=[Depends(verify_proxy_key)],
    )
    @app.api_route(
        "/v1/embeddings",
        methods=["POST"],
        dependencies=[Depends(verify_proxy_key)],
    )
    async def passthrough(request: Request):
        path = request.url.path.removeprefix("/v1/")
        return await _passthrough_backend(
            request, backend, path, metrics=server_metrics
        )

    return app


def anthropic_to_openai_chat_body(request: MessagesRequest) -> dict[str, Any]:
    messages = convert_anthropic_to_internal(
        request,
        preserve_images=True,
        native_reasoning_content=True,
        force_native_tool_calling=bool(request.tools),
    )
    for message in messages:
        message.pop("_preserve_role_boundary", None)
    _normalize_openai_tool_call_history(messages)

    body: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.top_k is not None:
        body["top_k"] = request.top_k
    if request.stop_sequences:
        body["stop"] = request.stop_sequences
    if request.tools:
        tools = convert_anthropic_tools_to_internal(request.tools)
        if tools:
            body["tools"] = tools
    if request.tool_choice:
        body["tool_choice"] = _map_anthropic_tool_choice(request.tool_choice)
    if request.chat_template_kwargs:
        body["chat_template_kwargs"] = request.chat_template_kwargs
    if request.thinking:
        body.setdefault("chat_template_kwargs", {})
        if request.thinking.type == "disabled":
            body["chat_template_kwargs"]["enable_thinking"] = False
        elif request.thinking.type in {"enabled", "adaptive"}:
            body["chat_template_kwargs"]["enable_thinking"] = True
        if request.thinking.budget_tokens:
            body["thinking_budget"] = request.thinking.budget_tokens
    return body


def _normalize_openai_tool_call_history(messages: list[dict[str, Any]]) -> None:
    """Normalize structured tool history to OpenAI Chat Completions wire shape."""
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        normalized_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue

            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments_json = arguments or "{}"
            elif arguments is None:
                arguments_json = "{}"
            else:
                arguments_json = json.dumps(arguments, ensure_ascii=False)

            normalized = dict(tool_call)
            normalized["type"] = normalized.get("type") or "function"
            normalized["function"] = {
                **function,
                "arguments": arguments_json,
            }
            normalized_tool_calls.append(normalized)

        if normalized_tool_calls:
            message["tool_calls"] = normalized_tool_calls
        else:
            message.pop("tool_calls", None)


def _gemma_recovery_enabled(model: str, text: str) -> bool:
    return (
        is_gemma4_model(model)
        or "<|tool_call>" in text
        or "<tool_call|>" in text
        or '<|"|>' in text
    )


def _clean_protocol_text_and_calls(
    text: str,
    *,
    model: str,
) -> tuple[str, list[ToolCall] | None]:
    processor = Gemma4StreamProcessor(
        enable_tool_recovery=_gemma_recovery_enabled(model, text)
    )
    events = processor.feed(text) + processor.flush()
    cleaned_text = "".join(value for kind, value in events if kind == "text")
    tool_calls: list[ToolCall] = []
    for kind, value in events:
        if kind != "tool_call" or not isinstance(value, dict):
            continue
        name = value.get("name")
        arguments = value.get("arguments")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        try:
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(name=name, arguments=arguments or "{}"),
                )
            )
        except ValueError:
            continue
    return cleaned_text, tool_calls or None


def _body_with_stream_usage(body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    stream_options = body.get("stream_options")
    if (
        isinstance(stream_options, dict)
        and stream_options.get("include_usage") is True
    ):
        return body, False

    updated = dict(body)
    updated_options = dict(stream_options) if isinstance(stream_options, dict) else {}
    updated_options["include_usage"] = True
    updated["stream_options"] = updated_options
    return updated, True


def _should_retry_without_stream_usage(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code in {400, 422}


async def _stream_anthropic_response(
    backend: OpenAIBackend,
    config: ProxyConfig,
    request: MessagesRequest,
    openai_body: dict[str, Any],
    inbound_authorization: str | None,
    metrics: ServerMetrics | None = None,
) -> AsyncIterator[str]:
    last_usage: tuple[int, int, int] = (0, 0, 0)
    request_start = time.monotonic()
    first_chunk_at: float | None = None
    seen_model = ""
    finish_reason: str | None = None
    raw_text_parts: list[str] = []
    raw_reasoning_parts: list[str] = []
    tool_call_deltas: list[Any] = []
    tools = openai_body.get("tools")
    if not isinstance(tools, list):
        tools = None

    text_marker_filter = MarkerStripper() if not tools else None
    thinking_marker_filter = MarkerStripper() if not tools else None
    uses_cache_control = request_has_cache_control(request)

    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield create_message_start_event(message_id, request.model)

    next_block_index = 0
    open_block_type: str | None = None
    open_block_index: int | None = None
    emitted_content_block = False

    def start_block(block_type: str) -> list[str]:
        nonlocal next_block_index
        nonlocal open_block_type
        nonlocal open_block_index
        nonlocal emitted_content_block

        events: list[str] = []
        if open_block_type == block_type:
            return events
        if open_block_type is not None:
            events.append(create_content_block_stop_event(open_block_index or 0))
            next_block_index += 1
            open_block_type = None
            open_block_index = None

        open_block_type = block_type
        open_block_index = next_block_index
        emitted_content_block = True
        events.append(create_content_block_start_event(open_block_index, block_type))
        return events

    def close_open_block() -> list[str]:
        nonlocal next_block_index
        nonlocal open_block_type
        nonlocal open_block_index

        if open_block_type is None:
            return []
        events = [create_content_block_stop_event(open_block_index or 0)]
        next_block_index += 1
        open_block_type = None
        open_block_index = None
        return events

    try:
        try:
            stream_body, injected_stream_usage = _body_with_stream_usage(openai_body)
            retried_without_stream_usage = False

            while True:
                try:
                    stream = backend.stream_chat_completion(
                        stream_body,
                        inbound_authorization,
                    )
                    async for item in stream:
                        if item == "[DONE]":
                            break
                        if not isinstance(item, dict):
                            continue
                        if first_chunk_at is None and item.get("choices"):
                            first_chunk_at = time.monotonic()
                        if item.get("model"):
                            seen_model = str(item["model"])

                        usage = item.get("usage") or {}
                        prompt_details = usage.get("prompt_tokens_details") or {}
                        prompt_tokens = int(usage.get("prompt_tokens") or 0)
                        completion_tokens = int(usage.get("completion_tokens") or 0)
                        cached_tokens = int(prompt_details.get("cached_tokens") or 0)
                        if prompt_tokens or completion_tokens or cached_tokens:
                            last_usage = (
                                prompt_tokens,
                                completion_tokens,
                                cached_tokens,
                            )

                        choices = item.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0] or {}
                        delta = choice.get("delta") or {}
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])

                        reasoning_delta = (
                            delta.get("reasoning_content")
                            if delta.get("reasoning_content") is not None
                            else delta.get("reasoning")
                        ) or ""
                        if reasoning_delta:
                            raw_reasoning_parts.append(reasoning_delta)
                            if not tools:
                                if thinking_marker_filter:
                                    reasoning_delta = thinking_marker_filter.feed(
                                        reasoning_delta
                                    )
                                if reasoning_delta:
                                    for event in start_block("thinking"):
                                        yield event
                                    yield create_thinking_delta_event(
                                        open_block_index or 0,
                                        reasoning_delta,
                                    )

                        text_delta = delta.get("content") or ""
                        if text_delta:
                            raw_text_parts.append(text_delta)
                            if not tools:
                                if text_marker_filter:
                                    text_delta = text_marker_filter.feed(text_delta)
                                if text_delta:
                                    for event in start_block("text"):
                                        yield event
                                    yield create_text_delta_event(
                                        open_block_index or 0,
                                        text_delta,
                                    )

                        delta_tool_calls = delta.get("tool_calls")
                        if isinstance(delta_tool_calls, list):
                            tool_call_deltas.extend(delta_tool_calls)
                    break
                except Exception as exc:
                    if (
                        injected_stream_usage
                        and not retried_without_stream_usage
                        and first_chunk_at is None
                        and last_usage == (0, 0, 0)
                        and not raw_text_parts
                        and not raw_reasoning_parts
                        and not tool_call_deltas
                        and _should_retry_without_stream_usage(exc)
                    ):
                        retried_without_stream_usage = True
                        stream_body = openai_body
                        continue
                    raise

        except Exception as exc:
            yield create_error_event("api_error", str(exc))
            yield create_message_stop_event()
            return

        tool_calls = coalesce_tool_call_deltas(tool_call_deltas)
        if tools:
            raw_text = "".join(raw_text_parts)
            protocol_model = seen_model or str(openai_body.get("model") or "")
            protocol_text, protocol_tool_calls = _clean_protocol_text_and_calls(
                raw_text,
                model=protocol_model,
            )
            protocol_tool_calls = filter_tool_calls_by_tools(protocol_tool_calls, tools)
            protocol_reasoning = strip_markers("".join(raw_reasoning_parts))
            cleaned_text, cleaned_reasoning, recovered_tool_calls = (
                recover_text_tool_calls(
                    protocol_text,
                    reasoning_content=protocol_reasoning or None,
                    tools=tools,
                )
            )
            if not tool_calls:
                tool_calls = protocol_tool_calls or recovered_tool_calls
            if cleaned_reasoning:
                for event in start_block("thinking"):
                    yield event
                yield create_thinking_delta_event(
                    open_block_index or 0,
                    cleaned_reasoning,
                )
            if cleaned_text:
                for event in start_block("text"):
                    yield event
                yield create_text_delta_event(open_block_index or 0, cleaned_text)
        else:
            if thinking_marker_filter:
                remaining_thinking = thinking_marker_filter.flush()
                if remaining_thinking:
                    for event in start_block("thinking"):
                        yield event
                    yield create_thinking_delta_event(
                        open_block_index or 0,
                        remaining_thinking,
                    )
            if text_marker_filter:
                remaining_text = text_marker_filter.flush()
                if remaining_text:
                    for event in start_block("text"):
                        yield event
                    yield create_text_delta_event(open_block_index or 0, remaining_text)

        for event in close_open_block():
            yield event

        if tool_calls:
            for tc in tool_calls:
                index = next_block_index
                yield create_content_block_start_event(
                    index,
                    "tool_use",
                    id=tc.id,
                    name=tc.function.name,
                )
                yield create_input_json_delta_event(
                    index,
                    _tool_call_arguments_json(tc),
                )
                yield create_content_block_stop_event(index)
                next_block_index += 1
        elif not emitted_content_block:
            index = next_block_index
            yield create_content_block_start_event(index, "text")
            yield create_content_block_stop_event(index)
            next_block_index += 1

        prompt, completion, cached = last_usage
        yield create_message_delta_event(
            stop_reason=map_finish_reason_to_stop_reason(
                finish_reason or "stop",
                bool(tool_calls),
            ),
            output_tokens=scale_token_count(completion, config),
            input_tokens=scale_token_count(prompt, config),
            cached_tokens=scale_token_count(cached, config),
            request_uses_cache_control=uses_cache_control,
        )
        yield create_message_stop_event()
    finally:
        end = time.monotonic()
        prefill_duration = 0.0
        generation_duration = 0.0
        if first_chunk_at is not None:
            prefill_duration = max(0.0, first_chunk_at - request_start)
            generation_duration = max(0.0, end - first_chunk_at)
        record_request(
            metrics,
            model_id=seen_model or str(openai_body.get("model") or ""),
            prompt_tokens=last_usage[0],
            completion_tokens=last_usage[1],
            cached_tokens=last_usage[2],
            prefill_duration=prefill_duration,
            generation_duration=generation_duration,
        )


def _tool_call_arguments_json(tool_call: ToolCall) -> str:
    raw_arguments = getattr(tool_call.function, "arguments", "{}") or "{}"
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError, ValueError):
            arguments = {}
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return json.dumps(arguments, ensure_ascii=False)


def apply_proxy_request_defaults(
    body: dict[str, Any],
    state: Any,
    *,
    include_chat_template: bool = False,
    context_limit: int | None = None,
    inject_max_tokens: bool = True,
) -> bool:
    """Apply saved admin sampling defaults before forwarding to the backend.

    Mutates ``body`` in place. Returns True when a default ``max_tokens``
    was injected (callers use this to retry without it on a backend
    context-length rejection).
    """
    if state is None or not isinstance(body, dict):
        return False

    model_id = body.get("model")
    model_settings = {}
    if model_id is not None:
        model_settings = state.model_settings.get(str(model_id), {}) or {}
    overrides = state.global_overrides or {}
    force = bool(model_settings.get("force_sampling"))
    injected_max_tokens = False

    sampling_fields = (
        ("max_tokens", "sampling_max_tokens"),
        ("temperature", "sampling_temperature"),
        ("top_p", "sampling_top_p"),
        ("top_k", "sampling_top_k"),
        ("min_p", "sampling_min_p"),
        ("presence_penalty", "sampling_presence_penalty"),
        ("repetition_penalty", "sampling_repetition_penalty"),
    )
    for request_key, global_key in sampling_fields:
        value = _first_configured(
            model_settings.get(request_key), overrides.get(global_key)
        )
        if value is None:
            continue
        if request_key == "max_tokens":
            if not inject_max_tokens:
                continue
            value = _positive_int_or_none(value)
            if value is None:
                # 0/empty means "no output cap configured" — let the
                # backend apply its own limit.
                continue
            if context_limit and value >= context_limit:
                # An output cap at or above the context window guarantees
                # rejections on strict backends (vLLM enforces
                # prompt + max_tokens <= max_model_len). Leave the
                # request untouched; the admin UI surfaces the
                # misconfiguration.
                continue
            if force or body.get(request_key) is None:
                body[request_key] = value
                injected_max_tokens = True
            continue
        if force or body.get(request_key) is None:
            body[request_key] = value

    if include_chat_template:
        chat_template_kwargs = _dict_value(model_settings.get("chat_template_kwargs"))
        enable_thinking = model_settings.get("enable_thinking")
        if isinstance(enable_thinking, bool):
            chat_template_kwargs["enable_thinking"] = enable_thinking
        if chat_template_kwargs:
            merged = _dict_value(body.get("chat_template_kwargs"))
            merged.update(chat_template_kwargs)
            body["chat_template_kwargs"] = merged

        budget_enabled = bool(model_settings.get("thinking_budget_enabled"))
        budget = model_settings.get("thinking_budget_tokens")
        if budget_enabled and _configured_value(budget):
            body["thinking_budget"] = budget
    return injected_max_tokens


def _body_with_proxy_defaults(
    body: bytes,
    state: Any,
    *,
    include_chat_template: bool = False,
    context_limit: int | None = None,
    inject_max_tokens: bool = True,
) -> tuple[bytes, bool]:
    """Encode ``body`` with proxy defaults; also report max_tokens injection."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body, False
    if not isinstance(payload, dict):
        return body, False
    injected_max_tokens = apply_proxy_request_defaults(
        payload,
        state,
        include_chat_template=include_chat_template,
        context_limit=context_limit,
        inject_max_tokens=inject_max_tokens,
    )
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return encoded, injected_max_tokens


async def _request_context_limit(backend: OpenAIBackend) -> int | None:
    """Context window used to guard injected sampling defaults.

    Prefers what the backend itself reports; managed sidecar stacks fall
    back to the launch-time context length (OMLX_ACTUAL_CONTEXT_SIZE).
    """
    try:
        limit = await backend_context_limit(backend)
    except Exception:
        limit = None
    if limit:
        return limit
    if os.getenv("OMLX_SIDECAR_BACKEND", "").strip():
        return backend.config.actual_context_size or None
    return None


def _positive_int_or_none(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _first_configured(*values: Any) -> Any | None:
    for value in values:
        if _configured_value(value):
            return value
    return None


def _configured_value(value: Any) -> bool:
    return value is not None and value != ""


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


async def _passthrough_backend(
    request: Request,
    backend: OpenAIBackend,
    path: str,
    metrics: ServerMetrics | None = None,
) -> Response:
    raw_body = await request.body()
    body = raw_body
    track_stats = metrics is not None and path in {"chat/completions", "completions"}
    injected_max_tokens = False
    admin_state = getattr(request.app.state, "proxy_admin_state", None)
    if path in {"chat/completions", "completions"}:
        context_limit = await _request_context_limit(backend)
        body, injected_max_tokens = _body_with_proxy_defaults(
            raw_body,
            admin_state,
            include_chat_template=path == "chat/completions",
            context_limit=context_limit,
        )
    headers = _backend_headers(request, backend)
    wants_stream = _body_requests_stream(body)

    def _body_without_max_tokens() -> bytes:
        # Rebuild with all defaults except the injected output cap; used
        # when the backend rejects the request for context-length reasons.
        rebuilt, _ = _body_with_proxy_defaults(
            raw_body,
            admin_state,
            include_chat_template=path == "chat/completions",
            inject_max_tokens=False,
        )
        return rebuilt

    if wants_stream:
        send_body = body
        injected_usage = False
        if track_stats:
            send_body, injected_usage = inject_include_usage(body)
        request_start = time.monotonic()
        try:
            response = await backend.raw_stream(
                request.method,
                path,
                send_body,
                headers,
            )
            if injected_usage and response.status_code >= 400:
                # Backend rejected stream_options; retry untouched so the
                # client still gets a response (stats lose usage counts).
                await response.aclose()
                injected_usage = False
                response = await backend.raw_stream(
                    request.method,
                    path,
                    body,
                    headers,
                )
            if injected_max_tokens and response.status_code == 400:
                # The injected default max_tokens may exceed the room the
                # prompt leaves (vLLM rejects with a context-length error);
                # retry once without the cap so the client still gets a
                # response.
                await response.aclose()
                injected_max_tokens = False
                body = _body_without_max_tokens()
                send_body = body
                if track_stats:
                    send_body, injected_usage = inject_include_usage(body)
                response = await backend.raw_stream(
                    request.method,
                    path,
                    send_body,
                    headers,
                )
        except Exception as exc:
            raise _backend_http_exception(exc)

        async def iterator() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        stream: AsyncIterator[bytes] = iterator()
        if path == "chat/completions" and response.status_code < 400:
            stream = _normalize_openai_reasoning_stream(stream)
        if track_stats and response.status_code < 400:
            stream = track_usage_stream(
                stream,
                metrics=metrics,
                model_id=model_from_body(body),
                request_start=request_start,
                strip_usage_chunk=injected_usage,
            )

        return StreamingResponse(
            stream,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/event-stream"),
        )

    try:
        response = await backend.raw_request(request.method, path, body, headers)
        if injected_max_tokens and response.status_code == 400:
            # Same context-length safety net as the streaming path.
            response = await backend.raw_request(
                request.method, path, _body_without_max_tokens(), headers
            )
    except Exception as exc:
        raise _backend_http_exception(exc)
    if track_stats and response.status_code < 400:
        try:
            data = json.loads(response.content.decode("utf-8"))
        except Exception:
            data = None
        record_chat_response(metrics, data, fallback_model=model_from_body(body))
    content = response.content
    if path == "chat/completions" and response.status_code < 400:
        content = _normalize_openai_reasoning_body(content)
    return Response(
        content=content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


def _add_reasoning_content_aliases(payload: dict[str, Any]) -> bool:
    changed = False
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for key in ("message", "delta"):
            part = choice.get(key)
            if not isinstance(part, dict):
                continue
            if (
                "reasoning_content" in part
                or "reasoning" not in part
                or part["reasoning"] is None
            ):
                continue
            part["reasoning_content"] = part["reasoning"]
            changed = True
    return changed


def _normalize_openai_reasoning_body(content: bytes) -> bytes:
    try:
        payload = json.loads(content.decode("utf-8"))
    except Exception:
        return content
    if not isinstance(payload, dict):
        return content
    if not _add_reasoning_content_aliases(payload):
        return content
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


async def _normalize_openai_reasoning_stream(
    stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    buffer = b""
    async for raw in stream:
        buffer += raw
        out = b""
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace")
            try:
                parsed = parse_sse_line(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and _add_reasoning_content_aliases(parsed):
                line = (
                    "data: "
                    + json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
                ).encode("utf-8")
            out += line + b"\n"
        if out:
            yield out
    if buffer:
        yield buffer


def _backend_headers(request: Request, backend: OpenAIBackend) -> dict[str, str]:
    headers = {"content-type": request.headers.get("content-type", "application/json")}
    auth = _backend_authorization(request, backend.config)
    headers.update(backend.headers(auth))
    return headers


def _backend_authorization(request: Request, config: ProxyConfig) -> str | None:
    if config.backend_api_key:
        return None
    if config.proxy_api_key:
        return None
    return request.headers.get("authorization")


def _body_requests_stream(body: bytes) -> bool:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return False
    return bool(isinstance(payload, dict) and payload.get("stream"))


def _map_anthropic_tool_choice(value: Any) -> str | dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    if not isinstance(value, dict):
        return value
    choice_type = value.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        return {
            "type": "function",
            "function": {"name": value.get("name", "")},
        }
    return value


def estimate_tokens(messages: list[dict[str, Any]], tools: Any = None) -> int:
    text = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)
    return max(1, len(text) // 4)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _enrich_model_list(data: dict[str, Any], admin_state: Any) -> dict[str, Any]:
    """Advertise each model's context window so clients size requests correctly.

    The window is the backend's reported ``max_model_len`` — exactly what the
    serving backend enforces (vLLM ``--max-model-len`` == the sidecar Context
    Length). That is the single source of truth: it is never capped by a
    separate admin setting, which would let clients (Codex, etc.) either
    overflow the real window or waste half of it.
    """
    items = data.get("data")
    if not isinstance(items, list):
        return data
    enriched = []
    for item in items:
        if not isinstance(item, dict):
            enriched.append(item)
            continue
        ctx = _int_or_none(item.get("max_model_len"))
        if ctx is not None:
            item = {**item, "context_window": ctx, "max_context_window": ctx}
        enriched.append(item)
    return {**data, "data": enriched}


def _backend_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if hasattr(exc, "response") and getattr(exc, "response") is not None:
        response = exc.response
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        return HTTPException(status_code=response.status_code, detail=detail)
    if isinstance(exc, BackendError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))
