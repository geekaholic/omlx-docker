# SPDX-License-Identifier: Apache-2.0
"""FastAPI application for the MLX-free oMLX proxy."""

from __future__ import annotations

import asyncio
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
from omlx.api.adapters.anthropic import AnthropicAdapter
from omlx.api.adapters.base import StreamChunk
from omlx.api.anthropic_models import (
    MessagesRequest,
    TokenCountRequest,
    TokenCountResponse,
)
from omlx.api.anthropic_utils import (
    convert_anthropic_to_internal,
    convert_anthropic_tools_to_internal,
    convert_internal_to_anthropic_response,
    request_has_cache_control,
)
from omlx.server_metrics import ServerMetrics

from .backend import (
    BackendError,
    OpenAIBackend,
    openai_chunk_to_stream_chunk,
    openai_response_to_internal,
)
from .admin import configure_admin
from .config import ProxyConfig
from .metrics import backend_context_limit
from .responses_adapter import (
    non_streaming_responses_response,
    responses_to_chat_body,
    stream_responses_events,
)
from .scaling import anthropic_keepalive_frame, scale_token_count
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
            internal = openai_response_to_internal(data)
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
    )

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


async def _stream_anthropic_response(
    backend: OpenAIBackend,
    config: ProxyConfig,
    request: MessagesRequest,
    openai_body: dict[str, Any],
    inbound_authorization: str | None,
    metrics: ServerMetrics | None = None,
) -> AsyncIterator[str]:
    adapter = AnthropicAdapter()
    first = True
    sent_last = False
    last_usage: tuple[int, int, int] = (0, 0, 0)
    keepalive = anthropic_keepalive_frame(config)
    request_start = time.monotonic()
    first_chunk_at: float | None = None
    seen_model = ""

    try:
        try:
            async for item in backend.stream_chat_completion(
                openai_body,
                inbound_authorization,
            ):
                if item == "[DONE]":
                    break
                if not isinstance(item, dict):
                    continue
                if first_chunk_at is None and item.get("choices"):
                    first_chunk_at = time.monotonic()
                if item.get("model"):
                    seen_model = str(item["model"])

                chunk = openai_chunk_to_stream_chunk(item, is_first=first)
                first = False
                if (
                    chunk.prompt_tokens
                    or chunk.completion_tokens
                    or chunk.cached_tokens
                ):
                    last_usage = (
                        chunk.prompt_tokens,
                        chunk.completion_tokens,
                        chunk.cached_tokens,
                    )
                if chunk.is_last:
                    chunk.prompt_tokens = scale_token_count(last_usage[0], config)
                    chunk.completion_tokens = scale_token_count(last_usage[1], config)
                    chunk.cached_tokens = scale_token_count(last_usage[2], config)
                    sent_last = True
                formatted = adapter.format_stream_chunk(chunk, request)
                if formatted:
                    yield formatted
                elif keepalive:
                    await asyncio.sleep(0)

            if not sent_last:
                prompt, completion, cached = last_usage
                yield adapter.format_stream_chunk(
                    StreamChunk(
                        is_first=first,
                        is_last=True,
                        finish_reason="stop",
                        prompt_tokens=scale_token_count(prompt, config),
                        completion_tokens=scale_token_count(completion, config),
                        cached_tokens=scale_token_count(cached, config),
                    ),
                    request,
                )
        except Exception as exc:
            yield adapter.format_error_event(str(exc))
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
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


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
    """Add context_window / max_context_window to each /v1/models entry.

    Codex (and other clients) use these fields to determine context limits
    for custom providers. Priority: admin per-model setting > admin global
    override > backend-reported max_model_len.
    """
    items = data.get("data")
    if not isinstance(items, list):
        return data
    enriched = []
    for item in items:
        if not isinstance(item, dict):
            enriched.append(item)
            continue
        model_id = str(item.get("id", ""))
        ctx: int | None = None
        if admin_state is not None:
            per_model = (admin_state.model_settings or {}).get(model_id, {}) or {}
            ctx = _int_or_none(per_model.get("max_context_window"))
            if ctx is None:
                global_ov = admin_state.global_overrides or {}
                ctx = _int_or_none(global_ov.get("sampling_max_context_window"))
        if ctx is None:
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
