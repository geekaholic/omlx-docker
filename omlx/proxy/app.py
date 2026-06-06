# SPDX-License-Identifier: Apache-2.0
"""FastAPI application for the MLX-free oMLX proxy."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
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

from .backend import (
    BackendError,
    OpenAIBackend,
    openai_chunk_to_stream_chunk,
    openai_response_to_internal,
)
from .admin import configure_admin
from .config import ProxyConfig
from .scaling import anthropic_keepalive_frame, scale_token_count

security = HTTPBearer(auto_error=False)


def create_app(
    config: ProxyConfig | None = None,
    backend: OpenAIBackend | None = None,
) -> FastAPI:
    proxy_config = config or ProxyConfig.from_env()
    backend = backend or OpenAIBackend(proxy_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.proxy_config = backend.config
        app.state.backend = backend
        yield
        await backend.close()

    app = FastAPI(
        title="oMLX Proxy",
        version=__version__,
        lifespan=lifespan,
    )
    configure_admin(app, backend, proxy_config)

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
            return await backend.get_models(
                _backend_authorization(request, backend.config)
            )
        except Exception as exc:
            raise _backend_http_exception(exc)

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
        )
        if anth_request.stream:
            return StreamingResponse(
                _stream_anthropic_response(
                    backend,
                    backend.config,
                    anth_request,
                    openai_body,
                    _backend_authorization(request, backend.config),
                ),
                media_type="text/event-stream",
            )

        try:
            data = await backend.chat_completion(
                openai_body,
                _backend_authorization(request, backend.config),
            )
            internal = openai_response_to_internal(data)
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
        return await _passthrough_backend(request, backend, path)

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
) -> AsyncIterator[str]:
    adapter = AnthropicAdapter()
    first = True
    sent_last = False
    last_usage: tuple[int, int, int] = (0, 0, 0)
    keepalive = anthropic_keepalive_frame(config)

    try:
        async for item in backend.stream_chat_completion(
            openai_body,
            inbound_authorization,
        ):
            if item == "[DONE]":
                break
            if not isinstance(item, dict):
                continue

            chunk = openai_chunk_to_stream_chunk(item, is_first=first)
            first = False
            if chunk.prompt_tokens or chunk.completion_tokens or chunk.cached_tokens:
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


def apply_proxy_request_defaults(
    body: dict[str, Any],
    state: Any,
    *,
    include_chat_template: bool = False,
) -> dict[str, Any]:
    """Apply saved admin sampling defaults before forwarding to the backend."""
    if state is None or not isinstance(body, dict):
        return body

    model_id = body.get("model")
    model_settings = {}
    if model_id is not None:
        model_settings = state.model_settings.get(str(model_id), {}) or {}
    overrides = state.global_overrides or {}
    force = bool(model_settings.get("force_sampling"))

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
        value = _first_configured(model_settings.get(request_key), overrides.get(global_key))
        if value is None:
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
    return body


def _body_with_proxy_defaults(
    body: bytes,
    state: Any,
    *,
    include_chat_template: bool = False,
) -> bytes:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body
    if not isinstance(payload, dict):
        return body
    apply_proxy_request_defaults(
        payload,
        state,
        include_chat_template=include_chat_template,
    )
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


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
) -> Response:
    body = await request.body()
    if path in {"chat/completions", "completions"}:
        body = _body_with_proxy_defaults(
            body,
            getattr(request.app.state, "proxy_admin_state", None),
            include_chat_template=path == "chat/completions",
        )
    headers = _backend_headers(request, backend)
    wants_stream = _body_requests_stream(body)

    if wants_stream:
        try:
            response = await backend.raw_stream(
                request.method,
                path,
                body,
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

        return StreamingResponse(
            iterator(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/event-stream"),
        )

    try:
        response = await backend.raw_request(request.method, path, body, headers)
    except Exception as exc:
        raise _backend_http_exception(exc)
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
