"""
Client gọi OpenAI API – singleton client, hỗ trợ streaming (SSE).
"""

from __future__ import annotations

import logging
import re
from typing import Any, AsyncGenerator, List, Optional

from openai import AsyncOpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL, MAX_TOKENS

log = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(text: str) -> str:
    """Strip control characters that break JSON serialization."""
    if not text:
        return text
    return _CONTROL_CHARS_RE.sub("", text)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        kwargs: dict = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        _client = AsyncOpenAI(**kwargs)
    return _client


async def generate_with_messages_stream(
    messages: List[dict[str, Any]],
    temperature: float = 0.5,
    model: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Chat completion streaming — yield từng delta text."""
    client = _get_client()
    safe: List[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = _sanitize_text(str(m.get("content", "")))
        if content:
            safe.append({"role": role, "content": content})
    if not safe:
        return
    stream = await client.chat.completions.create(
        model=model or OPENAI_MODEL,
        messages=safe,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def generate_stream(
    prompt: str,
    system: str = "",
    temperature: float = 0.5,
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Gọi OpenAI ChatCompletion với stream=True.
    Yield từng token (text) cho SSE.
    """
    client = _get_client()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": _sanitize_text(system)})
    messages.append({"role": "user", "content": _sanitize_text(prompt)})

    stream = await client.chat.completions.create(
        model=model or OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def generate_json_object(
    user_prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Chat completion với ``response_format: json_object`` — trả dict đã parse."""
    import json as _json

    client = _get_client()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": _sanitize_text(system)})
    messages.append({"role": "user", "content": _sanitize_text(user_prompt)})

    resp = await client.chat.completions.create(
        model=model or OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        return _json.loads(raw)
    except Exception:
        log.warning("generate_json_object: invalid JSON from model")
        return {}


async def generate(
    prompt: str,
    system: str = "",
    temperature: float = 0.5,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Gọi OpenAI và trả toàn bộ response (non-stream)."""
    client = _get_client()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": _sanitize_text(system)})
    messages.append({"role": "user", "content": _sanitize_text(prompt)})

    try:
        mt = max_tokens if max_tokens is not None else MAX_TOKENS
        resp = await client.chat.completions.create(
            model=model or OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=mt,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        log.error("OpenAI generate failed: %s", exc)
        raise


async def generate_with_messages(
    messages: List[dict[str, Any]],
    temperature: float = 0.5,
    model: Optional[str] = None,
) -> str:
    """Chat completion với mảng messages (system + lịch sử + user)."""
    client = _get_client()
    safe: List[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = _sanitize_text(str(m.get("content", "")))
        if content:
            safe.append({"role": role, "content": content})
    if not safe:
        return ""
    try:
        resp = await client.chat.completions.create(
            model=model or OPENAI_MODEL,
            messages=safe,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        log.error("OpenAI generate_with_messages failed: %s", exc)
        raise


async def list_models() -> list[str]:
    """Lấy danh sách model từ OpenAI."""
    client = _get_client()
    models = await client.models.list()
    return [m.id for m in models.data]
