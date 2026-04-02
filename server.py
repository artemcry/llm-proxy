import uvicorn
import httpx
import asyncio
import base64
import hashlib
import json
import os
import sys
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("llmproxy")

load_dotenv()

MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY")
GEMINI_KEY  = os.environ.get("GEMINI_API_KEY")

if not MINIMAX_KEY: sys.exit("ERROR: MINIMAX_API_KEY not found in .env")
if not GEMINI_KEY:  sys.exit("ERROR: GEMINI_API_KEY not found in .env")

gemini_client = genai.Client(api_key=GEMINI_KEY)

SERVER_HOST          = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT          = int(os.environ.get("SERVER_PORT", "8087"))

# MiniMax endpoints (two formats)
MINIMAX_OPENAI_URL    = os.environ.get("MINIMAX_URL", "https://api.minimax.io/v1/chat/completions")
MINIMAX_ANTHROPIC_URL = os.environ.get("MINIMAX_ANTHROPIC_URL", "https://api.minimax.io/anthropic")
MINIMAX_MODEL         = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")

IMAGE_CACHE_MAX      = int(os.environ.get("IMAGE_CACHE_MAX", "256"))
GEMINI_MODEL         = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_API_URL    = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com")

# Models that should be intercepted and rerouted to MiniMax.
# Any model containing these substrings (case-insensitive) gets redirected.
# Default: "haiku" — so claude-3-haiku, claude-3-5-haiku, etc. all go to MiniMax.
# Set INTERCEPT_MODELS="" in .env to disable interception.
INTERCEPT_MODELS = [s.strip().lower() for s in os.environ.get("INTERCEPT_MODELS", "haiku").split(",") if s.strip()]

_image_cache: OrderedDict[str, str] = OrderedDict()
http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=300)
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)


def should_route_to_minimax(model: str) -> bool:
    """Check if the requested model should be routed to MiniMax."""
    m = model.lower()
    logger.info("MODEL", m)
    logger.info("MODEL CHECK", m.startswith("minimax"))
    if m.startswith("minimax"):
        return True
    for pattern in INTERCEPT_MODELS:
        if pattern in m:
            return True
    return False


def sanitize_body_for_minimax(body: dict, original_model: str) -> dict:
    """Convert Claude Code request body into something MiniMax accepts.

    Uses a whitelist: only fields MiniMax Anthropic API supports are kept.
    Everything else (metadata, thinking, budget_tokens, etc.) is dropped.
    Also replaces model identity in system prompt so MiniMax doesn't claim to be Claude.
    """
    # Whitelist of fields MiniMax Anthropic API supports
    ALLOWED_FIELDS = {
        "model", "messages", "system", "max_tokens", "temperature",
        "top_p", "stream", "stop_sequences", "tools", "tool_choice",
    }

    # Log what Claude Code sent vs what we keep
    all_keys = set(body.keys())
    dropped = all_keys - ALLOWED_FIELDS
    if dropped:
        logger.info(f"    Dropping unsupported fields: {dropped}")

    clean = {k: v for k, v in body.items() if k in ALLOWED_FIELDS}

    # ── Convert system from array to plain string ──
    system = clean.get("system")
    if isinstance(system, list):
        parts = []
        for item in system:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        clean["system"] = "\n\n".join(p for p in parts if p)
        logger.info(f"    Converted system[] ({len(system)} parts) -> string ({len(clean['system'])} chars)")

    # ── Replace model identity in system prompt ──
    # Claude Code puts "You are claude-haiku-4-5-20251001" etc. in system.
    # MiniMax reads that and says "I'm Claude Haiku" — we need to fix this.
    if isinstance(clean.get("system"), str) and original_model:
        sys_text = clean["system"]
        # Replace specific model ID
        if original_model in sys_text:
            sys_text = sys_text.replace(original_model, MINIMAX_MODEL)
            logger.info(f"    Replaced model id '{original_model}' in system prompt")
        # Replace common Claude model family names
        for pattern in ("claude-haiku", "claude-sonnet", "claude-opus", "Claude Haiku", "Claude Sonnet", "Claude Opus"):
            if pattern in sys_text:
                sys_text = sys_text.replace(pattern, MINIMAX_MODEL)
        clean["system"] = sys_text

    # ── Remove thinking blocks and cache_control from message content parts ──
    # thinking blocks carry Anthropic-specific signatures that MiniMax doesn't
    # understand — sending them causes 400 errors or silent corruption.
    messages = clean.get("messages", [])
    cleaned_messages = []
    for msg in messages:
        msg = {**msg}
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if not isinstance(part, dict):
                    new_content.append(part)
                    continue
                # Drop thinking blocks entirely — MiniMax rejects them
                if part.get("type") == "thinking":
                    logger.info("    Dropped thinking block from message history")
                    continue
                # Strip cache_control
                if "cache_control" in part:
                    part = {k: v for k, v in part.items() if k != "cache_control"}
                new_content.append(part)
            # Avoid sending an assistant message with empty content
            if not new_content:
                new_content = [{"type": "text", "text": ""}]
            msg["content"] = new_content
        cleaned_messages.append(msg)
    clean["messages"] = cleaned_messages

    logger.info(f"    Sanitized body keys: {sorted(clean.keys())}")
    return clean


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {"object": "list", "data": [{"id": MINIMAX_MODEL, "object": "model"}]}


# ══════════════════════════════════════════════════════════════════════════════
#  Debug endpoint — test connectivity
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/debug/status")
async def debug_status():
    """Quick connectivity test for both MiniMax endpoints."""
    results = {"proxy": "ok", "intercept_models": INTERCEPT_MODELS}

    # Test MiniMax Anthropic endpoint
    try:
        resp = await http_client.post(
            f"{MINIMAX_ANTHROPIC_URL.rstrip('/')}/v1/messages",
            headers={
                "Authorization": f"Bearer {MINIMAX_KEY}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": MINIMAX_MODEL,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=15,
        )
        results["minimax_anthropic"] = {
            "status": resp.status_code,
            "ok": resp.status_code == 200,
            "body_preview": resp.text[:200],
        }
    except Exception as e:
        results["minimax_anthropic"] = {"error": str(e)}

    # Test MiniMax OpenAI endpoint
    try:
        resp = await http_client.post(
            MINIMAX_OPENAI_URL,
            headers={"Authorization": f"Bearer {MINIMAX_KEY}"},
            json={
                "model": MINIMAX_MODEL,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=15,
        )
        results["minimax_openai"] = {
            "status": resp.status_code,
            "ok": resp.status_code == 200,
            "body_preview": resp.text[:200],
        }
    except Exception as e:
        results["minimax_openai"] = {"error": str(e)}

    return JSONResponse(results)


# ══════════════════════════════════════════════════════════════════════════════
#  Image helpers (shared by both API formats)
# ══════════════════════════════════════════════════════════════════════════════

async def describe_image(image_bytes: bytes, mime_type: str) -> str:
    key = hashlib.sha256(image_bytes).hexdigest()
    if key in _image_cache:
        _image_cache.move_to_end(key)
        logger.info(f"    [cache hit] {key[:12]}")
        return _image_cache[key]

    logger.info(f"    Gemini <- {mime_type} {len(image_bytes)//1024}KB")
    response = await gemini_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "Describe in detail what you see in this image. "
            "If it contains code, UI, error messages, or terminal output — "
            "reproduce the text content exactly, then describe the layout."
        ],
    )
    description = response.text
    logger.info(f"    Gemini -> {len(description)} chars")
    _image_cache[key] = description
    if len(_image_cache) > IMAGE_CACHE_MAX:
        _image_cache.popitem(last=False)
    return description


QUOTA_ERROR_MSG = (
    "Limit Gemini Vision API вичерпано. "
    "Зачекайте до скидання квоти або увімкніть білінг в Google AI Studio."
)


# ══════════════════════════════════════════════════════════════════════════════
#  Anthropic format — image handling  (Claude Code / Anthropic clients)
# ══════════════════════════════════════════════════════════════════════════════

def has_images_anthropic(messages: list) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    return True
    return False


async def extract_image_bytes_anthropic(source: dict) -> tuple[bytes, str]:
    src_type = source.get("type")
    if src_type == "base64":
        return base64.b64decode(source.get("data", "")), source.get("media_type", "image/png")
    if src_type == "url":
        resp = await http_client.get(source.get("url", ""))
        resp.raise_for_status()
        return resp.content, resp.headers.get("content-type", "image/png").split(";")[0]
    raise ValueError(f"Unsupported image source type: {src_type}")


async def replace_images_anthropic(messages: list) -> tuple[list, bool]:
    """Replace Anthropic image blocks with Gemini text descriptions."""
    image_tasks = []
    msg_structures = []

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            msg_structures.append((msg, None))
            continue
        parts_info = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                try:
                    img_bytes, mime_type = await extract_image_bytes_anthropic(part.get("source", {}))
                    parts_info.append(("img", len(image_tasks)))
                    image_tasks.append(describe_image(img_bytes, mime_type))
                except Exception as e:
                    parts_info.append(("text", f"[Image error: {e}]"))
            else:
                parts_info.append(("keep", part))
        msg_structures.append((msg, parts_info))

    results = await asyncio.gather(*image_tasks, return_exceptions=True) if image_tasks else []

    quota_exceeded = False
    new_messages = []
    for msg, parts_info in msg_structures:
        if parts_info is None:
            new_messages.append(msg)
            continue
        new_parts = []
        for kind, value in parts_info:
            if kind == "keep":
                new_parts.append(value)
            elif kind == "text":
                new_parts.append({"type": "text", "text": value})
            elif kind == "img":
                result = results[value]
                if isinstance(result, Exception):
                    if "429" in str(result) or "RESOURCE_EXHAUSTED" in str(result):
                        quota_exceeded = True
                    new_parts.append({"type": "text", "text": f"[Image could not be described: {result}]"})
                else:
                    new_parts.append({"type": "text", "text": f"<image>\n{result}\n</image>"})
        if len(new_parts) == 1 and new_parts[0].get("type") == "text":
            new_messages.append({**msg, "content": new_parts[0]["text"]})
        else:
            new_messages.append({**msg, "content": new_parts})

    return new_messages, quota_exceeded


# ══════════════════════════════════════════════════════════════════════════════
#  OpenAI format — image handling  (standalone MiniMax client)
# ══════════════════════════════════════════════════════════════════════════════

def has_images_openai(messages: list) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    return True
    return False


async def extract_image_bytes_openai(image_url_value: dict) -> tuple[bytes, str]:
    url = image_url_value.get("url", "")
    if url.startswith("data:"):
        header, b64data = url.split(",", 1)
        mime_type = header.split(";")[0].replace("data:", "")
        return base64.b64decode(b64data), mime_type
    if url.startswith(("http://", "https://")):
        resp = await http_client.get(url)
        resp.raise_for_status()
        mime_type = resp.headers.get("content-type", "image/png").split(";")[0]
        return resp.content, mime_type
    raise ValueError(f"Unsupported image URL: {url[:60]}")


async def replace_images_openai(messages: list) -> tuple[list, bool]:
    """Replace OpenAI image_url parts with Gemini descriptions. Flattens content to string."""
    image_tasks = []
    msg_structures = []

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            msg_structures.append((msg, None))
            continue

        parts_info = []
        for part in content:
            if not isinstance(part, dict):
                parts_info.append(("lit", str(part)))
            elif part.get("type") == "image_url":
                try:
                    img_bytes, mime_type = await extract_image_bytes_openai(part["image_url"])
                    parts_info.append(("img", len(image_tasks)))
                    image_tasks.append(describe_image(img_bytes, mime_type))
                except (ValueError, httpx.HTTPError) as e:
                    parts_info.append(("lit", f"[Image error: {e}]"))
            elif part.get("type") == "text":
                parts_info.append(("lit", part.get("text", "")))
            else:
                parts_info.append(("lit", str(part)))
        msg_structures.append((msg, parts_info))

    results = await asyncio.gather(*image_tasks, return_exceptions=True) if image_tasks else []

    quota_exceeded = False
    new_messages = []
    for msg, parts_info in msg_structures:
        if parts_info is None:
            new_messages.append(msg)
            continue
        text_parts = []
        for kind, value in parts_info:
            if kind == "lit":
                text_parts.append(value)
            elif kind == "img":
                result = results[value]
                if isinstance(result, Exception):
                    if "429" in str(result) or "RESOURCE_EXHAUSTED" in str(result):
                        quota_exceeded = True
                    text_parts.append(f"[Image could not be described: {result}]")
                else:
                    text_parts.append(f"[Image description by Gemini Vision]:\n{result}")
        new_messages.append({**msg, "content": "\n\n".join(text_parts)})
    return new_messages, quota_exceeded


# ══════════════════════════════════════════════════════════════════════════════
#  Streaming helpers
# ══════════════════════════════════════════════════════════════════════════════

async def _forward_stream(url: str, headers: dict, body: dict):
    """Forward an Anthropic-format stream (MiniMax path — body is a dict)."""
    first_chunk = True
    try:
        async with http_client.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                logger.error(f"  Stream error {resp.status_code}: {err[:300]}")
                err_msg = f"Upstream {resp.status_code}: {err[:100].decode(errors='replace')}"
                yield f"data: {json.dumps({'type':'error','error':{'type':'api_error','message':err_msg}})}\n\n".encode()
                return
            async for chunk in resp.aiter_bytes():
                if first_chunk:
                    logger.info(f"  Stream first chunk: {len(chunk)}B — {chunk[:120]}")
                    first_chunk = False
                yield chunk
    except httpx.ReadTimeout:
        logger.error("  Stream timeout")
        yield f"data: {json.dumps({'type':'error','error':{'type':'api_error','message':'timeout'}})}\n\n".encode()
    except httpx.HTTPError as e:
        logger.error(f"  Stream HTTP error: {e}")
        yield f"data: {json.dumps({'type':'error','error':{'type':'api_error','message':str(e)}})}\n\n".encode()


async def _forward_stream_raw(url: str, headers: dict, raw_body: bytes):
    """Forward an Anthropic-format stream using raw bytes (Anthropic passthrough).

    Uses content= instead of json= so thinking block signatures are never
    re-serialized and cannot be corrupted.
    """
    first_chunk = True
    try:
        async with http_client.stream("POST", url, headers=headers, content=raw_body) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                logger.error(f"  Stream error {resp.status_code}: {err[:300]}")
                err_msg = f"Upstream {resp.status_code}: {err[:100].decode(errors='replace')}"
                yield f"data: {json.dumps({'type':'error','error':{'type':'api_error','message':err_msg}})}\n\n".encode()
                return
            async for chunk in resp.aiter_bytes():
                if first_chunk:
                    logger.info(f"  Stream first chunk: {len(chunk)}B — {chunk[:120]}")
                    first_chunk = False
                yield chunk
    except httpx.ReadTimeout:
        logger.error("  Stream timeout")
        yield f"data: {json.dumps({'type':'error','error':{'type':'api_error','message':'timeout'}})}\n\n".encode()
    except httpx.HTTPError as e:
        logger.error(f"  Stream HTTP error: {e}")
        yield f"data: {json.dumps({'type':'error','error':{'type':'api_error','message':str(e)}})}\n\n".encode()


async def _stream_openai(body: dict):
    """Forward an OpenAI-format stream to MiniMax."""
    first_chunk = True
    try:
        async with http_client.stream(
            "POST", MINIMAX_OPENAI_URL,
            headers={"Authorization": f"Bearer {MINIMAX_KEY}"},
            json=body,
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.error(f"  MiniMax stream error: {resp.status_code} {error_body[:300]}")
                yield f"data: {json.dumps({'error': {'message': f'MiniMax error {resp.status_code}', 'code': resp.status_code}})}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return
            async for chunk in resp.aiter_bytes():
                if first_chunk:
                    logger.info(f"  OpenAI stream first chunk: {len(chunk)}B — {chunk[:120]}")
                    first_chunk = False
                yield chunk
    except httpx.ReadTimeout:
        logger.error("  MiniMax stream ReadTimeout")
        yield f"data: {json.dumps({'error': {'message': 'MiniMax read timeout', 'code': 504}})}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except httpx.HTTPError as e:
        logger.error(f"  MiniMax stream error: {e}")
        yield f"data: {json.dumps({'error': {'message': str(e), 'code': 502}})}\n\n".encode()
        yield b"data: [DONE]\n\n"


# ══════════════════════════════════════════════════════════════════════════════
#  Anthropic passthrough (for real Claude models — sonnet/opus etc.)
# ══════════════════════════════════════════════════════════════════════════════

async def proxy_to_anthropic(request: Request, raw_body: bytes, stream: bool):
    """Forward request directly to Anthropic API using original raw bytes.

    Critically, we do NOT re-serialize the body — passing raw bytes preserves
    thinking block signatures exactly as Claude Code sent them, preventing
    'Invalid signature in thinking block' 400 errors on multi-turn conversations.
    """
    target = f"{ANTHROPIC_API_URL.rstrip('/')}/v1/messages"

    api_key = request.headers.get("x-api-key", "")
    auth_header = request.headers.get("authorization", "")
    is_oauth = auth_header.lower().startswith("bearer ")

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
    }

    if is_oauth:
        # OAuth: forward Authorization: Bearer <token> as-is
        headers["authorization"] = auth_header
        logger.info("  Auth: OAuth Bearer token")
    elif api_key:
        # API key: use x-api-key header
        headers["x-api-key"] = api_key
        logger.info("  Auth: x-api-key")
    else:
        logger.warning("  Auth: no credentials found in request!")

    for key, value in request.headers.items():
        if key.lower().startswith("anthropic-") and key.lower() not in headers:
            headers[key] = value

    body_preview = json.loads(raw_body)
    model = body_preview.get("model", "?")
    logger.info(f"  -> Anthropic API ({model})")

    if stream:
        return StreamingResponse(
            _forward_stream_raw(target, headers, raw_body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        resp = await http_client.post(target, headers=headers, content=raw_body)
        logger.info(f"  Anthropic {resp.status_code} ({len(resp.content)}B)")
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception as e:
        logger.error(f"  Anthropic ERROR: {e}")
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": str(e)}},
            status_code=502,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Quota error responses
# ══════════════════════════════════════════════════════════════════════════════

def _quota_error_anthropic(stream: bool):
    if stream:
        async def _qs():
            yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':'msg_quota','type':'message','role':'assistant','model':MINIMAX_MODEL,'content':[],'stop_reason':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':QUOTA_ERROR_MSG}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{'output_tokens':0}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"
        return StreamingResponse(_qs(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    return JSONResponse({
        "id": "msg_quota", "type": "message", "role": "assistant", "model": MINIMAX_MODEL,
        "content": [{"type": "text", "text": QUOTA_ERROR_MSG}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 0, "output_tokens": 0}
    })


def _quota_error_openai(stream: bool):
    if stream:
        async def _qs():
            chunk = {
                "id": "quota_error", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": QUOTA_ERROR_MSG}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_qs(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    return JSONResponse({
        "id": "quota_error", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": QUOTA_ERROR_MSG}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: Anthropic /v1/messages  (Claude Code + any Anthropic client)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/messages")
@app.post("/messages")
async def proxy_messages(request: Request):
    raw_body = await request.body()          # keep original bytes for Anthropic passthrough
    body = json.loads(raw_body)              # parse for routing logic
    model = body.get("model", "")
    messages = body.get("messages", [])
    images = has_images_anthropic(messages)
    stream = body.get("stream", False)

    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    route_minimax = should_route_to_minimax(model)

    # Diagnostic: log what Claude Code actually sends
    system_type = type(body.get("system")).__name__
    has_beta = any(k.lower().startswith("anthropic-beta") for k in request.headers.keys())
    logger.info(
        f"\n[{ts}] [Anthropic] {model} -> {'MiniMax' if route_minimax else 'Anthropic'} "
        f"| msgs={len(messages)} img={'Y' if images else 'N'} stream={'Y' if stream else 'N'} "
        f"| system={system_type} beta={'Y' if has_beta else 'N'}"
    )
    if has_beta:
        beta_val = request.headers.get("anthropic-beta", "")
        logger.info(f"    anthropic-beta: {beta_val}")

    # ── Route: real Claude models go straight to Anthropic ──
    if not route_minimax:
        return await proxy_to_anthropic(request, raw_body, stream)

    # ── Route: MiniMax (explicit or intercepted haiku etc.) ──
    logger.info(f"    Intercepted '{model}' -> {MINIMAX_MODEL}")

    if images:
        logger.info("  Describing images via Gemini...")
        messages, quota_exceeded = await replace_images_anthropic(messages)
        if quota_exceeded:
            return _quota_error_anthropic(stream)
        body = {**body, "messages": messages}

    # Sanitize body: remove Anthropic-beta features MiniMax doesn't support
    body = sanitize_body_for_minimax(body, original_model=model)
    body["model"] = MINIMAX_MODEL

    target = f"{MINIMAX_ANTHROPIC_URL.rstrip('/')}/v1/messages"
    # NOTE: do NOT forward anthropic-beta headers to MiniMax
    headers = {
        "Authorization": f"Bearer {MINIMAX_KEY}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    logger.info(f"    -> {target} (stream={stream})")

    if stream:
        return StreamingResponse(
            _forward_stream(target, headers, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        resp = await http_client.post(target, headers=headers, json=body)
        logger.info(f"  MiniMax {resp.status_code} ({len(resp.content)}B)")
        if resp.status_code != 200:
            logger.error(f"  MiniMax error body: {resp.text[:300]}")
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except httpx.HTTPStatusError as e:
        logger.error(f"  MiniMax {e.response.status_code}: {e.response.text[:200]}")
        return JSONResponse(e.response.json(), status_code=e.response.status_code)
    except Exception as e:
        logger.error(f"  ERROR: {e}")
        return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(e)}}, status_code=502)


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: OpenAI /v1/chat/completions  (standalone MiniMax client)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    images = has_images_openai(messages)
    stream = body.get("stream", False)

    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    tools = body.get("tools", [])
    tool_names = ", ".join(t.get("function", {}).get("name", "?") for t in tools[:5]) if tools else "-"
    logger.info(
        f"\n[{ts}] [OpenAI] {body.get('model','?')} | msgs={len(messages)} img={'Y' if images else 'N'} "
        f"stream={'Y' if stream else 'N'} tools={tool_names}"
    )

    if images:
        logger.info("  Describing images via Gemini...")
        messages, quota_exceeded = await replace_images_openai(messages)
        if quota_exceeded:
            return _quota_error_openai(stream)
        body["messages"] = messages

    body = {**body, "model": MINIMAX_MODEL, "stream": stream}

    if stream:
        return StreamingResponse(
            _stream_openai(body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        resp = await http_client.post(
            MINIMAX_OPENAI_URL,
            headers={"Authorization": f"Bearer {MINIMAX_KEY}"},
            json={**body, "stream": False},
        )
        logger.info(f"  MiniMax {resp.status_code} ({len(resp.content)}B)")
        resp.raise_for_status()
        result = resp.json()
        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        reply = msg.get("content", "") or ""
        tc = msg.get("tool_calls")
        logger.info(f"  < {choice.get('finish_reason')} | tc={'Y' if tc else 'N'} | {len(reply)}ch")
        if tc and msg.get("content"):
            msg["content"] = None
        return JSONResponse(result)
    except httpx.HTTPStatusError as e:
        logger.error(f"  MiniMax {e.response.status_code}: {e.response.text[:200]}")
        return JSONResponse({"error": str(e)}, status_code=502)
    except Exception as e:
        logger.error(f"  ERROR: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    intercept_info = ", ".join(INTERCEPT_MODELS) if INTERCEPT_MODELS else "none"
    logger.info(
        f"LLMProxy on {SERVER_HOST}:{SERVER_PORT} | MiniMax ...{MINIMAX_KEY[-8:]} "
        f"| Gemini ...{GEMINI_KEY[-8:]} | intercept: [{intercept_info}]"
    )
    logger.info(f"  Anthropic passthrough: {ANTHROPIC_API_URL}")
    logger.info(f"  MiniMax Anthropic:     {MINIMAX_ANTHROPIC_URL}")
    logger.info(f"  MiniMax OpenAI:        {MINIMAX_OPENAI_URL}")
    logger.info(f"  Debug:                 http://{SERVER_HOST}:{SERVER_PORT}/debug/status")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)