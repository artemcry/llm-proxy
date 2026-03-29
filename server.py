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
MINIMAX_ANTHROPIC_URL = os.environ.get("MINIMAX_ANTHROPIC_URL", "https://api.minimax.io/anthropic")
MINIMAX_MODEL        = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")
IMAGE_CACHE_MAX      = int(os.environ.get("IMAGE_CACHE_MAX", "256"))
GEMINI_MODEL         = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_API_URL    = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com")

_image_cache: OrderedDict[str, str] = OrderedDict()
http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=300)
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)


def is_minimax_model(model: str) -> bool:
    """Check if the requested model should be routed to MiniMax."""
    return model.lower().startswith("minimax")


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {"object": "list", "data": [{"id": MINIMAX_MODEL, "object": "model"}]}


# ── Image helpers ─────────────────────────────────────────────────────────────

def has_images(messages: list) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    return True
    return False


async def extract_image_bytes(source: dict) -> tuple[bytes, str]:
    src_type = source.get("type")
    if src_type == "base64":
        return base64.b64decode(source.get("data", "")), source.get("media_type", "image/png")
    if src_type == "url":
        resp = await http_client.get(source.get("url", ""))
        resp.raise_for_status()
        return resp.content, resp.headers.get("content-type", "image/png").split(";")[0]
    raise ValueError(f"Unsupported image source type: {src_type}")


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


async def replace_images(messages: list) -> tuple[list, bool]:
    """Replace Anthropic image blocks with Gemini text. Keep all other blocks intact."""
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
                    img_bytes, mime_type = await extract_image_bytes(part.get("source", {}))
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
                    new_parts.append({
                    "type": "text",
                    "text": f"<image>\n{result}\n</image>"
                })
        # Simplify to plain string if only one text block
        if len(new_parts) == 1 and new_parts[0].get("type") == "text":
            new_messages.append({**msg, "content": new_parts[0]["text"]})
        else:
            new_messages.append({**msg, "content": new_parts})

    return new_messages, quota_exceeded


QUOTA_ERROR_MSG = (
    "Limit Gemini Vision API вичерпано. "
    "Зачекайте до скидання квоти або увімкніть білінг в Google AI Studio."
)


# ── Anthropic passthrough (for Claude models) ────────────────────────────────

async def proxy_to_anthropic(request: Request, body: dict, stream: bool):
    """Forward request directly to Anthropic API, using the auth token from the original request."""
    target = f"{ANTHROPIC_API_URL.rstrip('/')}/v1/messages"

    # Extract API key: Claude Code sends "Authorization: Bearer sk-..." but
    # Anthropic API expects "x-api-key: sk-..." header
    api_key = request.headers.get("x-api-key", "")
    if not api_key:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            api_key = auth_header[7:].strip()

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
    }
    if api_key:
        headers["x-api-key"] = api_key

    # Forward any other anthropic-* headers
    for key, value in request.headers.items():
        if key.lower().startswith("anthropic-") and key.lower() not in headers:
            headers[key] = value

    model = body.get("model", "?")
    logger.info(f"  -> Anthropic API ({model})")

    if stream:
        return StreamingResponse(
            _forward_stream(target, headers, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        resp = await http_client.post(target, headers=headers, json=body)
        logger.info(f"  Anthropic {resp.status_code} ({len(resp.content)}B)")
        # Pass through the raw response (including errors) so Claude Code sees real messages
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception as e:
        logger.error(f"  Anthropic ERROR: {e}")
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": str(e)}},
            status_code=502,
        )


# ── Proxy endpoint ────────────────────────────────────────────────────────────

@app.post("/v1/messages")
@app.post("/messages")
async def proxy_messages(request: Request):
    body = await request.json()
    model = body.get("model", "")
    messages = body.get("messages", [])
    images = has_images(messages)
    stream = body.get("stream", False)

    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    logger.info(f"\n[{ts}] {model} | msgs={len(messages)} img={'Y' if images else 'N'} stream={'Y' if stream else 'N'}")

    # ── Route: non-MiniMax models go straight to Anthropic ──
    if not is_minimax_model(model):
        return await proxy_to_anthropic(request, body, stream)

    # ── Route: MiniMax — handle images via Gemini, then forward ──
    if images:
        logger.info("  Describing images via Gemini...")
        messages, quota_exceeded = await replace_images(messages)
        if quota_exceeded:
            return _quota_error(stream)
        body = {**body, "messages": messages}

    body = {**body, "model": MINIMAX_MODEL}
    target = f"{MINIMAX_ANTHROPIC_URL.rstrip('/')}/v1/messages"
    headers = {
        "Authorization": f"Bearer {MINIMAX_KEY}",
        "Content-Type": "application/json",
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
    }

    if stream:
        return StreamingResponse(
            _forward_stream(target, headers, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        resp = await http_client.post(target, headers=headers, json=body)
        logger.info(f"  MiniMax {resp.status_code} ({len(resp.content)}B)")
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except httpx.HTTPStatusError as e:
        logger.error(f"  MiniMax {e.response.status_code}: {e.response.text[:200]}")
        return JSONResponse(e.response.json(), status_code=e.response.status_code)
    except Exception as e:
        logger.error(f"  ERROR: {e}")
        return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(e)}}, status_code=502)


async def _forward_stream(url: str, headers: dict, body: dict):
    try:
        async with http_client.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                logger.error(f"  Stream error {resp.status_code}: {err[:200]}")
                yield f"data: {json.dumps({'type':'error','error':{'type':'api_error','message':f'Upstream {resp.status_code}'}})}\n\n".encode()
                return
            async for chunk in resp.aiter_bytes():
                yield chunk
    except httpx.ReadTimeout:
        logger.error("  Stream timeout")
        yield f"data: {json.dumps({'type':'error','error':{'type':'api_error','message':'timeout'}})}\n\n".encode()
    except httpx.HTTPError as e:
        logger.error(f"  Stream HTTP error: {e}")
        yield f"data: {json.dumps({'type':'error','error':{'type':'api_error','message':str(e)}})}\n\n".encode()


def _quota_error(stream: bool):
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


if __name__ == "__main__":
    logger.info(f"LLMProxy on {SERVER_HOST}:{SERVER_PORT} | MiniMax ...{MINIMAX_KEY[-8:]} | Gemini ...{GEMINI_KEY[-8:]}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)