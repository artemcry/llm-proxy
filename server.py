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

# Setup logger - will be configured by server_bg.py or use defaults
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("llmproxy")

load_dotenv()

MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

if not MINIMAX_KEY:
    sys.exit("ERROR: MINIMAX_API_KEY not found in .env")
if not GEMINI_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not found in .env")

gemini_client = genai.Client(api_key=GEMINI_KEY)

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8087"))

MINIMAX_URL = os.environ.get("MINIMAX_URL", "https://api.minimax.io/v1/chat/completions")
MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")
IMAGE_CACHE_MAX = int(os.environ.get("IMAGE_CACHE_MAX", "256"))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_image_cache: OrderedDict[str, str] = OrderedDict()

http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=300)
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MINIMAX_MODEL, "object": "model"}]
    }


def has_images(messages: list) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    return True
    return False


def last_message_preview(messages: list) -> str:
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")[:150]
        return str(content)[:150]
    return str(content)[:150]


async def extract_image_bytes(image_url_value: dict) -> tuple[bytes, str]:
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
        config=types.GenerateContentConfig()
    )
    description = response.text
    logger.info(f"    Gemini -> {len(description)} chars")

    _image_cache[key] = description
    if len(_image_cache) > IMAGE_CACHE_MAX:
        _image_cache.popitem(last=False)
    return description


async def replace_images(messages: list) -> tuple[list, bool]:
    """Replace image parts with Gemini descriptions (parallel). Flattens content to string."""
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
                    img_bytes, mime_type = await extract_image_bytes(part["image_url"])
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
                    err = str(result)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        quota_exceeded = True
                    logger.info(f"    Image error: {result}")
                    text_parts.append(f"[Image could not be described: {result}]")
                else:
                    text_parts.append(f"[Image description by Gemini Vision]:\n{result}")

        new_messages.append({**msg, "content": "\n\n".join(text_parts)})
    return new_messages, quota_exceeded


async def forward_to_minimax(body: dict) -> dict:
    body = {**body, "model": MINIMAX_MODEL, "stream": False}
    resp = await http_client.post(
        MINIMAX_URL,
        headers={"Authorization": f"Bearer {MINIMAX_KEY}"},
        json=body
    )
    logger.info(f"  MiniMax {resp.status_code} ({len(resp.content)}B)")
    resp.raise_for_status()
    return resp.json()


async def stream_from_minimax(body: dict):
    body = {**body, "model": MINIMAX_MODEL, "stream": True}
    try:
        async with http_client.stream(
            "POST", MINIMAX_URL,
            headers={"Authorization": f"Bearer {MINIMAX_KEY}"},
            json=body
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.error(f"  MiniMax stream error: {resp.status_code} {error_body[:200]}")
                yield f"data: {json.dumps({'error': {'message': f'MiniMax error {resp.status_code}', 'code': resp.status_code}})}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return
            async for chunk in resp.aiter_bytes():
                yield chunk
    except httpx.ReadTimeout:
        logger.error("  MiniMax stream ReadTimeout")
        yield f"data: {json.dumps({'error': {'message': 'MiniMax read timeout', 'code': 504}})}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except httpx.HTTPError as e:
        logger.error(f"  MiniMax stream error: {e}")
        yield f"data: {json.dumps({'error': {'message': str(e), 'code': 502}})}\n\n".encode()
        yield b"data: [DONE]\n\n"


QUOTA_ERROR_MSG = (
    "Limit Gemini Vision API вичерпано. "
    "Зачекайте до скидання квоти або увімкніть білінг в Google AI Studio."
)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    images = has_images(messages)
    stream = body.get("stream", False)

    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    tools = body.get("tools", [])
    tool_names = ", ".join(t.get("function", {}).get("name", "?") for t in tools[:5]) if tools else "-"
    logger.info(f"\n[{ts}] {body.get('model','?')} | msgs={len(messages)} img={'Y' if images else 'N'} "
          f"stream={'Y' if stream else 'N'} tools={tool_names}")

    if images:
        logger.info("  Describing images...")
        messages, quota_exceeded = await replace_images(messages)
        if quota_exceeded:
            logger.warning("  Quota exceeded!")
            if stream:
                chunk_response = {
                    "id": "quota_error",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": QUOTA_ERROR_MSG}, "finish_reason": "stop"}],
                }
                async def _qs():
                    yield f"data: {json.dumps(chunk_response)}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(_qs(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
            error_response = {
                "id": "quota_error",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": QUOTA_ERROR_MSG}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
            return JSONResponse(error_response)
        body["messages"] = messages

    if stream:
        return StreamingResponse(
            stream_from_minimax(body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        result = await forward_to_minimax(body)
        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        reply = msg.get("content", "") or ""
        tc = msg.get("tool_calls")
        logger.info(f"  < {choice.get('finish_reason')} | tc={'Y' if tc else 'N'} | {len(reply)}ch")
        if tc and msg.get("content"):
            msg["content"] = None
        return JSONResponse(result)

    except httpx.HTTPStatusError as e:
        logger.error(f"  ERROR: MiniMax {e.response.status_code}: {e.response.text[:200]}")
        return JSONResponse({"error": str(e)}, status_code=502)
    except Exception as e:
        logger.error(f"  ERROR: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)


if __name__ == "__main__":
    logger.info(f"LLMProxy on {SERVER_HOST}:{SERVER_PORT} | MiniMax ...{MINIMAX_KEY[-8:]} | Gemini ...{GEMINI_KEY[-8:]}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
