import os
import json
import secrets
from typing import Optional
from dotenv import load_dotenv
from fastapi import (FastAPI, Depends, Header, HTTPException,Request,)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from llm_client import make_async_client, get_model
from sdk_calls import (count_tokens, calculate_cost, log_usage, show_today_usage,)

load_dotenv()

app = FastAPI(title="Company Chat Service")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "x-api-key"],
)


SESSIONS = {}

class Settings(BaseModel):
    api_key_1: str = ""
    api_key_2: str = ""


def get_settings():
    return Settings(
        api_key_1=os.getenv("API_KEY_1", ""),
        api_key_2=os.getenv("API_KEY_2", ""),
    )


def get_llm_client():
    return make_async_client()


def get_store():
    return SESSIONS


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None),
    settings: Settings = Depends(get_settings),
):
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
        )

    valid_keys = [
        key for key in (
            settings.api_key_1,
            settings.api_key_2
        ) if key
    ]

    if not any(
        secrets.compare_digest(x_api_key, key)
        for key in valid_keys
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )
    return True


class CreateSessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str

@app.exception_handler(HTTPException)
async def handle_http_error(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "detail": exc.detail,
            "request_id": request.headers.get(
                "x-request-id", "not-provided"
            ),
        },
    )

@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
            "request_id": request.headers.get(
                "x-request-id", "not-provided"
            ),
        },
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post(
    "/sessions",
    response_model=CreateSessionResponse,
)
async def create_session(
    authenticated=Depends(require_api_key),
    store=Depends(get_store),
):
    session_id = secrets.token_urlsafe(12)

    store[session_id] = [
        {
            "role": "system",
            "content": (
                "You are a helpful company support assistant"
            ),
        }
    ]
    return {"session_id": session_id}


async def get_answer(client, messages):
    try:
        response = await client.chat.completions.create(
            model=get_model(),
            messages=messages,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content or ""
        return answer, response

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed: {type(error).__name__}",
        )

@app.post(
    "/sessions/{session_id}/chat",
    response_model=ChatResponse,
)
async def chat(
    session_id: str,
    body: ChatRequest,
    authenticated=Depends(require_api_key),
    store=Depends(get_store),
    client=Depends(get_llm_client),
):
    if session_id not in store:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    messages = store[session_id]

    new_messages = messages + [
        {"role": "user", "content": body.message}
    ]

    if count_tokens(new_messages) > 3000:
        raise HTTPException(
            status_code=413,
            detail="Conversation token limit exceeded",
        )
    messages.append({
        "role": "user",
        "content": body.message,
    })
    answer, response = await get_answer(
        client, messages
    )
    messages.append({
        "role": "assistant",
        "content": answer,
    })

    usage = response.usage

    input_tokens = (
        usage.prompt_tokens
        if usage else count_tokens(messages)
    )
    output_tokens = (
        usage.completion_tokens
        if usage else 0
    )
    log_usage(
        response.model,
        input_tokens,
        output_tokens,
    )
    return {
        "session_id": session_id,
        "answer": answer,
    }

@app.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: str,
    authenticated=Depends(require_api_key),
    store=Depends(get_store),
):
    if session_id not in store:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    return {
        "session_id": session_id,
        "messages": store[session_id],
    }

@app.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    authenticated=Depends(require_api_key),
    client=Depends(get_llm_client),
):
    messages = [
        {"role": "user", "content": body.message}
    ]

    if count_tokens(messages) > 3000:
        raise HTTPException(
            status_code=413,
            detail="Input token limit exceeded",
        )

    async def generate():
        try:
            stream = await client.chat.completions.create(
                model=get_model(),
                messages=messages,
                stream=True,
                max_tokens=1000,
            )

            async for chunk in stream:
                text = chunk.choices[0].delta.content

                if text:
                    yield (
                        "data: "
                        + json.dumps({"text": text})
                        + "\n\n"
                    )

            yield "data: [DONE]\n\n"

        except Exception:
            yield (
                "event: error\n"
                'data: {"error":"LLM request failed"}\n\n'
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/usage")
async def usage_today(
    authenticated=Depends(require_api_key),
):
    return show_today_usage()