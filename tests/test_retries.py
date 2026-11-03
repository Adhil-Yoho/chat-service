from unittest.mock import AsyncMock
import httpx
import pytest
from openai import APIStatusError
from tenacity import(retry, retry_if_exception, wait_exponential, stop_after_attempt)


def make_error(status_code):
    request = httpx.Request(
        "POST",
        "https://example.com/v1/chat/completions"
    )
    response = httpx.Response(
        status_code=status_code,
        request=request,
        json={"error":"Mock error"}
    )

    return APIStatusError(
        "Mock error",
        response=response,
        body={"error":"Mock error"}
    )


def should_retry(error):
    return (
        isinstance(error, APIStatusError)
        and (
            error.status_code == 429
            or 500 <= error.status_code <= 599
        )
    )


def create_retry_function(mock_call):
    @retry(
        retry=retry_if_exception(should_retry),
        wait=wait_exponential(multiplier=0, min=0, max=0),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def call():
        return await mock_call()
    return call

@pytest.mark.asyncio
async def test_429_twice_then_success():
    mock_call = AsyncMock(
        side_effect=[
            make_error(429),
            make_error(429),
            "success"
        ]
    )

    call = create_retry_function(mock_call)

    result = await call()

    assert result == "success"
    assert mock_call.await_count == 3

@pytest.mark.asyncio
async def test_401_is_not_retried():
    mock_call = AsyncMock(
        side_effect=make_error(401)
    )

    call = create_retry_function(mock_call)

    with pytest.raises(APIStatusError):
        await call()

    assert mock_call.await_count == 1