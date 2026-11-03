import pytest
from fastapi.testclient import TestClient
from app import (app, get_settings, get_llm_client, get_store, Settings,)

API_KEY = "test-key"

class FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5

class FakeMessage:
    content = "Test answer"

class FakeChoice:
    message = FakeMessage()

class FakeResponse:
    model = "fake-model"
    usage = FakeUsage()
    choices = [FakeChoice()]

class FakeCompletions:
    async def create(self, **kwargs):
        return FakeResponse()

class FakeChat:
    completions = FakeCompletions()

class FakeClient:
    chat = FakeChat()

class FakeDelta:
    def __init__(self, content):
        self.content = content


class FakeStreamChoice:
    def __init__(self, content):
        self.delta = FakeDelta(content)


class FakeStreamChunk:
    def __init__(self, content):
        self.choices = [FakeStreamChoice(content)]


class FakeAsyncStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        async def generate():
            for chunk in self.chunks:
                yield chunk

        return generate()


class FakeStreamingCompletions:
    async def create(self, **kwargs):
        return FakeAsyncStream([
            FakeStreamChunk("Hello"),
            FakeStreamChunk(" there"),
            FakeStreamChunk(None),
        ])


class FakeStreamingChat:
    completions = FakeStreamingCompletions()


class FakeStreamingClient:
    chat = FakeStreamingChat()


class FakeFailingCompletions:
    async def create(self, **kwargs):
        raise Exception("Simulated LLM failure")


class FakeFailingChat:
    completions = FakeFailingCompletions()


class FakeFailingClient:
    chat = FakeFailingChat()


@pytest.fixture
def client():
    test_score ={}
    app.dependency_overrides[get_settings] = (
        lambda: Settings(api_key_1=API_KEY)
    )

    app.dependency_overrides[get_llm_client] = (
        lambda: FakeClient()
    )

    app.dependency_overrides[get_store] = (
        lambda: test_score
    )

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def headers(key=API_KEY):
    return {"x-api-key": key}


def test_health_is_public(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_missing_key_returns_401(client):
    response = client.post("/sessions")
    assert response.status_code == 401


def test_wrong_key_returns_403(client):
    response = client.post(
        "/sessions",
        headers=headers("wrong-key"),
    )
    assert response.status_code == 403


def test_create_session(client):
    response = client.post(
        "/sessions",
        headers=headers(),
    )
    assert response.status_code == 200
    assert "session_id" in response.json()


def test_empty_message_returns_422(client):
    response = client.post(
        "/sessions/abc/chat",
        headers=headers(),
        json={"message": ""},
    )
    assert response.status_code == 422


def test_over_2000_characters_returns_422(client):
    response = client.post(
        "/sessions/abc/chat",
        headers=headers(),
        json={"message": "x" * 2001},
    )
    assert response.status_code == 422


def test_unknown_session_returns_404(client):
    response = client.get(
        "/sessions/unknown/messages",
        headers=headers(),
    )
    assert response.status_code == 404


def test_chat_and_history(client):
    session = client.post(
        "/sessions",
        headers=headers(),
    ).json()["session_id"]

    response = client.post(
        f"/sessions/{session}/chat",
        headers=headers(),
        json={"message": "Hello"},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "Test answer"

    history = client.get(
        f"/sessions/{session}/messages",
        headers=headers(),
    )
    assert history.status_code == 200
    assert len(history.json()["messages"]) == 3


def test_usage_endpoint(client):
    response = client.get(
        "/usage",
        headers=headers(),
    )
    assert response.status_code == 200
    assert "total_tokens" in response.json()

def test_chat_stream_success(client):
    app.dependency_overrides[get_llm_client] = (
        lambda: FakeStreamingClient()
    )

    response = client.post(
        "/chat/stream",
        headers=headers(),
        json={"message": "Hello"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )
    assert 'data: {"text": "Hello"}' in response.text
    assert 'data: {"text": " there"}' in response.text
    assert "[DONE]" in response.text


def test_chat_stream_llm_failure(client):
    app.dependency_overrides[get_llm_client] = (
        lambda: FakeFailingClient()
    )

    response = client.post(
        "/chat/stream",
        headers=headers(),
        json={"message": "Hello"},
    )
    assert response.status_code == 200
    assert "event: error" in response.text
    assert "LLM request failed" in response.text


def test_chat_stream_requires_api_key(client):
    response = client.post(
        "/chat/stream",
        json={"message": "Hello"},
    )
    assert response.status_code == 401