# Company Chat Service
A Python based GenAI chat application built using FastAPI and the NVIDIA API. This project demonstrates LLM integration, streaming responses, token and cost tracking, retries, asynchronous requests, tool calling, conversation memory, and API authentication.

It uses the OpenAI Python SDK against NVIDIA's OpenAI-compatible endpoint `(https://integrate.api.nvidia.com/v1)`.

## Features

LLM integration using the OpenAI Python SDK with NVIDIA's API endpoint.
Synchronous and asynchronous LLM requests.
Streaming responses using Server-Sent Events (SSE).
Token counting using tiktoken.
Estimated API cost calculation and daily usage tracking.
Automatic retries for HTTP 429 and 5xx errors using Tenacity.
Asynchronous execution with concurrency limited by a semaphore.
Function calling with three tools:
Order status lookup
Customer lookup
Discount calculation
Conversation history, sliding window and summarisation.
FastAPI session management and protected endpoints.
API-key authentication using secure comparison.
Automated tests using Pytest.

## Tech Stack

Python
FastAPI
NVIDIA API
OpenAI Python SDK
tiktoken
Tenacity
Pydantic
Pytest
pytest-cov

## Project structure

```
chat-service/
├── app.py                  FastAPI backend (Task 4, plus /chat/stream and /usage)
├── llm_client.py           sync and async clients, ask_openai, ask_claude
├── sdk_calls.py            count_tokens, cost, usage log, streaming timing (Task 1)
├── retries_async.py        tenacity retries, sequential vs parallel calls (Task 2)
├── conversation_mem.py     3 tools, tool loop, Conversation class (Task 3)
├── memory_demo.py          15-turn conversation demo (Task 3)
├── tests/
│   ├── test_app.py         12 API tests with a fake LLM
│   └── test_retries.py     2 retry tests (429 retried, 401 not)
├── requirements.txt
├── .env.example            template for the keys and model names
├── .gitignore
├── README.md
├── .env                    your real keys, local only, not committed
├── usage.jsonl             created at run time, not committed
└── myenv/                  virtual environment, not committed
```

## Setup

```
py -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Clone the repository:
```
git clone https://github.com/Adhil-Yoho/chat-service.git
cd chat-service
```

Create a virtual environment:
```
python -m venv myenv
```

Activate it on Windows:
```
.\myenv\Scripts\Activate.ps1
```

Install dependencies:
```
pip install -r requirements.txt
```

Create a `.env` file using `.env.example` and add your API key and model configuration.

### Run the Application
```
uvicorn app:app --reload
Open the API documentation:
http://127.0.0.1:8000/docs
```

---

## Files

| File | What it holds |
|---|---|
| `llm_client.py` | sync and async clients, `ask_openai`, `ask_claude` |
| `sdk_calls.py` | `count_tokens`, cost, usage log, streaming timing |
| `retries_async.py` | tenacity retry, 10 questions run one by one and in parallel |
| `conversation_mem.py` | 3 tools, tool loop, `Conversation` class |
| `memory_demo.py` | 15-turn conversation |
| `app.py` | FastAPI backend |
| `tests/test_app.py`, `tests/test_retries.py` | pytest tests |

## Task 1: SDK calls, streaming and cost

Run `py sdk_calls.py` and pick 1 (normal call), 2 (streaming) or 3 (today's usage).

```
Model: nvidia/nemotron-3.5-bolt-e4b
Input tokens: 15
Output tokens: 62
Estimated cost USD: 0.0001407
```

Streaming: the first token arrived after 0.52 s and the full answer took 24.3 s. A normal call prints nothing until the whole answer is ready.

`count_tokens` uses tiktoken. Requests above 3000 input tokens are refused. Cost is tokens times the price per million (0.70 input, 2.10 output). Every call is appended to `usage.jsonl`, and option 3 adds up today's lines:

```
{'date': '2026-11-03', 'total_tokens': 77, 'estimated_cost_usd': 0.0001407}
```

API routes (start the server with `uvicorn app:app --reload`):

- `POST /chat/stream` returns Server-Sent Events. Test it with `curl.exe -N -X POST http://127.0.0.1:8000/chat/stream -H "x-api-key: <key>" -H "Content-Type: application/json" -d "{\"message\":\"Hello\"}"`
- `GET /usage` returns today's tokens and cost. After the manual API test it returned 1596 tokens and 0.0025774 USD.

## Task 2: Retries and async

Run `py retries_async.py`. Run the mock tests with `py -m pytest -v tests/test_retries.py`.

- The call is wrapped with tenacity: `wait_exponential`, `stop_after_attempt(5)`, retry only on 429 and 5xx. Each retry is logged with `before_sleep_log`.
- The mock test fails twice with 429 and then succeeds (3 calls). A 401 is not retried (1 call). Both tests pass.
- Ten questions with `AsyncOpenAI`:

```
Sequential time: 449.81 seconds
Parallel time (limit 3): 115.84 seconds
Number of results: 10
```

- `asyncio.Semaphore(3)` limits the parallel run to 3 calls at a time. A failing call returns `{"question": ..., "error": ...}` and the other nine answers are kept.

Exponential backoff and jitter in 4 lines:

1. When the server answers 429 or 5xx, the client waits and tries again.
2. Each wait doubles: 1 s, 2 s, 4 s, 8 s.
3. Jitter adds a small random time to each wait.
4. Without jitter, many clients retry at the same moment. This code uses `wait_exponential` and has no jitter. tenacity's `wait_random_exponential` adds it.

## Task 3: Tools and conversation memory

Run `py conversation_mem.py` for the tool loop and `py memory_demo.py` for the 15-turn chat.

- Tools: `get_order_status`, `get_customer`, `calculate_discount`, each with a JSON schema.
- Tool loop: up to 5 rounds. All tool calls of a round run together with `asyncio.gather`. Each result goes back with its `tool_call_id`.
- Output for "Check order 1001, look up customer C001, and calculate 15% off 200": `Tool calls in round 1: 3`, and the final answer used all three results.
- Tool errors are returned as results, so the model can recover. Four cases from the terminal:

```
{'error': "Invalid argument: get_order_status() got an unexpected keyword argument 'orer_id'. Did you mean 'order_id'?"}
{'error': "Invalid argument: get_customer() missing 1 required positional argument: 'customer_id'"}
{'error': 'Amount must be non-negative and percent must be between 0 and 100'}
{'error': 'Unknown tool: unknown_tool'}
```

- `Conversation` stores `[{"role": ..., "content": ...}]` and sends the full history on every call. Budgets: system 500, history 3000, output 1000 tokens (`max_tokens`). When the history is over budget, the last 8 messages stay and older ones are summarised into a short summary.
- In the 15-turn run, turn 1 gave the ticket reference ABC123 and turn 15 answered "Your original ticket reference ... was ABC123". The history stayed under the budget in this run, so the saved summary was empty.

## Task 4: Secured FastAPI backend

Run `uvicorn app:app --reload` and open `http://127.0.0.1:8000/docs`.
## Main API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET	| /health | Public health check |
| POST	| /sessions | Create a chat session |
| POST	| /sessions/{session_id}/chat | Send a chat message |
| GET	| /sessions/{session_id}/messages | Retrieve conversation history |
| POST	| /chat/stream | Stream an LLM response |
| GET	| /usage | Get daily token and cost usage |

| Status | When |
|---|---|
| 401 | no `x-api-key` header |
| 403 | wrong key (checked with `secrets.compare_digest`) |
| 404 | unknown session |
| 413 | conversation or input above 3000 tokens |
| 422 | empty message or more than 2000 characters |
| 502 | the LLM call failed |

All errors use one JSON body: `{"error": "http_error", "detail": "Session not found", "request_id": "not-provided"}`. CORS allows two origins: `http://localhost:3000` and `http://localhost:5500`. Settings, the LLM client and the in-memory store come from `Depends()`.

Tests use `TestClient` with a fake LLM set through `dependency_overrides`:

```
py -m pytest --cov=app --cov-report=term-missing
14 passed
app.py 97 statements 8 missed 92%
```

12 tests are in `tests/test_app.py` and 2 in `tests/test_retries.py`.
