import json
import time
from datetime import date
from pathlib import Path
import tiktoken
from llm_client import make_client, get_model


USAGE_FILE = Path("usage.jsonl")
MAX_INPUT_TOKENS = 3000

INPUT_PRICE = 0.70
OUTPUT_PRICE = 2.10


def count_tokens(messages):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total = 0
    for message in messages:
        for value in message.values():
            if isinstance(value, str):
                total += len(encoding.encode(value))
    return total


def calculate_cost(input_tokens, output_tokens):
    input_cost = input_tokens * INPUT_PRICE / 1_000_000
    output_cost = output_tokens * OUTPUT_PRICE / 1_000_000
    return input_cost + output_cost


def log_usage(model, input_tokens, output_tokens):
    cost = calculate_cost(input_tokens, output_tokens)

    record = {
        "date": date.today().isoformat(),
        "time": time.time(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost, 8)
    }

    with USAGE_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return cost


def ask_with_cost(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]

    tokens = count_tokens(messages)

    if tokens > MAX_INPUT_TOKENS:
        raise ValueError("Input token limit exceeded")

    client = make_client()

    try:
        response = client.chat.completions.create(
            model=get_model(),
            messages=messages
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage

        input_tokens = (
            usage.prompt_tokens if usage else tokens
        )
        output_tokens = (
            usage.completion_tokens if usage else 0
        )

        cost = log_usage(
            response.model,
            input_tokens,
            output_tokens
        )

        print("Answer:", answer)
        print("Model:", response.model)
        print("Input tokens:", input_tokens)
        print("Output tokens:", output_tokens)
        print("Estimated cost USD:", round(cost, 8))

        return answer
    finally:
        client.close()


def stream_answer(prompt):
    client = make_client()
    start = time.perf_counter()
    first_token = None

    try:
        with client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True
        ) as stream:

            print("Answer: ", end="", flush=True)

            for chunk in stream:
                text = chunk.choices[0].delta.content

                if text:
                    if first_token is None:
                        first_token = (
                            time.perf_counter() - start
                        )

                    print(text, end="", flush=True)

        print()

        print(
            "Time to first token:",
            round(first_token or 0, 3),
            "seconds"
        )

        print(
            "Total time:",
            round(time.perf_counter() - start, 3),
            "seconds"
        )

    finally:
        client.close()


def show_today_usage():
    total_tokens = 0
    total_cost = 0.0

    if USAGE_FILE.exists():
        with USAGE_FILE.open(
            "r", encoding="utf-8"
        ) as file:

            for line in file:
                row = json.loads(line)

                if row["date"] == date.today().isoformat():
                    total_tokens += row["total_tokens"]
                    total_cost += row["cost_usd"]

    result = {
        "date": date.today().isoformat(),
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 8)
    }

    print(result)
    return result


if __name__ == "__main__":
    print("1.Normal LLM call")
    print("2.Streaming call")
    print("3.Today's usage")

    choice = input("Choose 1, 2 or 3: ")

    if choice == "1":
        prompt = input("Enter prompt: ")
        ask_with_cost(prompt)

    elif choice == "2":
        prompt = input("Enter prompt: ")
        stream_answer(prompt)

    elif choice == "3":
        show_today_usage()

    else:
        print("Invalid choice")