import os
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

BASE_URL = "https://integrate.api.nvidia.com/v1"


def get_model():
    return os.getenv("NVIDIA_MODEL", "")


def make_client():
    key = os.getenv("API_KEY")

    return OpenAI(
        api_key=key,
        base_url=BASE_URL
    )


def make_async_client():
    key = os.getenv("API_KEY")

    return AsyncOpenAI(
        api_key=key,
        base_url=BASE_URL
    )


def ask_openai(prompt):
    client = make_client()

    try:
        response = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content or ""

        print("Answer:", answer)
        print("Model:", response.model)
        print("Usage:", response.usage)
        return answer
    finally:
        client.close()


def ask_claude(prompt: str, system: str = "You are a helpful assistant"):
    model = os.getenv("CLAUDE_MODEL", "")
    client = make_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content or ""

        print("Answer:", answer)
        print("Model:", response.model)
        print("Usage:", response.usage)
        return answer
    finally:
        client.close()