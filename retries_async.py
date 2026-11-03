import asyncio
import time
from openai import APIStatusError
from tenacity import (retry, retry_if_exception, wait_exponential, stop_after_attempt,before_sleep_log)
import logging
from llm_client import make_async_client, get_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def should_retry(error):
    return (
        isinstance(error, APIStatusError)
        and (
            error.status_code == 429
            or 500 <= error.status_code <= 599
        )
    )


@retry(
    retry=retry_if_exception(should_retry),
    wait=wait_exponential(
        multiplier=1,
        min=1,
        max=8
    ),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(
        logger, logging.WARNING
    ),
    reraise=True
)
async def reliable_question(client, question):
    return await client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "user", "content": question}
        ]
    )


async def safe_question(client, question, semaphore):
    async with semaphore:
        try:
            response = await reliable_question(
                client, question
            )

            return {
                "question": question,
                "answer": (
                    response.choices[0].message.content
                    or ""
                )
            }

        except Exception as error:
            return {
                "question": question,
                "error": str(error)
            }


async def main():
    client = make_async_client()

    questions = [
        "Explain Python variables with one example",
        "Explain Python data types with one example",
        "Explain Python operators with one example",
        "Explain if-else statement with one example",
        "Explain for loops with one example",
        "Explain while loops with one example.",
        "Explain fucntions with one example.",
        "Explain lists an tuples with one example.",
        "Explain dictionaries with one example.",
        "Explain exception handling wit one example",
    ]

    start = time.perf_counter()
    sequential_results = []

    for question in questions:
        try:
            response = await reliable_question(
                client, question
            )

            sequential_results.append(
                response.choices[0].message.content
            )

        except Exception as error:
            sequential_results.append(
                f"ERROR: {error}"
            )

    sequential_time = time.perf_counter() - start

    semaphore = asyncio.Semaphore(3)

    start = time.perf_counter()

    parallel_results = await asyncio.gather(
        *[
            safe_question(client, question, semaphore)
            for question in questions
        ]
    )

    parallel_time = time.perf_counter() - start

    print(
        "Sequential time:",
        round(sequential_time, 2),
        "seconds"
    )

    print(
        "Parallel time (limit 3):",
        round(parallel_time, 2),
        "seconds"
    )

    print("Number of results:", len(parallel_results))

    for result in parallel_results:
        print(result)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())