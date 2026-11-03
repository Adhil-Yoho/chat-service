import asyncio
from llm_client import make_async_client
from conversation_mem import Conversation


async def main():
    client = make_async_client()
    conversation = Conversation()

    try:
        first = await conversation.send(
            "My ticket reference is ABC123.",
            client
        )
        print("Turn 1:", first)

        for number in range(2, 15):
            answer = await conversation.send(
                f"Turn {number}: Explain concept {number} briefly.",
                client
            )
            print(f"Turn {number}:", answer)

        final_answer = await conversation.send(
            "What was my original ticket reference?",
            client
        )

        print("Turn 15:", final_answer)
        print("Saved summary:", conversation.summary)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())