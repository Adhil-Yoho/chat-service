import asyncio
import json
from llm_client import make_async_client, get_model
from sdk_calls import count_tokens


ORDERS = {
    "1001": {
        "status": "shipped",
        "eta": "tomorrow"
    },
    "1002": {
        "status": "processing",
        "eta": "not available yet"
    }
}

CUSTOMERS = {
    "C001": {
        "name": "Asha",
        "tier": "gold"
    },
    "C002": {
        "name": "Ravi",
        "tier": "standard"
    }
}


def get_order_status(order_id):
    if order_id not in ORDERS:
        return {"error": f"Order {order_id} not found"}
    return ORDERS[order_id]


def get_customer(customer_id):
    if customer_id not in CUSTOMERS:
        return {
            "error": f"Customer {customer_id} not found"
        }
    return CUSTOMERS[customer_id]


def calculate_discount(amount, percent):
    if amount < 0 or not 0 <= percent <= 100:
        return {
            "error": (
                "Amount must be non-negative and "
                "percent must be between 0 and 100"
            )
        }

    discount = amount * percent / 100
    return {
        "original": amount,
        "percent": percent,
        "discount": round(discount, 2),
        "final": round(amount - discount, 2)
    }

FUNCTIONS = {
    "get_order_status": get_order_status,
    "get_customer": get_customer,
    "calculate_discount": calculate_discount
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up an order by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Example: 1001"
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Look up a customer by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Example: C001"
                    }
                },
                "required": ["customer_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_discount",
            "description": "Calculate a discount and final price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "minimum": 0
                    },
                    "percent": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["amount", "percent"],
                "additionalProperties": False
            }
        }
    }
]


def execute_tool(name, arguments):
    if name not in FUNCTIONS:
        return {"error": f"Unknown tool: {name}"}
    
    if not isinstance(arguments, dict):
        return {"error": "Arguments must be a JSON object"}

    try:
        return FUNCTIONS[name](**arguments)

    except (TypeError, ValueError) as error:
        return {"error": f"Invalid argument: {error}"}


async def run_tool_loop(client, messages):
    for round_number in range(1, 6):
        response = await client.chat.completions.create(
            model=get_model(),
            messages=messages,
            tools=TOOLS,
            parallel_tool_calls=True
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls

        if not tool_calls:
            return assistant_message.content or ""

        print(
            f"Tool calls in round {round_number}:",
            len(tool_calls)
        )

        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        async def execute_one(call):
            try:
                arguments = json.loads(
                    call.function.arguments or "{}"
                )
            except json.JSONDecodeError:
                arguments = None

            result = await asyncio.to_thread(
                execute_tool,
                call.function.name,
                arguments
            )

            return call, result

        results = await asyncio.gather(
            *(execute_one(call) for call in tool_calls)
        )

        for call, result in results:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })
    return "Stopped: maximum five tool rounds reached"


class Conversation:
    def __init__(self):
        self.system_prompt = (
            "You are a helpful company support assistant"
        )

        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        self.summary = ""

        self.system_budget = 500
        self.history_budget = 3000
        self.output_budget = 1000

    def make_messages(self):
        system_text = self.system_prompt

        if self.summary:
            system_text += (
                "\nSummary of earlier conversation: "
                + self.summary
            )

        return [
            {"role": "system", "content": system_text}
        ] + self.messages[1:]

    async def summarise_old_messages(self, client):
        old_messages = self.messages[1:-8]

        if not old_messages:
            return

        response = await client.chat.completions.create(
            model=get_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarise the conversation briefly "
                        "Preserve names, reference codes, "
                        "decisions and unresolved questions."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps(old_messages)
                }
            ],
            max_tokens=500
        )

        new_summary = response.choices[0].message.content or ""

        if new_summary:
            self.summary = (
                self.summary + "\n" + new_summary
            ).strip()

        self.messages = [
            self.messages[0]
        ] + self.messages[-8:]

    async def send(self, text, client):
        self.messages.append({
            "role": "user",
            "content": text
        })

        messages = self.make_messages()

        while (
            count_tokens(messages)
            > self.system_budget + self.history_budget
        ):
            old_count = len(self.messages)

            await self.summarise_old_messages(client)

            messages = self.make_messages()

            if len(self.messages) >= old_count:

                raise ValueError(
                    "Cannot reduce history to the token budget"
                )

        response = await client.chat.completions.create(
            model=get_model(),
            messages=messages,
            max_tokens=self.output_budget
        )

        answer = response.choices[0].message.content or ""

        self.messages.append({
            "role": "assistant",
            "content": answer
        })
        return answer


async def main():
    client = make_async_client()

    try:
        messages = [
            {
                "role": "user",
                "content": (
                    "Check order 1001, look up customer C001, "
                    "and calculate 15% off 200."
                )
            }
        ]

        answer = await run_tool_loop(client, messages)

        print("Final answer:", answer)

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())