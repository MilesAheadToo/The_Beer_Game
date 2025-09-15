import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env if present
load_dotenv()

# Initialize OpenAI client with API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key)

def call_beer_game_gpt(user_message: str):
    """Call the Beer Game GPT and return the order and any structured data.

    Uses a custom GPT model specified by the ``GPT_ID`` environment variable,
    defaulting to ``gpt-50`` if unset.
    """

    gpt_id = os.getenv("GPT_ID", "gpt-50")

    request_args = {
        "model": "gpt-4.1-mini",
        "gpt": gpt_id,
        "instructions": "You are Daybreak Beer Game Strategist.",
        "input": user_message,
        "tools": [{"type": "code_interpreter"}],
    }

    response = client.responses.create(**request_args)
    reply = response.output_text
    print("FULL RESPONSE:\n", reply)

    order_match = re.search(r"ORDER:\s*(\d+)", reply)
    order = int(order_match.group(1)) if order_match else None

    data_match = re.search(r"```json\s*(\{.*?\})\s*```", reply, re.DOTALL)
    data = json.loads(data_match.group(1)) if data_match else None

    return order, data
