"""Utility script that prints custom OpenAI model identifiers.

Loads the API key from the project-level .env file so developers do not
need to hardcode credentials inside the script.
"""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI


OUTPUT_PATH = Path(__file__).with_name("find_keys.txt")


def get_openai_api_key() -> str:
    """Read OPENAI_API_KEY from environment or the repository .env file."""

    # allow overriding via environment variable for flexibility
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key.strip().strip("\"\'")

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "OPENAI_API_KEY":
                return value.strip().strip("\"\'")

    raise RuntimeError("OPENAI_API_KEY not found in environment or .env file")


client = OpenAI(api_key=get_openai_api_key())
models = client.models.list()

lines = [f"client: {client}", "models:"]

for model in models.data:
    lines.append(model.id)

OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote OpenAI metadata to {OUTPUT_PATH}")
