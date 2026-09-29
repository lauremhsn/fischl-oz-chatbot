"""
Smoke test: confirm the OpenAI SDK can talk to the local Ollama server.

Run with the venv active:
    python check_ollama.py

This is a throwaway diagnostic, not part of the app. Delete it later.
"""

from openai import OpenAI

MODEL = "llama3.1:8b"
BASE_URL = "http://localhost:11434/v1"

client = OpenAI(
    base_url=BASE_URL,
    api_key="ollama",  # Ollama ignores this, but the SDK requires a non-empty value
)

print(f"Connecting to {BASE_URL} ...")

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are terse. Answer in one short sentence."},
            {"role": "user", "content": "What is the capital of Lebanon?"},
        ],
    )
except Exception as exc:
    print("FAILED")
    print(f"{type(exc).__name__}: {exc}")
    raise SystemExit(1)

print("OK")
print("-" * 40)
print(response.choices[0].message.content)
print("-" * 40)

usage = response.usage
if usage is not None:
    print(f"prompt tokens:     {usage.prompt_tokens}")
    print(f"completion tokens: {usage.completion_tokens}")
    print(f"total tokens:      {usage.total_tokens}")
else:
    print("(no usage data returned)")