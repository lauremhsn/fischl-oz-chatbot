"""
Smoke test 2: can we override Ollama's default num_ctx through the
OpenAI-compatible endpoint?

Ollama defaults to a 4096-token context regardless of what the model
supports. The OpenAI API has no field for this, so we smuggle Ollama's
native options through `extra_body`.

Run with the venv active:
    python check_context.py

Then, while the model is still loaded, run `ollama ps` in another window
and look at the CONTEXT column. That is the real verification -- if it
still says 4096, extra_body did not take effect.

Throwaway diagnostic. Delete it later.
"""

from openai import OpenAI

MODEL = "llama3.1:8b"
BASE_URL = "http://localhost:11434/v1"
TARGET_CTX = 8192

client = OpenAI(base_url=BASE_URL, api_key="ollama")

print(f"Requesting num_ctx={TARGET_CTX} ...")

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        extra_body={"options": {"num_ctx": TARGET_CTX}},
    )
except Exception as exc:
    print("FAILED")
    print(f"{type(exc).__name__}: {exc}")
    raise SystemExit(1)

print("Request succeeded.")
print(f"Model said: {response.choices[0].message.content!r}")
print()
print("Now run this in another terminal while the model is still loaded:")
print("    ollama ps")
print()
print(f"CONTEXT should read {TARGET_CTX}, not 4096.")