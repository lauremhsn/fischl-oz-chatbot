"""
Smoke test 3: does the persona hold, and what does few-shot actually buy?

Runs a handful of prompts through the Fischl stage only (no Oz, no UI), once
with the few-shot examples and once without, so the difference is visible
rather than assumed.

Run with the venv active:
    python check_persona.py

Throwaway diagnostic. Delete it later.
"""

from openai import OpenAI

from persona import FISCHL_FEWSHOT, FISCHL_SYSTEM

MODEL = "fischl-llama"
BASE_URL = "http://localhost:11434/v1"

PROMPTS = [
    "Hello, who are you?",
    "What is the boiling point of water?",
    "Can you help me debug my Python code?",
    "What's 17 times 23?",
]

client = OpenAI(base_url=BASE_URL, api_key="ollama")


def ask(user_message: str, use_fewshot: bool) -> str:
    messages = [{"role": "system", "content": FISCHL_SYSTEM}]
    if use_fewshot:
        messages.extend(FISCHL_FEWSHOT)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.8,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


for prompt in PROMPTS:
    print("=" * 70)
    print(f"USER: {prompt}")
    print()
    print("--- system prompt only -------------------------------------------")
    print(ask(prompt, use_fewshot=False))
    print()
    print("--- system prompt + few-shot -------------------------------------")
    print(ask(prompt, use_fewshot=True))
    print()