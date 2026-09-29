"""
Smoke test 4: does the two-stage chain work, and is the token estimate honest?

Runs a short multi-turn conversation through the full Fischl -> Oz chain,
printing both voices and the estimate-vs-actual token accounting after each
turn.

Two things to watch:
  1. Does Oz actually restate Fischl, or does he answer independently?
  2. How far off is the 4-chars-per-token estimate? If it errs low by a lot,
     the reserve in chat.py needs to grow.

Run with the venv active:
    python check_chain.py

Throwaway diagnostic. Delete it later.
"""

from chat import CONTEXT_LIMIT, last_estimate_error, respond

CONVERSATION = [
    "Hello! What should I call you?",
    "I'm trying to decide what to have for dinner.",
    "What did I just ask you about?",
]

history = []

for user_message in CONVERSATION:
    print("=" * 70)
    print(f"YOU: {user_message}")
    print()

    turn = respond(history, user_message)

    print("FISCHL:")
    print(f"  {turn.fischl}")
    print()
    print("OZ:")
    print(f"  {turn.oz}")
    print()

    estimated, actual, error = last_estimate_error()
    print(
        f"  [prompt tokens -- estimated {estimated}, actual {actual}, "
        f"error {error:+.1f}%  |  window {actual}/{CONTEXT_LIMIT}]"
    )
    print()

    history.append(turn)

print("=" * 70)
print(f"Turns kept in history: {len(history)}")