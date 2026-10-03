"""
Smoke test 7: does reasoning mode actually change correctness?

The assignment asks for two example prompts where reasoning mode improves the
answer. Which prompts those are is not something to guess at -- an 8B model
gets plenty of multi-step questions right without help, and picking examples
by intuition risks presenting two that would have been correct either way.

So this runs a set of multi-step problems with checkable answers, N times each
in both modes, and reports the hit rate for each. Whatever shows a real gap is
what belongs in the README.

Run with the venv active:
    python check_reasoning.py

It makes RUNS x 2 x len(CASES) calls and reasoning mode is the slower path, so
expect a few minutes.

Throwaway diagnostic. Delete it later.
"""

import re

from chat import Conversation, fischl_reply, reasoning_stream

RUNS = 4

# (question, acceptable answer fragments)
#
# Both digit and spelled-out forms are accepted. The first version of this
# checked only for digits, which scored a correct "twenty-eight" or "sixty-six"
# as a failure -- and because the persona spells numbers out far more often
# with reasoning OFF, that bug inflated the apparent benefit of reasoning mode.
# A measurement bug that flatters the thing being measured is the dangerous
# kind. Same trap as check_accuracy.py; fixed there first.
CASES = [
    (
        "A shelf has 3 rows of 7 books. I take 5 away, then add 12. "
        "How many books are on the shelf now?",
        ["28", "twenty-eight", "twenty eight"],
    ),
    (
        "I am 3 times as old as my sister. In 6 years I will be twice her age. "
        "How old is my sister now?",
        ["6 ", "6.", "6,", "six"],
    ),
    (
        "A train leaves at 14:25 and the journey takes 3 hours and 50 minutes. "
        "What time does it arrive?",
        ["18:15", "18.15", "6:15", "quarter past six", "six fifteen"],
    ),
    (
        "If 5 machines take 5 minutes to make 5 widgets, how long do 100 "
        "machines take to make 100 widgets?",
        ["5 minute", "five minute"],
    ),
    (
        "A jacket costs 80 dollars. It is discounted 25 percent, then a "
        "further 10 percent is taken off the sale price. What is the final "
        "price?",
        ["54", "fifty-four", "fifty four"],
    ),
    (
        "There are 12 people at a party and everyone shakes hands with "
        "everyone else exactly once. How many handshakes happen?",
        ["66", "sixty-six", "sixty six"],
    ),
]


def answer_without_reasoning(question: str) -> str:
    return fischl_reply(Conversation().turns, question)


def answer_with_reasoning(question: str) -> tuple[str, str]:
    trace = ""
    for trace in reasoning_stream(question):
        pass
    reply = fischl_reply(Conversation().turns, question, reasoning=trace)
    return trace, reply


def correct(text: str, accepted: list[str]) -> bool:
    low = re.sub(r"\s+", " ", text.lower())
    return any(a.lower() in low for a in accepted)


print(f"{RUNS} runs per condition per question.\n")

off_total = on_total = 0

for question, accepted in CASES:
    print("=" * 72)
    print(question)

    off_hits = 0
    on_hits = 0
    first_gain = None

    for _ in range(RUNS):
        plain = answer_without_reasoning(question)
        off_hits += correct(plain, accepted)

        trace, reply = answer_with_reasoning(question)
        got = correct(reply, accepted)
        on_hits += got

        if first_gain is None and got and not correct(plain, accepted):
            first_gain = (plain, trace, reply)

    off_total += off_hits
    on_total += on_hits
    delta = on_hits - off_hits
    flag = "  <-- reasoning helps" if delta > 0 else ("  <-- reasoning hurts" if delta < 0 else "")
    print(f"  reasoning OFF: {off_hits}/{RUNS}   ON: {on_hits}/{RUNS}{flag}")

    if first_gain:
        plain, trace, reply = first_gain
        print("\n  --- a case where it flipped -------------------------------")
        print(f"  OFF: {re.sub(r'[ ]+', ' ', plain)}")
        print(f"  TRACE: {re.sub(r'[ ]+', ' ', trace)}")
        print(f"  ON:  {re.sub(r'[ ]+', ' ', reply)}")
    print()

n = RUNS * len(CASES)
print("=" * 72)
print(f"OVERALL   off: {off_total}/{n} ({off_total / n * 100:.0f}%)   "
      f"on: {on_total}/{n} ({on_total / n * 100:.0f}%)")
print()
print("Pick the two questions with the largest positive gap for the README.")