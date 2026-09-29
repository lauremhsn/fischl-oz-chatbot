"""
Smoke test 6: does the persona degrade factual accuracy?

The persona rewrite introduced a regression that the other tests missed:
17 x 23 came back as "three hundred and eleven" (it is 391). Earlier, less
ornate versions got it right. So heavier persona load appears to cost
accuracy, and nothing in the test suite was watching for it.

This runs a set of questions with checkable answers, several times each,
and reports a hit rate. Several times each because a single correct answer
at temperature 0.8 proves nothing.

Run with the venv active:
    python check_accuracy.py

Throwaway diagnostic. Delete it later.
"""

import re

from openai import OpenAI

from persona import FISCHL_FEWSHOT, FISCHL_SYSTEM

MODEL = "fischl-llama"
BASE_URL = "http://localhost:11434/v1"
RUNS = 5

client = OpenAI(base_url=BASE_URL, api_key="ollama")

# (question, list of acceptable substrings -- any one counts as correct)
#
# Both digit and spelled-out forms are accepted. An earlier version of this
# test checked only for digits and scored every correct "nineteen forty-five"
# as a failure, which made accuracy look 13 points worse than it was. A test
# that reports false failures is worse than no test, because it sends you
# hunting for a bug in the thing being measured.
CASES = [
    ("What's 17 times 23?", ["391", "three hundred and ninety-one",
                             "three hundred ninety-one"]),
    ("What is the boiling point of water in Celsius?",
     ["100", "one hundred", "a hundred"]),
    ("How many days are in a leap year?", ["366", "three hundred and sixty-six",
                                           "three hundred sixty-six"]),
    ("What year did the Second World War end?",
     ["1945", "nineteen forty-five", "nineteen forty five",
      "nineteen hundred and forty-five", "nineteen hundred forty-five"]),
    ("What's the square root of 144?", ["12", "twelve"]),
]

# Every wrong answer so far has spelled the number out in words. Not one has
# written digits and got them wrong. Tracked separately because it points at
# the mechanism rather than just the rate: the model appears to lose the
# arithmetic while composing the words for it.
DIGIT_PATTERN = re.compile(r"\d")


def ask(question: str, use_fewshot: bool) -> str:
    messages = [{"role": "system", "content": FISCHL_SYSTEM}]
    if use_fewshot:
        messages.extend(FISCHL_FEWSHOT)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.8,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def correct(reply: str, accepted: list[str]) -> bool:
    lowered = reply.lower()
    return any(a.lower() in lowered for a in accepted)


for label, use_fewshot in [("system only", False), ("system + few-shot", True)]:
    print("=" * 70)
    print(f"CONDITION: {label}")
    print()

    hits = 0
    total = 0
    wrong_in_digits = 0
    wrong_in_words = 0

    for question, accepted in CASES:
        results = []
        for _ in range(RUNS):
            reply = ask(question, use_fewshot)
            ok = correct(reply, accepted)
            results.append((ok, reply))
            hits += ok
            total += 1
            if not ok:
                if DIGIT_PATTERN.search(reply):
                    wrong_in_digits += 1
                else:
                    wrong_in_words += 1

        passed = sum(1 for ok, _ in results if ok)
        print(f"  {question}")
        print(f"    {passed}/{RUNS} correct")
        for ok, reply in results:
            if not ok:
                # Print the whole reply, not a prefix. An earlier version cut
                # this at 160 characters, which meant a failure whose answer
                # appeared late looked identical to one that never answered at
                # all -- the display hid the evidence needed to diagnose it.
                collapsed = re.sub(r"\s+", " ", reply)
                print(f"      WRONG: {collapsed}")
        print()

    print(f"  OVERALL: {hits}/{total} ({hits / total * 100:.0f}%)")
    if wrong_in_digits or wrong_in_words:
        print(
            f"  of the {wrong_in_digits + wrong_in_words} wrong: "
            f"{wrong_in_digits} contained digits, {wrong_in_words} were "
            f"spelled out in words only"
        )
    print()