"""
Smoke test 5: does context compaction work, and does memory survive it?

Waiting for the real 6830-token budget to fill would take ~65 turns. This
test shrinks the budget so compaction triggers after a handful of exchanges,
then checks the two things that actually matter:

  1. Does a fact stated in turn 1 survive after that turn has been folded
     into the chronicle and dropped? (Summarising beats truncating only if
     the answer is yes.)
  2. Does the chronicle stay bounded? It fills toward its ~80-word cap over
     the first few folds, so early compactions can be net-positive in tokens.
     What matters is that it CONVERGES: once the chronicle is at its cap,
     each fold must be net-negative. A summariser that keeps growing is worse
     than none at all, because it fails slowly enough to look like it works.

Test data below is fictional.

Run with the venv active:
    python check_context_mgmt.py

Throwaway diagnostic. Delete it later.
"""

import chat
from chat import Conversation, respond

# Force compaction early. The real budget is ~6830; this makes it trigger
# within a few turns instead of sixty-five.
#
# 350 is deliberate. At 200 the budget was below the structural floor --
# chronicle (~95 tokens) + dossier (~40) + the KEEP_VERBATIM turns (~150)
# already exceed it, so the loop could never satisfy the target and every
# turn looked like a failure. The loop was in fact doing the right thing:
# it stops rather than stripping the recent turns the model needs to resolve
# a follow-up. A budget below the floor tests nothing except the test.
chat.HISTORY_BUDGET = 350
chat.SUMMARISE_BATCH = 2
chat.KEEP_VERBATIM = 2

# The smallest history the system can hold: chronicle at its cap, plus the
# dossier, plus the turns that are never eligible for folding. Printed so a
# budget set below it is obvious rather than mysterious.
print(f"HISTORY_BUDGET lowered to {chat.HISTORY_BUDGET} to force compaction.")
print(
    f"Structural floor is roughly "
    f"{chat.CHRONICLE_WORD_CAP * 4 // 3 + 40 + chat.KEEP_VERBATIM * 75} tokens "
    f"(chronicle + dossier + {chat.KEEP_VERBATIM} verbatim turns).\n"
)

CONVERSATION = [
    # Turn 1 plants the facts we test for at the end.
    "My name is Wren and I'm restoring a 1970s synthesizer.",
    "What's a good way to stay focused on a long project?",
    "I'm also learning to make sourdough.",
    "Any advice on juggling two hobbies at once?",
    "What's the capital of France?",
    "I work night shifts on Tuesdays and Thursdays.",
    "How do I stop procrastinating?",
    "The synth has a broken filter board.",
    "What's a good soldering iron temperature for old PCBs?",
    "My starter smells like acetone, is that bad?",
    "Do you think I'm taking on too much?",
    "What's 15 percent of 240?",
    # The test: turn 1 was folded into the chronicle many turns ago.
    "What is my name, and what am I restoring?",
]

conversation = Conversation()
previous_tokens = 0

for i, user_message in enumerate(CONVERSATION, start=1):
    turn = respond(conversation, user_message)
    folded = conversation.last_folded

    print("=" * 70)
    print(f"TURN {i}")
    print(f"YOU:    {user_message}")
    print(f"FISCHL: {turn.fischl}")
    print(f"OZ:     {turn.oz}")
    print()

    tokens = conversation.history_tokens()

    if folded:
        delta = tokens - previous_tokens
        chronicle_words = len(conversation.chronicle.split())
        direction = "DOWN" if delta < 0 else "UP"
        if delta < 0:
            verdict = "DOWN (good)"
        elif chronicle_words >= 65:
            verdict = "UP (BAD -- chronicle is at its cap)"
        else:
            verdict = "UP, chronicle still filling (expected)"
        print(f"  >>> COMPACTED: {folded} turn(s) folded")
        print(f"      tokens {previous_tokens} -> {tokens}  [{verdict}]")

    print(
        f"  [verbatim turns: {len(conversation.turns)} | "
        f"history tokens: {tokens}/{chat.HISTORY_BUDGET}]"
    )
    if conversation.facts:
        print(f"  [dossier]\n    " + conversation.facts.replace("\n", "\n    "))
    if conversation.chronicle:
        words = len(conversation.chronicle.split())
        print(f"  [chronicle: {words} words] {conversation.chronicle}")
    print()

    previous_tokens = tokens

print("=" * 70)
print("FINAL STATE")
print(f"  verbatim turns kept: {len(conversation.turns)}")
print(f"  chronicle words:     {len(conversation.chronicle.split())}")
print(f"  history tokens:      {conversation.history_tokens()}/{chat.HISTORY_BUDGET}")
print(f"  dossier:\n    " + (conversation.facts or "(empty)").replace("\n", "\n    "))
print()
print("PASS CONDITIONS")
print("  1. The last answer names Wren and the synthesizer.")
print("  2. The dossier holds name and project, with no questions recorded")
print("     as facts and no padding lines.")
print("  3. Chronicle stayed at or under 70 words for the whole run.")
print("  4. Once the chronicle reached its cap, every fold was net-negative.")
print("  5. Fischl never mentioned a chronicle, dossier or record.")