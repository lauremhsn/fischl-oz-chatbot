"""
Chat engine for the Fischl & Oz chatbot.

Two stages, chained:

    user message
        -> stage 1: FISCHL  (system + pinned few-shot + history + user)
        -> stage 2: OZ      (system + user message + Fischl's reply)

Every Fischl call assembles the same four parts, in this order:

    [ system prompt ][ pinned few-shot ][ conversation history ][ new message ]
      ~350 tokens      ~450 tokens        grows                   varies

Only the third part shrinks. The few-shot examples are pinned so they are
never evicted when space runs short.
"""

from dataclasses import dataclass

from openai import OpenAI

from persona import FISCHL_FEWSHOT, FISCHL_SYSTEM, OZ_EXAMPLES, OZ_SYSTEM

MODEL = "fischl-llama"
BASE_URL = "http://localhost:11434/v1"

# Must match PARAMETER num_ctx in the Modelfile. The model supports 128k; 8192
# is what the KV cache fits in 8 GB of VRAM alongside the weights.
CONTEXT_LIMIT = 8192

# Held back for the model's own reply.
RESERVED_FOR_REPLY = 512

# Measured: system prompt plus pinned few-shot came to 815 prompt tokens on an
# empty history. Rounded up.
FIXED_OVERHEAD = 850

# What is left for conversation history, chronicle and dossier.
HISTORY_BUDGET = CONTEXT_LIMIT - FIXED_OVERHEAD - RESERVED_FOR_REPLY  # 6830

# Oldest turns folded into the chronicle per compaction pass.
SUMMARISE_BATCH = 6

# Most recent turns, never summarised, so follow-ups and pronouns resolve.
KEEP_VERBATIM = 4

client = OpenAI(base_url=BASE_URL, api_key="ollama")


@dataclass
class Turn:
    """One exchange. Fischl's reply is what goes back into history."""

    user: str
    fischl: str
    oz: str = ""
    reasoning: str = ""


def estimate_tokens(text: str) -> int:
    """
    Rough token count without loading a tokenizer: English averages close to
    4 characters per token. Checked against the server's own prompt_tokens on
    every non-streamed call, and errs high by roughly 2%.
    """
    return len(text) // 4 + 1


def count_messages(messages: list[dict]) -> int:
    """Estimated tokens for a message list, plus 4 per message for role and
    delimiter tokens."""
    return sum(estimate_tokens(m["content"]) + 4 for m in messages)


# Populated after each Fischl call so the UI can show estimate vs actual.
_last_estimated = 0
_last_actual = 0


def last_estimate_error() -> tuple[int, int, float]:
    """
    (estimated, actual, percent_error) for the most recent Fischl call.
    `actual` is 0 when the server reported no usage, which is always the case
    for a streamed reply.
    """
    if _last_actual == 0:
        return (_last_estimated, 0, 0.0)
    error = (_last_estimated - _last_actual) / _last_actual * 100
    return (_last_estimated, _last_actual, error)


FACTS_SYSTEM = """You maintain a short factual dossier about the person in a \
conversation. You will be given the current dossier and some new exchanges.

Output the updated dossier as plain lines, one fact per line, no bullets or \
numbering.

RECORD ONLY stable facts they stated about themselves:
- their name, if they gave one, recorded as "name: " followed by it
- what they are working on, studying, building or dealing with
- ongoing commitments, constraints and circumstances
- stated preferences and decisions

DO NOT RECORD:
- questions they asked. "How do I stay focused?" is a question, not a fact \
about them. Asking about a topic does not make the topic a property of the \
person.
- anything Fischl said, advised or decreed
- anything you inferred rather than were told
- transient states, pleasantries or conversational filler
- padding: never state the same fact twice in different words.

NEVER invent a value. If the person has not told you their name, the dossier
has no name line at all. Do not fill a field with a plausible placeholder.

Rules:
- Merge new information into existing lines rather than adding duplicates.
- If a fact was corrected, keep the corrected version only.
- 8 lines maximum. If you must cut, cut the least specific.
- Output the dossier only. No preamble, no commentary."""

# The dossier is small and pinned, and is never summarised away: the chronicle
# records the shape of a conversation rather than the user's facts.
FACTS_MAX_TOKENS = 128


def build_facts(existing: str, turns: list[Turn]) -> str:
    """Update the pinned dossier of facts about the user."""
    transcript = "\n".join(f"Person: {t.user}" for t in turns)

    prompt = (
        f"Current dossier:\n{existing or '(empty)'}\n\n"
        f"New messages from the person:\n{transcript}\n\n"
        f"Output the updated dossier."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": FACTS_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=FACTS_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


CHRONICLE_SYSTEM = """You maintain a compact factual record of a conversation \
between Fischl and the person she is speaking with.

You will be given the current record and some further exchanges. Output a \
SINGLE REWRITTEN RECORD covering everything, old and new together.

THIS IS A REWRITE, NOT AN APPENDIX. Do not add a paragraph to what you were \
given. Merge the new exchanges into the existing record and compress the \
whole thing. If the record is already near its limit, drop the least useful \
older details to make room. The record must not grow longer over time.

KEEP:
- Concrete facts the person stated about themselves: names, projects, plans, \
preferences, problems, decisions.
- Anything Fischl promised, advised, or agreed to, stated in plain terms.

DISCARD:
- Fischl's ornate phrasing. Record that she advised a regular routine, not \
that she invoked the sacred oil of routine.
- Pleasantries, flourishes, and anything with no bearing on what comes next.

- FORMAT: one paragraph of plain prose, third person, past tense.
- 70 WORDS MAXIMUM. This is a hard limit and it is checked. Count before you
  answer. If the merged record would run longer, cut older detail until it
  fits.
- No headings, no bullet points, no preamble. Output the record only."""

# 70 words is roughly 95 tokens.
CHRONICLE_MAX_TOKENS = 160
CHRONICLE_WORD_CAP = 70


def build_chronicle(existing: str, turns: list[Turn]) -> str:
    """
    Fold a batch of old turns into the running chronicle and return the
    rewritten record. Folded turns are then dropped from verbatim history, so
    what survives is the facts rather than the phrasing.
    """
    transcript = "\n".join(
        f"Person: {t.user}\nFischl: {t.fischl}" for t in turns
    )

    prompt = (
        f"Current record:\n{existing or '(nothing recorded yet)'}\n\n"
        f"Further exchanges to merge in:\n{transcript}\n\n"
        f"Output the single rewritten record, 80 words maximum."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CHRONICLE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=CHRONICLE_MAX_TOKENS,
    )
    text = response.choices[0].message.content.strip()

    # The word limit is enforced here as well as asked for in the prompt.
    words = text.split()
    if len(words) > CHRONICLE_WORD_CAP:
        text = " ".join(words[:CHRONICLE_WORD_CAP]).rstrip(",;:") + "."

    return text


@dataclass
class Conversation:
    """
    Holds everything that has to fit in the context window, and keeps it
    fitting.

      1. Each turn adds roughly 105 tokens of history (measured).
      2. While history stays under HISTORY_BUDGET nothing is discarded.
      3. Once it crosses the budget, the oldest SUMMARISE_BATCH turns are
         folded into the chronicle and dropped from verbatim history. The most
         recent KEEP_VERBATIM turns are never eligible.
      4. The chronicle is rewritten rather than appended to, and capped at
         CHRONICLE_WORD_CAP words, so it does not grow without bound.

    The system prompt and few-shot examples are never evicted, which a naive
    sliding window over the whole message list would do first.
    """

    turns: list[Turn]
    chronicle: str = ""
    facts: str = ""
    last_folded: int = 0
    last_error: str = ""

    def __init__(self) -> None:
        self.turns = []
        self.chronicle = ""
        self.facts = ""
        self.last_folded = 0
        self.last_error = ""

    def history_tokens(self) -> int:
        """Estimated tokens currently held by history, chronicle and dossier."""
        total = estimate_tokens(self.chronicle) if self.chronicle else 0
        total += estimate_tokens(self.facts) if self.facts else 0
        for turn in self.turns:
            total += estimate_tokens(turn.user) + estimate_tokens(turn.fischl) + 8
        return total

    def needs_compaction(self) -> bool:
        return self.history_tokens() > HISTORY_BUDGET

    def compact(self) -> int:
        """
        Fold the oldest turns into the chronicle until the budget is met, and
        return the number folded so the interface can report it.

        The loop also terminates on the KEEP_VERBATIM floor, so the budget can
        legitimately be exceeded if the last few turns are enormous on their
        own.
        """
        folded = 0

        while self.needs_compaction():
            eligible = len(self.turns) - KEEP_VERBATIM
            if eligible <= 0:
                break

            batch_size = min(SUMMARISE_BATCH, eligible)
            batch = self.turns[:batch_size]

            self.chronicle = build_chronicle(self.chronicle, batch)
            self.turns = self.turns[batch_size:]
            folded += batch_size

        return folded

    def add(self, turn: Turn) -> int:
        """
        Record a turn, refresh the dossier, and compact if needed.

        The dossier is refreshed on every turn rather than only during
        compaction, so a fact stated on turn three appears in the panel
        immediately. The extra call runs after both voices have finished
        streaming.
        """
        self.turns.append(turn)
        self.last_error = ""
        try:
            self.facts = build_facts(self.facts, [turn])
        except Exception as exc:
            self.last_error = f"dossier update failed — {type(exc).__name__}: {exc}"

        try:
            self.last_folded = self.compact()
        except Exception as exc:
            self.last_error = f"compaction failed — {type(exc).__name__}: {exc}"

        return self.last_folded


REASONING_SYSTEM = """You work problems out carefully, in plain language, \
before anyone answers them.

Given the question, write the steps needed to reach the answer. Number them. \
Be brief -- a few short lines, not an essay. Do arithmetic one step at a time \
and check each step before moving on. If the question needs no working out, \
say so in one line.

Finish with a final line in exactly this form:

ANSWER: <the answer, stated plainly>

No character, no flourish, no archaic language. This is working, not speech."""


def reasoning_stream(user_message: str, facts: str = "", temperature: float = 0.2):
    """
    Optional first stage: work the problem out in a plain voice, before Fischl
    speaks. Keeps the arithmetic out of the ornate voice, which measurably
    costs accuracy. Temperature is low because this stage wants the likeliest
    next token rather than an interesting one.
    """
    messages = [{"role": "system", "content": REASONING_SYSTEM}]
    if facts:
        messages.append(
            {"role": "system", "content": f"What you know about them:\n{facts}"}
        )
    messages.append({"role": "user", "content": user_message})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=400,
        stream=True,
    )

    text = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            text += delta
            yield text


def build_fischl_messages(
    history: list[Turn],
    user_message: str,
    chronicle: str = "",
    facts: str = "",
    reasoning: str = "",
) -> list[dict]:
    """
    Assemble the message list for a Fischl call.

    The dossier and chronicle sit after the few-shot examples and before the
    verbatim history, so they read as established background rather than as
    dialogue to continue.
    """
    messages = [{"role": "system", "content": FISCHL_SYSTEM}]
    messages.extend(FISCHL_FEWSHOT)

    if facts:
        messages.append(
            {
                "role": "system",
                "content": (
                    "You remember the following about the person you are "
                    "speaking with. They told you these things themselves. "
                    "Treat them as your own memory and never mention how you "
                    "came to remember them.\n\n"
                    f"{facts}"
                ),
            }
        )

    if chronicle:
        messages.append(
            {
                "role": "system",
                "content": (
                    "You also remember what has already passed between you "
                    "earlier in this conversation. Again, this is simply your "
                    "memory.\n\n"
                    f"{chronicle}"
                ),
            }
        )

    # Only Fischl's replies go back into history, never Oz's gloss and never a
    # reasoning trace: those are derived from her turn rather than part of the
    # conversation.
    for turn in history:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.fischl})

    if reasoning:
        messages.append(
            {
                "role": "system",
                "content": (
                    "You have already worked this out, privately:\n\n"
                    f"{reasoning}\n\n"
                    "Announce that conclusion in your own voice. The figures "
                    "above are correct — carry them across exactly as they "
                    "stand and do not work them out again."
                ),
            }
        )

    messages.append({"role": "user", "content": user_message})
    return messages


def fischl_reply(
    history: list[Turn],
    user_message: str,
    chronicle: str = "",
    facts: str = "",
    reasoning: str = "",
    temperature: float = 0.8,
) -> str:
    """Stage 1: the in-character reply. Parameters are kept in step with
    `fischl_reply_stream`."""
    global _last_estimated, _last_actual

    messages = build_fischl_messages(
        history, user_message, chronicle, facts, reasoning
    )
    _last_estimated = count_messages(messages)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=RESERVED_FOR_REPLY,
    )

    if response.usage is not None:
        _last_actual = response.usage.prompt_tokens

    return response.choices[0].message.content.strip()


def oz_reply(user_message: str, fischl_text: str, temperature: float = 0.7) -> str:
    """
    Stage 2: Oz.

    Receives one user message and one reply, and deliberately not the
    conversation history, which would invite him to summarise the conversation
    instead of answering the turn in front of him. Temperature is 0.7 because
    at 0.3 the model picked the most probable restatement every time and
    ignored the few-shot examples. He gets examples of his own for the same
    reason Fischl does.
    """
    messages = [{"role": "system", "content": OZ_SYSTEM}]
    messages.extend(OZ_EXAMPLES)
    messages.append(
        {
            "role": "user",
            "content": (
                f"They asked: {user_message}\n\n"
                f"She replied: {fischl_text}"
            ),
        }
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=128,
    )
    return response.choices[0].message.content.strip()


def respond(conversation: Conversation, user_message: str) -> Turn:
    """Run the full chain, record the turn, and compact if needed."""
    fischl_text = fischl_reply(
        conversation.turns,
        user_message,
        conversation.chronicle,
        conversation.facts,
    )
    oz_text = oz_reply(user_message, fischl_text)
    turn = Turn(user=user_message, fischl=fischl_text, oz=oz_text)
    conversation.add(turn)
    return turn


# --------------------------------------------------------------------------
# Streaming variants, used by the web UI
# --------------------------------------------------------------------------
#
# The diagnostics use the blocking `respond` above, which reports exact
# prompt_tokens from the server. The UI uses the streaming pair below so a
# reply appears a word at a time. A streamed response carries no usage block,
# so the UI falls back to the 4-chars-per-token estimate and labels it "≈".


def fischl_reply_stream(
    history: list[Turn],
    user_message: str,
    chronicle: str = "",
    facts: str = "",
    reasoning: str = "",
    temperature: float = 0.8,
):
    """Stage 1, yielding the reply text as it grows."""
    global _last_estimated, _last_actual

    messages = build_fischl_messages(
        history, user_message, chronicle, facts, reasoning
    )
    _last_estimated = count_messages(messages)
    _last_actual = 0  # no usage block arrives with a stream

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=RESERVED_FOR_REPLY,
        stream=True,
    )

    text = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            text += delta
            yield text


def oz_reply_stream(user_message: str, fischl_text: str, temperature: float = 0.7):
    """Stage 2, yielding the reply text as it grows."""
    messages = [{"role": "system", "content": OZ_SYSTEM}]
    messages.extend(OZ_EXAMPLES)
    messages.append(
        {
            "role": "user",
            "content": f"They asked: {user_message}\n\nShe replied: {fischl_text}",
        }
    )

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=128,
        stream=True,
    )

    text = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            text += delta
            yield text


def stream_turn(conversation: Conversation, user_message: str, reasoning: bool = False):
    """
    Run the chain, yielding (stage, text) as it goes.

        ("wait",   name)  -- a call has been made and nothing has come back
                             yet; the interface shows a typing indicator
        ("reason", text)  -- the plain-voice working, growing (reasoning mode)
        ("fischl", text)  -- her reply, growing
        ("oz",     text)  -- his gloss, growing
        ("done",   "")    -- turn recorded, dossier updated, compaction run

    The turn is recorded, the dossier updated and compaction run only once
    every stage has finished, so a half-generated exchange never enters
    history.
    """
    reasoning_text = ""
    if reasoning:
        yield "wait", "reason"
        for reasoning_text in reasoning_stream(user_message, conversation.facts):
            yield "reason", reasoning_text

    yield "wait", "fischl"
    fischl_text = ""
    for fischl_text in fischl_reply_stream(
        conversation.turns,
        user_message,
        conversation.chronicle,
        conversation.facts,
        reasoning_text,
    ):
        yield "fischl", fischl_text

    yield "wait", "oz"
    oz_text = ""
    for oz_text in oz_reply_stream(user_message, fischl_text):
        yield "oz", oz_text

    conversation.add(
        Turn(
            user=user_message,
            fischl=fischl_text,
            oz=oz_text,
            reasoning=reasoning_text,
        )
    )
    yield "done", ""