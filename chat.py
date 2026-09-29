"""
Chat engine for the Fischl & Oz chatbot.

Owns the model client, the two-stage prompt chain, and the token accounting
that the context-management layer will sit on top of.

THE CHAIN
---------
    user message
        -> stage 1: FISCHL  (system + pinned few-shot + history + user)
        -> stage 2: OZ      (system + user message + Fischl's reply)

Stage 2 consumes stage 1's output. Each call has exactly one job, which is
why this is a chain and not one call asked to produce two voices at once.

WHAT LIVES IN THE CONTEXT WINDOW
--------------------------------
Every Fischl call assembles the same four parts, in this order:

    [ system prompt ][ pinned few-shot ][ conversation history ][ new message ]
      ~350 tokens      ~450 tokens        grows                   varies

Only the third part is allowed to shrink. The few-shot examples are pinned:
dropping them is what causes the persona to drift, so they must never be the
thing that gets evicted when space runs short.
"""

from dataclasses import dataclass

from openai import OpenAI

from persona import FISCHL_FEWSHOT, FISCHL_SYSTEM, OZ_EXAMPLES, OZ_SYSTEM

MODEL = "fischl-llama"
BASE_URL = "http://localhost:11434/v1"

# Must match PARAMETER num_ctx in the Modelfile. The model itself supports
# 128k; 8192 is what the KV cache will fit in 8 GB of VRAM alongside the
# weights. See the Modelfile for the arithmetic.
CONTEXT_LIMIT = 8192

# Room reserved for the model's own reply, so we never fill the window so
# full that there is nowhere left to generate into.
RESERVED_FOR_REPLY = 512

# Measured fixed overhead: system prompt plus pinned few-shot examples came to
# 815 prompt tokens on an empty history, of which the first user message was a
# small part. Rounded up.
FIXED_OVERHEAD = 850

# What is left for conversation history and the chronicle.
HISTORY_BUDGET = CONTEXT_LIMIT - FIXED_OVERHEAD - RESERVED_FOR_REPLY  # 6830

# Once history exceeds the budget, this many of the oldest turns are folded
# into the chronicle at a time. Summarising in batches rather than one turn at
# a time avoids paying for a summarisation call on every single message once
# the window is full.
SUMMARISE_BATCH = 6

# Turns to always keep verbatim, however full the window gets. Recent context
# is what the model needs for pronouns and follow-ups to resolve, so it is
# never summarised away.
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
    Rough token count without loading a tokenizer.

    Llama 3.1 uses a 128k-vocabulary BPE tokenizer that we would have to pull
    in transformers to use properly, which is a heavy dependency for a
    budgeting heuristic. English averages close to 4 characters per token, so
    that is the estimate used here.

    This is deliberately an estimate and it is deliberately checked: every
    call compares it against the true prompt_tokens the server reports, and
    `last_estimate_error()` exposes the difference. The heuristic errs low on
    ornate vocabulary and on German, both of which this persona produces in
    quantity, so the reserve above exists partly to absorb that.
    """
    return len(text) // 4 + 1


def count_messages(messages: list[dict]) -> int:
    """Estimated tokens for a full message list, including per-message overhead."""
    # Each message carries role and delimiter tokens on top of its content;
    # 4 per message is the usual approximation.
    return sum(estimate_tokens(m["content"]) + 4 for m in messages)


# Populated after each Fischl call so the UI can show estimate vs actual.
_last_estimated = 0
_last_actual = 0


def last_estimate_error() -> tuple[int, int, float]:
    """
    (estimated, actual, percent_error) for the most recent Fischl call.

    `actual` is 0 when the server reported no usage, which is always the case
    for a streamed reply. The estimate is still returned in that case: an
    earlier version returned (0, 0, 0.0) whenever actual was missing, which
    silently zeroed the estimate too and left the interface reporting an empty
    context window on every turn.
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

# The dossier is small and pinned. It is never summarised away, because the
# chronicle demonstrably cannot be trusted to hold user facts: it records the
# shape of the conversation, and in testing it kept Fischl's advice while
# dropping the user's name -- which then produced a bot that denied knowing a
# name sitting in its own context.
#
# The prompt above carries no worked example containing a name. It used to:
#   - padding. "name: Wren" is a fact; "Wren's name is Wren" states it twice.
# The model lifted the name straight out of the illustration and filed it as
# the user's, so anyone who had not introduced themselves was greeted as Wren.
# This is the same failure as every other example-leak in this project -- a
# concrete noun in a prompt is a candidate for output -- and it is worth
# noting that it happened in a prompt with no persona and a temperature of
# 0.1, which is to say the effect is not a quirk of creative sampling.
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

# Hard ceiling on the chronicle, enforced in code rather than trusted to the
# prompt. 70 words is roughly 95 tokens.
CHRONICLE_MAX_TOKENS = 160
CHRONICLE_WORD_CAP = 70


def build_chronicle(existing: str, turns: list[Turn]) -> str:
    """
    Fold a batch of old turns into the running chronicle.

    This is the third prompting technique in the app, and the reason the
    conversation can outlive its own context window. Turns that get folded in
    are then dropped from the verbatim history: what survives is the facts,
    not the phrasing.

    The rewrite-don't-append instruction is emphatic because the first
    version of this prompt asked for an "updated chronicle" and the model
    appended a new paragraph every time. The chronicle then grew faster than
    compaction reclaimed space, and total context went UP after each
    compaction (392 -> 411 -> 474 tokens in testing) instead of down. A
    summariser that grows without bound is worse than no summariser, because
    it fails slowly enough to look like it is working.
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
    # In testing the model treated 80 words as a target to approach and then
    # drift past (47 -> 87 words over four folds). A summariser that creeps
    # past its cap defeats the purpose of having one, and a prompt instruction
    # is a request, not a guarantee.
    words = text.split()
    if len(words) > CHRONICLE_WORD_CAP:
        text = " ".join(words[:CHRONICLE_WORD_CAP]).rstrip(",;:") + "."

    return text


@dataclass
class Conversation:
    """
    Holds everything that has to fit in the context window, and keeps it
    fitting.

    What happens as the window fills:

      1. Each turn adds roughly 105 tokens of history (measured).
      2. While history stays under HISTORY_BUDGET, nothing is discarded --
         every turn goes to the model verbatim.
      3. When history crosses the budget, the oldest SUMMARISE_BATCH turns
         are folded into the chronicle and dropped from verbatim history.
         The most recent KEEP_VERBATIM turns are never eligible, so
         follow-ups and pronouns still resolve against real text.
      4. The chronicle itself is capped by its own prompt (under 120 words)
         and is rewritten rather than appended to, so it does not grow without
         bound the way a transcript does.

    The result is that the system prompt and few-shot examples are never the
    thing that gets evicted. Dropping those is what makes a persona drift, and
    a naive sliding window over the whole message list would drop them first.
    """

    turns: list[Turn]
    chronicle: str = ""
    facts: str = ""
    last_folded: int = 0

    def __init__(self) -> None:
        self.turns = []
        self.chronicle = ""
        self.facts = ""
        self.last_folded = 0

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
        Fold the oldest turns into the chronicle until the budget is met.

        Returns the number of turns folded, so the caller can tell the user it
        happened. Silent truncation is how conversations mysteriously lose
        their memory; this is deliberately observable.

        This loops rather than folding one batch per message. A single fold
        per turn cannot catch up once the budget is already exceeded -- it
        reclaims a fixed amount while new turns keep arriving -- so the window
        creeps past its limit and stays there. The loop also terminates on the
        KEEP_VERBATIM floor, which means the budget can legitimately be
        exceeded if the last few turns are enormous on their own. That is the
        correct failure: better to overrun slightly than to strip the recent
        context the model needs to resolve a follow-up.
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
        Record a turn, update the dossier, and compact if needed.

        The dossier is refreshed on EVERY turn, not only when compaction runs.
        It was originally built during compaction, on the reasoning that its
        job is to rescue facts from turns about to be evicted -- which is
        true, but meant it stayed empty for the first sixty-odd turns of any
        real conversation. Someone who says "I like pizza" on turn three and
        looks at the panel should see it there, not a promise that it will
        appear once the window fills. The cost is one short extra call per
        turn, which runs after both voices have finished streaming and so is
        not in the way of anything the reader is waiting for.
        """
        self.turns.append(turn)
        self.facts = build_facts(self.facts, [turn])
        self.last_folded = self.compact()
        return self.last_folded


def build_fischl_messages(
    history: list[Turn],
    user_message: str,
    chronicle: str = "",
    facts: str = "",
) -> list[dict]:
    """
    Assemble the message list for a Fischl call.

    Order matters. The dossier and chronicle sit after the few-shot examples
    and before the verbatim history, so they read as established background
    rather than as dialogue the model is being asked to continue.
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

    for turn in history:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.fischl})

    messages.append({"role": "user", "content": user_message})
    return messages


def fischl_reply(
    history: list[Turn],
    user_message: str,
    chronicle: str = "",
    facts: str = "",
    temperature: float = 0.8,
) -> str:
    """Stage 1: the in-character reply."""
    global _last_estimated, _last_actual

    messages = build_fischl_messages(history, user_message, chronicle, facts)
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

    Deliberately does NOT receive the conversation history. His job is to
    respond to one reply, and giving him the backlog invites him to summarise
    the conversation instead of answering the turn in front of him.

    Temperature is 0.7, not the 0.3 it was for several rounds. Low temperature
    was chosen on the theory that this stage should be faithful rather than
    creative -- but faithful and bland turned out to be the same setting. At
    0.3 the model picked the most probable phrasing every time, which for a
    restatement task is "She says that you should...", and it produced that
    shape in thirteen consecutive turns while ignoring six few-shot examples
    that did no such thing. Sampling temperature was quietly overriding the
    examples. Character lives in the less-probable choice, so this stage needs
    room to make one.

    An earlier version also gave Oz a verification mandate -- correct her if
    she got a fact wrong -- on the theory that a second pass would catch
    errors the ornate first pass introduced. It failed badly. Told to look for
    errors, the model invented them: it "corrected" her about her own Vision
    with fabricated lore, criticised her prose style instead of translating,
    and in one case emitted no translation at all. An 8B model asked to
    find mistakes will produce mistakes to find.

    Oz gets few-shot examples of his own for the same reason Fischl does.
    Instructing him to vary his opening produced "Mein Fräulein is saying
    that" in every single reply -- naming any opening in the prompt made that
    opening universal. The examples below show three different shapes,
    including a one-word answer, without naming any of them.
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
# The blocking `respond` above is what the diagnostics use: it is simpler to
# reason about and it reports exact prompt_tokens from the server. The UI uses
# the streaming pair below instead, because a reply that appears a word at a
# time reads as speech, and one that appears all at once after a long pause
# reads as a page load.
#
# The tradeoff is token accounting. A streamed response carries no usage
# block, so the exact prompt_tokens the server counted is not available until
# the stream ends -- and Ollama does not return one at all. The UI therefore
# falls back to the 4-chars-per-token estimate and labels it with a "≈".
# That estimate was measured against the server's own count at roughly +2%,
# erring high, so the meter slightly over-reports rather than hiding an
# overflow.


def fischl_reply_stream(
    history: list[Turn],
    user_message: str,
    chronicle: str = "",
    facts: str = "",
    temperature: float = 0.8,
):
    """Stage 1, yielding the reply text as it grows."""
    global _last_estimated, _last_actual

    messages = build_fischl_messages(history, user_message, chronicle, facts)
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


def stream_turn(conversation: Conversation, user_message: str):
    """
    Run the chain, yielding (fischl_so_far, oz_so_far) as each stage streams.

    `oz_so_far` has three states, which is how the UI knows what to draw:
        None  -- Fischl is still speaking
        ""    -- Fischl has finished, Oz has been asked and is thinking
        text  -- Oz is speaking

    The empty-string state exists because the gap before Oz's first token is
    dead air inside `oz_reply_stream`, and nothing would be yielded during it.
    Announcing the gap explicitly is what lets the interface show a second
    typing indicator instead of freezing on Fischl's finished reply.

    The turn is recorded and compaction runs only once both stages finish, so
    a half-generated exchange never enters the history.
    """
    fischl_text = ""
    for fischl_text in fischl_reply_stream(
        conversation.turns, user_message, conversation.chronicle, conversation.facts
    ):
        yield fischl_text, None

    yield fischl_text, ""

    oz_text = ""
    for oz_text in oz_reply_stream(user_message, fischl_text):
        yield fischl_text, oz_text

    conversation.add(Turn(user=user_message, fischl=fischl_text, oz=oz_text))
    yield fischl_text, oz_text