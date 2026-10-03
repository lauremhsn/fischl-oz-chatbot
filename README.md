# ✦ Prinzessin der Verurteilung

A persona-driven web chat app built on an open-source LLM running entirely on
your own machine.

**Fischl** answers every question in self-declared royal grandeur. Her raven
**Oz** follows with one dry line telling you what she actually meant.

> **You:** What's 12 times 14?
> **Fischl:** A trifling sum. 12 times 14 is 168. So shall it be.
> **Oz:** She's right, for what it's worth.

Built for **EECE 503P / 798S: Agentic Systems**, Assignment 2.
Model: **Llama 3.1 8B**, self-hosted via [Ollama](https://ollama.com). No API
key, no quota, nothing leaves the machine.

---

## Contents

- [Quick start](#quick-start)
- [Running it in Google Colab](#running-it-in-google-colab)
- [Why this model](#why-this-model)
- [The persona twist](#the-persona-twist)
- [Prompting techniques](#prompting-techniques)
- [Context handling](#context-handling)
- [Reasoning mode](#reasoning-mode-stretch-goal)
- [Project layout](#project-layout)
- [Diagnostics](#diagnostics)
- [Known limitations](#known-limitations)
- [Credits](#credits)

---

## Quick start

These steps assume nothing. Follow them in order.

### What you need first

- **Python 3.10 or newer.** Check with `python --version`. If that fails,
  install it from [python.org](https://www.python.org/downloads/) and tick
  *"Add Python to PATH"* during setup.
- **About 6 GB of free disk space** for the model.
- A GPU helps but is not required. The app was developed on an 8 GB RTX 4070
  laptop GPU; on CPU it works but replies are slow.

### 1. Install Ollama

Ollama is what runs the model on your machine. Download it from
[ollama.com/download](https://ollama.com/download) and install it.

Verify it is running:

```bash
ollama --version
```

If that prints a version number, you are set. On Windows, Ollama runs in the
background after install — look for its icon in the system tray.

### 2. Get the code

```bash
git clone https://github.com/lauremhsn/fischl-oz-chatbot
cd fischl-oz-chatbot
```

### 3. Build the model

Two steps. First download the base model — this is the slow part, roughly
4.9 GB:

```bash
ollama pull llama3.1:8b
```

Then build the derived model this app uses:

```bash
ollama create fischl-llama -f Modelfile
```

<details>
<summary><b>Why the second step exists</b> (click to expand)</summary>

The `Modelfile` is two lines:

```
FROM llama3.1:8b
PARAMETER num_ctx 8192
```

It sets the context window to 8192 tokens. Ollama defaults to **4096**, and it
**silently ignores** `num_ctx` when passed through the OpenAI-compatible
endpoint this app uses — the request succeeds, the setting does nothing. A
`Modelfile` is the only reliable way to set it, and everything in
[Context handling](#context-handling) depends on the window actually being
8192.

If you already had the model loaded at the old context size, run `ollama stop
fischl-llama` first — a resident model will not reload with a new window.

</details>

Confirm it worked:

```bash
ollama run fischl-llama "hi"
ollama ps
```

The `ollama ps` output should show `CONTEXT  8192`. If it says 4096, the
Modelfile step did not take.

### 4. Set up Python

```bash
python -m venv .venv
```

Activate it:

| Platform | Command |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Your prompt should now start with `(.venv)`. Then:

```bash
pip install -r requirements.txt
```

### 5. Run it

```bash
python app.py
```

Open **http://127.0.0.1:7860** in a browser.

> **If you get `ModuleNotFoundError: No module named 'gradio'`** — the virtual
> environment is not active. Look for `(.venv)` at the start of your prompt and
> redo step 4.

> **If replies never arrive** — Ollama is not running. Check with `ollama ps`.

### Optional flags

Inside `app.py`, the `demo.launch(...)` call at the bottom accepts:

- `share=True` — also returns a public URL (lives ~72 hours) that tunnels to
  your machine. Useful for showing the app to someone without asking them to
  install anything.
- `server_name="0.0.0.0"` — makes it reachable from a phone on the same wifi.
- `pwa=True` — already enabled; lets the app install to a phone home screen.

---

## Running it in Google Colab

`Mohsen_LLM_Chatbot.ipynb` runs the whole thing from a notebook, including
installing Ollama and pulling the model. Nothing needs to be installed locally.

Open it directly:

```
https://colab.research.google.com/github/lauremhsn/fischl-oz-chatbot/blob/main/Mohsen_LLM_Chatbot.ipynb
```

Set **Runtime → Change runtime type → T4 GPU** first. Colab's T4 has 16 GB,
comfortably more than the 8 GB the context budget was tuned for.

The notebook also walks through the persona, the context-management machinery
and the reasoning-mode comparison with live output, then launches the app with
a public link.

---

## Why this model

**Llama 3.1 8B**, open weights, served locally.

| Consideration | Reasoning |
|---|---|
| Open source | Required by the assignment. Llama 3.1's weights are published and runnable offline. |
| Size | 8B at 4-bit quantisation is ~4.9 GB — it fits on a laptop GPU. A 70B model would not. |
| Local hosting | No API key, no rate limit, no per-token cost, and no conversation data leaving the machine. |
| Instruction tuning | The chat-tuned variant follows system prompts and multi-turn structure, which a base model does not. |

**What was tried first:** the Hugging Face Inference API. The free tier is now
$0.10/month of credit, and HF-Inference serves mostly CPU or small models, so
an 8B chat model was not realistically available on it. Local hosting removed
the constraint entirely.

---

## The persona twist

Fischl is a character who never breaks from an elaborate self-appointed title.
That is funny for one message and exhausting for ten — you stop being able to
*use* the bot. Oz exists to solve that: he restates her in plain words, so
every exchange is both in character and actually informative.

The implementation point is that this is **two model calls with two system
prompts**, not one call asked to produce two voices. Each call has exactly one
job. The interface renders them as two separate bubbles so the chain is visible
rather than hidden — which also required `group_consecutive_messages=False`,
since Gradio merges consecutive same-role messages by default and would glue
the two voices back together.

```
user message
    │
    ├─▶ stage 1: FISCHL   (system + pinned few-shot + history + message)
    │                              │
    │                              ▼
    └─▶ stage 2: OZ       (system + examples + message + Fischl's reply)
```

Oz deliberately does **not** receive the conversation history. Given the
backlog he summarises the conversation instead of answering the turn in front
of him.

---

## Prompting techniques

Three, with the reasoning for each.

### 1. Few-shot prompting

`FISCHL_FEWSHOT` and `OZ_EXAMPLES` in `persona.py`.

The examples do most of the work. They carry register, length range and the
knowledge boundary without naming any of it.

**This was learned by getting it wrong.** An earlier version ran to roughly 900
words of rules for Fischl alone, accumulated by adding a rule each time
something looked off. The output got steadily *worse*, and it failed in a
consistent direction: whatever vocabulary appeared in the prompt came back as a
tic.

- Banning one stock phrase produced a different stock phrase in its place.
- Listing alternative imagery "to vary it" put that imagery in every reply.
- Forbidding the model to mention a "chronicle" taught it the word. It
  announced one in turn one, before any chronicle existed.
- Telling Oz to vary his opening produced the same opening in every single
  reply.

A small model does not weigh a long instruction list evenly. It fixates on
salient terms, and **a prohibition is a mention**. The current prompts state the
minimum, name as few specific phrases as possible, and let the examples
demonstrate what a prohibition cannot.

The same effect appeared at temperature 0.1 in a prompt with no persona at all:
the dossier prompt once contained `name: Wren` as a worked example, and the
model filed "Wren" as the user's actual name. Any concrete noun in a prompt is a
candidate for output.

### 2. Prompt chaining

`FISCHL_SYSTEM` → `OZ_SYSTEM`.

Oz's reply is a second call consuming Fischl's output, as diagrammed above.

Two findings worth recording:

**Temperature matters more than expected.** Oz ran at 0.3 for several rounds on
the theory that a restatement task should be faithful rather than creative. But
faithful and bland turned out to be the same setting: at 0.3 the model picked
the most probable phrasing every time — *"She says that you should…"* — and
produced that shape in thirteen consecutive turns while ignoring six few-shot
examples that did no such thing. Sampling temperature was quietly overriding the
examples. Character lives in the less-probable choice, so Oz now runs at 0.7.

**Do not ask a small model to find errors.** An earlier version gave Oz a
verification mandate — correct her if she got a fact wrong. Told to look for
errors, it *invented* them: it "corrected" her about her own abilities with
fabricated detail, criticised her prose style instead of translating, and
occasionally emitted no translation at all. The mandate was removed.

### 3. Chain-of-thought

`REASONING_SYSTEM`, behind the reasoning-mode toggle. See
[Reasoning mode](#reasoning-mode-stretch-goal).

---

## Context handling

### The window

**8192 tokens**, set via `num_ctx` in the `Modelfile`.

Llama 3.1 supports 128k, but the KV cache decides what actually fits. At
roughly 128 KB per token (32 layers × 8 KV heads × 128 head dim × 2 tensors ×
2 bytes), 8192 tokens is about 1 GB of cache, which sits alongside 4.9 GB of
weights inside an 8 GB card.

### How it is spent

Every call is assembled in the same order, and only one part may shrink:

```
[ system prompt ][ pinned few-shot ][ conversation history ][ new message ]
  ~350 tokens      ~450 tokens        grows                   varies
```

| Constant | Value | Meaning |
|---|---|---|
| `CONTEXT_LIMIT` | 8192 | the window, matching the Modelfile |
| `FIXED_OVERHEAD` | 850 | measured: system prompt + pinned few-shot |
| `RESERVED_FOR_REPLY` | 512 | held back so there is room to generate into |
| `HISTORY_BUDGET` | 6830 | what remains for history |
| `SUMMARISE_BATCH` | 6 | turns folded per compaction pass |
| `KEEP_VERBATIM` | 4 | most recent turns, never summarised |
| `CHRONICLE_WORD_CAP` | 70 | hard ceiling on the running summary |

### What happens as the limit approaches

A turn costs about **104 tokens**, so roughly **65 turns** fit before anything
is discarded. Then:

1. While history stays under `HISTORY_BUDGET`, nothing is dropped — every turn
   goes to the model verbatim.
2. Once it crosses, the oldest six turns are folded into a running **chronicle**
   and removed from verbatim history.
3. The four most recent turns are never eligible, so pronouns and follow-ups
   still resolve against real text.
4. The system prompt and few-shot examples are **never** evicted. Dropping them
   is what makes a persona drift, and a naive sliding window over the message
   list would drop them first.

Compaction loops rather than folding one batch per message: a single fold per
turn cannot catch up once the budget is already exceeded, so the window creeps
past its limit and stays there.

The budget can legitimately be exceeded if the last few turns are enormous on
their own. That is the correct failure — better a slight overrun than stripping
the recent context the model needs to resolve a follow-up.

### Two memories, not one

| | Dossier | Chronicle |
|---|---|---|
| Holds | stable facts the person stated | what has happened in the conversation |
| Updated | every turn | only during compaction |
| Size | 8 lines max | 70 words max |
| Evicted? | never | rewritten each fold |

They are separate because the chronicle demonstrably cannot be trusted to hold
user facts. In testing it preserved Fischl's advice while dropping the user's
name — producing a bot that denied knowing a name sitting in its own context.

The dossier is also refreshed on *every* turn rather than only during
compaction. It was originally built during compaction, on the reasoning that
its job is to rescue facts from turns about to be evicted — true, but it meant
it stayed empty for the first sixty-odd turns of any real conversation. Someone
who says "I like pizza" on turn three and looks at the panel should see it
there.

### The word cap is enforced in code

`build_chronicle` truncates to 70 words after the call, in addition to asking
for it in the prompt. Two reasons, both measured:

- Asked for an "updated chronicle", the model **appended** a new paragraph every
  time. Total context went *up* after each compaction — 392 → 411 → 474 tokens —
  instead of down. A summariser that grows without bound is worse than none,
  because it fails slowly enough to look like it is working. The prompt now says
  **"THIS IS A REWRITE, NOT AN APPENDIX"** in those words.
- Even then, asked for "80 words maximum", the model treated it as a target to
  approach and then drift past: 47 → 87 words over four folds. A prompt
  instruction is a request, not a guarantee.

### Watching it happen

The context panel on the right shows window usage, the dossier and the
chronicle live. Context management is invisible until it fails, and a
conversation that silently forgets is indistinguishable from one that never
knew.

Since 65 turns is more than a demo can show, **Demo controls → History budget**
lowers the budget so the same machinery runs after a few exchanges. Set it to
`350` and send five messages.

---

## Reasoning mode (stretch goal)

A checkbox above the input. When on, a **plain-voice** pass works the problem
out first and ends with a single `ANSWER:` line; Fischl then announces that
conclusion in character. The trace appears in its own collapsible panel above
her reply, visually separate from the answer.

### Why the reasoning happens outside the persona

An accuracy harness (`check_accuracy.py`) found that heavier persona load costs
correctness. 17 × 23 came back as *"three hundred and eleven"*. And every wrong
answer was one the model had **spelled out in words** while decorating it —
not one wrote digits and got them wrong.

Reasoning in the ornate voice would put the arithmetic back inside the thing
that breaks it. Reasoning first, plainly, then handing Fischl a finished
conclusion keeps the performance away from the part that has to be right.
Temperature is 0.2 for that stage — it wants the likeliest next token, not an
interesting one.

### Measured effect

`check_reasoning.py` runs six multi-step problems four times each in both modes.

| Question | OFF | ON |
|---|---|---|
| Train, 14:25 + 3h50m | 2/4 | **4/4** |
| 5 machines / 5 widgets | 3/4 | **4/4** |
| 12 people shaking hands | 3/4 | **4/4** |
| Shelf arithmetic | 4/4 | 4/4 |
| Age word problem | 4/4 | 4/4 |
| Two successive discounts | 4/4 | 4/4 |
| **Overall** | **20/24 (83%)** | **24/24 (100%)** |

#### Example 1 — arrival time

> **Off:** *"A simple arithmetic, really. Subtract 3 hours, 50 minutes from
> 14:25. The answer, when revealed, shall be: 10:35."*

It subtracted instead of adding. With reasoning on, the trace is two plain
lines — add 3 hours to get 17:25, add 50 minutes to get 18:15 — and the answer
is correct.

#### Example 2 — handshakes

With reasoning off you can watch it come apart mid-sentence:

> *"Eleven times twelve… no, wait, we count not each greeting twice. Eleven
> times twelve, minus the dozen hands I have already counted, and we find…
> sixty, no. Sixty was too soon. Sixty, minus twelve, for the duplicate
> counting… Ah, done. Sixty, minus twelve, is… forty-eight."*

That is persona load eating the arithmetic in real time. The trace does
12 × 11 = 132, ÷ 2 = 66, and the answer survives into her voice intact.

> **A measurement note.** The first version of this harness accepted only digit
> forms, scoring a correct *"twenty-eight"* as a failure. Because the persona
> spells numbers out far more often with reasoning **off**, that bug
> systematically inflated the benefit it was measuring. The table above is from
> the corrected version, which accepts both forms. A measurement bug that
> flatters the thing being measured is the dangerous kind.

---

## Project layout

```
fischl-oz-chatbot/
├── app.py                      Gradio interface, theme, streaming, context panel
├── chat.py                     model client, the two-stage chain, context management
├── persona.py                  system prompts and few-shot examples for both voices
├── Modelfile                   sets num_ctx 8192 on top of llama3.1:8b
├── requirements.txt
├── Mohsen_LLM_Chatbot.ipynb    runnable notebook (local or Colab)
├── assets/                     avatars
└── check_*.py                  diagnostics (see below)
```

---

## Diagnostics

Every measured claim above is reproducible. Run any of these with the venv
active and Ollama up.

| Script | What it checks |
|---|---|
| `check_ollama.py` | the endpoint responds and the model answers |
| `check_context.py` | the window really is 8192, not 4096 |
| `check_persona.py` | register, length range, knowledge boundary |
| `check_chain.py` | both stages, end to end |
| `check_context_mgmt.py` | compaction, the chronicle, the dossier |
| `check_accuracy.py` | does persona load cost correctness? |
| `check_reasoning.py` | does reasoning mode change the answer? |

They are kept in the repository rather than deleted because they are the
evidence behind the numbers.

---

## Known limitations

Stated plainly rather than hidden.

- **Oz occasionally calls the user "mein Fräulein".** That name is Fischl's
  alone. The system prompt says so explicitly; an 8B model still slips.
- **Oz occasionally attributes the user's possessions to Fischl.** Same cause.
- **Token counts in the panel are estimates while streaming.** A streamed
  response carries no usage block, so the panel falls back to a
  4-characters-per-token heuristic and marks it with `≈`. Measured against the
  server's own count it errs high by roughly 2%, so the meter over-reports
  slightly rather than hiding an overflow.
- **The chronicle is lossy by design.** It keeps facts and discards phrasing.
  Something said once in passing sixty turns ago may not survive.
- **An 8B model is an 8B model.** It can state something confidently wrong.
  Reasoning mode helps on multi-step problems; it is not a correctness
  guarantee.

---

## Credits

- **Fischl and Oz** are characters from *Genshin Impact* by HoYoverse. This is
  a non-commercial student project; no game assets are used.
- **The avatars** in `assets/` were drawn for this project by a friend and are
  used with permission.
- **The starfield and raven constellation** in the background are drawn in SVG
  in `app.py`.
- **Llama 3.1** by Meta, under the Llama 3.1 Community License.
- **Ollama** for local model serving; **Gradio** for the interface.