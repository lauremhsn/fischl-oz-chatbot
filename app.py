"""
Web interface for the Fischl & Oz chatbot.

    python app.py

Opens on http://127.0.0.1:7860 with Ollama serving fischl-llama locally, so
nothing leaves the machine and no API key is involved.

WHY THE LAYOUT LOOKS LIKE THIS
------------------------------
Fischl and Oz get separate bubbles rather than one combined reply. They are
two model calls with two system prompts, and showing them as one block would
hide the prompt chain that produced them. This needs
`group_consecutive_messages=False`: Gradio merges consecutive same-role
messages by default, which silently glues the two voices back together.

Replies stream in a word at a time, with a typing indicator in between, so
the exchange reads as speech rather than as a page load. The panel on the
right shows the context state -- window usage, the dossier, the chronicle --
because context management is invisible until it fails, and a conversation
that silently forgets is indistinguishable from one that never knew.

GRADIO 6 NOTES
--------------
Written against Gradio 6.28. Four things that differ from 5.x examples, each
found by reading the installed signatures rather than guessing:
  - `theme` and `css` are arguments to `launch()`, not to `Blocks()`.
  - `gr.Chatbot` has no `type=`. The tuple format is gone and the
    {"role", "content"} list is the only one.
  - `group_consecutive_messages` defaults to True.
  - Message HTML is sanitized: `class` survives, `width`/`height` attributes
    do not. Avatars are therefore sized from CSS, not from the tag.

The avatars are inline images rather than `avatar_images`, because that
argument takes a single (user, bot) pair and there are two distinct speakers
on the bot side. Both are drawn here in SVG -- a crowned eye and a raven --
so the repo carries no third-party artwork.
"""

import base64

import gradio as gr

import chat
from chat import CONTEXT_LIMIT, Conversation, last_estimate_error, stream_turn

# --------------------------------------------------------------------------
# Avatars, drawn here so there are no asset files and no licensing questions
# --------------------------------------------------------------------------

FISCHL_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<circle cx="32" cy="32" r="32" fill="#2a1b3d"/>
<path d="M14 26 L20 14 L26 22 L32 10 L38 22 L44 14 L50 26 Z" fill="#e8c86a"/>
<rect x="14" y="26" width="36" height="5" rx="2" fill="#e8c86a"/>
<ellipse cx="32" cy="44" rx="13" ry="8" fill="none" stroke="#c9a7f5" stroke-width="2.5"/>
<circle cx="32" cy="44" r="4.5" fill="#c9a7f5"/></svg>"""

OZ_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<circle cx="32" cy="32" r="32" fill="#14101f"/>
<path d="M20 46 C14 38 15 26 23 20 C31 14 42 17 45 25 C47 30 45 35 41 37
L44 48 C38 50 26 52 20 46 Z" fill="#7a6bad"/>
<path d="M45 24 L60 28 L45 31 Z" fill="#e8c86a"/>
<circle cx="37" cy="25" r="3.2" fill="#f4eeff"/>
<circle cx="38" cy="25" r="1.5" fill="#14101f"/></svg>"""


def _uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


FISCHL_AV = _uri(FISCHL_SVG)
OZ_AV = _uri(OZ_SVG)


def _avatar(uri: str, px: int) -> str:
    """
    An avatar tag that is the right size without needing the stylesheet.

    Gradio sanitizes message HTML: `width` and `height` attributes are
    stripped, `class` and `style` are not. Sizing from a CSS class works, but
    only once the custom stylesheet is in the page -- and a browser holding a
    cached copy of an earlier version of this app renders a 400px raven
    instead. Inline styles survive sanitization and cannot miss, so the
    geometry lives here and the stylesheet only adds animation.
    """
    return (
        f'<img src="{uri}" style="display:inline-block;width:{px}px;'
        f"height:{px}px;border-radius:50%;margin-right:9px;"
        f'vertical-align:-{px // 3}px">'
    )


# Dots are styled inline for the same reason, so a stylesheet that fails to
# load degrades to three static dots rather than to nothing at all. The CSS
# below only makes them blink.
_DOT = (
    '<i style="display:inline-block;width:9px;height:9px;margin-right:5px;'
    'border-radius:50%;background:#c9a7f5"></i>'
)
TYPING = f'<span class="typing">{_DOT * 3}</span>'

CSS = """
.who { font-weight:700; letter-spacing:.02em; }
.typing { display:inline-block; animation: popin .28s ease-out; }
.typing i { animation: ozblink 1.3s infinite; }
.typing i:nth-child(2) { animation-delay:.22s; }
.typing i:nth-child(3) { animation-delay:.44s; }
@keyframes ozblink {
  0%,60%,100% { opacity:.25; transform:translateY(0); }
  30%         { opacity:1;   transform:translateY(-3px); }
}
@keyframes popin {
  from { opacity:0; transform:translateY(6px) scale(.9); }
  to   { opacity:1; transform:translateY(0)   scale(1); }
}
.fischl-panel { font-size: 0.85rem; }
"""

INTRO = (
    "**Fischl von Luftschloss Narfidort**, Prinzessin der Verurteilung, "
    "with her familiar **Oz**, who translates.\n\n"
    "She answers in character. He tells you what she meant."
)


def typing_bubble() -> dict:
    """
    The waiting state: a bare bubble of dots, the way a messaging app does it.

    Deliberately carries no avatar and no name. An earlier version showed the
    speaker's portrait and label while the dots blinked, which announced who
    was about to talk before they had said anything and made the bubble tall
    and empty. The portrait now arrives with the first words, so the message
    itself is the thing that appears.
    """
    return {"role": "assistant", "content": TYPING}


def fischl_bubble(text: str) -> dict:
    return {
        "role": "assistant",
        "content": (
            f'{_avatar(FISCHL_AV, 28)}<span class="who">Fischl</span>\n\n{text}'
        ),
    }


def oz_bubble(text: str) -> dict:
    return {
        "role": "assistant",
        "content": f'{_avatar(OZ_AV, 22)}<span class="who">Oz</span> — {text}',
    }


def context_panel(conversation: Conversation) -> str:
    """Markdown for the side panel: window usage, dossier, chronicle."""
    estimated, actual, _ = last_estimate_error()

    # Streamed replies carry no usage block, so fall back to the estimate and
    # say so rather than silently showing a stale exact figure.
    used, mark = (actual, "") if actual else (estimated, "≈ ")
    pct = used / CONTEXT_LIMIT * 100 if used else 0.0

    filled = int(pct // 5)
    bar = "█" * filled + "░" * (20 - filled)

    lines = [
        "### Context",
        f"`{bar}`",
        f"{mark}{used} / {CONTEXT_LIMIT} tokens ({pct:.0f}%)",
        "",
        f"**Verbatim turns:** {len(conversation.turns)}",
    ]

    if conversation.last_folded:
        lines.append(
            f"**Just compacted:** {conversation.last_folded} turn(s) folded "
            f"into the chronicle"
        )

    lines += ["", "### What she remembers about you"]
    if conversation.facts:
        lines += [f"- {ln}" for ln in conversation.facts.splitlines() if ln.strip()]
    else:
        lines.append("*nothing yet*")

    lines += ["", "### Chronicle"]
    if conversation.chronicle:
        words = len(conversation.chronicle.split())
        lines.append(f"*{words} words, capped at {chat.CHRONICLE_WORD_CAP}*")
        lines.append("")
        lines.append(conversation.chronicle)
    else:
        lines.append("*empty until the window fills*")

    return "\n".join(lines)


def on_submit(message: str, history: list, conversation: Conversation):
    """
    One turn, as a generator so the UI can grow each reply in.

    Shape of the yields: put up the user's message and a bubble of dots, swap
    the dots for Fischl's message once her first word arrives and grow it,
    then put up a second bubble of dots while Oz thinks and swap that for his
    line. The panel only refreshes meaningfully at the end, because the turn
    is not recorded -- and compaction has not run -- until both stages finish.
    """
    message = (message or "").strip()
    if not message:
        yield "", history, conversation, context_panel(conversation)
        return

    history = history + [{"role": "user", "content": message}, typing_bubble()]
    yield "", history, conversation, context_panel(conversation)

    oz_pending = False
    oz_text = ""
    for fischl_text, oz_text in stream_turn(conversation, message):
        if oz_text is None:
            history[-1] = fischl_bubble(fischl_text)
        elif oz_text == "":
            if not oz_pending:
                history = history + [typing_bubble()]
                oz_pending = True
        else:
            history[-1] = oz_bubble(oz_text)
        yield "", history, conversation, context_panel(conversation)

    # If Oz returned nothing at all, the dots would otherwise blink forever.
    if history[-1]["content"] == TYPING:
        history[-1] = oz_bubble(oz_text or "...")

    yield "", history, conversation, context_panel(conversation)


def on_clear():
    """Fresh conversation, fresh panel."""
    conversation = Conversation()
    return [], conversation, context_panel(conversation)


def set_budget(value):
    """
    Shrink the history budget so compaction can be watched happening.

    At its real value the window holds roughly sixty turns before anything is
    folded, which makes the chronicle impossible to demonstrate without a very
    long conversation. Lowering this forces the same machinery to run after a
    few exchanges. It writes a module-level global, which is fine for a local
    single-user app and would not be for a shared one.
    """
    try:
        chat.HISTORY_BUDGET = max(150, int(value))
    except (TypeError, ValueError):
        pass


with gr.Blocks(title="Prinzessin der Verurteilung") as demo:
    conversation_state = gr.State(Conversation())

    gr.Markdown("# ✦ Prinzessin der Verurteilung")
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=480,
                show_label=False,
                # Without this, Fischl's bubble and Oz's are merged into one,
                # which hides the two-stage chain the app is built around.
                group_consecutive_messages=False,
                # No per-message copy/share buttons. They sit under every
                # bubble and, with two bubbles per turn, produce more controls
                # than content.
                buttons=[],
            )
            with gr.Row():
                box = gr.Textbox(
                    placeholder="Speak, and be welcome.",
                    show_label=False,
                    scale=6,
                    autofocus=True,
                )
                send = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("Begin anew", size="sm")

        with gr.Column(scale=1, elem_classes="fischl-panel"):
            panel = gr.Markdown(context_panel(Conversation()))
            with gr.Accordion("Demo controls", open=False):
                gr.Markdown(
                    "The real budget holds about 60 turns before anything is "
                    "folded. Lower it to watch compaction run after a few."
                )
                budget = gr.Number(
                    value=chat.HISTORY_BUDGET,
                    label="History budget (tokens)",
                    precision=0,
                )

    inputs = [box, chatbot, conversation_state]
    outputs = [box, chatbot, conversation_state, panel]

    # show_progress="hidden" turns off Gradio's own loading treatment, which
    # dims every output component and overlays a spinner and an elapsed-time
    # counter for as long as the handler runs. On a streaming generator that
    # covers the whole exchange, including the typing indicator it is meant to
    # be replaced by. The typing bubble IS the progress indicator here.
    box.submit(on_submit, inputs, outputs, show_progress="hidden")
    send.click(on_submit, inputs, outputs, show_progress="hidden")
    clear.click(on_clear, None, [chatbot, conversation_state, panel],
                show_progress="hidden")
    budget.change(set_budget, budget, None, show_progress="hidden")


if __name__ == "__main__":
    # theme and css belong to launch() in Gradio 6, not to Blocks().
    demo.launch(theme=gr.themes.Soft(primary_hue="purple"), css=CSS)