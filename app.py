"""
Web interface for the Fischl & Oz chatbot.

    python app.py

Opens on http://127.0.0.1:7860 with Ollama serving fischl-llama locally, so
nothing leaves the machine and no API key is involved.

Fischl and Oz get separate bubbles rather than one combined reply, because
they are two model calls with two system prompts. Replies stream in a word at
a time with a typing indicator in between. The panel on the right shows the
context state -- window usage, the dossier, the chronicle -- so context
management is visible rather than invisible until it fails.

Written against Gradio 6.28, which differs from 5.x examples in four ways:
`theme` and `css` are arguments to `launch()` rather than `Blocks()`;
`gr.Chatbot` has no `type=`; `group_consecutive_messages` defaults to True;
and message HTML is sanitized, so `class` and `style` survive while `width`
and `height` attributes do not.

Avatars are inline SVG images rather than `avatar_images`, which takes a
single (user, bot) pair and cannot distinguish two speakers on the bot side.
Drawing them here also means the repo carries no third-party artwork.
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
    An avatar tag sized by inline style rather than by a CSS class or a width
    attribute, so it renders correctly even against a stale cached stylesheet.
    The stylesheet below only adds animation.
    """
    return (
        f'<img src="{uri}" style="display:inline-block;width:{px}px;'
        f"height:{px}px;border-radius:50%;margin-right:9px;"
        f'vertical-align:-{px // 3}px">'
    )


# Styled inline for the same reason, so a stylesheet that fails to load
# degrades to three static dots rather than to nothing.
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
    """The waiting state: a bare bubble of dots, with no avatar and no name, so
    the portrait arrives with the first words."""
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


def reasoning_bubble(text: str) -> dict:
    """The working, above the answer. `metadata={"title": ...}` is what makes
    Gradio render a message as a titled, collapsible panel."""
    return {
        "role": "assistant",
        "content": text,
        "metadata": {"title": "✦ Reasoning — the Auge der Verurteilung"},
    }


def context_panel(conversation: Conversation) -> str:
    """Markdown for the side panel: window usage, dossier, chronicle."""
    estimated, actual, _ = last_estimate_error()

    # Streamed replies carry no usage block, so fall back to the estimate and
    # mark it as one.
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


BUBBLE_FOR = {
    "reason": reasoning_bubble,
    "fischl": fischl_bubble,
    "oz": oz_bubble,
}


def on_submit(message: str, history: list, conversation: Conversation, reasoning: bool):
    """
    One turn, as a generator so the interface can grow each reply in.

    `stream_turn` yields (stage, text). A "wait" appends a fresh bubble of
    dots; any content stage replaces that bubble and keeps replacing it as the
    text grows.
    """
    message = (message or "").strip()
    if not message:
        yield "", history, conversation, context_panel(conversation)
        return

    history = history + [{"role": "user", "content": message}]
    yield "", history, conversation, context_panel(conversation)

    for stage, text in stream_turn(conversation, message, reasoning=reasoning):
        if stage == "wait":
            history = history + [typing_bubble()]
        elif stage in BUBBLE_FOR:
            history[-1] = BUBBLE_FOR[stage](text)
        yield "", history, conversation, context_panel(conversation)

    # A stage that produced nothing at all would leave its dots blinking.
    if history and history[-1].get("content") == TYPING:
        history.pop()

    yield "", history, conversation, context_panel(conversation)


def on_clear():
    """Fresh conversation, fresh panel."""
    conversation = Conversation()
    return [], conversation, context_panel(conversation)


def set_budget(value):
    """
    Shrink the history budget so compaction can be watched happening. At its
    real value the window holds roughly sixty turns before anything is folded.
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
                # Without this, Fischl's bubble and Oz's are merged into one.
                group_consecutive_messages=False,
                # No per-message copy/share buttons.
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
            with gr.Row():
                reasoning = gr.Checkbox(
                    value=False,
                    label="Reasoning mode — work it out first, then answer",
                    scale=4,
                )
                clear = gr.Button("Begin anew", size="sm", scale=1)

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

    inputs = [box, chatbot, conversation_state, reasoning]
    outputs = [box, chatbot, conversation_state, panel]

    # show_progress="hidden" turns off Gradio's own loading treatment, which
    # dims every output and overlays a spinner for as long as the handler runs.
    # The typing bubble is the progress indicator here.
    box.submit(on_submit, inputs, outputs, show_progress="hidden")
    send.click(on_submit, inputs, outputs, show_progress="hidden")
    clear.click(on_clear, None, [chatbot, conversation_state, panel],
                show_progress="hidden")
    budget.change(set_budget, budget, None, show_progress="hidden")


if __name__ == "__main__":
    # theme and css belong to launch() in Gradio 6, not to Blocks().
    demo.launch(theme=gr.themes.Soft(primary_hue="purple"), css=CSS)