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

Avatars are loaded from assets/ and embedded in the stylesheet rather than in
each message, so the image data is sent once instead of once per bubble.
`avatar_images` is not used because it takes a single (user, bot) pair and
cannot distinguish two speakers on the bot side.
"""

import base64
from pathlib import Path

import gradio as gr

import chat
from chat import CONTEXT_LIMIT, Conversation, last_estimate_error, stream_turn

ASSETS = Path(__file__).parent / "assets"

# --------------------------------------------------------------------------
# Palette -- Fischl's own: Immernachtreich night, electro violet, gold trim
# --------------------------------------------------------------------------

NIGHT_0 = "#160b2d"   # deepest, page edges
NIGHT_1 = "#2a1550"   # body
NIGHT_2 = "#2d1d55"   # raised surfaces
PANEL = "#34235f"     # blocks, bubbles
EDGE = "#4d3a85"      # borders
VIOLET = "#8a63d8"    # primary accent
LILAC = "#c9aef5"     # Oz, secondary text accent
GOLD = "#f0d487"      # Fischl, headings
TEXT = "#f2ecff"
MUTED = "#b0a1d6"


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()

_FAVICON = ASSETS / "bow.png"
if not _FAVICON.exists():
    _FAVICON = ASSETS / "fischl.png"


FISCHL_AV = _data_uri(ASSETS / "fischl.png", "image/png")
OZ_AV = _data_uri(ASSETS / "oz.png", "image/png")

# A tiled star field for the page background. Drawn here so the repo carries
# no extra asset file for it.
_STARS = (
    '<circle cx="161.9" cy="75.4" r="1.6" opacity="0.21"/><circle cx="410.6" cy="47.1" r="1.3" opacity="0.21"/>'
    '<circle cx="253.7" cy="18.7" r="1.0" opacity="0.42"/><circle cx="120.3" cy="275.5" r="0.6" opacity="0.65"/>'
    '<circle cx="61.9" cy="111.6" r="1.6" opacity="0.51"/><circle cx="30.9" cy="292.8" r="0.6" opacity="0.74"/>'
    '<circle cx="23.3" cy="429.2" r="0.9" opacity="0.42"/><circle cx="270.3" cy="285.5" r="1.3" opacity="0.65"/>'
    '<circle cx="90.4" cy="290.8" r="1.6" opacity="0.29"/><circle cx="48.7" cy="356.1" r="1.3" opacity="0.21"/>'
    '<circle cx="103.0" cy="340.2" r="1.0" opacity="0.62"/><circle cx="232.8" cy="461.7" r="0.9" opacity="0.35"/>'
    '<circle cx="397.2" cy="349.5" r="0.7" opacity="0.23"/><circle cx="150.1" cy="247.6" r="0.9" opacity="0.6"/>'
    '<circle cx="144.0" cy="490.1" r="0.6" opacity="0.47"/><circle cx="82.5" cy="171.0" r="1.0" opacity="0.42"/>'
    '<circle cx="481.0" cy="38.8" r="1.3" opacity="0.51"/><circle cx="437.7" cy="156.9" r="1.6" opacity="0.38"/>'
    '<circle cx="248.3" cy="398.4" r="0.6" opacity="0.66"/><circle cx="472.3" cy="237.0" r="1.6" opacity="0.22"/>'
    '<circle cx="365.6" cy="154.8" r="1.3" opacity="0.75"/><circle cx="411.0" cy="142.3" r="1.0" opacity="0.69"/>'
    '<circle cx="173.5" cy="470.3" r="0.9" opacity="0.28"/><circle cx="58.5" cy="29.5" r="0.9" opacity="0.25"/>'
    '<circle cx="123.8" cy="195.5" r="1.0" opacity="0.23"/><circle cx="224.6" cy="274.7" r="0.7" opacity="0.65"/>'
    '<circle cx="432.0" cy="139.2" r="1.0" opacity="0.74"/><circle cx="341.4" cy="190.2" r="0.7" opacity="0.27"/>'
    '<circle cx="88.1" cy="116.0" r="0.7" opacity="0.19"/><circle cx="415.5" cy="91.2" r="0.9" opacity="0.18"/>'
    '<circle cx="209.5" cy="184.6" r="1.3" opacity="0.36"/><circle cx="62.7" cy="429.6" r="1.3" opacity="0.55"/>'
    '<circle cx="369.9" cy="228.3" r="1.6" opacity="0.63"/><circle cx="196.2" cy="199.5" r="0.6" opacity="0.45"/>'
    '<circle cx="200.2" cy="95.3" r="0.7" opacity="0.43"/><circle cx="55.0" cy="300.4" r="0.6" opacity="0.18"/>'
    '<circle cx="75.6" cy="50.7" r="0.9" opacity="0.53"/><circle cx="35.2" cy="104.0" r="1.0" opacity="0.26"/>'
    '<circle cx="126.1" cy="173.7" r="0.9" opacity="0.45"/><circle cx="57.7" cy="244.0" r="1.0" opacity="0.45"/>'
    '<circle cx="155.9" cy="72.1" r="1.6" opacity="0.38"/><circle cx="132.4" cy="414.4" r="0.7" opacity="0.47"/>'
    '<circle cx="102.6" cy="476.0" r="0.9" opacity="0.26"/><circle cx="271.6" cy="13.5" r="1.3" opacity="0.35"/>'
    '<circle cx="321.5" cy="45.5" r="0.9" opacity="0.48"/><circle cx="454.1" cy="177.8" r="0.7" opacity="0.48"/>'
    '<circle cx="389.5" cy="164.8" r="0.7" opacity="0.53"/><circle cx="394.2" cy="379.2" r="0.7" opacity="0.64"/>'
    '<circle cx="409.2" cy="369.9" r="0.7" opacity="0.29"/><circle cx="246.4" cy="365.5" r="0.6" opacity="0.63"/>'
    '<circle cx="236.1" cy="96.8" r="1.3" opacity="0.73"/><circle cx="223.6" cy="468.5" r="0.9" opacity="0.72"/>'
    '<circle cx="182.3" cy="110.2" r="0.7" opacity="0.45"/><circle cx="168.9" cy="241.3" r="1.3" opacity="0.66"/>'
    '<circle cx="239.7" cy="326.5" r="1.6" opacity="0.23"/><circle cx="330.3" cy="454.9" r="1.6" opacity="0.61"/>'
    '<circle cx="239.0" cy="89.3" r="1.6" opacity="0.37"/><circle cx="400.4" cy="485.8" r="1.0" opacity="0.44"/>'
    '<circle cx="371.7" cy="42.5" r="0.7" opacity="0.28"/><circle cx="63.5" cy="75.6" r="1.0" opacity="0.64"/>'
    '<circle cx="73.1" cy="413.3" r="1.0" opacity="0.55"/><circle cx="175.2" cy="274.3" r="0.7" opacity="0.19"/>'
    '<circle cx="399.7" cy="363.2" r="0.6" opacity="0.48"/><circle cx="466.8" cy="216.9" r="0.7" opacity="0.65"/>'
    '<circle cx="105.5" cy="125.9" r="0.9" opacity="0.47"/><circle cx="381.8" cy="163.0" r="1.3" opacity="0.42"/>'
    '<circle cx="65.5" cy="455.0" r="0.9" opacity="0.69"/><circle cx="331.2" cy="407.5" r="1.3" opacity="0.42"/>'
    '<circle cx="458.9" cy="250.8" r="1.3" opacity="0.27"/><circle cx="255.3" cy="436.4" r="0.7" opacity="0.53"/>'
)
STARFIELD = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="500">'
        f'<g fill="#efe6ff">{_STARS}</g></svg>'.encode()
    ).decode()
)

# A raven rather than a generic bird: the three things that separate a corvid
# from a songbird are a deep hooked beak roughly as long as the head, a flat
# crown with no domed forehead, and shaggy throat hackles. Drawn here, line art
# only. Laure's own hand-traced version is kept in git history if she prefers it.
_RV_STROKES = [
    # beak: deep wedge, upper mandible hooking past the lower at the tip
    "M108 188 C136 164 170 150 200 146 L202 192 C170 186 138 186 108 188 Z",
    "M108 188 C115 193 118 198 115 203",
    # flat crown back to the nape
    "M200 146 C220 126 252 118 284 126 C312 134 328 154 332 180",
    # throat: a smooth edge, with separate feather ticks for the shaggy
    # hackles. Curving the throat line inward notched the silhouette and read
    # as two bites taken out of the bird.
    "M202 192 C207 222 206 258 208 290 C209 300 210 306 212 312",
    "M205 214 L189 223", "M206 244 L190 254", "M208 274 L192 284",
    "M212 312 C226 336 242 354 260 366",
    "M262 366 C286 380 312 386 338 386 C356 386 372 382 386 374",
    "M332 180 C356 202 382 230 404 262",
    # wedge tail
    "M404 262 C432 288 462 316 490 344",
    "M490 344 L468 374",
    "M468 374 C438 366 410 368 386 374",
    # folded wing and two primaries
    "M324 192 C360 214 396 250 426 296 C402 304 374 294 350 274 "
    "C332 258 320 228 324 192 Z",
    "M350 274 C378 292 404 310 428 326",
    "M336 236 C364 258 392 280 416 298",
    # legs and feet
    "M300 384 L296 430", "M296 430 L274 440", "M296 430 L318 439",
    "M340 386 L338 428", "M338 428 L316 438", "M338 428 L360 437",
]
_RV_PERCH = "M246 442 C300 434 364 434 414 444"
_RV_N = [
    (112, 188, 4.0), (200, 146, 4.4), (284, 126, 5.6), (332, 180, 4.6),
    (212, 312, 4.4), (404, 262, 5.0), (490, 344, 5.4), (338, 386, 4.0),
    (296, 430, 3.2), (338, 428, 3.2),
]
RAVEN = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 470">'
            '<g fill="none" stroke="#d8c2fb" stroke-width="2.2" '
            'stroke-linecap="round" stroke-linejoin="round" opacity=".27">'
            + "".join(f'<path d="{d}"/>' for d in _RV_STROKES)
            + "</g>"
            f'<path d="{_RV_PERCH}" fill="none" stroke="#c9a7f5" '
            'stroke-width="1.8" stroke-linecap="round" opacity=".16"/>'
            '<circle cx="232" cy="162" r="4.6" fill="#fffdf4" opacity=".42"/>'
            '<g fill="#fff6dc" opacity=".11">'
            + "".join(
                f'<circle cx="{x}" cy="{y}" r="{r * 2.5}"/>' for x, y, r in _RV_N
            )
            + "</g>"
            '<g fill="#fffdf4" opacity=".50">'
            + "".join(f'<circle cx="{x}" cy="{y}" r="{r}"/>' for x, y, r in _RV_N)
            + "</g></svg>"
        ).encode()
    ).decode()
)

# --------------------------------------------------------------------------
# Theme and stylesheet
# --------------------------------------------------------------------------

THEME = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="violet",
    neutral_hue="slate",
    font=["Spectral", "Georgia", "serif"],
).set(
    body_background_fill=NIGHT_1,
    body_text_color=TEXT,
    body_text_color_subdued=MUTED,
    background_fill_primary=PANEL,
    background_fill_secondary=NIGHT_2,
    block_background_fill=PANEL,
    block_border_color=EDGE,
    block_label_text_color=LILAC,
    block_title_text_color=LILAC,
    border_color_primary=EDGE,
    input_background_fill=NIGHT_2,
    input_border_color=EDGE,
    button_primary_background_fill=VIOLET,
    button_primary_text_color="#ffffff",
    button_secondary_background_fill=NIGHT_2,
    button_secondary_text_color=LILAC,
    button_secondary_border_color=EDGE,
    color_accent=LILAC,
    color_accent_soft=PANEL,
)

# The sky is painted on ONE element and every Gradio surface above it is made
# transparent. Painting it on the container as well produced a visible seam
# down the page where the container's own copy ended.
CSS = f"""
gradio-app {{
  display:block; height:100vh; height:100dvh; color:{TEXT};
  background:
    url("{STARFIELD}") repeat,
    radial-gradient(900px 620px at 20% 16%, rgba(158,96,214,.46), transparent 62%),
    radial-gradient(780px 560px at 80% 26%, rgba(112,64,176,.44), transparent 64%),
    radial-gradient(900px 660px at 60% 92%, rgba(82,44,146,.50), transparent 66%),
    radial-gradient(640px 500px at 6% 82%, rgba(138,74,194,.32), transparent 64%),
    linear-gradient(162deg, {NIGHT_1} 0%, #231146 46%, {NIGHT_0} 100%)
    !important;
  background-attachment: fixed !important;
}}

/* Everything Gradio paints above the sky is cleared so nothing seams. */
.gradio-container, .gradio-container > div, .main, .app, .fillable,
.gradio-container .block, .gradio-container .form {{
  background: transparent !important; border-color: transparent !important;
}}

/* Lock the app to the viewport: the transcript scrolls, the page does not. */
.gradio-container {{
  height:100dvh !important; max-height:100dvh !important; overflow:hidden !important;
  max-width:1520px !important; padding:4px 22px 12px !important;
  display:flex !important; flex-direction:column;
}}
.gradio-container > div, .gradio-container .main, .gradio-container .fillable {{
  flex:1 1 auto; min-height:0; display:flex; flex-direction:column;
}}
.main-row {{
  flex:1 1 auto !important; min-height:0 !important;
  align-items:stretch !important;
}}
.chat-col {{ min-height:0 !important; display:flex; flex-direction:column; }}
/* flex-basis 0, not auto: with `auto` the transcript's base size is its own
   content, so a long conversation inflated the panel past the viewport, pushed
   the input box off-screen and left nothing to scroll inside. Basis 0 makes it
   take the free space instead, which bounds .bubble-wrap so it can scroll. */
.chat-col > .chat-wrap {{
  flex:1 1 0 !important; min-height:0 !important;
}}
.chat-wrap > .wrap {{ height:100% !important; }}
.chat-wrap .bubble-wrap {{
  height:100% !important; overflow-y:auto !important;
  scrollbar-width: thin;
  scrollbar-color: rgba(201,174,245,.24) transparent;
}}
.chat-wrap .bubble-wrap::-webkit-scrollbar {{ width:10px; }}
.chat-wrap .bubble-wrap::-webkit-scrollbar-track {{ background:transparent; }}
.chat-wrap .bubble-wrap::-webkit-scrollbar-thumb {{
  background: rgba(201,174,245,.22); border-radius:10px;
  border:3px solid transparent; background-clip:content-box;
}}
.chat-wrap .bubble-wrap::-webkit-scrollbar-thumb:hover {{
  background: rgba(201,174,245,.40); background-clip:content-box;
}}
/* Gradio's dark theme paints .bubble-wrap a solid slate, which turned the
   transcript into an opaque slab on a fresh load but not after a turn. */
.chat-wrap > .wrap, .chat-wrap .wrapper, .chat-wrap .bubble-wrap,
.chat-wrap .placeholder-content {{ background: transparent !important; }}
/* Gradio gives rows flex:1 0 auto, so the two control rows grow and starve the
   transcript. They are fixed-size; only the transcript flexes. */
.fixed-row {{ flex:0 0 auto !important; align-items:center !important; }}
.side-col {{ min-height:0; overflow-y:auto; overflow-x:hidden; padding-right:4px; }}
.side-col::-webkit-scrollbar {{ width:6px; }}
.side-col::-webkit-scrollbar-thumb {{ background:{EDGE}; border-radius:3px; }}

/* Header */
.crest {{ text-align:center; margin:2px 0 0; }}
.crest h1 {{
  font-size:1.72rem; letter-spacing:.03em; margin:0; font-weight:600;
  background: linear-gradient(180deg, #fff6d9 0%, {GOLD} 55%, #b4914a 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.crest .sub {{
  color:{MUTED}; font-size:.82rem; margin-top:3px; letter-spacing:.05em;
}}
.crest .rule {{
  width:210px; height:1px; margin:8px auto 2px;
  background: linear-gradient(90deg, transparent, {GOLD}, transparent);
  opacity:.6;
}}

/* Chat surface -- barely there, so the sky reads through it */
.chat-wrap {{
  background:
    url("{RAVEN}") no-repeat center / auto 68%,
    rgba(26,13,52,.26) !important;
  border:1px solid rgba(140,105,210,.20) !important;
  border-radius:14px; backdrop-filter: blur(1px);
}}
.chat-wrap .message-wrap {{ gap:6px; }}
.chat-wrap .bot-row .message, .chat-wrap .message.bot {{
  background: linear-gradient(170deg, rgba(58,40,104,.62), rgba(36,22,70,.62)) !important;
  border:1px solid rgba(150,115,220,.30) !important;
  box-shadow: 0 2px 16px rgba(14,6,34,.30);
}}
.chat-wrap .user-row .message, .chat-wrap .message.user {{
  background: linear-gradient(170deg, rgba(126,92,196,.60), rgba(88,60,154,.60)) !important;
  border:1px solid rgba(178,146,240,.36) !important;
}}

/* Speaker line */
.who {{ font-weight:700; letter-spacing:.06em; font-size:.82rem;
        text-transform:uppercase; }}
.who-f {{ color:{GOLD}; }}
.who-o {{ color:{LILAC}; }}

/* Avatars: geometry inline on the tag, image from here, so a stale cached
   stylesheet degrades to an empty circle rather than a full-width image. */
.av {{
  background-size: cover; background-position: center top;
  box-shadow: 0 0 0 1px {EDGE}, 0 0 12px rgba(150,110,230,.42);
}}
.av-f {{ background-image:url("{FISCHL_AV}"); }}
.av-o {{ background-image:url("{OZ_AV}"); background-position:center; }}

/* Typing indicator */
.typing {{ display:inline-block; animation: popin .28s ease-out; }}
.typing i {{ animation: ozblink 1.3s infinite; }}
.typing i:nth-child(2) {{ animation-delay:.22s; }}
.typing i:nth-child(3) {{ animation-delay:.44s; }}
@keyframes ozblink {{
  0%,60%,100% {{ opacity:.25; transform:translateY(0); }}
  30%         {{ opacity:1;   transform:translateY(-3px); }}
}}
@keyframes popin {{
  from {{ opacity:0; transform:translateY(6px) scale(.9); }}
  to   {{ opacity:1; transform:translateY(0)   scale(1); }}
}}

/* Inputs need their surface back after the blanket transparency above. */
.gradio-container textarea, .gradio-container input[type="text"],
.gradio-container input[type="number"] {{
  background: rgba(26,13,52,.55) !important;
  border:1px solid rgba(140,105,210,.28) !important;
  color:{TEXT} !important; border-radius:10px;
}}
.gradio-container .accordion, .gradio-container .label-wrap {{
  background: rgba(26,13,52,.42) !important;
  border:1px solid rgba(140,105,210,.22) !important; border-radius:10px;
}}
.gradio-container button {{ border-radius:10px; }}
/* The dark theme overrides the secondary-button and checkbox tokens, so these
   are set directly rather than through the theme. */
.gradio-container button.secondary {{
  background: rgba(26,13,52,.55) !important; color:{LILAC} !important;
  border:1px solid rgba(140,105,210,.30) !important;
}}
.gradio-container button.secondary:hover {{
  background: rgba(60,36,112,.70) !important;
}}
.gradio-container input[type="checkbox"] {{
  background: rgba(26,13,52,.55) !important;
  border:1px solid rgba(140,105,210,.42) !important;
}}
.gradio-container input[type="checkbox"]:checked {{
  background: {VIOLET} !important; border-color:{VIOLET} !important;
}}

/* The transcript's own clear button comes through as cold slate in dark mode. */
.chat-wrap .icon-button-wrapper, .chat-wrap button.icon-button {{
  background: rgba(26,13,52,.60) !important;
  border:1px solid rgba(140,105,210,.28) !important;
}}

/* Side panel */
.fischl-panel {{ font-size:.85rem; }}
.fischl-panel h3 {{
  color:{GOLD} !important; font-size:.75rem !important; font-weight:700;
  letter-spacing:.12em; text-transform:uppercase;
  border-bottom:1px solid {EDGE}; padding-bottom:5px; margin:14px 0 8px;
}}
.fischl-panel em {{ color:{MUTED}; }}
.fischl-panel strong {{ color:{TEXT}; }}
.meter {{
  height:7px; border-radius:4px; background:#241c47;
  border:1px solid {EDGE}; overflow:hidden; margin:8px 0 7px;
}}
.meter span {{ display:block; height:100%;
  background:linear-gradient(90deg, {VIOLET}, {GOLD}); }}
.meter-label {{ color:{MUTED}; font-size:.78rem; letter-spacing:.03em; }}

footer {{ display:none !important; }}

"""

# Gradio rewrites whatever is passed to launch(css=...), prefixing selectors
# with `.gradio-container ... .contain`. That silently neutralises any `html` or
# `body` rule and every rule inside an @media block, which is why the first
# attempt at a mobile layout did nothing at all. This sheet is injected raw
# through gr.HTML(head=...), which Gradio leaves alone. Selectors are repeated
# to out-weigh the specificity that prefixing adds to the rules above.
def _sp(selector: str, times: int = 7) -> str:
    return selector * times


MOBILE_CSS = f"""
@media (max-width: 860px) {{
  html, body {{ height:auto !important; overflow:auto !important; }}
  html body gradio-app {{
    height:auto !important; min-height:100dvh !important;
    background-attachment: scroll !important;
  }}
  {_sp(".gradio-container")} {{
    height:auto !important; max-height:none !important;
    overflow:visible !important; padding:4px 12px 24px !important;
  }}
  {_sp(".main-row")} {{ flex-direction:column !important; }}
  {_sp(".chat-col")} {{ min-height:0 !important; }}
  {_sp(".chat-col")} > .chat-wrap {{
    flex:0 0 auto !important; height:56dvh !important; min-height:300px !important;
    background-size: 86% auto, auto !important;
  }}
  {_sp(".side-col")} {{
    overflow:visible !important; padding-right:0 !important; margin-top:6px;
  }}
  .crest h1 {{ font-size:1.2rem !important; }}
  .crest .sub {{ font-size:.72rem !important; line-height:1.4 !important; }}
  .crest .rule {{ width:150px !important; margin:7px auto 2px !important; }}
}}
"""

INTRO = (
    "Fischl von Luftschloss Narfidort, Prinzessin der Verurteilung &nbsp;·&nbsp; "
    "and her familiar Oz, who translates"
)


def _av(kind: str, px: int) -> str:
    """An avatar span: size and shape inline, image from the stylesheet."""
    return (
        f'<span class="av av-{kind}" style="display:inline-block;'
        f"width:{px}px;height:{px}px;border-radius:50%;margin-right:9px;"
        f'vertical-align:-{px // 3}px"></span>'
    )


_DOT = (
    '<i style="display:inline-block;width:9px;height:9px;margin-right:5px;'
    'border-radius:50%;background:#c0a3f0"></i>'
)
TYPING = f'<span class="typing">{_DOT * 3}</span>'


def typing_bubble() -> dict:
    """The waiting state: a bare bubble of dots, with no avatar and no name, so
    the portrait arrives with the first words."""
    return {"role": "assistant", "content": TYPING}


def fischl_bubble(text: str) -> dict:
    return {
        "role": "assistant",
        "content": (
            f'{_av("f", 34)}<span class="who who-f">Fischl</span>\n\n{text}'
        ),
    }


def oz_bubble(text: str) -> dict:
    return {
        "role": "assistant",
        "content": (
            f'{_av("o", 26)}<span class="who who-o">Oz</span> &nbsp;{text}'
        ),
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

    lines = [
        "### Context",
        f'<div class="meter"><span style="width:{min(pct, 100):.1f}%"></span></div>',
        f'<div class="meter-label">{mark}{used} / {CONTEXT_LIMIT} tokens '
        f"&nbsp;·&nbsp; {pct:.0f}%</div>",
        "",
        f"**Verbatim turns:** {len(conversation.turns)}",
    ]

    if conversation.last_folded:
        lines.append(
            f"**Just compacted:** {conversation.last_folded} turn(s) folded "
            f"into the chronicle"
        )

    if getattr(conversation, "last_error", ""):
        lines.append(f"**⚠ {conversation.last_error}**")

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


with gr.Blocks(title="Prinzessin der Verurteilung", fill_height=True) as demo:
    conversation_state = gr.State(Conversation())

    gr.HTML("", head=f"<style>{MOBILE_CSS}</style>")

    gr.HTML(
        '<div class="crest"><h1>✦ Prinzessin der Verurteilung ✦</h1>'
        f'<div class="sub">{INTRO}</div><div class="rule"></div></div>'
    )

    with gr.Row(elem_classes="main-row", equal_height=True):
        with gr.Column(scale=3, elem_classes="chat-col"):
            chatbot = gr.Chatbot(
                height="100%",
                show_label=False,
                elem_classes="chat-wrap",
                # Without this, Fischl's bubble and Oz's are merged into one.
                group_consecutive_messages=False,
                # No per-message copy/share buttons.
                buttons=[],
            )
            with gr.Row(elem_classes="fixed-row"):
                box = gr.Textbox(
                    placeholder="Speak, and be welcome.",
                    show_label=False,
                    scale=6,
                    autofocus=True,
                )
                send = gr.Button("Send", variant="primary", scale=1)
            with gr.Row(elem_classes="fixed-row"):
                reasoning = gr.Checkbox(
                    value=False,
                    label="Reasoning mode — work it out first, then answer",
                    scale=4,
                )
                clear = gr.Button("Begin anew", size="sm", scale=1)

        with gr.Column(scale=1, elem_classes=["fischl-panel", "side-col"]):
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
    demo.launch(
        theme=THEME,
        css=CSS,
        favicon_path=str(_FAVICON),
        pwa=True,
    )