"""Build the additive slides as a real .pptx.

Every slide here is NEW — none of them replaces a slide someone else already made.
Numbers come from results/task1_panel50.csv (DiT-XL/2, K=16, 250 steps, cfg=1.0).

    python scripts/build_pptx.py    ->  Task1_additional_slides.pptx
"""
from __future__ import annotations

import copy
import os
import re

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "figures")

# ---- palette (matches the team's deck) -------------------------------------
AMBER = RGBColor(0xFF, 0xC2, 0x0E)
AMBER_D = RGBColor(0xE0, 0xA8, 0x00)
INK = RGBColor(0x12, 0x10, 0x0C)
BODY = RGBColor(0x3A, 0x36, 0x2E)
MUTED = RGBColor(0x6E, 0x68, 0x5C)
ALERT = RGBColor(0xB0, 0x3A, 0x2E)
ALERT_BG = RGBColor(0xFB, 0xEE, 0xEB)
PAGE = RGBColor(0xEF, 0xEC, 0xE4)
OK = RGBColor(0x2F, 0x6F, 0x4F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Poppins"          # the deck's face; Google Slides has it natively
MONO = "Consolas"

W, H = Inches(13.333), Inches(7.5)
BAND_H = Inches(1.15)
MARGIN = Inches(0.55)
BODY_W = W - 2 * MARGIN

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]

_n = 0


# --------------------------------------------------------------------------- #
# rich text: "U_{ens}(t) = x^{(k)}"  ->  runs with real sub/superscript
# --------------------------------------------------------------------------- #
_TOK = re.compile(r"(\*\*.+?\*\*|[_^]\{.*?\})")


def _baseline(run, value):
    run.font._rPr.set("baseline", str(value))


def rich(para, text, size=15, color=BODY, bold=False, font=FONT):
    for tok in _TOK.split(text):
        if not tok:
            continue
        r = para.add_run()
        if tok.startswith("**") and tok.endswith("**"):
            r.text = tok[2:-2]
            r.font.bold = True
            r.font.color.rgb = INK if color is BODY else color
        elif tok.startswith("_{"):
            r.text = tok[2:-1]
            _baseline(r, -25000)
            r.font.color.rgb = color
        elif tok.startswith("^{"):
            r.text = tok[2:-1]
            _baseline(r, 30000)
            r.font.color.rgb = color
        else:
            r.text = tok
            r.font.color.rgb = color
        r.font.size = Pt(size)
        r.font.name = font
        if bold:
            r.font.bold = True
    return para


def slide(title, crit=False):
    """New slide with the amber band + title. Returns the slide."""
    global _n
    _n += 1
    s = prs.slides.add_slide(BLANK)

    band = s.shapes.add_shape(1, 0, 0, W, BAND_H)          # 1 = rectangle
    band.fill.solid()
    band.fill.fore_color.rgb = ALERT if crit else AMBER
    band.line.fill.background()
    band.shadow.inherit = False
    tf = band.text_frame
    tf.margin_left, tf.margin_right = MARGIN, MARGIN
    tf.margin_top, tf.margin_bottom = Inches(0.1), Inches(0.1)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size = Pt(28)
    r.font.name = FONT
    r.font.bold = False
    r.font.color.rgb = WHITE if crit else INK

    num = s.shapes.add_textbox(W - Inches(0.9), H - Inches(0.5), Inches(0.5), Inches(0.3))
    pn = num.text_frame.paragraphs[0]
    pn.alignment = PP_ALIGN.RIGHT
    rn = pn.add_run()
    rn.text = f"+{_n}"
    rn.font.size = Pt(11)
    rn.font.name = FONT
    rn.font.color.rgb = MUTED
    return s


def body_box(s, top=1.55, height=5.4):
    tb = s.shapes.add_textbox(MARGIN, Inches(top), BODY_W, Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def bullets(tf, items, size=15, gap=8):
    """items: list of (level, text). First paragraph reuses tf.paragraphs[0]."""
    first = True
    for lvl, text in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.space_after = Pt(gap)
        p.line_spacing = 1.15
        rich(p, ("•  " if lvl == 0 else "–  ") + text, size=size)
    return tf


def para(tf, text, size=15, first=False, space=8, color=BODY, bold=False):
    p = tf.paragraphs[0] if (first and not tf.paragraphs[0].runs) else tf.add_paragraph()
    p.space_after = Pt(space)
    p.line_spacing = 1.15
    rich(p, text, size=size, color=color, bold=bold)
    return p


def math(s, text, top, size=17, height=0.55):
    box = s.shapes.add_shape(1, MARGIN, Inches(top), BODY_W, Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = PAGE
    box.line.fill.background()
    box.shadow.inherit = False
    tf = box.text_frame
    tf.margin_left = Inches(0.2)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    rich(tf.paragraphs[0], text, size=size, color=INK)
    return box


def callout(s, kicker, text, top, height=1.0, tone="alert"):
    fill, edge = (ALERT_BG, ALERT) if tone == "alert" else (PAGE, AMBER_D)
    box = s.shapes.add_shape(1, MARGIN, Inches(top), BODY_W, Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = edge
    box.line.width = Pt(1)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.18)
    tf.margin_top = tf.margin_bottom = Inches(0.1)
    p0 = tf.paragraphs[0]
    p0.space_after = Pt(3)
    rich(p0, kicker.upper(), size=10, color=edge, bold=True)
    p1 = tf.add_paragraph()
    p1.line_spacing = 1.12
    rich(p1, text, size=13)
    return box


def table(s, rows, left, top, width, col_w, size=12, header=True):
    nr, nc = len(rows), len(rows[0])
    shp = s.shapes.add_table(nr, nc, left, top, width, Inches(0.3 * nr))
    tbl = shp.table
    for i, w in enumerate(col_w):
        tbl.columns[i].width = w
    # kill the default banded blue style
    tbl._tbl.find(qn("a:tblPr")).set("firstRow", "1" if header else "0")
    tbl._tbl.find(qn("a:tblPr")).set("bandRow", "0")
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            if c > 0 and r > 0:
                p.alignment = PP_ALIGN.RIGHT
            hi = val.startswith("!")
            txt = val[1:] if hi else val
            rich(p, txt, size=size,
                 color=MUTED if (header and r == 0) else (ALERT if hi else BODY),
                 bold=(header and r == 0) or hi)
    return tbl


def picture(s, name, top, height=None, width=None):
    """Place a centred picture; return its BOTTOM edge in inches so callers can
    lay out beneath it instead of guessing (the figures have very different
    aspect ratios, and guessing overlaps the text)."""
    path = os.path.join(FIG, name)
    if width:
        pic = s.shapes.add_picture(path, 0, Inches(top), width=width)
    else:
        pic = s.shapes.add_picture(path, 0, Inches(top), height=Inches(height))
    pic.left = int((W - pic.width) / 2)
    return top + pic.height / 914400.0


def caption(s, text, top):
    tb = s.shapes.add_textbox(MARGIN, Inches(top), BODY_W, Inches(0.4))
    tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    rich(p, text, size=11, color=MUTED)
    return tb


# =========================================================================== #
# 1 — Related Work & Novelty
# =========================================================================== #
s = slide("Related Work & Novelty")
tf = body_box(s, 1.5, 4.0)
bullets(tf, [
    (0, "**TREAD** (Krause et al., 2025) — token routing for efficient DiT training. Presented purely "
        "as an efficiency method; the stochastic-depth training it induces is never analysed."),
    (0, "**Generative Uncertainty in Diffusion** (Jazbec et al., 2025) — Bayesian UQ via last-layer "
        "Laplace, for U-Net / flow matching. No DiT, no per-timestep profile; aleatoric left as future work."),
    (0, "**HyperDM** (Chan et al., NeurIPS 2024) — epistemic + aleatoric via a hyper-network ensemble. "
        "Rigorous but computationally heavy; no timestep-resolved analysis."),
    (0, "**Timestep-Aware Block Masking** (2025) — different DiT blocks activate at different timesteps, "
        "supporting the early-vs-late hypothesis."),
    (0, "**Attention entropy** — an established UQ proxy in NLP and ViTs, never applied to the diffusion "
        "timestep axis."),
], size=14)
callout(s, "Our gap",
        "TREAD presents routing as an efficiency mechanism. **We are the first to use its routing as an "
        "inference-time uncertainty-quantification substrate** — combined with attention entropy, resolved "
        "per timestep, decomposed into epistemic / aleatoric, and linked to generation quality. "
        "Training-free throughout.",
        top=5.55, height=1.25, tone="amber")

# =========================================================================== #
# 2 — Method: the three proxies
# =========================================================================== #
s = slide("Method — the three proxies")
tf = body_box(s, 1.42, 0.35)
para(tf, "All three are inference-time and training-free.  D_{θ} is the x̂_{0}-prediction.",
     size=13, first=True, color=MUTED)

math(s, "1 ·  U_{ens}(t, c)  =  Var_{k} [ D_{θ}( x_{t}^{(k)}, t, c ) ],   z^{(1..K)} ~ N(0, I)  through the full path", 1.85)
tf = body_box(s, 2.42, 0.3)
para(tf, "Captures **data stochasticity** — different noise, different image.", size=13, first=True)

math(s, "2 ·  U_{route}(t, c)  =  Var_{m} [ D_{θ}^{r(m)}( x_{t}, t, c )  |  z ],   M = 16 random TREAD routes", 2.82)
tf = body_box(s, 3.39, 0.3)
para(tf, "Captures **model / sub-network ambiguity** — the only **epistemic** proxy.", size=13, first=True)

math(s, "3 ·  U_{attn}(t, ℓ)  =  − Σ_{ij}  A_{ij}^{(ℓ)}  log A_{ij}^{(ℓ)}", 3.79)
tf = body_box(s, 4.36, 0.55)
para(tf, "We report the **mean per-query** entropy — bounded by ln N = ln 256 = 5.55, which gives the "
         "curve an interpretable ceiling (on the ceiling = attention is uniform = the block selects nothing).",
     size=13, first=True)

math(s, "U_{epi}  =  Var_{r}[ D | z ]        U_{ale}  =  E_{r}[ Var_{z}[ D ] ]        (law of total variance)", 5.05)
tf = body_box(s, 5.68, 0.6)
para(tf, "**Seeds alone cannot separate model ambiguity from data stochasticity.** Routes + seeds can — "
         "that is the whole point of having two independent noise sources.", size=13, first=True)

# =========================================================================== #
# 3 — Experimental setup
# =========================================================================== #
s = slide("Experimental setup")
rows = [
    ["", ""],
    ["Model", "DiT-XL/2 — 675 M params, 28 blocks, patch 2 → 256 tokens"],
    ["Checkpoint", "official DiT-XL-2-256x256.pt — inference only, no retraining"],
    ["VAE", "stabilityai/sd-vae-ft-ema, latent 32 × 32 × 4"],
    ["Sampler", "DDPM ancestral, 250 steps"],
    ["Guidance", "cfg_scale = 1.0 — pure conditional"],
    ["Seeds", "K = 16 per class, batched in one reverse pass"],
    ["Classes", "50, stratified easy / medium / hard"],
    ["Bins", "20 timestep bins over t ∈ [0, 1000)"],
    ["Hardware", "1 × RTX 4070 (12 GB) — ≈ 1 min per class"],
]
t = table(s, rows, MARGIN, Inches(1.45), BODY_W,
          [Inches(2.6), BODY_W - Inches(2.6)], size=13, header=False)
for r in range(1, len(rows)):
    p = t.cell(r, 0).text_frame.paragraphs[0]
    for run in p.runs:
        run.font.bold = True
        run.font.color.rgb = INK
    t.cell(r, 1).text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT

callout(s, "Correction to slide 3",
        "“Data: ImageNet 256×256” is misleading. **No ImageNet images are loaded anywhere.** Every proxy "
        "is measured on trajectories the model generates from z ~ N(0, I) conditioned on a class label — "
        "“50 ImageNet classes” means 50 label integers. The ImageNet-ness lives in the checkpoint’s weights.",
        top=5.55, height=1.25, tone="amber")

# =========================================================================== #
# 4 — Routing variance: the blocker  (CRITICAL)
# =========================================================================== #
s = slide("Routing variance — the proxy we could not compute", crit=True)
math(s, "U_{route}(t, c)  =  Var_{m} [ D_{θ}^{r(m)}( x_{t}, t, c )  |  z ]", 1.5)
tf = body_box(s, 2.2, 1.5)
bullets(tf, [
    (0, "It is the **only epistemic proxy**. The proposal calls it “the linchpin” — the epistemic / "
        "aleatoric split in Tasks 3 and 5 is unjustified without it."),
    (0, "It is **only legitimate on a TREAD-trained model**: routing variance means something because the "
        "network was trained as a distribution over sub-networks. Injecting routes — or MC-Dropout — into "
        "a vanilla DiT that never saw them yields uninterpretable variance."),
], size=14)
callout(s, "Blocker",
        "There is **no publicly released TREAD DiT-XL/2 checkpoint.** We cannot run it, and we cannot "
        "substitute vanilla DiT.", top=3.85, height=0.75)
tf = body_box(s, 4.8, 1.6)
bullets(tf, [
    (0, "**What is ready:** a model-agnostic reference implementation — routing_variance, route_sanity, "
        "epi_ale_decomposition — validated on a small TREAD-trained DiT. Swapping in a real checkpoint is "
        "a one-function change."),
    (0, "**Decision needed from the curator:** obtain a TREAD checkpoint, or train a TREAD DiT-B/2 "
        "ourselves? **This gates Tasks 2, 3 and 5.**"),
], size=14)

# =========================================================================== #
# 5 — Task 1.4: the 50-class panel
# =========================================================================== #
s = slide("All proxies over 50 ImageNet classes × 20 timestep bins")
bot = picture(s, "fig1_class_bin_heatmap.png", 1.32, height=4.25)
caption(s, "Row-normalised. Blue = easy, orange = medium, red = hard.  "
           "U_route column exists in the CSV but is NaN — see the blocker slide.", bot + 0.06)
tf = body_box(s, bot + 0.48, 6.9 - (bot + 0.48))
para(tf, "**Stratification — expected difficulty of p(x|c):**  easy = one canonical object, plain "
         "background (lemon, golf ball);  medium = real pose / background variation (retriever, zebra);  "
         "hard = polysemous label or cluttered scene (crane ×2, restaurant, coral reef).",
     size=12, first=True)
para(tf, "The proposal asks to stratify **by expected FID contribution** — that needs real ImageNet "
         "reference images. We stratify semantically and flag FID-based stratification as future work.",
     size=12, color=MUTED)

# =========================================================================== #
# 6 — Timestep profiles by difficulty
# =========================================================================== #
s = slide("Timestep profiles by class difficulty")
bot = picture(s, "fig2_difficulty_curves.png", 1.5, width=Inches(11.6))
tf = body_box(s, bot + 0.15, 7.25 - bot)
para(tf, "Mean ± 1 std over classes.", size=13, first=True, color=MUTED)
bullets(tf, [
    (0, "**Left:** easy sits cleanly above medium above hard, at every timestep — the anti-correlation, drawn."),
    (0, "**Right:** the three attention curves lie on top of each other — the null, drawn."),
    (0, "U_{attn} decays exactly as the proposal predicts, and it is a clean, low-variance signal — but it "
        "is the **same** signal for every class. It is a property of the denoising schedule, not of the "
        "conditioning."),
], size=13)

# =========================================================================== #
# 7 — Attention entropy across 50 classes
# =========================================================================== #
s = slide("Attention entropy — across all 50 classes")
bot = picture(s, "fig3_attn_per_block.png", 1.5, width=Inches(11.0))
tf = body_box(s, bot + 0.15, 7.25 - bot)
para(tf, "Averaged over all 50 classes, ± 1 std. Dashed line = ln 256 = 5.55, the maximum possible "
         "entropy over 256 tokens.", size=13, first=True, color=MUTED)
bullets(tf, [
    (0, "**Block 0 sits pinned on the ceiling.** Entropy at the maximum means attention is uniform — the "
        "early block is not selecting anything, it passes everything through."),
    (0, "Selection emerges with **depth**: blocks 14 and 27 fall to ≈3.4 nats as the image resolves. The "
        "early-vs-late structure shows up along the **block axis** as clearly as along the timestep axis."),
], size=13)

# =========================================================================== #
# 8 — Key finding (a)
# =========================================================================== #
s = slide("Key finding (a) — U_ens rises along the denoising path")
tf = body_box(s, 1.5, 1.5)
para(tf, "**The proposal predicts the opposite.**", size=17, first=True, color=ALERT)
para(tf, "Hypothesis: early steps (high σ) = mode selection = **high** seed variance, decaying late.",
     size=15)
para(tf, "Measured: starts at ≈0.03–0.05, dips further around t ≈ 900, then **grows monotonically to "
         "≈0.50–0.65** at t → 0.  Reproduced independently by two team members on separate code — this is "
         "a property of the metric, not a bug.", size=15)
callout(s, "Why",
        "At high σ the model resolves nothing and predicts x̂_{0} ≈ the conditional mean — “the average "
        "lemon” — so all seeds agree. By the end each seed has committed to **its own** lemon, so they "
        "disagree maximally.", top=3.95, height=1.0, tone="amber")
callout(s, "Consequence",
        "Var_{k}[x̂_{0}] in latent space measures **divergence of the final images**, not mode selection. "
        "To test the proposal’s actual claim, variance must be taken in a space where “mode” is meaningful "
        "— e.g. CLIP / DINO embeddings of x̂_{0} — not raw latents.", top=5.2, height=1.1)

# =========================================================================== #
# 9 — Key finding (b)
# =========================================================================== #
s = slide("Key finding (b) — U_ens anti-correlates with class difficulty")
table(s, [
    ["Proxy", "Spearman", "p"],
    ["U_ens  early (t ≥ 800)", "!−0.62", "<0.001"],
    ["U_ens  late (t ≤ 200)", "!−0.72", "<0.001"],
    ["U_attn late", "+0.05", "0.72"],
], MARGIN, Inches(1.45), Inches(6.0), [Inches(3.2), Inches(1.5), Inches(1.3)], size=12)

table(s, [
    ["Proposal’s key comparison", "early", "late"],
    ["lemon — unimodal", "!0.0470", "0.7065"],
    ["crane, bird — ambiguous", "0.0232", "0.4618"],
    ["crane, machine — ambiguous", "0.0304", "0.4946"],
], Inches(6.9), Inches(1.45), Inches(5.9), [Inches(3.1), Inches(1.4), Inches(1.4)], size=12)

tf = body_box(s, 3.05, 1.15)
para(tf, "Easy classes carry the **highest** seed variance. Attention entropy carries **no class "
         "information at all** — a clean null. And the proposal’s own key comparison fails in the "
         "predicted direction: **lemon has ~2× the early variance of either crane.**", size=15, first=True)

callout(s, "What it means",
        "Var[x̂_{0}] in latent space is dominated by **how much the class’s dominant low-frequency layout "
        "swings**, not by semantic ambiguity. A lemon is one big saturated blob; a coral reef is generic "
        "clutter in every sample. Normalising each curve by its own late value shrinks the effect "
        "(−0.62 → −0.48) but does **not** remove it — the shape is genuinely class-dependent.",
        top=4.35, height=1.25)
callout(s, "Honest caveat",
        "Our difficulty tiers are **semantic hand-labels**, confounded with “clutter”. To claim "
        "anti-correlation with **difficulty** proper we need per-class FID — which does require real "
        "ImageNet reference images. That is the one place in this project where the dataset is genuinely "
        "needed.", top=5.85, height=1.15, tone="amber")

# --- (b) drawn ---
s = slide("Key finding (b) — lemon vs crane, drawn")
bot = picture(s, "fig4_lemon_vs_crane.png", 1.6, width=Inches(11.6))
tf = body_box(s, bot + 0.2, 7.2 - bot)
para(tf, "**Lemon (blue) sits above both cranes at every timestep.** The unimodal class — the one the "
         "proposal expects to be *least* uncertain early — carries the most seed variance.",
     size=15, first=True)
para(tf, "Right panel: all three attention-entropy curves coincide. U_{attn} does not distinguish the "
         "ambiguous class from the unimodal one either.", size=15, color=MUTED)

# =========================================================================== #
# 10 — Spatial
# =========================================================================== #
s = slide("Spatial — where the seeds disagree")
picture(s, "fig5_patch_heatmaps.png", 1.35, height=5.0)
tf = body_box(s, 6.45, 0.9)
para(tf, "**Early:** uncertainty is diffuse and low — no structure has been committed to.  "
         "**Late:** it concentrates on object boundaries and texture. The seeds have agreed on what to "
         "draw and now disagree on detail.", size=13, first=True)
para(tf, "Next (Task 4): compare these maps against SAM / DINO masks and test whether high-uncertainty "
         "patches co-localise with semantically ambiguous regions.", size=13, color=MUTED)

# =========================================================================== #
# 11 — Roadmap
# =========================================================================== #
s = slide("Roadmap — mapped onto the curator's tasks")
tf = body_box(s, 1.45, 4.0)
para(tf, "DONE — TASK 1", size=12, first=True, color=OK, bold=True)
para(tf, "Two of three proxies implemented on pretrained DiT-XL/2, training-free, logged over 50 classes "
         "× 20 timestep bins. Two clean negative results.", size=14)
para(tf, "BLOCKED", size=12, color=ALERT, bold=True, space=2)
para(tf, "U_{route} — needs a TREAD checkpoint. Gates Tasks 2, 3 and 5.", size=14)
para(tf, "NEXT, PER THE PROPOSAL", size=12, color=INK, bold=True, space=2)
bullets_items = [
    (0, "**Task 2** — validate U_{route} as epistemic: OOD sensitivity, capacity trend across DiT-B/L/XL, "
        "non-redundancy vs U_{ens}, route-sanity ablation."),
    (0, "**Task 3** — epistemic / aleatoric decomposition; locate the mode-transition point t*."),
    (0, "**Task 4** — spatial heatmaps vs SAM / DINO masks."),
    (0, "**Task 5** — link uncertainty to per-sample quality (CLIP, IS contribution, intra-class LPIPS)."),
    (0, "**Fix the confound** — per-class FID for honest stratification; re-measure variance in a semantic "
        "embedding space."),
]
for lvl, text in bullets_items:
    p = tf.add_paragraph()
    p.space_after = Pt(5)
    p.line_spacing = 1.1
    rich(p, "•  " + text, size=13)

callout(s, "Keep slide 6 as it is",
        "Slide 6 already proposes uncertainty-adaptive sampling and entropy-guided routing. That stays. "
        "This slide sits **in front of it**, so the deck answers the curator’s brief first and pitches the "
        "extension second.", top=5.95, height=0.95, tone="amber")

# --------------------------------------------------------------------------- #
out = os.path.join(REPO, "Task1_additional_slides.pptx")
prs.save(out)
print(f"saved -> {out}   ({_n} slides, 16:9)")
