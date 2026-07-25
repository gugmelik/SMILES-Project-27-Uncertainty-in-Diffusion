import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import tiers as T

CELL = 256          
GAP = 2          
LABEL_W = 200       
HEAD_H = 34   
PAD = 14
BG = (255, 255, 255)
INK = (20, 20, 20)


def _font(size, bold=False):
    for name in (["arialbd.ttf", "seguisb.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_ROW = _font(20, bold=True)
F_SUB = _font(15)
F_HEAD = _font(18, bold=True)


def load_row(grid_dirs, cls, t, mode, k, noise="gaussian"):
    """Return the saved grid image for one (class, t, k), searching each dir."""
    if mode == "denoise":
        name = f"cls{cls}_t{t:.1f}_denoise_k0.png"
    else:
        name = f"cls{cls}_t{t:.1f}_perturb_k{k}_{noise}.png"
    for gd in ([grid_dirs] if isinstance(grid_dirs, (str, Path)) else grid_dirs):
        p = Path(gd) / name
        if p.is_file():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"{name} not found in {grid_dirs}")


def cell_at(row_img, i):
    x0 = i * (CELL + GAP) + GAP
    return row_img.crop((x0, GAP, x0 + CELL, GAP + CELL))


def compose(rows, col_headers, outfile, title, thumb=180, n_members=8):
    ncol = 1 + n_members  # reference + members
    cw = thumb
    W = LABEL_W + ncol * cw + (ncol + 1) * PAD
    row_h = thumb + PAD
    H = HEAD_H + 30 + len(rows) * row_h + PAD
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)

    d.text((PAD, 8), title, fill=INK, font=F_HEAD)
    # column headers
    y_head = HEAD_H + 4
    for j, htxt in enumerate(col_headers):
        x = LABEL_W + PAD + j * (cw + PAD) + cw // 2
        w = d.textlength(htxt, font=F_SUB)
        d.text((x - w / 2, y_head), htxt, fill=INK, font=F_SUB)

    y = HEAD_H + 30
    for label, sub, row_img in rows:
        d.text((PAD, y + thumb // 2 - 18), label, fill=INK, font=F_ROW)
        if sub:
            d.text((PAD, y + thumb // 2 + 6), sub, fill=(110, 110, 110), font=F_SUB)
        for j in range(ncol):
            cell = cell_at(row_img, j).resize((cw, thumb), Image.LANCZOS)
            x = LABEL_W + PAD + j * (cw + PAD)
            canvas.paste(cell, (x, y))
            if j == 0:  # box the reference so it reads as the anchor
                d.rectangle([x - 2, y - 2, x + cw + 1, y + thumb + 1],
                            outline=(200, 60, 80), width=3)
        y += row_h
    # photo montages -> JPEG (photographic content compresses ~10x vs PNG,
    # keeps the repo light); the small analytical plots stay PNG elsewhere.
    save_kwargs = {}
    if str(outfile).lower().endswith((".jpg", ".jpeg")):
        save_kwargs = dict(quality=88, optimize=True)
    canvas.save(outfile, **save_kwargs)
    print("wrote", outfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+",
                    default=["outputs/exp1_tiers/grids", "outputs/exp1_densek/grids"],
                    help="grid dirs, searched in order (tier run + dense run)")
    ap.add_argument("--outdir", default="../../exp1_tiers/images")
    ap.add_argument("--reps", type=int, nargs=3, default=[63, 974, 13],
                    help="representative class for hardest / medium / easiest")
    ap.add_argument("--t", type=float, default=0.3)
    ap.add_argument("--dense-ks", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 7, 10, 15, 20, 30],
                    help="k values for the dense strip (needs the dense run's grids)")
    ap.add_argument("--gallery-k", type=int, default=5,
                    help="k used for the all-classes gallery panel")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reps = dict(zip(T.TIER_ORDER, args.reps))
    headers = ["reference"] + [f"member {i+1}" for i in range(8)]

    # Panel 1: divergence grows with k (one class per tier, three k rows each)
    rows = []
    for tier in T.TIER_ORDER:
        cls = reps[tier]
        for k in (1, 2, 5):
            rows.append((f"{tier}", f"class {cls} · k={k}",
                         load_row(args.grids, cls, args.t, "perturb", k)))
    compose(rows, headers, outdir / "photo_k_divergence.jpg",
            f"Backward perturbation at t={args.t}: ensemble spreads as k grows "
            f"(reference boxed in red)")

    # Panel 2: tiers side by side at the most sensitive setting
    rows = []
    for tier in T.TIER_ORDER:
        cls = reps[tier]
        rows.append((f"{tier}", f"class {cls} · k=5",
                     load_row(args.grids, cls, args.t, "perturb", 5)))
    compose(rows, headers, outdir / "photo_tier_compare.jpg",
            f"Same setting (t={args.t}, k=5) across difficulty tiers — "
            f"visually similar spread")

    # Panel 3: dense-k strip per tier (needs grids from the dense run).
    # One figure per tier: rows = every k value, so the full divergence sweep
    # is visible. Skipped gracefully if the dense grids aren't there yet.
    for tier in T.TIER_ORDER:
        cls = reps[tier]
        rows, missing = [], False
        for k in args.dense_ks:
            try:
                rows.append((f"k={k}", "",
                             load_row(args.grids, cls, args.t, "perturb", k)))
            except FileNotFoundError:
                missing = True
                break
        if missing:
            print(f"[skip] dense strip for {tier} (class {cls}): "
                  f"run exp1_densek.py first to generate its grids")
            continue
        compose(rows, headers, outdir / f"photo_densek_{tier}.jpg",
                f"{tier} (class {cls}), t={args.t}: ensemble across dense k")

    # Panel 4: gallery — every class (grouped by tier) at one fixed setting, so
    # the reviewer sees a lot of actual generations across the whole subset.
    gk = args.gallery_k
    rows = []
    for tier in T.TIER_ORDER:
        for cls in T.TIER_CLASSES[tier]:
            try:
                rows.append((f"{tier}", f"class {cls}",
                             load_row(args.grids, cls, args.t, "perturb", gk)))
            except FileNotFoundError:
                continue  # class not in this run's subset
    if rows:
        compose(rows, headers, outdir / "photo_gallery_classes.jpg",
                f"All classes at t={args.t}, k={gk} — reference + 8 generations each")


if __name__ == "__main__":
    main()
