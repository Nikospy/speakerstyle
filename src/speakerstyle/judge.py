"""Model-in-the-loop: render a sheet of candidates, let Claude pick and steer, repeat.

Design choices:
  * the model compares candidates against the style reference instead of scoring one image;
    vision models are far more reliable at "which is closer" than at absolute scores
  * the model never outputs numbers for parameters: it returns a direction (-1/0/+1) per
    parameter and the code applies fixed steps, so a bad answer cannot wreck the render
  * the previous winner is always on the next sheet, so a round can never make things worse
  * runs through the local `claude` CLI on the user's subscription (no API key needed)
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from . import pipeline as P

STEPS = {
    "ink": 0.04,
    "paper": 0.07,
    "grey": 0.06,
    "smooth": 0.3,
    "detail": 0.4,
    "pop": 0.35,
    "ink_lines": 0.35,
    "dot_cell": 0.002,
    "fine": 0.5,
    "skin_smooth": 0.4,
}

ISSUES = [
    "face_too_dark",
    "face_too_bright",
    "eyes_lost",
    "features_unclear",
    "too_flat_posterized",
    "too_noisy_speckled",
    "hair_blob",
    "dots_too_coarse",
    "dots_too_fine",
    "background_leftovers",
    "cutout_damaged",
    "unflattering",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "best": {"type": "integer", "description": "number of the best candidate"},
        "good_enough": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string", "enum": ISSUES}},
        "adjust": {
            "type": "object",
            "properties": {k: {"type": "integer", "enum": [-1, 0, 1]} for k in STEPS},
            "required": list(STEPS),
        },
        "note": {"type": "string"},
    },
    "required": ["best", "good_enough", "issues", "adjust", "note"],
}

NO_CLAUDE = ("Tuning by Claude needs Claude Code installed and signed in (the `claude` command). "
             "Everything else works without it.")

PROMPT = """You are the art director checking speaker portraits for an event poster.

Read these images with the Read tool:
1. Style references (examples of the look): {ref}
2. Candidate sheet: numbered head close-ups of the same person (the halftone dots are real, judge them
   at this zoom), on this event's background colour, plus a bottom strip of the full cut-outs in the same
   order at thumbnail size: {sheet}

The references set the DIRECTION, not the ceiling: they are rough examples with flaws (muddy patches,
random specks, a few features lost in black). Aim for the same look done better:
hard cut-out, three flat levels only (solid black ink, one fine halftone-dot grey, plain white),
a crisp engraving/photocopy feel: clean flat skin with white highlights and grey modelling, crisp
black lines on eyes, lids, nostrils and lips, hair and clothing drawn as fine strands and folds
(white lines on black, black lines on white), black for deep shadows. Better than the reference means: eyes, brows, nose and
mouth read instantly even at thumbnail size, shapes are clean and intentional rather than noisy, the
person looks like themselves and looks good (this promotes them), and the silhouette reads well on
this background colour.

Automatic checks on the current best: {flags}

Keep the person's real skin tone: a darker-skinned person must not be bleached to look light-skinned.

Pick the strongest candidate by that standard. Then say how to move the parameters from the BEST
candidate for the next round (-1 lower, 0 keep, +1 raise):
- ink: share of solid black on the face
- paper: share of plain white on the face
- grey: darkness of the halftone grey
- smooth: flatter, simpler shapes (+) vs more raw texture (-)
- detail: local contrast
- pop: lifts hair strands / features out of dark areas
- ink_lines: thin black lines on eyes, brows, lips, wrinkles
- dot_cell: halftone dot size
- fine: strand/fold line detail in hair and clothes
- skin_smooth: how flat and clean the skin is
Set good_enough=true when the best candidate could go on the poster as is: no lost features, no
ugly artefacts, flattering. Small matters of taste do not block it. If the candidates are nearly
identical and none fixes the remaining issue, still answer good_enough honestly and say so in note.
Answer only through the structured output."""

PROMPT_OBJECT = """You are the art director checking an object illustration for an event poster
(a product, a prop, a thing the speaker brings, an animal: anything that is not a portrait).

Read these images with the Read tool:
1. Style references (examples of the look, shown on portraits): {ref}
2. Candidate sheet: numbered renders of the same object on this event's background colour (the halftone
   dots are real, judge them at this zoom), plus a bottom strip of the same renders at thumbnail size: {sheet}

Apply the house style of the references to the object: hard cut-out, three flat levels only (solid black
ink, one fine halftone-dot grey, plain white), a crisp engraving/photocopy feel. Better means: the object
is recognisable at once even at thumbnail size, its silhouette and main forms read clearly, the material
shows (metal, glass, fabric, stone, ceramic: highlights in white, form in grey, deep shadow and contours in
black), printed text or a logo on it stays legible where the photo allows, and the shapes look intentional
rather than noisy. The references set the direction, not the ceiling.

Automatic checks on the current best: {flags}

The issue names were written for portraits; read them for the object: face_too_dark / face_too_bright =
the object too dark / too bright, eyes_lost = key details or text lost, hair_blob = a fine-textured area
turned into one blob, unflattering = the object looks cheap, dirty or broken.

Pick the strongest candidate by that standard. Then say how to move the parameters from the BEST
candidate for the next round (-1 lower, 0 keep, +1 raise):
- ink: share of solid black on the object
- paper: share of plain white on the object
- grey: darkness of the halftone grey
- smooth: flatter, simpler shapes (+) vs more raw texture (-)
- detail: local contrast
- pop: lifts detail out of dark areas
- ink_lines: thin black contour and edge lines
- dot_cell: halftone dot size
- fine: line detail in textured areas (fabric, stone, wood grain)
- skin_smooth: no effect on objects, always 0
Set good_enough=true when the best candidate could go on the poster as is. Small matters of taste do not
block it. Answer only through the structured output."""


# The judge keeps preferring "the whitest face", which drifts toward bleaching darker skin.
# Its moves on ink/paper are bounded; tone_keep stays out of its reach.
JUDGE_BOUNDS = {"ink": (0.08, 0.24), "paper": (0.30, 0.52)}


def bounded(p: P.Params) -> P.Params:
    p = p.clamp()
    for k, (lo, hi) in JUDGE_BOUNDS.items():
        setattr(p, k, float(np.clip(getattr(p, k), lo, hi)))
    return p


# --------------------------------------------------------------------------- candidates
def neighbours(center: P.Params, keys: list[str], scale: float = 1.0) -> list[P.Params]:
    out = []
    for k in keys:
        for d in (+1, -1):
            out.append(bounded(replace(center, **{k: getattr(center, k) + d * STEPS[k] * scale})))
    return out


def first_round(base: P.Params) -> list[P.Params]:
    c = [base]
    c += neighbours(base, ["ink", "paper"], scale=1.5)
    c.append(bounded(replace(base, fine=base.fine + 0.8, ink_lines=base.ink_lines + 0.5)))
    return c  # 6 candidates, spread wide enough to look different on the sheet


def next_round(best: P.Params, adjust: dict, scale: float) -> list[P.Params]:
    moved = {k: getattr(best, k) + int(v) * STEPS[k] * scale for k, v in adjust.items() if int(v)}
    center = bounded(replace(best, **moved)) if moved else best
    keys = [k for k, v in adjust.items() if int(v)] or ["ink", "paper"]
    c = [best]  # the previous winner always stays in the race
    if center != best:
        c.append(center)
    c += neighbours(center, keys[:3], scale)
    # dedupe
    uniq, seen = [], set()
    for p in c:
        key = tuple(p.to_dict().values())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq[:6]


# --------------------------------------------------------------------------- sheet
def make_sheet(prep: P.Prepared, cands: list[P.Params], bg: str, path: Path) -> None:
    tiles, thumbs = [], []
    for i, prm in enumerate(cands, 1):
        tone, _ = P.render(prep, prm)
        rgba = P.to_rgba(tone, P.finish_alpha(prep))
        img = P.composite(rgba, bg)
        H, W = img.shape[:2]
        # head close-up: the halftone dots must stay visible after the model's own downscaling;
        # an object is shown whole (its frame is already tight around it)
        head = img if prep.kind == "object" else img[int(0.10 * H):int(0.78 * H), int(0.12 * W):int(0.88 * W)]
        h = 560
        head = cv2.resize(head, (int(head.shape[1] * h / head.shape[0]), h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(head, (0, 0), (64, 56), (0, 0, 0), -1)
        cv2.putText(head, str(i), (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
        tiles.append(head)
        thumbs.append(img)
    cols = 3
    w = max(t.shape[1] for t in tiles)
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        row = [cv2.copyMakeBorder(t, 4, 4, 4, 4 + w - t.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in row]
        while len(row) < cols:
            row.append(np.full_like(row[0], 40))
        rows.append(np.hstack(row))
    # thumbnail strip: the poster/social avatar size, where features must still read
    th = [cv2.resize(t, (int(t.shape[1] * 130 / t.shape[0]), 130), interpolation=cv2.INTER_AREA) for t in thumbs]
    strip = np.hstack([cv2.copyMakeBorder(t, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in th])
    rows.append(cv2.copyMakeBorder(strip, 0, 0, 0, max(0, rows[0].shape[1] - strip.shape[1]),
                                   cv2.BORDER_CONSTANT, value=(40, 40, 40))[:, :rows[0].shape[1]])
    sheet = np.vstack(rows)
    if sheet.shape[1] > 1800:
        s = 1800 / sheet.shape[1]
        sheet = cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), sheet)


# --------------------------------------------------------------------------- model call
def ask_claude(sheet: Path, refs: list[Path], flags: list[str], model: str, kind: str = "face",
               timeout: int = 300) -> dict:
    if not shutil.which("claude"):
        raise RuntimeError(NO_CLAUDE)
    prompt = (PROMPT_OBJECT if kind == "object" else PROMPT).format(ref=", ".join(map(str, refs)), sheet=sheet, flags="; ".join(flags) or "none")
    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--allowedTools", "Read",
        "--output-format", "json",
        "--json-schema", json.dumps(SCHEMA),
        "--no-session-persistence",
        "--disable-slash-commands",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--add-dir", str(sheet.parent),
    ] + [x for r in {str(r.parent) for r in refs} for x in ("--add-dir", r)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(sheet.parent))
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI failed ({r.returncode}): {r.stderr[-500:] or r.stdout[-500:]}")
    env = json.loads(r.stdout)
    data = env.get("structured_output")
    if data is None:  # older CLI: JSON inside the text result
        txt = env.get("result", "")
        data = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    data["_cost_usd"] = env.get("total_cost_usd")
    return data


def tune(prep: P.Prepared, prm: P.Params, bg: str, refs: list[Path], model: str, rounds: int,
         workdir: Path, name: str, on_round=None) -> tuple[P.Params, list[dict]]:
    """The model-in-the-loop search. Returns the best params and one history entry per round.
    on_round(entry) is called after every round (progress for the lab)."""
    workdir.mkdir(parents=True, exist_ok=True)
    cands = first_round(prm)
    best = prm
    history: list[dict] = []
    for rnd in range(1, rounds + 1):
        sheet = workdir / f"{name}-r{rnd}.jpg"
        make_sheet(prep, cands, bg, sheet)
        _, info_b = P.render(prep, best)
        flags = P.metrics(prep, info_b, best)["flags"]
        try:
            ans = ask_claude(sheet.resolve(), [r.resolve() for r in refs], flags, model, prep.kind)
        except Exception as e:  # model unavailable: keep what we have
            history.append({"round": rnd, "error": str(e)[:300]})
            break
        idx = min(max(int(ans["best"]), 1), len(cands)) - 1
        best = cands[idx]
        entry = {"round": rnd, "sheet": str(sheet), "candidates": [c.to_dict() for c in cands],
                 "picked": idx + 1, **{k: ans[k] for k in ("good_enough", "issues", "adjust", "note")},
                 "cost_usd": ans.get("_cost_usd")}
        history.append(entry)
        if on_round:
            on_round(entry)
        if ans["good_enough"]:
            break
        if rnd >= 2 and idx == 0:  # previous winner kept: the search has converged
            entry["converged"] = True
            break
        cands = next_round(best, ans["adjust"], scale=0.8 ** (rnd - 1))
    return best, history


def rule_fix(prm: P.Params, flags: list[str]) -> P.Params | None:
    """No-model fallback: nudge parameters for flags the metrics can see."""
    upd = {}
    if any("lost in shadow" in f for f in flags):
        upd["ink"] = prm.ink - 0.03
        upd["pop"] = prm.pop + 0.3
    if any("washed out" in f for f in flags):
        upd["ink_lines"] = prm.ink_lines + 0.4
    if any("noisy" in f for f in flags):
        upd["smooth"] = prm.smooth + 0.3
        upd["despeckle"] = prm.despeckle + 0.4
    return replace(prm, **upd).clamp() if upd else None
