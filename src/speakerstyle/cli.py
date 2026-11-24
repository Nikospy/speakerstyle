"""speakerstyle: turn any portrait or object photo into the event's halftone cut-out style.

  speakerstyle process photo.jpg [more.jpg ...] --bg "#6b2a9a" --out out/
  speakerstyle process photos/ --no-judge          # metrics-only, no model calls
  speakerstyle render photo.jpg --params '{"ink":0.15}' --bg "#0d0d0d"
  speakerstyle render photos/ --settings my-look-settings.json   # a look saved in the lab
  speakerstyle lab                                  # local web page at http://localhost:8770/
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
import time
from pathlib import Path

import cv2

from . import judge as J
from . import pipeline as P

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SEVERE = {"eyes_lost", "cutout_damaged", "background_leftovers", "unflattering"}
DEFAULT_REFS = P.style_refs()


def _inputs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS)
        elif p.suffix.lower() in EXTS:
            out.append(p)
        else:
            print(f"skip (not an image): {p}", file=sys.stderr)
    return out


def _write_outputs(prep, prm, out: Path, name: str, bg: str, ink: str, paper: str) -> dict:
    tone, info = P.render(prep, prm)
    rgba = P.to_rgba(tone, P.finish_alpha(prep), P.hex_to_rgb(ink), P.hex_to_rgb(paper))
    cv2.imwrite(str(out / f"{name}.png"), rgba)
    cv2.imwrite(str(out / f"{name}-preview.jpg"), P.composite(rgba, bg), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return P.metrics(prep, info, prm)


def process_one(path: Path, a) -> dict:
    name = path.stem
    rep: dict = {"input": str(path), "name": name}
    t0 = time.time()
    img = P.load(path)
    g = P.gate(img, a.subject)
    rep["subject"] = g.kind
    rep["gate"] = {"ok": g.ok, "problems": g.problems, "warnings": g.warnings,
                   "stats": {k: (float(v) if hasattr(v, "item") else v) for k, v in g.stats.items()}}
    if not g.ok:
        rep["status"] = "rejected"
        return rep
    try:
        prep = P.prepare(img, g)
    except P.NoSubject as e:
        rep["gate"]["problems"].append(str(e))
        rep["status"] = "rejected"
        return rep
    rep["gate"]["warnings"] = g.warnings
    rep["gate"]["stats"] = {k: (float(v) if hasattr(v, "item") else v) for k, v in g.stats.items()}
    prm = P.Params.from_dict(json.loads(a.params) if a.params else {}, prep.kind)

    tone, info = P.render(prep, prm)
    m = P.metrics(prep, info, prm)
    history = []

    if a.judge:
        def log(e):
            print(f"  round {e['round']}: picked {e['picked']}/{len(e['candidates'])} good_enough={e['good_enough']} "
                  f"issues={e['issues']} note={e['note'][:120]}", flush=True)
        prm, history = J.tune(prep, prm, a.bg, a.ref, a.model, a.rounds, a.out / "_rounds", name, log)
    elif a.cmd == "process":  # render keeps the settings exactly as given
        for _ in range(3):
            fix = J.rule_fix(prm, m["flags"])
            if not fix:
                break
            prm = fix
            _, info = P.render(prep, prm)
            m = P.metrics(prep, info, prm)
            history.append({"rule_fix": prm.to_dict(), "flags": m["flags"]})

    m = _write_outputs(prep, prm, a.out, name, a.bg, a.ink, a.paper)
    if getattr(a, "experimental", False):
        rep["live"] = str(write_live(prep, prm, a.out, name, a.bg))
    rep.update(status="ok", params=prm.to_dict(), metrics=m, history=history,
               seconds=round(time.time() - t0, 1))
    last = history[-1] if history else {}
    severe = SEVERE & set(last.get("issues", [])) or any(
        k in f for f in m["flags"] for k in ("lost in shadow", "frame edges", "will not read"))
    if severe and not last.get("good_enough", False):
        rep["status"] = "needs_review"
    return rep


def write_live(prep, prm, out: Path, name: str, bg: str) -> Path:
    """Experimental: layers + an animated page with four variants (own halftone shader + Paper shaders)."""
    layers = out / f"{name}-layers"
    meta = P.export_layers(prep, prm, layers)
    tpl = (Path(__file__).parent / "live_template.html").read_text()
    page = (tpl.replace("__NAME__", html.escape(name)).replace("__BG__", bg)
            .replace("__LAYERS__", layers.name).replace("__META__", json.dumps(meta))
            .replace("__ASPECT__", f"{meta['width']} / {meta['height']}"))
    path = out / f"{name}-live.html"
    path.write_text(page)
    return path


def serve(out: Path, port: int) -> None:
    """The live pages load images into WebGL, which browsers block on file:// URLs."""
    import functools
    import http.server

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    print(f"serving {out} at http://localhost:{port}/  (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


def write_index(out: Path, reports: list[dict], bg: str) -> None:
    rows = []
    for r in reports:
        n = html.escape(r["name"])
        notes = r["gate"]["problems"] + r["gate"]["warnings"] + r.get("metrics", {}).get("flags", [])
        judge = [h.get("note") or h.get("error") for h in r.get("history", []) if h.get("note") or h.get("error")]
        img = f'<img src="{n}-preview.jpg">' if r["status"] != "rejected" else '<div class="rej">rejected</div>'
        if r.get("live"):
            img = f'<a href="{n}-live.html">{img}</a>'
        rows.append(f"""<figure class="{r['status']}">{img}<figcaption><b>{n}</b> · {r['status']}
<ul>{''.join(f'<li>{html.escape(x)}</li>' for x in notes + judge)}</ul></figcaption></figure>""")
    page = f"""<!DOCTYPE html><html lang="en"><meta charset="utf-8"><title>speakerstyle results</title>
<style>body{{font:14px system-ui;margin:24px;background:#111;color:#eee}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}}
figure{{margin:0}}img{{width:100%;display:block;background:{bg}}}
.needs_review figcaption b::after{{content:" ⚠"}}.rej{{aspect-ratio:1;display:grid;place-items:center;background:#300}}
ul{{margin:4px 0 0;padding-left:18px;color:#aaa}}</style>
<h1>speakerstyle · {len(reports)} photos</h1><main>{''.join(rows)}</main></html>"""
    (out / "index.html").write_text(page)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="speakerstyle", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("process", "render"):
        s = sub.add_parser(cmd)
        s.add_argument("inputs", nargs="+", help="image files or folders")
        s.add_argument("--out", type=Path, default=Path("out"))
        s.add_argument("--bg", help="background colour of this event, for previews and judging (default #6b2a9a)")
        s.add_argument("--ink", help="ink colour (default black)")
        s.add_argument("--paper", help="paper colour (default white)")
        s.add_argument("--params", help="JSON with parameter overrides, e.g. '{\"ink\":0.15}'")
        s.add_argument("--settings", type=Path,
                       help="settings file saved in the lab (Save settings): its look and colours apply to every photo; "
                            "--bg, --ink and --paper override its colours")
        s.add_argument("--enhanced-experimental", dest="experimental", action="store_true",
                       help="also export layers and an animated NAME-live.html (view with: speakerstyle serve OUT)")
        s.add_argument("--subject", choices=("auto", "face", "object"), default="auto",
                       help="auto: portrait when a face fills enough of the photo, otherwise object")
        s.add_argument("--ref", type=Path, action="append", help="style reference image (repeatable); default: every image in reference/")
    sub.choices["process"].add_argument("--no-judge", dest="judge", action="store_false",
                                        help="metrics and rule fixes only, no model calls")
    sub.choices["process"].add_argument("--model", default="sonnet", help="claude model alias for judging")
    sub.choices["process"].add_argument("--rounds", type=int, default=3)
    lb = sub.add_parser("lab", help="local web page: import a photo, pick a version, fine-tune, export")
    lb.add_argument("--port", type=int, default=8770)
    sv = sub.add_parser("serve", help="serve an output folder so the live pages work")
    sv.add_argument("out", type=Path, nargs="?", default=Path("out"))
    sv.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    if a.cmd == "lab":
        from . import lab
        lab.serve(a.port)
        return 0
    if a.cmd == "serve":
        serve(a.out, a.port)
        return 0
    if a.cmd == "render":
        a.judge, a.rounds, a.model = False, 0, None
    st = json.loads(a.settings.read_text()) if getattr(a, "settings", None) else {}
    if st:
        a.params = json.dumps(st.get("params", {}))
    a.bg = a.bg or st.get("bg", "#6b2a9a")
    a.ink = a.ink or st.get("ink", "#000000")
    a.paper = a.paper or st.get("paper", "#ffffff")
    a.ref = a.ref or DEFAULT_REFS
    if a.judge and not shutil.which("claude"):
        print("Claude Code (the `claude` command) is not installed, so the photos are tuned by rules only.\n"
              "Install Claude Code to let Claude pick the best version.", file=sys.stderr)
        a.judge = False
    a.out.mkdir(parents=True, exist_ok=True)

    reports = []
    for path in _inputs(a.inputs):
        print(f"{path.name}", flush=True)
        try:
            rep = process_one(path, a)
        except Exception as e:
            rep = {"input": str(path), "name": path.stem, "status": "error", "error": str(e),
                   "gate": {"problems": [str(e)], "warnings": []}}
        print(f"  -> {rep['status']}" + (f" ({'; '.join(rep['gate']['problems'])})" if rep["status"] in ("rejected", "error") else ""),
              flush=True)
        P.save_json(a.out / f"{path.stem}.json", rep)
        reports.append(rep)
    write_index(a.out, reports, a.bg)
    print(f"done: {a.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
