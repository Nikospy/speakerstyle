"""Speaker Style Lab: a local web page to style one photo by eye.

  speakerstyle lab            # then open http://localhost:8770/

The page only draws controls. Everything heavy runs here in Python: the cut-out once per photo
(~10 s), then a render per slider change (~0.4 s). Affinity (optional) is driven through its MCP
server by affinity.py.
"""
from __future__ import annotations

import base64
import dataclasses
import http.server
import json
import re
import threading
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from . import judge as J
from . import pipeline as P

PAGE = Path(__file__).with_name("lab.html")
WORK = P.ROOT / "out" / "lab"
REFS = P.style_refs()

# What the sliders show. Order = order on the page; "main" ones are always visible.
KNOBS = [
    # id, label, min, max, step, main, face-only
    ("ink", "Black", 0.04, 0.35, 0.01, True, False),
    ("paper", "White", 0.15, 0.65, 0.01, True, False),
    ("pop", "Shadow detail", 0.0, 1.5, 0.05, True, False),
    ("grey", "Dot darkness", 0.2, 0.75, 0.01, True, False),
    ("dot_cell", "Dot size", 0.0065, 0.02, 0.0005, False, False),
    ("dot_angle", "Screen angle", 0.0, 90.0, 1.0, False, False),
    ("smooth", "Smooth shapes", 0.0, 1.5, 0.05, False, False),
    ("ink_lines", "Feature lines", 0.0, 2.0, 0.05, False, False),
    ("fine", "Hair and fabric lines", 0.0, 3.0, 0.05, False, False),
    ("skin_smooth", "Smooth skin", 0.0, 2.0, 0.05, False, True),
    ("detail", "Local contrast", 0.0, 2.0, 0.05, False, False),
    ("despeckle", "Clean up specks", 0.0, 1.5, 0.05, False, False),
    ("outline", "Outline", 0.0, 3.0, 0.25, False, False),
    ("paint", "Brush strokes (1 = on)", 0.0, 1.0, 1.0, False, False),
]

# The six versions to pick from: named moves away from the current settings.
VERSIONS = [
    ("Current", {}),
    ("More black", {"ink": +0.05}),
    ("Lighter", {"ink": -0.04, "paper": +0.08}),
    ("Smoother", {"smooth": +0.4, "fine": -0.6, "despeckle": +0.3}),
    ("Crisper features", {"ink_lines": +0.6, "pop": +0.4}),
    ("Brush strokes", {"paint": +1, "grey": -0.10, "outline": +1}),
]
# Shown instead of "Brush strokes" when the current settings already use them.
UNDO_PAINT = ("No brush strokes", {"paint": -1, "grey": +0.10, "outline": -1})


@dataclasses.dataclass
class Photo:
    id: str
    name: str
    prep: P.Prepared
    gate: dict
    renders: OrderedDict = dataclasses.field(default_factory=OrderedDict)
    doc: object = None          # Affinity document handle, once opened there
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    tuning: bool = False


PHOTOS: "OrderedDict[str, Photo]" = OrderedDict()
JOBS: dict[str, dict] = {}
MAX_PHOTOS = 4             # each keeps its cut-out and renders in memory; older ones are dropped
MAX_UPLOAD = 60 * 2 ** 20  # bytes
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


class UserError(Exception):
    """Shown to the user as is: what happened and what to do."""


# --------------------------------------------------------------------------- core actions
def load_photo(data: bytes, filename: str, subject: str) -> dict:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise UserError("This file is not a photo the lab can read. Use JPG, PNG or WebP.")
    # cut-out work is sized from the face or subject, but a huge original slows every step
    if max(img.shape[:2]) > 4000:
        s = 4000 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = P.gate(img, subject)
    gate = {"problems": list(g.problems), "warnings": list(g.warnings), "kind": g.kind}
    if not g.ok:
        raise UserError(_plain(g.problems[0]))
    try:
        prep = P.prepare(img, g)
    except P.NoSubject as e:
        raise UserError(_plain(str(e)))
    gate["warnings"] = [_plain(w) for w in g.warnings]
    pid = uuid.uuid4().hex[:10]
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(filename).stem).strip("-")[:40] or "photo"
    PHOTOS[pid] = Photo(pid, name, prep, gate)
    while len(PHOTOS) > MAX_PHOTOS:
        PHOTOS.popitem(last=False)
    prm = P.Params.for_kind(prep.kind)
    return {"id": pid, "name": name, "kind": prep.kind, "warnings": gate["warnings"],
            "params": prm.to_dict(), "knobs": knobs(prep.kind),
            "size": [int(prep.alpha.shape[1]), int(prep.alpha.shape[0])]}


def _plain(msg: str) -> str:
    """Gate messages are written for a log; make the first letter a capital and end with a full stop."""
    msg = msg.strip()
    return (msg[:1].upper() + msg[1:]).rstrip(".") + "."


def knobs(kind: str) -> list[dict]:
    return [{"id": k, "label": lab, "min": lo, "max": hi, "step": st, "main": main}
            for k, lab, lo, hi, st, main, face in KNOBS if not (face and kind != "face")]


def params_from(d, kind: str) -> P.Params:
    if d is None:
        d = {}
    if not isinstance(d, dict) or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in d.values()):
        raise UserError("Settings must be numbers. Reload the page and try again.")
    return P.Params.from_dict(d, kind)


def colour(v, default: str) -> str:
    v = default if v is None else v
    if not isinstance(v, str) or not HEX.match(v):
        raise UserError("Colours must look like #1a2b3c.")
    return v.lower()


def render_png(photo: Photo, prm: P.Params, ink: str, paper: str, height: int | None = None) -> bytes:
    key = json.dumps([dataclasses.asdict(prm), ink, paper, height], sort_keys=True)
    with photo.lock:
        if key in photo.renders:
            photo.renders.move_to_end(key)
            return photo.renders[key]
    tone, _ = P.render(photo.prep, prm)
    rgba = P.to_rgba(tone, P.finish_alpha(photo.prep), P.hex_to_rgb(ink), P.hex_to_rgb(paper))
    if height and height < rgba.shape[0]:
        s = height / rgba.shape[0]
        rgba = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    png = cv2.imencode(".png", rgba, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tobytes()
    with photo.lock:
        photo.renders[key] = png
        while len(photo.renders) > 40:
            photo.renders.popitem(last=False)
    return png


def versions(photo: Photo, base: P.Params) -> list[dict]:
    out = []
    for label, moves in VERSIONS:
        if "paint" in moves and base.paint:
            label, moves = UNDO_PAINT
        p = dataclasses.replace(base, **{k: getattr(base, k) + v for k, v in moves.items()}).clamp()
        out.append({"label": label, "params": p.to_dict()})
    return out


def start_tune(photo: Photo, prm: P.Params, bg: str) -> str:
    if photo.tuning:  # two runs would share the round sheets and read each other's candidates
        raise UserError("Claude is already tuning this photo. Wait for it to finish.")
    photo.tuning = True
    jid = uuid.uuid4().hex[:10]
    job = JOBS[jid] = {"state": "running", "round": 0, "rounds": 3, "note": "", "params": None, "error": None}

    def run():
        try:
            def on_round(e):
                job.update(round=e["round"], note=e.get("note", ""))
            best, hist = J.tune(photo.prep, prm, bg, REFS, "sonnet", 3, WORK / photo.id / "_rounds",
                                photo.name, on_round)
            errors = [h["error"] for h in hist if "error" in h]
            if errors and len(hist) == 1:
                raise UserError("Claude did not answer. Check that the claude command works in a terminal, "
                                "then try again.")
            last = next((h for h in reversed(hist) if "note" in h), {})
            job.update(state="done", params=best.to_dict(), note=last.get("note", ""),
                       good_enough=bool(last.get("good_enough")))
        except UserError as e:
            job.update(state="error", error=str(e))
        except Exception as e:  # keep the page alive; details go to the terminal
            traceback.print_exc()
            job.update(state="error", error=str(e) if str(e) == J.NO_CLAUDE else f"Tuning stopped: {e}")
        finally:
            photo.tuning = False
    threading.Thread(target=run, daemon=True).start()
    return jid


def export_file(photo: Photo, prm: P.Params, fmt: str, bg: str, ink: str, paper: str) -> tuple[bytes, str, str]:
    """Returns (bytes, mime type, file name)."""
    if fmt == "json":
        data = {"params": prm.to_dict(), "subject": photo.prep.kind, "bg": bg, "ink": ink, "paper": paper}
        return json.dumps(data, indent=2).encode(), "application/json", f"{photo.name}-settings.json"
    tone, _ = P.render(photo.prep, prm)
    rgba = P.to_rgba(tone, P.finish_alpha(photo.prep), P.hex_to_rgb(ink), P.hex_to_rgb(paper))
    if fmt == "png":
        return cv2.imencode(".png", rgba)[1].tobytes(), "image/png", f"{photo.name}-cutout.png"
    if fmt == "jpg":
        jpg = cv2.imencode(".jpg", P.composite(rgba, bg), [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tobytes()
        return jpg, "image/jpeg", f"{photo.name}-{bg.lstrip('#')}.jpg"
    raise UserError("Unknown export format.")


# --------------------------------------------------------------------------- Affinity
def _aff():
    try:
        from . import affinity
        return affinity
    except ImportError:
        return None


def affinity_status() -> dict:
    A = _aff()
    if A is None:
        return {"connected": False, "message": "Affinity support is not installed in this copy of the lab.", "effects": {}}
    try:
        return {**A.status(), "effects": A.EFFECTS}
    except Exception as e:  # a broken or half-updated Affinity module must not break the page
        traceback.print_exc()
        return {"connected": False, "message": f"Affinity support failed to start: {e}", "effects": {}}


def affinity_open(photo: Photo, prm: P.Params, bg: str, ink: str, paper: str, effects: dict) -> dict:
    A = _aff()
    if A is None:
        raise UserError("Affinity support is not installed in this copy of the lab.")
    tone, info = P.render(photo.prep, prm)
    folder = WORK / photo.id / "layers"
    folder.mkdir(parents=True, exist_ok=True)
    layers = []
    for name, arr in P.split_layers(photo.prep, info, P.hex_to_rgb(ink), P.hex_to_rgb(paper)):
        path = folder / f"{name.lower()}.png"
        cv2.imwrite(str(path), arr)
        layers.append((name, path))
    h, w = photo.prep.alpha.shape
    if photo.doc is not None:  # re-opening replaces the lab's previous document for this photo
        try:
            A.close_doc(photo.doc)
        except Exception:
            pass
        photo.doc = None
    try:
        res = A.open_layers(photo.name, layers, (w, h), bg, effects or None)
    except A.AffinityError as e:
        raise UserError(str(e))
    photo.doc = res["doc"]
    return {"folder": res.get("folder", "")}


def affinity_call(photo: Photo, what: str, payload: dict):
    A = _aff()
    if A is None or photo.doc is None:
        raise UserError("Open the photo in Affinity first.")
    try:
        if what == "effects":
            A.apply_effects(photo.doc, payload.get("effects") or {})
            return {}
        if what == "preview":
            out = A.preview(photo.doc, WORK / photo.id / "affinity-preview.jpg")
            return {"jpeg": base64.b64encode(Path(out).read_bytes()).decode()}
        if what == "export":
            preset = {"png": "PNG", "psd": "PSD (preserve editability)", "af": "AFFINITY"}.get(payload.get("format"), "PNG")
            ext = {"PNG": "png", "PSD (preserve editability)": "psd", "AFFINITY": "af"}[preset]
            out = A.export(photo.doc, WORK / photo.id / f"{photo.name}-affinity.{ext}", preset)
            return {"path": str(out)}
    except A.AffinityError as e:
        raise UserError(str(e))
    raise UserError("Unknown Affinity action.")


# --------------------------------------------------------------------------- HTTP
class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "SpeakerStyleLab/1"
    timeout = 60  # a stalled upload must not hold its thread forever

    def log_message(self, fmt, *args):  # quiet: only errors are printed
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body(self, limit: int = 2 ** 20) -> bytes:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > limit:
            raise UserError("This file is too big for the lab (limit 60 MB)." if limit == MAX_UPLOAD
                            else "The request is too big.")
        data = self.rfile.read(n) if n else b""
        if len(data) != n:
            raise UserError("The upload was cut off. Try again.")
        return data

    def _photo(self, d: dict) -> Photo:
        ph = PHOTOS.get(str(d.get("id", "")))
        if ph is not None:
            PHOTOS.move_to_end(ph.id)
        if ph is None:
            raise UserError("The lab was restarted, so this photo is gone. Choose the photo again.")
        return ph

    def do_GET(self):
        try:
            self._get()
        except Exception as e:
            traceback.print_exc()
            self._json({"error": f"Something went wrong in the lab: {e}. Details are in the terminal."}, 500)

    def _get(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/affinity/status":
            return self._json(affinity_status())
        if self.path.startswith("/api/job/"):
            job = JOBS.get(self.path.rsplit("/", 1)[-1])
            return self._json(job or {"state": "error", "error": "This task is gone. Try again."})
        self._send(404, b"not found", "text/plain")

    ROUTES = {"/api/photo", "/api/render", "/api/versions", "/api/tune", "/api/export", "/api/affinity/open",
              "/api/affinity/effects", "/api/affinity/preview", "/api/affinity/export"}

    def do_POST(self):
        try:
            if self.path not in self.ROUTES:
                return self._send(404, b"not found", "text/plain")
            if self.path == "/api/photo":
                name = self.headers.get("X-Filename", "photo.jpg")
                subject = self.headers.get("X-Subject", "auto")
                if subject not in ("auto", "face", "object"):
                    subject = "auto"
                data = self._body(MAX_UPLOAD)
                if not data:
                    raise UserError("The file is empty. Choose another photo.")
                return self._json(load_photo(data, name, subject))
            try:
                d = json.loads(self._body() or b"{}")
            except ValueError:
                raise UserError("The lab could not read this request. Reload the page.")
            if not isinstance(d, dict):
                raise UserError("The lab could not read this request. Reload the page.")
            ph = self._photo(d)
            prm = params_from(d.get("params"), ph.prep.kind)
            ink, paper = colour(d.get("ink"), "#000000"), colour(d.get("paper"), "#ffffff")
            bg = colour(d.get("bg"), "#6b2a9a")
            if self.path == "/api/render":
                height = d.get("height")
                if height is not None and not (isinstance(height, int) and 16 <= height <= 4000):
                    raise UserError("Preview height must be a whole number of pixels.")
                return self._send(200, render_png(ph, prm, ink, paper, height), "image/png")
            if self.path == "/api/versions":
                return self._json(versions(ph, prm))
            if self.path == "/api/tune":
                return self._json({"job": start_tune(ph, prm, bg)})
            if self.path == "/api/export":
                body, mime, fname = export_file(ph, prm, d.get("format", "png"), bg, ink, paper)
                return self._send(200, body, mime, {"Content-Disposition": f'attachment; filename="{fname}"'})
            if self.path == "/api/affinity/open":
                return self._json(affinity_open(ph, prm, bg, ink, paper, d.get("effects") or {}))
            if self.path.startswith("/api/affinity/"):
                return self._json(affinity_call(ph, self.path.rsplit("/", 1)[-1], d))
            self._send(404, b"not found", "text/plain")
        except UserError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": f"Something went wrong in the lab: {e}. Details are in the terminal."}, 500)


def serve(port: int = 8770) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Speaker Style Lab: http://localhost:{port}/  (Ctrl+C to stop)", flush=True)
    srv.serve_forever()
