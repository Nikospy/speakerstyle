"""Drive Affinity by Canva 3.x through its local MCP server (stdlib only).

Transport: SSE at http://localhost:6767/sse. GET /sse -> first `data:` line is the POST endpoint;
JSON-RPC is POSTed there and the answers come back on the SSE stream. Only protocolVersion
"2025-11-25" is accepted. Every session must first read the doc topic "preamble".

NEVER call the tools search_sdk_hints / add_sdk_hint / report_sdk_issue: they send text to a shared pool.
"""
from __future__ import annotations

import base64
import json
import queue
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://localhost:6767"
DESKTOP_DIR = Path.home() / "Desktop" / "speakerstyle-lab"
_PROTOCOL = "2025-11-25"


class AffinityError(RuntimeError):
    """Message is plain English a user can act on."""


# --------------------------------------------------------------------------- MCP client

class _Session:
    def __init__(self) -> None:
        self.q: queue.Queue[str] = queue.Queue()
        self.n = 10
        self.resp = None
        try:
            self.resp = urllib.request.urlopen(BASE + "/sse", timeout=300)
        except (urllib.error.URLError, OSError) as e:
            raise AffinityError(_not_reachable_message()) from e
        threading.Thread(target=self._reader, daemon=True).start()
        try:
            d = self.q.get(timeout=10)
        except queue.Empty as e:
            raise AffinityError("Affinity did not answer on its MCP port. Restart Affinity.") from e
        self.endpoint = d if d.startswith("http") else BASE + d
        self._rpc("initialize", {"protocolVersion": _PROTOCOL, "capabilities": {},
                                 "clientInfo": {"name": "speakerstyle-lab", "version": "0"}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.call("read_sdk_documentation_topic", {"filename": "preamble"})

    def _reader(self) -> None:
        try:
            for raw in self.resp:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("data:"):
                    self.q.put(line[5:].strip())
        except Exception:
            pass
        self.q.put("__closed__")

    def _post(self, msg: dict) -> None:
        req = urllib.request.Request(self.endpoint, json.dumps(msg).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()

    def _rpc(self, method: str, params: dict, timeout: float = 180) -> dict:
        self.n += 1
        i = self.n
        self._post({"jsonrpc": "2.0", "id": i, "method": method, "params": params})
        while True:
            try:
                raw = self.q.get(timeout=timeout)
            except queue.Empty as e:
                raise AffinityError("Affinity took too long to answer. Is a dialog open in Affinity?") from e
            if raw == "__closed__":
                raise AffinityError("The connection to Affinity closed. Restart Affinity.")
            try:
                m = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(m, dict) and m.get("id") == i:   # the stream also carries non-JSON-RPC noise
                if "error" in m:
                    raise AffinityError(f"Affinity MCP error: {m['error'].get('message', m['error'])}")
                return m.get("result", {})

    def call(self, tool: str, args: dict, timeout: float = 180) -> list[dict]:
        res = self._rpc("tools/call", {"name": tool, "arguments": args}, timeout)
        content = res.get("content", [])
        if res.get("isError"):
            text = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
            raise AffinityError(f"Affinity refused the request: {text[:400]}")
        return content

    def close(self) -> None:
        try:
            self.resp.close()
        except Exception:
            pass


_lock = threading.RLock()
_sess: _Session | None = None


def _not_reachable_message() -> str:
    return ("Affinity is not running, or its MCP server is off. Open Affinity, then turn on "
            "Settings > Model Context Protocol > Enable Affinity MCP and restart Affinity")


def _call(tool: str, args: dict, timeout: float = 180) -> list[dict]:
    """Call one MCP tool on a persistent session; reconnect once if the session died."""
    global _sess
    with _lock:
        for attempt in (0, 1):
            try:
                if _sess is None:
                    _sess = _Session()
                return _sess.call(tool, args, timeout)
            except AffinityError as e:
                dead = "closed" in str(e) or "not running" in str(e) or isinstance(e.__cause__, OSError)
                if _sess is not None and dead:
                    _sess.close()
                    _sess = None
                if attempt == 1 or not dead:
                    raise
            except (urllib.error.URLError, OSError):
                if _sess is not None:
                    _sess.close()
                    _sess = None
                if attempt == 1:
                    raise AffinityError(_not_reachable_message())
    raise AffinityError("unreachable")


def run_script(js: str, timeout: float = 180) -> str:
    """Run JavaScript in Affinity; return what it console.log()ed. Errors raise AffinityError."""
    out = "\n".join(c.get("text", "") for c in _call("execute_script", {"script": js}, timeout)
                    if c.get("type") == "text")
    low = out.lower()
    if "not_allowed" in low:
        raise AffinityError("Affinity blocks scripts from using files. Allow it in Affinity: "
                            "Settings > Model Context Protocol (file access)")
    return out


# --------------------------------------------------------------------------- status

def status() -> dict:
    """{"connected": bool, "message": str}. Never raises."""
    try:
        run_script("console.log('ok')", timeout=15)
        return {"connected": True, "message": "Connected to Affinity"}
    except AffinityError as e:
        msg = str(e)
    except Exception as e:  # pragma: no cover - defensive
        msg = f"Could not talk to Affinity: {e}"
    return {"connected": False, "message": msg}


# --------------------------------------------------------------------------- effects

def _p(label, lo, hi, default, step):
    return {"label": label, "min": float(lo), "max": float(hi), "default": float(default), "step": float(step)}


# Order = stacking order, bottom to top. All of them are live layers INSIDE the "Artwork" group,
# so they act on the finished artwork (paper+halftone+ink composite) and never on the background:
# the group is forced to Normal blend (non-passthrough). With a passthrough group a filter inside it
# also hits everything below the group (verified: background got grain).
# Ranges come from param_ranges.min.json (AddNoise intensity 0-1, UnsharpMask factor 0-4 radius 0-1024,
# Halftone cellSize 2-500 / contrast 0-100 / angle 0-2pi, Posterise levels 2-256); the UI ranges below are
# narrower, sensible sub-ranges of those.
EFFECTS: dict = {
    "tones": {"label": "Snap to tones (posterise)", "params": {
        "levels": _p("Tone levels (3 = black, grey, white)", 2, 8, 3, 1)}},
    "halftone": {"label": "Affinity halftone screen", "params": {
        "cell": _p("Dot size (px)", 2, 40, 8, 1),
        "angle": _p("Screen angle (deg)", 0, 90, 45, 1),
        "contrast": _p("Contrast", 0, 100, 50, 1)}},
    "sharpen": {"label": "Sharpen (unsharp mask)", "params": {
        "amount": _p("Amount", 0, 4, 1.5, 0.1),
        "radius": _p("Radius (px)", 0.5, 20, 2, 0.5)}},
    "grain": {"label": "Print grain (noise)", "params": {
        "amount": _p("Amount", 0, 1, 0.12, 0.01)}},
}
_FX_PREFIX = "Effect: "


def _clean_effects(effects: dict | None) -> list[dict]:
    """Validate/clamp -> ordered list of {id, name, params} for the enabled effects."""
    out = []
    effects = effects or {}
    for fid, spec in EFFECTS.items():
        e = effects.get(fid)
        if not e or not e.get("on"):
            continue
        params = {}
        for pid, pspec in spec["params"].items():
            v = e.get(pid, pspec["default"])
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise AffinityError(f"Effect {spec['label']}: {pid} must be a number")
            params[pid] = min(max(v, pspec["min"]), pspec["max"])
        out.append({"id": fid, "name": _FX_PREFIX + spec["label"], "params": params})
    unknown = set(effects) - set(EFFECTS)
    if unknown:
        raise AffinityError(f"Unknown effect: {', '.join(sorted(unknown))}")
    return out


# JS shared by all scripts. Pitfalls:
# - console.log is the only output channel; every script must be self-contained.
# - a node's name is `description` (= userDescription once set via the definition).
# - doc.addNode(def, targetNode) inserts INTO a container target, on top of its children.
# - layers/children are iterables, not arrays (no [i], no .map).
# - filter nodes cannot be edited from the node (parameters getter is read-only); rebuild instead.
_JS_PRE = r"""
const {Document, NewDocumentOptions, FileExportOptions} = require('/document.js');
const {UnitType} = require('/units.js');
const {Application} = require('/application.js');
const {Bitmap, RasterFormat} = require('/rasterobject.js');
const N = require('/nodes.js');
const {ShapeRectangle} = require('/shapes.js');
const {Rectangle} = require('/geometry.js');
const {Colour} = require('/colours.js');
const {BlendMode} = require('affinity:common');
const FXP = %(prefix)s;
function findDoc(uuid) {
  for (const d of Document.all) if (d.sessionUuid === uuid) return d;
  throw new Error('NO_DOC: that Affinity document is closed');
}
function findGroup(doc) {
  for (const l of doc.currentSpread.layers) if (l.description === 'Artwork') return l;
  throw new Error('NO_GROUP: the Artwork group is missing');
}
const BUILD = {
  tones: p => { const q = N.PosteriseAdjustmentParameters.create(); q.levels = Math.round(p.levels);
    return N.PosteriseAdjustmentRasterNodeDefinition.create(q); },
  halftone: p => { const q = N.HalftoneFilterParameters.create(); q.cellSize = p.cell;
    q.screenAngle = p.angle * Math.PI / 180; q.contrast = p.contrast;
    q.screenType = N.HalftoneScreenType.Mono; q.dotType = N.HalftoneDotType.Round;
    return N.HalftoneFilterRasterNodeDefinition.create(q); },
  sharpen: p => { const q = N.UnsharpMaskFilterParameters.create(); q.factor = p.amount; q.radius = p.radius;
    q.threshold = 0; return N.UnsharpMaskFilterRasterNodeDefinition.create(q); },
  grain: p => { const q = N.AddNoiseFilterParameters.create(); q.intensity = p.amount;
    q.isMonochromatic = true; q.noiseType = N.AddNoiseType.Gaussian;
    return N.AddNoiseFilterRasterNodeDefinition.create(q); },
};
function rebuildEffects(doc, grp, fx) {
  const old = [];
  for (const c of grp.children) if (c.description.startsWith(FXP)) old.push(c);
  if (old.length) doc.deleteSelection(old);
  for (const e of fx) { const d = BUILD[e.id](e.params); d.userDescription = e.name; doc.addNode(d, grp); }
}
""" % {"prefix": json.dumps(_FX_PREFIX)}


def _js(body: str, **data) -> str:
    return _JS_PRE + f"const DATA = {json.dumps(data)};\n" + body


def _json_out(out: str) -> dict:
    """Scripts print one line `RESULT {json}`; failures print/raise errors."""
    for line in out.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    if "NO_DOC" in out:
        raise AffinityError("That Affinity document was closed. Open the layers again.")
    raise AffinityError(f"Affinity did not return a result: {out.strip()[:300] or 'empty answer'}")


def _safe(name: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in name).strip(" .")
    return s or "doc"


# --------------------------------------------------------------------------- document

def open_layers(name: str, layers: list[tuple[str, Path]], size: tuple[int, int], bg_hex: str,
                effects: dict[str, dict] | None = None) -> dict:
    """New Affinity document: background shape + group "Artwork" (PNG layers + effects)."""
    st = status()
    if not st["connected"]:
        raise AffinityError(st["message"])
    hexs = bg_hex.lstrip("#")
    if len(hexs) != 6 or any(c not in "0123456789abcdefABCDEF" for c in hexs):
        raise AffinityError("Background colour must look like #RRGGBB")
    rgb = [int(hexs[i:i + 2], 16) for i in (0, 2, 4)]
    folder = DESKTOP_DIR / _safe(name)
    folder.mkdir(parents=True, exist_ok=True)
    items = []
    for i, (lname, src) in enumerate(layers):
        src = Path(src)
        if not src.is_file():
            raise AffinityError(f"Layer file not found: {src}")
        dst = folder / f"{i + 1:02d}-{_safe(lname)}.png"
        shutil.copyfile(src, dst)
        items.append({"name": lname, "file": f"{_safe(name)}/{dst.name}"})
    w, h = int(size[0]), int(size[1])
    body = r"""
const base = Application.userDesktopPath + '/speakerstyle-lab/';
const o = NewDocumentOptions.createDefault();
o.units = UnitType.Pixel; o.width = DATA.w; o.height = DATA.h; o.dpi = 72; o.isTransparentBackground = true;
const doc = Document.create(o);
const bg = N.ShapeNodeDefinition.create(ShapeRectangle.create(), new Rectangle(0, 0, DATA.w, DATA.h),
  Colour.createRGBA8({r: DATA.rgb[0], g: DATA.rgb[1], b: DATA.rgb[2], alpha: 255}), null, null, null);
bg.userDescription = 'Event background';
doc.addNode(bg);
doc.addNode(N.ContainerNodeDefinition.create('Artwork'));
const grp = findGroup(doc);
doc.setBlendMode(BlendMode.Normal, false, grp);   // isolate: effects must not touch the background
for (const it of DATA.items) {
  const d = N.ImageNodeDefinition.create(RasterFormat.RGBA8);
  d.bitmap = Bitmap.loadFromFile(base + it.file, RasterFormat.RGBA8);
  d.userDescription = it.name;
  doc.addNode(d, grp);
}
rebuildEffects(doc, grp, DATA.fx);
console.log('RESULT ' + JSON.stringify({doc: doc.sessionUuid}));
"""
    res = _json_out(run_script(_js(body, w=w, h=h, rgb=rgb, items=items, fx=_clean_effects(effects))))
    return {"doc": res["doc"], "folder": str(folder)}


def apply_effects(doc, effects: dict[str, dict]) -> None:
    """Make the document's effect layers match `effects` (rebuilds them, so it is idempotent)."""
    body = "const d = findDoc(DATA.doc); rebuildEffects(d, findGroup(d), DATA.fx); console.log('RESULT {}');"
    _json_out(run_script(_js(body, doc=doc, fx=_clean_effects(effects))))


def preview(doc, out_path: Path) -> Path:
    """JPEG (max 1024 px on the long side) of the document as Affinity renders it."""
    for c in _call("render_spread", {"document_session_uuid": doc, "spread_index": 0}, 120):
        if c.get("type") == "image":
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(base64.b64decode(c["data"]))
            return out_path
    raise AffinityError("Affinity did not return a preview. Is the document still open?")


_EXT = {"PNG": ".png", "JPEG (Best quality)": ".jpg", "PSD (preserve editability)": ".psd", "AFFINITY": ".af"}


def export(doc, out_path: Path, preset: str = "PNG") -> Path:
    """Export through the SDK into DESKTOP_DIR/_export, then copy to out_path.

    preset: "PNG", "JPEG (Best quality)", "PSD (preserve editability)" (any name in
    FileExportOptions.allPresetNames) or "AFFINITY" (native .af via doc.saveAs)."""
    out_path = Path(out_path)
    ext = _EXT.get(preset)
    if ext is None:
        raise AffinityError(f"Unknown export preset: {preset}")
    tmp_dir = DESKTOP_DIR / "_export"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / (_safe(out_path.stem) + ext)
    tmp.unlink(missing_ok=True)
    body = r"""
const d = findDoc(DATA.doc);
const path = Application.userDesktopPath + '/speakerstyle-lab/_export/' + DATA.file;
if (DATA.preset === 'AFFINITY') d.saveAs(path);
else d.export(path, FileExportOptions.createWithPresetName(DATA.preset));
console.log('RESULT {}');
"""
    _json_out(run_script(_js(body, doc=doc, file=tmp.name, preset=preset), timeout=300))
    if not tmp.is_file():
        raise AffinityError("Affinity finished but wrote no file. Check that the Desktop is allowed in Affinity.")
    if tmp.resolve() != out_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp, out_path)
    return out_path


def close_doc(doc) -> None:
    """Close a document this module created (no save prompt: it is discarded)."""
    _json_out(run_script(_js("findDoc(DATA.doc).close(); console.log('RESULT {}');", doc=doc)))
