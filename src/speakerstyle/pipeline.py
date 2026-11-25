"""Speaker photo stylization: gate -> normalize -> cut out -> 3-level halftone render.

Two subject profiles share one renderer:
  * face: a portrait; framing, scale and tone targets come from the face (eyes, skin get extra care)
  * object: anything else (a product, a prop, an animal); framing and scale come from the cut-out
    silhouette, tone targets from the whole subject. `face_h` is then just the scale unit.

The style (measured on the event's original speaker graphics):
  * three flat levels inside the face: solid ink (~13%), one halftone grey (~45%), paper (~42%)
  * halftone: round dots on a 45-degree screen, one fixed coverage, cell ~ face height / 65
  * smooth, comic-like boundaries between levels (edge-preserving smoothing before thresholds)
  * hard cut-out silhouette, output is RGBA so any background colour can go behind it

Thresholds are not fixed numbers: they are quantiles of the face luminance, so every photo
hits the same ink/grey/paper proportions on the face no matter how dark or bright it was.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
os.environ.setdefault("U2NET_HOME", str(MODELS))

WORK_FACE_H = 420  # face box height in px at working resolution
# objects: the subject's longer side at working size. Chosen so that the scale unit (face_h) stays 420,
# which keeps every size constant of the renderer (dots, smoothing, strokes) calibrated for objects too.
OBJ_UNIT_PER_LONG = 2.4
SUBJECT_MIN_FACE = 0.10  # auto mode: a face this tall relative to the photo makes it a portrait
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
REFERENCE = ROOT / "reference"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def style_refs() -> list[Path]:
    """Every image in reference/ (subfolders too): the judge compares candidates against these."""
    return sorted(p for p in REFERENCE.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _yunet() -> Path:
    """The face detector is small (~230 KB): fetch it on first use, like rembg does with its models."""
    if not YUNET.exists():
        import urllib.request
        MODELS.mkdir(parents=True, exist_ok=True)
        print("Downloading the face detector (one time, ~230 KB)...", flush=True)
        tmp = YUNET.with_suffix(".part")
        urllib.request.urlretrieve(YUNET_URL, tmp)
        tmp.rename(YUNET)
    return YUNET


# --------------------------------------------------------------------------- params
@dataclass
class Params:
    ink: float = 0.14         # target share of solid ink on the face
    paper: float = 0.43       # target share of plain paper (white) on the face
    grey: float = 0.40        # dot coverage of the halftone level (0..1)
    dot_cell: float = 0.009  # halftone cell size relative to face height
    dot_angle: float = 45.0   # screen angle in degrees
    smooth: float = 0.25      # edge-preserving smoothing strength (0 = off)
    detail: float = 1.0       # local contrast (CLAHE) strength (0 = off)
    pop: float = 0.5          # large-radius high-pass: lifts hair strands and features out of shadow
    ink_lines: float = 1.3    # extra ink on fine dark edges (eyes, brows), 0 = off
    despeckle: float = 0.4    # removes tiny isolated islands, 0 = off
    fine: float = 1.6         # strand-like detail outside the face (hair, clothes): engraving lines
    skin_smooth: float = 1.0  # extra flattening on the skin only, so the face stays clean
    tone_keep: float = 0.5    # 0 = every face forced to the same ink/paper shares; 1 = thresholds fixed
                              # relative to the photo's white point, so darker skin stays darker
    paint: float = 0.0        # 1 = brush-stroke renderer (see _zones_paint), 0 = smoothing renderer
    outline: float = 0.0      # black rim inside the silhouette edge, 1 = about 1 px on a 1000 px portrait

    def clamp(self) -> "Params":
        self.paint = 1.0 if self.paint >= 0.5 else 0.0
        self.outline = float(np.clip(self.outline, 0, 3))
        self.ink = float(np.clip(self.ink, 0.02, 0.45))
        self.paper = float(np.clip(self.paper, 0.10, 0.75))
        if self.ink + self.paper > 0.92:
            s = 0.92 / (self.ink + self.paper)
            self.ink *= s
            self.paper *= s
        self.grey = float(np.clip(self.grey, 0.2, 0.8))
        self.dot_cell = float(np.clip(self.dot_cell, 1 / 160, 1 / 25))
        self.smooth = float(np.clip(self.smooth, 0, 3))
        self.detail = float(np.clip(self.detail, 0, 3))
        self.pop = float(np.clip(self.pop, 0, 3))
        self.ink_lines = float(np.clip(self.ink_lines, 0, 2))
        self.despeckle = float(np.clip(self.despeckle, 0, 3))
        self.tone_keep = float(np.clip(self.tone_keep, 0, 1))
        self.fine = float(np.clip(self.fine, 0, 3))
        self.skin_smooth = float(np.clip(self.skin_smooth, 0, 2))
        return self

    def to_dict(self) -> dict:
        return {k: round(v, 6) for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict, kind: str = "face") -> "Params":
        names = {f.name for f in dataclasses.fields(cls)}
        return dataclasses.replace(cls.for_kind(kind), **{k: float(v) for k, v in d.items() if k in names}).clamp()

    @classmethod
    def for_kind(cls, kind: str) -> "Params":
        # objects: the portrait's strand strokes turn foil, stone and ceramic into speckle; the judge
        # moved fine down and smooth up on every object sample, so start there. Brush strokes are
        # tuned on portraits and turn most objects too dark, so objects keep the smoothing renderer.
        return cls(fine=0.5, smooth=0.6) if kind == "object" else cls(paint=1.0, grey=0.30, outline=1.0)


# --------------------------------------------------------------------------- gate
@dataclass
class Face:
    box: np.ndarray          # x, y, w, h (source px)
    eyes: np.ndarray         # 2x2 right eye, left eye (source px)
    nose: np.ndarray
    score: float


@dataclass
class GateResult:
    ok: bool
    face: Face | None
    problems: list[str] = field(default_factory=list)   # reasons to reject
    warnings: list[str] = field(default_factory=list)   # usable, but expect weaker result
    stats: dict = field(default_factory=dict)
    kind: str = "face"                                  # "face" or "object"


class NoSubject(ValueError):
    """The cut-out found nothing usable (empty, tiny, or the whole frame)."""


def detect_faces(img: np.ndarray) -> list[Face]:
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > 1600:  # YuNet is faster and more stable on moderate sizes
        scale = 1600 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img
    det = cv2.FaceDetectorYN.create(str(_yunet()), "", (small.shape[1], small.shape[0]), 0.6, 0.3, 50)
    _, faces = det.detect(small)
    out = []
    for f in faces if faces is not None else []:
        f = f / scale
        f[14] *= scale  # score is not a coordinate
        out.append(Face(box=f[0:4], eyes=f[4:8].reshape(2, 2), nose=f[8:10], score=float(f[14])))
    out.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
    return out


def gate(img: np.ndarray, subject: str = "auto") -> GateResult:
    """subject: "face" (portrait only), "object" (never look for a face) or "auto"."""
    faces = [] if subject == "object" else detect_faces(img)
    if subject == "auto" and faces and faces[0].box[3] < SUBJECT_MIN_FACE * img.shape[0]:
        faces = []  # a small face in the background does not make a product shot a portrait
    if not faces:
        if subject == "face":
            return GateResult(False, None, ["no face found (profile, covered face, or too small)"])
        return _gate_object(img)
    face = faces[0]
    res = GateResult(True, face)
    x, y, w, h = face.box
    res.stats.update(face_px=int(h), faces=len(faces), score=round(face.score, 3))
    if len(faces) > 1 and faces[1].box[3] > 0.6 * h:
        res.warnings.append(f"{len(faces)} faces, similar size; using the largest")
    if h < 90:
        res.problems.append(f"face is only {int(h)} px tall; need at least 90, ideally 250+")
    elif h < 200:
        res.warnings.append(f"face is {int(h)} px tall; result will be soft (ideal 250+)")
    # sharpness on the face, measured at a fixed face size so it is comparable across photos
    crop = _crop_pad(img, int(x), int(y), int(w), int(h))
    g = cv2.cvtColor(cv2.resize(crop, (200, int(200 * h / max(w, 1)))), cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    res.stats["sharpness"] = round(sharp, 1)
    if sharp < 15:
        res.warnings.append("face looks blurry")
    # profile check: eye distance relative to face width
    eye_d = float(np.linalg.norm(face.eyes[0] - face.eyes[1]))
    res.stats["eye_ratio"] = round(eye_d / max(w, 1), 3)
    if eye_d / max(w, 1) < 0.25:
        res.warnings.append("strong side view; eyes may read poorly")
    lum = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    res.stats["face_mean"] = round(float(lum.mean()), 1)
    if lum.mean() < 40:
        res.warnings.append("face is very dark; shadows will merge")
    res.ok = not res.problems
    return res


def _gate_object(img: np.ndarray) -> GateResult:
    res = GateResult(True, None, kind="object")
    H, W = img.shape[:2]
    res.stats.update(image_px=[W, H])
    if min(H, W) < 200:
        res.problems.append(f"photo is only {W}x{H} px; need at least 200 on the short side")
    elif min(H, W) < 500:
        res.warnings.append(f"photo is {W}x{H} px; result will be soft (ideal 800+)")
    s = 800 / max(H, W)
    g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    res.stats["sharpness"] = round(sharp, 1)
    if sharp < 15:
        res.warnings.append("photo looks blurry")
    res.ok = not res.problems
    return res


def _crop_pad(img, x, y, w, h, value=0):
    """Crop that may extend past the image; outside area is filled."""
    H, W = img.shape[:2]
    out = np.full((h, w) + img.shape[2:], value, img.dtype)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 > x0 and y1 > y0:
        out[y0 - y:y1 - y, x0 - x:x1 - x] = img[y0:y1, x0:x1]
    return out


# --------------------------------------------------------------------------- prep
@dataclass
class Prepared:
    rgb: np.ndarray          # BGR uint8, head-and-shoulders crop at working size
    lum: np.ndarray          # float32 0..1, luminance (L of Lab)
    alpha: np.ndarray        # float32 0..1, person mask
    face_mask: np.ndarray    # bool, ellipse inside the face box
    skin_mask: np.ndarray    # bool, smooth skin connected to the face: face, bald scalp, ears, neck
    eye_pts: np.ndarray      # 2x2 eye centres at working size
    face_h: float            # face box height at working size; for objects the scale unit
    gate: GateResult
    kind: str = "face"       # "face" or "object"


_SESSIONS: dict = {}
SEG_FACE = os.environ.get("SPEAKERSTYLE_SEG", "birefnet-portrait")
SEG_OBJECT = os.environ.get("SPEAKERSTYLE_SEG_OBJECT", "birefnet-general")


def _segment(bgr: np.ndarray, model: str = SEG_FACE) -> np.ndarray:
    from rembg import new_session, remove
    from PIL import Image

    if model not in _SESSIONS:
        _SESSIONS[model] = new_session(model)
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    m = remove(pil, session=_SESSIONS[model], only_mask=True)
    return np.asarray(m, np.float32) / 255.0


def prepare(img: np.ndarray, g: GateResult) -> Prepared:
    return _prepare_face(img, g) if g.face is not None else _prepare_object(img, g)


def _main_parts(mask: np.ndarray, keep: float = 0.15) -> np.ndarray:
    """The subject plus anything at least `keep` of its size (a set of items stays together);
    crumbs of segmentation noise go."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    area = st[1:, cv2.CC_STAT_AREA]
    ok = np.flatnonzero(area >= keep * area.max()) + 1
    return np.isin(lab, ok)


def _prepare_object(img: np.ndarray, g: GateResult) -> Prepared:
    H, W = img.shape[:2]
    # 1. coarse cut-out on a small copy: where is the subject?
    s0 = min(1.0, 1024 / max(H, W))
    small = cv2.resize(img, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA) if s0 < 1 else img
    coarse = _main_parts(_segment(small, SEG_OBJECT) > 0.5)
    share = float(coarse.mean())
    g.stats["subject_share"] = round(share, 3)
    if share < 0.01:
        raise NoSubject("no clear subject found (cut-out is empty or tiny)")
    if share > 0.92:
        raise NoSubject("the cut-out takes the whole photo: no separate subject to cut out")
    edge = np.concatenate([coarse[0], coarse[-1], coarse[:, 0], coarse[:, -1]]).mean()
    if edge > 0.05:
        g.warnings.append("subject is cut off by the photo edge")
    ys, xs = np.where(coarse)
    bx0, by0, bx1, by1 = xs.min() / s0, ys.min() / s0, (xs.max() + 1) / s0, (ys.max() + 1) / s0
    long = max(bx1 - bx0, by1 - by0)
    # 2. frame: the bounding box plus a margin, at a fixed size for the subject's longer side
    m = 0.08 * long
    X, Y = int(bx0 - m), int(by0 - m)
    Wc, Hc = int(bx1 - bx0 + 2 * m), int(by1 - by0 + 2 * m)
    crop = _crop_pad(img, X, Y, Wc, Hc)
    valid = _crop_pad(np.full(img.shape[:2], 255, np.uint8), X, Y, Wc, Hc)
    unit = float(WORK_FACE_H)
    s = unit * OBJ_UNIT_PER_LONG / long
    size = (int(round(Wc * s)), int(round(Hc * s)))
    rgb = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4)
    valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST) > 0
    # 3. fine cut-out at working size, limited to where the coarse pass found the subject
    near = _crop_pad(cv2.resize(coarse.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST), X, Y, Wc, Hc)
    near = cv2.resize(near, size, interpolation=cv2.INTER_NEAREST)
    near = cv2.dilate(near, np.ones((int(unit * 0.08) | 1,) * 2, np.uint8)) > 0
    alpha = _segment(rgb, SEG_OBJECT) * valid * near
    alpha *= cv2.dilate(_main_parts(alpha > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    lum = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255.0
    lum = _decontaminate(lum, alpha, unit)
    none = np.zeros(alpha.shape, bool)
    return Prepared(rgb, lum, alpha, none, none.copy(), np.zeros((0, 2)), unit, g, kind="object")


def _prepare_face(img: np.ndarray, g: GateResult) -> Prepared:
    x, y, w, h = g.face.box
    cx = x + w / 2
    # head-and-shoulders frame: room for hair above, shoulders below
    top, bottom = y - 0.85 * h, y + 2.0 * h
    half_w = 1.25 * h
    X, Y, Wc, Hc = int(cx - half_w), int(top), int(2 * half_w), int(bottom - top)
    crop = _crop_pad(img, X, Y, Wc, Hc)
    valid = _crop_pad(np.full(img.shape[:2], 255, np.uint8), X, Y, Wc, Hc)

    s = WORK_FACE_H / h
    size = (int(round(Wc * s)), int(round(Hc * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
    rgb = cv2.resize(crop, size, interpolation=interp)
    valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST) > 0

    alpha = _segment(rgb) * valid
    lum = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255.0
    lum = _decontaminate(lum, alpha, WORK_FACE_H)

    fx, fy, fw, fh = (x - X) * s, (y - Y) * s, w * s, h * s
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    face_mask = (((xx - (fx + fw / 2)) / (fw * 0.42)) ** 2 + ((yy - (fy + fh * 0.55)) / (fh * 0.5)) ** 2) <= 1
    face_mask &= alpha > 0.5
    eyes = (g.face.eyes - np.array([X, Y])) * s
    skin_mask = _skin_mask(rgb, alpha, face_mask, fh)
    return Prepared(rgb, lum, alpha, face_mask, skin_mask, eyes, fh, g)


def _skin_mask(bgr: np.ndarray, alpha: np.ndarray, face_mask: np.ndarray, face_h: float) -> np.ndarray:
    """Skin outside the face ellipse (a bald or shaved scalp, ears, neck) would otherwise be treated
    like hair: strand strokes and no smoothing, which reads as a dirty, mottled head. Skin = the face's
    own chroma AND smooth texture AND connected to the face; textured hair of a similar colour fails
    the texture test, so blond strands keep their engraving."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    ref = face_mask & (L > np.percentile(L[face_mask], 35))          # lit skin, not beard or brows
    if ref.sum() < 50:
        return face_mask.copy()
    ma, mb = np.median(a[ref]), np.median(b[ref])
    da = np.sqrt((a - ma) ** 2 + (b - mb) ** 2)
    tol = max(6.0, 2.5 * np.percentile(np.sqrt((a[ref] - ma) ** 2 + (b[ref] - mb) ** 2), 75))
    sig = face_h * 0.012
    hp = L - cv2.GaussianBlur(L, (0, 0), sig)
    tex = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), face_h * 0.03))
    tex_lim = 2.0 * np.percentile(tex[ref], 60)
    cand = (da < tol) & (tex < tex_lim) & (alpha > 0.5) & (L > np.percentile(L[ref], 5) * 0.8)
    k = max(3, int(face_h * 0.02) | 1)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)
    n, lab_cc = cv2.connectedComponents((cand | face_mask).astype(np.uint8))
    keep = np.unique(lab_cc[face_mask])
    return (np.isin(lab_cc, keep[keep > 0]) & (alpha > 0.5)) | face_mask


def _decontaminate(lum: np.ndarray, alpha: np.ndarray, face_h: float) -> np.ndarray:
    """Edge pixels of the cut-out still carry background colour (a light wall turns into a
    white halo around hair). Replace the edge band with luminance spread from inside."""
    k = max(3, int(face_h * 0.012)) | 1
    core = cv2.erode((alpha > 0.5).astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)
    sig = face_h * 0.01
    num = cv2.GaussianBlur(lum * core, (0, 0), sig)
    den = cv2.GaussianBlur(core, (0, 0), sig)
    fill = np.where(den > 1e-3, num / np.maximum(den, 1e-3), lum)
    band = (alpha > 0.02) & (core < 0.5)
    out = lum.copy()
    out[band] = fill[band]
    return out


# --------------------------------------------------------------------------- render
ABS_INK, ABS_PAPER = 0.19, 0.66  # threshold / white point on typical well-lit samples
SS = 3  # supersampling factor for smooth zone edges and round dots


def tone_map(p: Prepared, prm: Params) -> tuple[np.ndarray, dict]:
    """Returns the smoothed luminance used for thresholds plus the two thresholds."""
    L = p.lum.copy()
    # local contrast so hair strands and features survive the flattening
    if prm.detail > 0:
        clahe = cv2.createCLAHE(clipLimit=1.0 + 1.5 * prm.detail, tileGridSize=(6, 6))
        Lc = clahe.apply((L * 255).astype(np.uint8)).astype(np.float32) / 255
        L = np.clip(L + (Lc - L) * min(prm.detail, 1.5), 0, 1)
    if prm.pop > 0:
        base = cv2.GaussianBlur(L, (0, 0), p.face_h * 0.12)
        L = L + prm.pop * (L - base)
    Ld = L
    # edge-preserving smoothing: gives the flat, comic-like shapes
    if prm.smooth > 0:
        sp = p.face_h * 0.012 * prm.smooth
        for _ in range(2):
            L = cv2.bilateralFilter(L, 0, 0.08 + 0.04 * prm.smooth, sp)
    L = cv2.GaussianBlur(L, (0, 0), 0.6 + 0.4 * prm.smooth)
    # the face gets extra flattening, everything else gets fine strand detail (hair, fabric folds):
    # clean skin + engraved hair is what the references do
    wface = cv2.GaussianBlur(p.skin_mask.astype(np.float32), (0, 0), p.face_h * 0.04)
    if prm.skin_smooth > 0:
        sk = Ld
        for _ in range(2):
            sk = cv2.bilateralFilter(sk, 0, 0.10 + 0.05 * prm.skin_smooth, p.face_h * 0.02 * prm.skin_smooth)
        sk = cv2.GaussianBlur(sk, (0, 0), 1.0 + prm.skin_smooth)
        w = wface * min(prm.skin_smooth, 1.0)
        L = L * (1 - w) + sk * w
    wp = float(np.percentile(L[p.alpha > 0.5], 99.5))  # white point before strand detail inflates it
    if prm.fine > 0:
        hp = Ld - cv2.GaussianBlur(Ld, (0, 0), p.face_h * 0.008)
        L = L + 1.6 * prm.fine * hp * (1 - wface)
    vals = L[p.face_mask]
    if vals.size < 100:
        vals = L[p.alpha > 0.5]
    t_ink = float(np.quantile(vals, prm.ink))
    t_paper = float(np.quantile(vals, 1 - prm.paper))
    # Pure quantiles would render every skin tone identically (a dark-skinned speaker comes out
    # as light as anyone else). Blend with thresholds anchored to the photo's own white point:
    # that still fixes exposure but keeps real tonal differences. ABS_* calibrated on samples.
    if prm.tone_keep > 0:
        # keep the user's ink/paper intent: shift the anchors by how far the targets moved from default
        a_ink = wp * (ABS_INK + (prm.ink - 0.12) * 1.2)
        a_paper = wp * (ABS_PAPER - (prm.paper - 0.45) * 0.8)
        k = prm.tone_keep
        t_ink = (1 - k) * t_ink + k * a_ink
        t_paper = (1 - k) * t_paper + k * a_paper
    return L, {"t_ink": t_ink, "t_paper": max(t_paper, t_ink + 1e-3)}


def _ink_lines(p: Prepared, prm: Params) -> np.ndarray:
    """Difference-of-Gaussians dark lines (eye contours, brows, lips). 1 = ink."""
    s = p.face_h / 420
    g1 = cv2.GaussianBlur(p.lum, (0, 0), 1.2 * s)
    g2 = cv2.GaussianBlur(p.lum, (0, 0), 3.0 * s)
    dog = g2 - g1  # positive where locally darker (thin dark features)
    thr = 0.035 / max(prm.ink_lines, 1e-3)
    return np.clip((dog - thr) / 0.02, 0, 1)


def _eye_disks(p: Prepared, shape, scale: float = 1.0) -> list[np.ndarray]:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    rx, ry = 0.16 * p.face_h * scale, 0.09 * p.face_h * scale
    return [(((xx - ex) / rx) ** 2 + ((yy - ey) / ry) ** 2) <= 1 for ex, ey in p.eye_pts]


def _rescue_eyes(zone: np.ndarray, p: Prepared) -> np.ndarray:
    """Deep sockets and dark skin make the eye and brow merge into one black blob. Inside an eye
    region that is mostly ink, re-threshold on local contrast so the eye always has a dark part
    (lashes, pupil) and a lighter part (white of the eye, lid)."""
    out = zone.copy()
    for disk in _eye_disks(p, zone.shape):
        if not disk.any() or (zone[disk] == 0).mean() < 0.40:
            continue
        ys, xs = np.where(disk)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        patch = (p.lum[y0:y1, x0:x1] * 255).astype(np.uint8)
        patch = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2)).apply(patch).astype(np.float32)
        patch = cv2.GaussianBlur(patch, (0, 0), max(1.0, p.face_h * 0.004))
        d = disk[y0:y1, x0:x1]
        t_ink, t_paper = np.quantile(patch[d], [0.40, 0.88])
        local = np.where(patch < t_ink, 0, np.where(patch < t_paper, 1, 2)).astype(np.uint8)
        region = out[y0:y1, x0:x1]
        region[d] = local[d]
    return out


def _despeckle(zone: np.ndarray, prm: Params, p: Prepared) -> np.ndarray:
    """Merge tiny islands of one level into their surroundings. On the skin the size limit is much
    larger (clean face), elsewhere small so hair strands and fabric lines survive. Eyes are spared."""
    if prm.despeckle <= 0:
        return zone
    face_h = p.face_h
    min_area = int((face_h * 0.012 * prm.despeckle) ** 2)
    min_skin = int((face_h * 0.06 * prm.despeckle) ** 2)
    skin = p.skin_mask.copy()
    for disk in _eye_disks(p, zone.shape, 1.1):
        skin &= ~disk
    out = zone.copy()
    for level in (0, 2, 1):
        m = (out == level).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        area = st[:, cv2.CC_STAT_AREA]
        on_skin = np.bincount(lab[skin], minlength=n) > 0.6 * area
        kill = (area < min_area) | (on_skin & (area < min_skin))
        kill[0] = False
        small = kill[lab] & (m > 0)
        if small.any():
            med = cv2.medianBlur(out.astype(np.uint8), 7)
            out[small] = med[small]
    return out


def _zones_classic(p: Prepared, prm: Params) -> tuple[np.ndarray, dict]:
    L, th = tone_map(p, prm)
    zone = np.where(L < th["t_ink"], 0, np.where(L < th["t_paper"], 1, 2)).astype(np.uint8)
    if prm.ink_lines > 0:
        lines = _ink_lines(p, prm) > 0.5
        zone[lines & (zone == 1)] = 0
        zone[lines & (zone == 2) & (_ink_lines(p, prm) > 0.9)] = 1
    return _rescue_eyes(zone, p), th


# Brush-stroke renderer: the filter stack of the designer's Photoshop template, rebuilt here.
# Each level gets its own copy of the photo: Shadows/Highlights -> noise reduction -> Oil Paint ->
# Threshold. The grey level's Oil Paint is about 3x stronger than the black level's, so grey sits in
# big calm patches while black keeps detail. A third copy (High Pass -> Threshold) draws thin lines
# over the whole subject. All sizes below are in face heights, measured on the template.
PAINT_LIFT_R, PAINT_LIFT_W, PAINT_LIFT_GAIN = 0.057, 0.65, 3.0   # Shadows/Highlights: radius, tonal width
PAINT_FLOW_SIGMA = 0.010         # how far stroke direction is averaged
PAINT_STROKE = 0.0035            # stroke half-length per unit of Oil Paint "stylization"
PAINT_STYLE_INK, PAINT_STYLE_GREY = 2.1, 7.1
PAINT_INK, PAINT_PAPER = 0.610, 0.796  # thresholds as a share of the white point (template: 138, 180)
PAINT_Q_INK, PAINT_Q_PAPER = 0.29, 0.45  # the shares those thresholds give on the template photo
PAINT_HP_R, PAINT_LINE = 0.0055, 0.0265  # High Pass radius; line = this much darker than surroundings
PAINT_IGNORES = ("detail", "fine", "skin_smooth")  # sliders the brush renderer does not use


def _shadows_lift(L: np.ndarray, face_h: float, amount: float) -> np.ndarray:
    """Like Photoshop's Shadows/Highlights (shadows only): brightens pixels whose neighbourhood is dark."""
    M = cv2.GaussianBlur(L, (0, 0), PAINT_LIFT_R * face_h)
    w = np.clip(1 - M / PAINT_LIFT_W, 0, 1)
    w = w * w * (3 - 2 * w)
    return 1 - (1 - L) ** (1 + PAINT_LIFT_GAIN * amount * w)


def _stroke_field(L: np.ndarray, face_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors along edges (perpendicular to the smoothed gradient): the direction of brush strokes."""
    s = PAINT_FLOW_SIGMA * face_h
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    j11 = cv2.GaussianBlur(gx * gx, (0, 0), s)
    j12 = cv2.GaussianBlur(gx * gy, (0, 0), s)
    j22 = cv2.GaussianBlur(gy * gy, (0, 0), s)
    th = 0.5 * np.arctan2(2 * j12, j11 - j22) + np.pi / 2
    return np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)


def _strokes(L: np.ndarray, tx: np.ndarray, ty: np.ndarray, half: float) -> np.ndarray:
    """Averages each pixel along its stroke line (line integral convolution): the Oil Paint look,
    where hair and fabric turn into streaks instead of blobs."""
    n = int(round(half))
    if n < 1:
        return L
    H, W = L.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sig = max(half / 2, 0.5)
    acc, wsum = L.copy(), 1.0
    for sign in (1, -1):
        px, py, vx, vy = xx.copy(), yy.copy(), tx * sign, ty * sign
        for k in range(1, n + 1):
            px += vx
            py += vy
            nx = cv2.remap(tx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            ny = cv2.remap(ty, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            flip = np.where(nx * vx + ny * vy < 0, -1.0, 1.0).astype(np.float32)  # the field has no sign
            vx, vy = nx * flip, ny * flip
            wk = float(np.exp(-k * k / (2 * sig * sig)))
            acc += wk * cv2.remap(L, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wsum += wk
    return acc / wsum


def _zones_paint(p: Prepared, prm: Params) -> tuple[np.ndarray, dict]:
    """Slider meaning here: pop = shadow lift (0.5 = the template's 100%), smooth = stroke length,
    ink_lines = line sensitivity. PAINT_IGNORES have no effect."""
    fh = p.face_h
    inside = p.alpha > 0.5
    L = _shadows_lift(p.lum.astype(np.float32), fh, 2 * prm.pop)
    L = cv2.bilateralFilter(L, 0, 0.06, fh * 0.006)  # the template's Camera Raw noise reduction
    wp = float(np.percentile(L[inside], 99.5)) if inside.any() else 1.0
    tx, ty = _stroke_field(L, fh)
    length = PAINT_STROKE * fh * (0.5 + 2 * prm.smooth)
    Lk = _strokes(L, tx, ty, PAINT_STYLE_INK * length)
    Lg = _strokes(L, tx, ty, PAINT_STYLE_GREY * length)
    t_ink = wp * (PAINT_INK + (prm.ink - 0.14) * 1.2)
    t_paper = wp * (PAINT_PAPER - (prm.paper - 0.43) * 0.8)
    if prm.tone_keep < 1:  # as in tone_map: blend with thresholds that hit target shares on the face
        sel = p.face_mask if p.face_mask.sum() >= 100 else inside
        q_ink = float(np.quantile(Lk[sel], np.clip(PAINT_Q_INK + prm.ink - 0.14, 0.01, 0.9)))
        q_paper = float(np.quantile(Lg[sel], np.clip(1 - PAINT_Q_PAPER - (prm.paper - 0.43), 0.05, 0.99)))
        k = prm.tone_keep
        t_ink, t_paper = k * t_ink + (1 - k) * q_ink, k * t_paper + (1 - k) * q_paper
    zone = np.where(Lk < t_ink, 0, np.where(Lg < t_paper, 1, 2)).astype(np.uint8)
    if prm.ink_lines > 0:
        hp = L - cv2.GaussianBlur(L, (0, 0), PAINT_HP_R * fh)
        zone[hp < -wp * PAINT_LINE * 1.3 / prm.ink_lines] = 0
    return zone, {"t_ink": t_ink, "t_paper": t_paper}


def render(p: Prepared, prm: Params) -> tuple[np.ndarray, dict]:
    """Returns tone image (float32, 0 = ink, 1 = paper) at working size, plus info."""
    prm = prm.clamp()
    zone, th = (_zones_paint if prm.paint else _zones_classic)(p, prm)
    zone = _despeckle(zone, prm, p)

    # supersample: smooth zone boundaries with a blurred per-level indicator, then screen dots
    H, W = zone.shape
    Hs, Ws = H * SS, W * SS
    ind = [cv2.resize((zone == k).astype(np.float32), (Ws, Hs), interpolation=cv2.INTER_CUBIC) for k in range(3)]
    ind = [cv2.GaussianBlur(i, (0, 0), SS * 0.7) for i in ind]
    zs = np.argmax(np.stack(ind), axis=0)

    cell = p.face_h * prm.dot_cell * SS
    a = np.radians(prm.dot_angle)
    yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
    u = (xx * np.cos(a) + yy * np.sin(a)) / cell
    v = (-xx * np.sin(a) + yy * np.cos(a)) / cell
    # round-dot spot function: distance to nearest cell centre, normalised so area matches coverage
    du, dv = u - np.round(u), v - np.round(v)
    r = np.sqrt(du * du + dv * dv)
    r_cov = np.sqrt(prm.grey / np.pi)  # radius of a dot covering `grey` of a unit cell
    dots = (r < r_cov).astype(np.float32)   # 1 = ink
    if prm.grey > 0.6:  # above ~0.6 round dots touch; switch to white dots between cell centres
        du2, dv2 = u - np.floor(u) - 0.5, v - np.floor(v) - 0.5
        dots = 1 - (np.sqrt(du2 ** 2 + dv2 ** 2) < np.sqrt((1 - prm.grey) / np.pi)).astype(np.float32)

    solid = (zs == 0).astype(np.float32)
    if prm.outline > 0:  # the template's 1 px inner stroke, drawn at the supersampled size
        sil = cv2.resize((p.alpha > 0.5).astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST)
        r = max(1, int(round(prm.outline * p.face_h / 527 * SS)))
        rim = sil - cv2.erode(sil, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        solid = np.maximum(solid, rim.astype(np.float32))
    dotted = np.where(zs == 1, dots, 0.0).astype(np.float32) * (1 - solid)
    down = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    ink_solid, ink_dots = down(solid), down(dotted)
    tone = 1 - (ink_solid + ink_dots)

    # ink_solid / ink_dots: coverage of the solid ink plate and of the halftone dots, for layered export
    info = {**{k: round(v, 4) for k, v in th.items()}, "zone": zone, "ink_solid": ink_solid, "ink_dots": ink_dots}
    return tone, info


def finish_alpha(p: Prepared) -> np.ndarray:
    """Hard silhouette edge (the reference has no soft halo), lightly anti-aliased."""
    a = cv2.GaussianBlur(p.alpha, (0, 0), 1.0)
    return np.clip((a - 0.5) * 6 + 0.5, 0, 1)


def to_rgba(tone: np.ndarray, alpha: np.ndarray, ink=(0, 0, 0), paper=(255, 255, 255)) -> np.ndarray:
    ink = np.array(ink, np.float32)
    paper = np.array(paper, np.float32)
    rgb = ink[None, None] + (paper - ink)[None, None] * tone[..., None]
    out = np.dstack([rgb, alpha[..., None] * 255]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)


def split_layers(p: Prepared, info: dict, ink=(0, 0, 0), paper=(255, 255, 255)) -> list[tuple[str, np.ndarray]]:
    """Three editable layers (BGRA, bottom to top) whose stack equals to_rgba(): paper fills the
    silhouette, then the halftone dots, then the solid ink plate."""
    a = finish_alpha(p)
    s = np.clip(info["ink_solid"], 0, 1)
    d = np.clip(info["ink_dots"], 0, 1 - s)
    # Source-over stacking must give alpha a and ink share (s + d), as in to_rgba(). Solving the
    # "over" equations top-down gives these layer alphas (edges and dot rims are partial coverage).
    a_ink = a * s
    a_dots = np.where(1 - a_ink > 1e-6, a * d / np.maximum(1 - a_ink, 1e-6), 0)
    a_paper = np.where(1 - a * (s + d) > 1e-6, a * (1 - s - d) / np.maximum(1 - a * (s + d), 1e-6), 0)

    def layer(colour, alpha):
        rgb = np.empty(a.shape + (3,), np.uint8)
        rgb[:] = colour
        return cv2.cvtColor(np.dstack([rgb, (np.clip(alpha, 0, 1) * 255).round().astype(np.uint8)]), cv2.COLOR_RGBA2BGRA)
    return [("Paper", layer(paper, a_paper)), ("Halftone", layer(ink, a_dots)), ("Ink", layer(ink, a_ink))]


def composite(rgba_bgra: np.ndarray, bg_hex: str) -> np.ndarray:
    bg = hex_to_bgr(bg_hex)
    a = rgba_bgra[..., 3:4].astype(np.float32) / 255
    base = np.empty_like(rgba_bgra[..., :3], dtype=np.float32)
    base[:] = bg
    return (base * (1 - a) + rgba_bgra[..., :3].astype(np.float32) * a).astype(np.uint8)


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    b, g, r = hex_to_bgr(h)
    return (r, g, b)


# --------------------------------------------------------------------------- metrics
def metrics(p: Prepared, info: dict, prm: Params) -> dict:
    """Cheap checks that catch most failures without a model."""
    zone = info["zone"]
    if p.kind == "object":
        return _metrics_object(p, zone)
    fm = p.face_mask
    m = {
        "face_ink": float((zone[fm] == 0).mean()) if fm.any() else None,
        "face_grey": float((zone[fm] == 1).mean()) if fm.any() else None,
        "face_paper": float((zone[fm] == 2).mean()) if fm.any() else None,
    }
    flags = []
    r = 0.11 * p.face_h
    H, W = zone.shape
    yy, xx = np.mgrid[0:H, 0:W]
    for name, (ex, ey) in zip(("right_eye", "left_eye"), p.eye_pts):
        disk = (xx - ex) ** 2 + (yy - ey) ** 2 <= r * r
        z = zone[disk]
        ink, paper = float((z == 0).mean()), float((z == 2).mean())
        m[f"{name}_ink"] = round(ink, 3)
        # an eye reads when it has dark AND lighter parts; all-black or all-white = lost
        if ink > 0.85:
            flags.append(f"{name} lost in shadow")
        if ink < 0.04:
            flags.append(f"{name} washed out (no dark pupil/lash line)")
    # specks: many tiny components = noisy texture
    # only the skin counts: strands in hair and fabric are wanted texture
    n = sum(cv2.connectedComponents(((zone == k) & fm).astype(np.uint8), connectivity=4)[0] - 1 for k in range(3))
    m["components_per_face"] = round(n / max(fm.sum() / 1e4, 1), 1)
    if m["components_per_face"] > 25:
        flags.append("noisy texture (many small specks)")
    # background leak: opaque pixels touching top/side borders far from the head
    a = p.alpha > 0.5
    border = np.concatenate([a[0, :], a[:, 0], a[:, -1]])
    m["border_opaque"] = round(float(border.mean()), 3)
    if m["border_opaque"] > 0.25:
        flags.append("cut-out touches the frame edges (background left or tight crop)")
    m["flags"] = flags
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}


def _metrics_object(p: Prepared, zone: np.ndarray) -> dict:
    sub = p.alpha > 0.5
    z = zone[sub]
    m = {f"subject_{n}": round(float((z == k).mean()), 3) for k, n in enumerate(("ink", "grey", "paper"))}
    flags = []
    # a level that (almost) vanishes or swallows the subject = flat, unreadable shape
    if m["subject_ink"] > 0.6:
        flags.append("subject mostly solid black (shape will not read)")
    if m["subject_paper"] > 0.75:
        flags.append("subject mostly white (washed out)")
    if m["subject_ink"] < 0.02:
        flags.append("no solid black on the subject (looks flat)")
    n = sum(cv2.connectedComponents(((zone == k) & sub).astype(np.uint8), connectivity=4)[0] - 1 for k in range(3))
    m["components_per_area"] = round(n / max(sub.sum() / 1e4, 1), 1)
    if m["components_per_area"] > 130:  # objects keep texture (stone, foil): samples/objects sit at 40-120
        flags.append("noisy texture (many small specks)")
    # the frame has a margin all round: an opaque border means the photo itself cut the subject
    border = np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]])
    m["border_opaque"] = round(float(border.mean()), 3)
    if m["border_opaque"] > 0.03:
        flags.append("cut-out touches the frame edges (subject cut off by the photo)")
    m["flags"] = flags
    return m


# --------------------------------------------------------------------------- io helpers
def load(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- layers (experimental)
def export_layers(p: Prepared, prm: Params, outdir: Path) -> dict:
    """Separate layers for animation: the page animates each level on its own, so black and white
    stay put (the likeness holds) while only the halftone moves."""
    outdir.mkdir(parents=True, exist_ok=True)
    prm = prm.clamp()
    tone, info = render(p, prm)
    zone = info["zone"]
    alpha = finish_alpha(p)
    soft = [cv2.GaussianBlur((zone == k).astype(np.float32), (0, 0), 0.8) for k in range(3)]
    # zones.png: R = ink, G = grey, B = paper, A = silhouette (straight alpha)
    rgba = np.dstack([soft[0], soft[1], soft[2], alpha])
    cv2.imwrite(str(outdir / "zones.png"), cv2.cvtColor((rgba * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA))
    # ink.png: black ink areas on transparent (shape for LiquidMetal)
    ink_a = (soft[0] * alpha * 255).astype(np.uint8)
    cv2.imwrite(str(outdir / "ink.png"), np.dstack([np.zeros_like(ink_a)] * 3 + [ink_a]))
    # silhouette.png: solid silhouette on transparent (shape for Heatmap)
    sil = (alpha * 255).astype(np.uint8)
    cv2.imwrite(str(outdir / "silhouette.png"), np.dstack([np.zeros_like(sil)] * 3 + [sil]))
    cv2.imwrite(str(outdir / "portrait.png"), to_rgba(tone, alpha))
    H, W = zone.shape
    meta = {
        "width": W, "height": H,
        "cell_px": float(p.face_h * prm.dot_cell), "angle": float(prm.dot_angle), "grey": float(prm.grey),
        "face_h": float(p.face_h),
        # objects have no eyes: park them far outside so the eye-safe zones never trigger
        "eyes": [[float(x) / W, float(y) / H] for x, y in p.eye_pts] or [[-10.0, -10.0], [-10.0, -10.0]],
        "kind": p.kind,
    }
    save_json(outdir / "meta.json", meta)
    return meta
