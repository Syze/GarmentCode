"""Reader for AAMA/ASTM garment DXF exports (Lectra Modaris / Richpeace).

The hyperdrop_data DXFs are industry pattern exports, not CLO exports, so the
layout conventions differ from the CLO reader in `pants_clo.py`:

  * every pattern piece is its own BLOCK, referenced by an INSERT in ENTITIES
  * the standard ASTM D6673 layer numbering carries the semantics:
        1  piece boundary (CUT line, seam allowance included)
        2  turn points  (boundary vertices that are hard corners)
        3  curve points (boundary vertices the curve passes smoothly through)
        4  notches
        5  grade reference point
        6  mirror line
        7  grain line
        8  internal lines (fold lines, dart legs, topstitch)
        11 internal cutouts / drill holes
        13 piece annotation
        14 SEW line (net finished pattern) -- what we actually want to simulate
  * piece metadata lives in TEXT entities inside the block
        'Piece Name: ...', 'Size: ...', 'Quantity: ...', 'Category: ...',
        'Fabric:'/'Material:' , 'Annotation: ...'
    and is frequently GBK-encoded Chinese.
  * pieces are stored in nesting orientation; the grain line (layer 7) gives the
    upright direction.

`read_pieces()` returns everything normalised: centimetres, grain pointing +y,
boundary closed and counter-clockwise, with the low-resolution Lectra outlines
re-smoothed through their marked curve points.
"""
from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pygarment as pyg
from scipy.spatial.transform import Rotation as R

# ASTM D6673 layer numbers we care about
L_CUT, L_TURN, L_CURVE, L_NOTCH, L_GRADE = '1', '2', '3', '4', '5'
L_MIRROR, L_GRAIN, L_INTERNAL, L_DRILL, L_SEW = '6', '7', '8', '11', '14'

MM_TO_CM = 0.1

# Exporters disagree about drawing units and the file header does not settle it:
# the Lectra/Richpeace files carry no $INSUNITS at all, and the Browzwear
# `.rul` beside the bonprix DXFs declares "UNITS: ENGLISH" while both the
# geometry and its own grade deltas are plainly centimetres (the blouse front
# half-panel grades +1.12 per size, i.e. a 4.5 cm bust step -- the EU standard;
# read as inches it would be 11 cm per size).
#
# Piece extents settle it instead. No pattern piece of a garment is over ~250 cm
# in any direction, and none is under ~2 cm, so the largest boundary extent in a
# file falls in a different decade for each unit. Detected once per file from the
# biggest piece, never per piece -- a cuff alone is ambiguous.
CM_TO_CM = 1.0
_UNIT_CM_MAX = 300.0     # biggest plausible piece extent in cm


def _unit_scale(blocks):
    """Scale bringing a file's drawing units to centimetres."""
    big = 0.0
    for blk in blocks:
        for e in blk['entities']:
            if e['type'] != 'POLYLINE' or e['layer'] not in (L_CUT, L_SEW):
                continue
            pts = _pts(e)
            if len(pts) >= 3:
                big = max(big, float(pts[:, 0].ptp()), float(pts[:, 1].ptp()))
    return MM_TO_CM if big > _UNIT_CM_MAX else CM_TO_CM


# --------------------------------------------------------------------------- #
#  Raw DXF parsing
# --------------------------------------------------------------------------- #
def _decode(path):
    raw = open(path, 'rb').read()
    for enc in ('utf-8', 'gbk', 'cp1252'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('latin-1')


def _pairs(path):
    """Yield (group_code, value) pairs of the DXF."""
    lines = _decode(path).replace('\r\n', '\n').replace('\r', '\n').split('\n')
    for i in range(0, len(lines) - 1, 2):
        code = lines[i].strip()
        if not code:
            continue
        try:
            code = int(code)
        except ValueError:
            continue
        yield code, lines[i + 1].strip()


_GEOM = ('POLYLINE', 'LINE', 'POINT', 'TEXT', 'ARC', 'CIRCLE', 'LWPOLYLINE', 'VERTEX')


def _raw_blocks(path):
    """Parse the BLOCKS section into [{'name', 'entities': [...]}].

    Entities are dicts (type, layer, pts, text). POLYLINE swallows the VERTEX
    entities that follow it, and LINE picks up its second endpoint (codes
    11/21) as a second point.
    """
    out, cur, ent = [], None, None

    def flush():
        nonlocal ent
        if ent is not None and cur is not None:
            if ent['type'] == 'VERTEX' and cur['entities'] \
                    and cur['entities'][-1]['type'] == 'POLYLINE':
                cur['entities'][-1]['pts'].extend(ent['pts'])
            else:
                cur['entities'].append(ent)
        ent = None

    for code, val in _pairs(path):
        if code == 0:
            flush()
            if val == 'BLOCK':
                cur = dict(name=None, entities=[])
            elif val == 'ENDBLK':
                if cur is not None:
                    out.append(cur)
                cur = None
            elif cur is not None and val in _GEOM:
                ent = dict(type=val, layer='?', pts=[], text=None)
        elif cur is None:
            continue
        elif ent is not None:
            if code == 8:
                ent['layer'] = val
            elif code in (10, 11):
                ent['pts'].append([float(val), None])
            elif code in (20, 21) and ent['pts']:
                ent['pts'][-1][1] = float(val)
            elif code == 1:
                ent['text'] = val
        elif code == 2 and cur['name'] is None:
            cur['name'] = val
    return out


def _pts(ent):
    return np.array([p for p in ent['pts'] if p[1] is not None], float)


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def _dedup(pts, tol=1e-4):
    out = [np.asarray(pts[0], float)]
    for p in np.asarray(pts, float)[1:]:
        if np.hypot(*(p - out[-1])) > tol:
            out.append(p)
    return np.array(out)


# --------------------------------------------------------------------------- #
#  Seam fitting and weld direction
#
#  These four came from `pants_clo.py`, which is a GARMENT program -- so every
#  module here was importing from a pair of CLO3D trousers to get a curve fitter
#  and a weld-direction test, and `aama_dxf` imported from `pants_clo` while
#  `hyperdrop` imported from both. They are generic, so they live at the bottom
#  of the stack now and the cycle is gone.
#
#  `SEAM_DEDUP` is explicit because it had to be: the fitters called a bare
#  `_dedup(pts)` and `pants_clo`'s default was 0.05 while the one already in
#  this module is 1e-4. Moving them without pinning it would have silently
#  refitted every seam in every garment.
# --------------------------------------------------------------------------- #
SEAM_DEDUP = 0.05


def _fit_cubic(pts):
    """Least-squares cubic Bezier control points (P1,P2) for a polyline with
    fixed endpoints. Chord-length parameterised."""
    pts = _dedup(pts, SEAM_DEDUP)
    P0, P3 = pts[0], pts[-1]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    t = d / d[-1]
    a = 3 * (1 - t) ** 2 * t
    b = 3 * (1 - t) * t ** 2
    base = np.outer((1 - t) ** 3, P0) + np.outer(t ** 3, P3)
    rhs = pts - base
    A = np.column_stack([a, b])                      # (n,2)
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)    # (2,2): rows P1,P2
    return sol[0], sol[1]


def _cubic_dev(pts, P1, P2):
    """Max distance from the GT points to the fitted cubic Bezier."""
    P0, P3 = pts[0], pts[-1]
    t = np.linspace(0, 1, 60)[:, None]
    bez = ((1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1
           + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3)
    return max(np.min(np.linalg.norm(bez - p, axis=1)) for p in pts)


def _piecewise(pts, tol=0.25):
    """Adaptive piecewise-cubic fit: one cubic where the seam is smooth, split
    into more segments only where the GT curve bends sharply. Faithful to the
    GT (unlike one cubic for the whole seam) yet few edges (unlike the dense
    polyline that broke stitching)."""
    pts = _dedup(pts, SEAM_DEDUP)
    if len(pts) <= 2:
        return [(pts[0], pts[-1], None)]
    chord = pts[-1] - pts[0]; L = np.linalg.norm(chord)
    if L > 1e-9:
        nrm = np.array([-chord[1], chord[0]]) / L
        if np.abs((pts - pts[0]) @ nrm).max() < 0.15:    # straight
            return [(pts[0], pts[-1], None)]
    P1, P2 = _fit_cubic(pts)
    if len(pts) < 5 or _cubic_dev(pts, P1, P2) < tol:
        return [(pts[0], pts[-1], (P1, P2))]
    # split at the point of largest deviation from the chord
    nrm = np.array([-chord[1], chord[0]]) / max(L, 1e-9)
    k = int(np.argmax(np.abs((pts - pts[0]) @ nrm)))
    k = min(max(k, 1), len(pts) - 2)
    return _piecewise(pts[:k + 1], tol) + _piecewise(pts[k:], tol)


def _seq_from_points(pts, label='', single=False):
    """EdgeSequence following the seam.

    single=True  -> ONE cubic edge (used for seams stitched between two
                    differently-shaped panels: outseam/inseam front<->back and
                    waist<->waistband. Multiple segments there get subdivided to
                    different vertex counts on each side and weld in a staircase
                    / tangled seam).
    single=False -> adaptive piecewise cubic (faithful; used for the rise, which
                    only stitches to its mirror-identical other leg, so both
                    sides have identical structure and weld 1:1)."""
    pts = np.asarray(pts, float)
    if single:
        dd = _dedup(pts, SEAM_DEDUP)
        P0, P3 = dd[0], dd[-1]
        chord = P3 - P0; L = np.linalg.norm(chord)
        nrm = np.array([-chord[1], chord[0]]) / max(L, 1e-9)
        if np.abs((dd - P0) @ nrm).max() < 0.15:
            return pyg.EdgeSequence(pyg.Edge(list(P0), list(P3), label=label))
        P1, P2 = _fit_cubic(pts)
        return pyg.EdgeSequence(pyg.CurveEdge(
            list(P0), list(P3), control_points=[list(P1), list(P2)],
            relative=False, label=label))
    edges = []
    for P0, P3, cps in _piecewise(pts):
        if cps is None:
            edges.append(pyg.Edge(list(P0), list(P3), label=label))
        else:
            edges.append(pyg.CurveEdge(list(P0), list(P3),
                         control_points=[list(cps[0]), list(cps[1])],
                         relative=False, label=label))
    return pyg.EdgeSequence(*edges)

def _auto_rw(int_ref, int_other):
    """Set int_other.right_wrong per-edge so every paired edge welds its
    physically-coincident endpoints, decided by which pairing is closer in 3D:
      start<->start + end<->end closer -> right_wrong=True  (swap=False)
      end<->start (the default)  closer -> right_wrong=False (swap=True)
    Coincidence is used rather than the edge-direction dot because near the
    crotch tip the rise edges are nearly perpendicular, where the dot SIGN is
    unreliable but the endpoint distances are not. Handles the mirror seams
    (crotch) whose left/right segments are a mix after the mirror."""
    out = pyg.Interface.from_multiple(int_other)
    rw = []
    n = min(len(int_ref.edges), len(out.edges))
    for i in range(n):
        pr, po = int_ref.panel[i], out.panel[i]
        er, eo = int_ref.edges[i], out.edges[i]
        rS = np.array(pr.point_to_3D(list(er.start))); rE = np.array(pr.point_to_3D(list(er.end)))
        oS = np.array(po.point_to_3D(list(eo.start))); oE = np.array(po.point_to_3D(list(eo.end)))
        d_noswap = np.linalg.norm(rS - oS) + np.linalg.norm(rE - oE)   # start<->start
        d_swap = np.linalg.norm(rE - oS) + np.linalg.norm(rS - oE)     # end<->start
        rw.append(bool(d_noswap < d_swap))
    out.right_wrong = rw + [False] * (len(out.edges) - n)
    return out

FRONT_PIECE, BACK_PIECE, YOKE_PIECE, POCKET_PIECE = 140, 150, 141, 184

# Per-DXF profiles: each file has its own block-naming convention, boundary
# layer and piece identifiers. Resolved by the DXF file's basename so the
# default dxf_3 path is byte-identical to before.
DXF_PROFILES = {
    'dxf_3.dxf': dict(
        name=lambda p, s: f'61251-{p}_{s}', layer='1',
        front=140, back=150, yoke=141, pocket=184),
    'next.dxf': dict(
        name=lambda p, s: f'W13854-3 {p}_{s}', layer='1',
        front='Front leg', back='Back Leg', yoke='Yoke', pocket='Pocket Bearer',
        geom_yoke=True, merge_bearer=True, reflect_left=True,
        # Target init clearance gap (cm) between the two legs' fly/CB edges.
        # _apply_pose_x_rotation measures the actual front-crotch-hook overlap on
        # the target body+size and pushes each leg out adaptively to hit this gap,
        # so the deep DXF hooks never interpenetrate. 0/absent -> feature off.
        x_sep=2.0),
}


def _open_loop(pts, tol=1e-3):
    """Drop a duplicated closing vertex so the loop is stored open."""
    pts = _dedup(pts)
    while len(pts) > 3 and np.hypot(*(pts[0] - pts[-1])) < tol:
        pts = pts[:-1]
    return pts


def _signed_area(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _catmull_rom(P, n_per_seg=6):
    """Resample an open point chain with a centripetal Catmull-Rom spline.

    Used to restore the smooth curve Lectra only stored as its control (curve)
    points -- with 24 vertices for a whole pant leg, straight-line interpolation
    visibly flattens the crotch and hip curves.
    """
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    ext = np.vstack([P[0] + (P[0] - P[1]), P, P[-1] + (P[-1] - P[-2])])
    d = np.linalg.norm(np.diff(ext, axis=0), axis=1) ** 0.5   # centripetal
    d = np.maximum(d, 1e-9)
    t = np.r_[0.0, np.cumsum(d)]
    out = []
    for i in range(1, len(ext) - 2):
        t0, t1, t2, t3 = t[i - 1:i + 3]
        p0, p1, p2, p3 = ext[i - 1:i + 3]
        for s in np.linspace(t1, t2, n_per_seg, endpoint=False):
            a1 = (t1 - s) / (t1 - t0) * p0 + (s - t0) / (t1 - t0) * p1
            a2 = (t2 - s) / (t2 - t1) * p1 + (s - t1) / (t2 - t1) * p2
            a3 = (t3 - s) / (t3 - t2) * p2 + (s - t2) / (t3 - t2) * p3
            b1 = (t2 - s) / (t2 - t0) * a1 + (s - t0) / (t2 - t0) * a2
            b2 = (t3 - s) / (t3 - t1) * a2 + (s - t1) / (t3 - t1) * a3
            out.append((t2 - s) / (t2 - t1) * b1 + (s - t1) / (t2 - t1) * b2)
    out.append(P[-1])
    return _dedup(np.array(out))


def inset_loop(loop, sa, fold=None, mitre_limit=5.0, tol=0.05, darts=None):
    """Move a closed CCW boundary inward by `sa` cm: cut line -> sew line.

    A proper polygon offset with MITRE joins, not a vertex-wise one. Offsetting
    each vertex along its own miter looks equivalent and is not: any edge shorter
    than about twice the offset reverses direction, and these outlines average
    0.8 cm between vertices against a 1.0 cm allowance, so on a 289-vertex leg
    almost every edge flips. Healing those flips then chops the corners -- the
    jogger's four square corners came back as 46-65 deg and its straight runs
    bowed. A mitre buffer holds them to a tenth of a degree (90.0 -> 90.0,
    109.5 -> 109.8) and leaves straight runs straight.

    Vertex COUNT is not preserved, and no longer needs to be: junctions are
    located by role -- edge bands, extremes, turn angles -- rather than by index.

    `fold` is the layer-6 mirror line. A folded edge is not a seam and gets no
    allowance, so the piece is reflected about it, offset as the whole doubled
    panel (which makes that edge interior), and cut back to its own side. Without
    this every cut-on-the-fold panel loses `sa` at the centre, i.e. the garment
    loses 2*sa of girth that was never taken off it.
    """
    P = np.asarray(loop, float)
    if len(P) < 3 or not sa:
        return P
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        return P

    # DARTS ARE NOT OFFSET. A dart is interior to the piece: the cut line adds
    # allowance around the OUTSIDE only, so the dart legs a pattern draws are
    # already the sewing lines. Offsetting them anyway destroys the dart -- it is
    # a zero-width spike, and 1 cm inward collapses it (the 8672609700 front's
    # two spikes, 172 and 163 deg, come out as a single 108 deg corner that no
    # threshold can tell from an ordinary one).
    #
    # So each dart is lifted out, the remaining outline is offset on its own, and
    # the dart is put back between its two offset base points with its apex where
    # the pattern drew it. The legs come out slightly shorter, which is right:
    # the dart's base sits on the seam line, which has moved inward.
    if darts:
        spans, apexes = [], []
        for a, apex, b in sorted(darts, key=lambda d: d[0], reverse=True):
            spans.append((int(a), int(b)))
            apexes.append(P[int(apex)].copy())
        flat = P.copy()
        keep = np.ones(len(P), bool)
        for a, b in spans:
            lo, hi = (a, b) if a <= b else (b, a)
            keep[lo + 1:hi] = False
        flat = P[keep]
        base = inset_loop(flat, sa, fold=fold, mitre_limit=mitre_limit, tol=tol)
        out = list(base)
        for (a, b), apex in zip(spans, apexes):
            ia = int(np.argmin(np.linalg.norm(np.asarray(out) - P[a], axis=1)))
            ib = int(np.argmin(np.linalg.norm(np.asarray(out) - P[b], axis=1)))
            # insert on the short side, so the dart lands where it was drawn
            at = max(ia, ib) if abs(ia - ib) == 1 else min(ia, ib) + 1
            out.insert(at, apex)
        res = np.asarray(out, float)
        return res if _signed_area(res) > 0 else res[::-1]
    poly = Polygon(P)
    if not poly.is_valid:
        poly = poly.buffer(0)
    half = None
    if fold is not None and len(np.asarray(fold, float)) >= 2:
        f = np.asarray(fold, float)
        a, d = f[0], f[-1] - f[0]
        n = np.linalg.norm(d)
        if n > 1e-9:
            d = d / n
            perp = np.array([-d[1], d[0]])
            side = np.sign(np.mean((P - a) @ perp)) or 1.0
            # mirror the loop about the fold and offset the union
            m = P - 2.0 * (((P - a) @ perp))[:, None] * perp
            mirrored = Polygon(m)
            if not mirrored.is_valid:
                mirrored = mirrored.buffer(0)
            poly = unary_union([poly, mirrored])
            if poly.geom_type != 'Polygon':
                # The two halves only TOUCH along the fold, and shapely does not
                # always merge that -- 8642610003's front waistband came back a
                # MultiPolygon while the back, same shape mirrored, came back a
                # Polygon. Unmerged, `buffer(-sa)` insets each half's fold edge
                # and the piece loses `sa` at the centre after all, which is the
                # very thing folding it is meant to avoid (the band came out
                # 31.0 cm instead of 33.0). Close the hairline seam first; eps
                # is 8 orders below the allowance, so the outline is unchanged.
                eps = 1e-6
                poly = poly.buffer(eps, join_style=2).buffer(
                    -eps, join_style=2)
            span = 10.0 * max(P[:, 0].ptp(), P[:, 1].ptp())
            half = Polygon([a - d * span, a + d * span,
                            a + d * span + perp * side * span,
                            a - d * span + perp * side * span])
    off = poly.buffer(-float(sa), join_style=2, mitre_limit=mitre_limit)
    if half is not None and not off.is_empty:
        off = off.intersection(half)
    if off.is_empty:
        return P
    if off.geom_type != 'Polygon':
        off = max(off.geoms, key=lambda g: g.area)
    out = np.asarray(off.exterior.coords, float)
    if len(out) > 1 and np.allclose(out[0], out[-1]):
        out = out[:-1]
    return out if _signed_area(out) > 0 else out[::-1]


def _smooth_boundary(loop, curve_mask, n_per_seg=6, min_spacing_mm=3.0):
    """Re-smooth a closed boundary: runs of consecutive CURVE vertices become
    Catmull-Rom arcs (anchored on the bounding turn points), turn-to-turn
    stretches stay straight.

    Only Lectra-style sparse outlines need this. Richpeace exports already
    store the curve as a dense polyline; resampling those would multiply an
    already-fine outline by n_per_seg for no gain, so they are left alone.
    """
    n = len(loop)
    if not curve_mask.any():
        return loop
    step = np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1)
    if np.median(step) < min_spacing_mm:
        return loop
    # rotate so index 0 is a turn point (so runs never wrap)
    turns = np.where(~curve_mask)[0]
    if len(turns) == 0:
        return _catmull_rom(np.vstack([loop, loop[:1]]), n_per_seg)[:-1]
    sh = turns[0]
    loop = np.roll(loop, -sh, axis=0)
    curve_mask = np.roll(curve_mask, -sh)
    out, i = [], 0
    while i < n:
        if not curve_mask[i]:
            out.append(loop[i])
            i += 1
            continue
        j = i
        while j < n and curve_mask[j]:
            j += 1
        # anchor the arc on the previous and next turn points
        seg = loop[np.arange(i - 1, min(j, n) + 1) % n]
        arc = _catmull_rom(seg, n_per_seg)
        out.extend(arc[1:-1])          # anchors are emitted by the turn branch
        i = j
    return _dedup(np.array(out))


# --------------------------------------------------------------------------- #
#  Piece
# --------------------------------------------------------------------------- #
# Keywords marking a piece as NOT part of the outer fabric shell.
#
# The Material/Fabric field is the reliable discriminator: the Richpeace
# exports carry several blocks under the SAME piece name (the shell panel plus
# its net template and its fusible), told apart only by Material -- 面料 (shell
# fabric) vs 实样/翻修样 (templates) vs 纸衬 (paper interfacing).
_MATERIAL_NON_SHELL = [
    ('衬', 'interfacing'), ('纸', 'interfacing'), ('TELA', 'interfacing'),
    ('里', 'lining'), ('样', 'template'), ('净', 'template'),
]
# The free-text annotation is NOT scanned: it routinely names the companion
# interfacing of a shell piece ('KEMER X2 TELA X2' is the waistband, cut in
# fabric, whose interfacing is a separate piece), so scanning it would throw
# away real panels. Only the piece name and category are.
_NAME_NON_SHELL = [
    ('衬', 'interfacing'), ('TELA', 'interfacing'), ('里', 'lining'),
    ('袋布', 'pocket bag'), ('袋唇', 'welt'),
    ('修片', 'template'), ('样', 'template'), ('净', 'template'),
    # Browzwear (bonprix) exports say it in English, in the piece ANNOTATION
    # rather than in a Material field -- those files have no Material/Fabric
    # TEXT at all, so the Chinese rules above never fire and every block,
    # zipper tape included, came back as 'fabric'.
    ('LINING', 'lining'), ('TEMPLATE', 'template'), ('DOUBLURE', 'lining'),
    ('FACING', 'facing'), ('PAREMENTURE', 'facing'),
]
# Blocks that are not pattern pieces at all. Browzwear writes the trims and
# construction aids of a marker as plain rectangles named 'Shape <n>' or
# 'zipper_<n>', annotated only 'CUT': drawcords, elastic, hanger loops, zip
# tape. They have no grain line and no seam, and one of them (a 1 x 25 cm
# sliver) is thinner than the mesh resolution.
_NOTION_BLOCK = re.compile(r'^(shape\s*\d+|zipper[_\s]*\d*)', re.I)
# reference-only drawings, identified from the annotation
_REFERENCE = ['CIZIMI']

# Chinese pattern-piece vocabulary -> English, for readable previews/reports
GLOSSARY = [
    ('修腰西装', 'fitted blazer'), ('后片修片样', 'back trim template'),
    ('司马克毛样', 'smocking template'), ('下脚领面', 'under collar'),
    ('后领贴', 'back neck facing'), ('后领袢', 'back neck loop'),
    ('领面面', 'collar'), ('袋口贴', 'pocket facing'), ('袋唇', 'welt'),
    ('大袋布', 'large pocket bag'), ('小袋布', 'small pocket bag'),
    ('大袖面', 'upper sleeve'), ('小袖面', 'under sleeve'),
    ('大袖里', 'upper sleeve lining'), ('小袖里', 'under sleeve lining'),
    ('前中', 'front centre'), ('前侧', 'front side'), ('后中', 'back centre'),
    ('后侧拼', 'back side'), ('后侧', 'back side'), ('后裙片', 'back skirt'),
    ('前裙侧', 'front skirt side'), ('前幅', 'front'), ('门襟', 'front placket'),
    ('后肩', 'back shoulder'), ('前片', 'front panel'), ('后片', 'back panel'),
    ('袖子', 'sleeve'), ('大货', 'bulk'), ('面料', 'shell fabric'),
    ('实纺', 'woven'), ('里', 'lining'), ('衬', 'interfacing'), ('净', 'net'),
    ('面', 'shell'), ('样', 'template'),
]


def translate(text):
    """Best-effort English rendering of a Chinese piece name."""
    for zh, en in GLOSSARY:
        text = text.replace(zh, ' ' + en + ' ')
    return re.sub(r'\s+', ' ', text).strip(' .')


@dataclass
class Piece:
    block: str
    name: str = ''
    size: str = ''
    quantity: str = ''
    category: str = ''
    fabric: str = ''
    annotation: str = ''
    cut: np.ndarray | None = None        # closed loop, cm, CCW, grain +y
    sew: np.ndarray | None = None
    notches: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    internal: list = field(default_factory=list)     # (layer, Nx2 cm)
    grain_deg: float = 0.0               # original grain angle in the DXF
    mirrored: bool = False               # cut as a mirrored pair
    fold: np.ndarray | None = None       # layer-6 mirror line, cm, or None

    @property
    def boundary(self):
        """The line to simulate: net sew line if the DXF has one, else cut."""
        return self.sew if self.sew is not None else self.cut

    @property
    def has_sew_line(self):
        return self.sew is not None

    def net(self, seam_allow):
        """The finished (sew-line) outline, for a cut-line-only piece.

        A no-op when the DXF already carries a layer-14 sew line, or when
        `seam_allow` is 0 -- so a caller can hand this the same value for every
        garment and the ones that need nothing get nothing.
        """
        if self.has_sew_line or not seam_allow:
            return self.boundary
        return inset_loop(self.boundary, seam_allow, self.fold)

    @property
    def kind(self):
        if _NOTION_BLOCK.match((self.block or '').strip()):
            return 'notion'
        for key, label in _MATERIAL_NON_SHELL:
            if key.upper() in self.fabric.upper():
                return label
        # The annotation is scanned for the ENGLISH keys only. The comment above
        # still holds for the Chinese ones: 'X2 TELA X2' in a Turkish blazer's
        # annotation names a shell piece's companion interfacing, and scanning
        # it would drop the shell piece itself. 'LINING'/'TEMPLATE' carry no
        # such companion idiom -- they name the piece they are written on.
        # .upper() alone does not fold the Turkish dotted capital I: these
        # exports write 'FRONTFACING-2X' as 'FRONTFACİNG-2X', whose upper form
        # keeps the U+0130, so a plain 'FACING' test misses it.
        _fold = lambda t: (t.upper().replace('\u0130', 'I').replace('\u0131', 'I')
                           .replace('\u015e', 'S').replace('\u015f', 'S'))
        hay = _fold(f'{self.name} {self.category}')
        ann = _fold(self.annotation)
        for key, label in _NAME_NON_SHELL:
            if _fold(key) in hay:
                return label
            if key.isascii() and _fold(key) in ann:
                return label
        if any(k in self.annotation.upper() for k in _REFERENCE):
            return 'reference'
        return 'fabric'

    @property
    def label(self):
        """Readable English-ish name for previews and reports."""
        return translate(self.name or self.block)

    def size_cm(self):
        b = self.boundary
        return float(b[:, 0].ptp()), float(b[:, 1].ptp())

    def perimeter(self):
        b = np.vstack([self.boundary, self.boundary[:1]])
        return float(np.sum(np.linalg.norm(np.diff(b, axis=0), axis=1)))

    def __repr__(self):
        w, h = self.size_cm()
        return (f'<Piece {self.name or self.block!r} [{self.kind}] '
                f'{w:.1f}x{h:.1f}cm n={len(self.boundary)} '
                f'{"sew" if self.has_sew_line else "CUT-ONLY"}>')


_META_KEYS = {
    'piece name': 'name', 'size': 'size', 'quantity': 'quantity',
    'category': 'category', 'fabric': 'fabric', 'material': 'fabric',
    'annotation': 'annotation', 'description': 'annotation',
}


def _meta(entities):
    out = {}
    for e in entities:
        if e['type'] != 'TEXT' or not e['text'] or ':' not in e['text']:
            continue
        k, _, v = e['text'].partition(':')
        key = _META_KEYS.get(k.strip().lower())
        if key and key not in out:
            out[key] = v.strip()
    return out


def _is_pair(meta):
    qty = meta.get('quantity', '').replace(' ', '')
    if qty in ('2', '2,0', '1,1'):
        return True
    hay = ' '.join(meta.get(k, '') for k in ('name', 'category', 'annotation'))
    # 'X2' is the Lectra idiom, '-2X'/'-4X' the Browzwear one ('CUFF-2X',
    # 'YOKE-4X'). The bonprix files carry Quantity 0 or 1 on every block
    # regardless of how many are cut, so the suffix is the only signal there.
    # 4X is a pair too as far as placement goes: a 4-up piece is the yoke, cut
    # twice per side (outer + facing), one mirrored pair of shells.
    return bool(re.search(r'X\s*2\b', hay, re.I)
                or re.search(r'\b[24]\s*X\b', hay, re.I))


def _mark_curves(loop, curve_pts, tol=0.05):
    """Flag boundary vertices that coincide with a layer-3 CURVE point."""
    mask = np.zeros(len(loop), bool)
    if len(curve_pts) == 0:
        return mask
    for i, p in enumerate(loop):
        if np.min(np.linalg.norm(curve_pts - p, axis=1)) < tol:
            mask[i] = True
    return mask


def read_pieces(path, size=None, smooth=True, upright=True, grade_to=None,
                unit=None):
    """Read every pattern piece of an AAMA DXF.

    size     -- keep only pieces of this size label (the size-set DXFs hold all
                of them); None keeps everything.
    smooth   -- re-interpolate curve-point runs with a Catmull-Rom spline.
    upright  -- rotate each piece so its grain line points +y.
    grade_to -- grade the geometry to this size label using the `.rul` beside the
                DXF. Only needed when the DXF holds a single sample size.
    unit     -- drawing-unit -> cm scale. None auto-detects from piece extents
                (see `_unit_scale`); pass MM_TO_CM or CM_TO_CM to force it.
    """
    grade = None
    if grade_to:
        rul = grade_path(path)
        if rul is None:
            raise FileNotFoundError(f'no .rul beside {path} to grade with')
        labels, sample, rules = read_grade_rules(rul)
        if grade_to not in labels:
            raise ValueError(f'{grade_to!r} not in {labels}')
        if grade_to != sample:
            col = labels.index(grade_to)
            grade = lambda rule: rules.get(rule, [(0.0, 0.0)] * len(labels))[col]
    pieces = []
    blocks = _raw_blocks(path)
    scale = _unit_scale(blocks) if unit is None else float(unit)
    for blk in blocks:
        ents = blk['entities']
        polys = [e for e in ents if e['type'] == 'POLYLINE' and len(_pts(e)) >= 3]
        if not polys:
            continue

        def longest(layer):
            c = [_pts(e) for e in polys if e['layer'] == layer]
            return max(c, key=len) if c else None

        cut, sew = longest(L_CUT), longest(L_SEW)
        if cut is None and sew is None:
            continue

        meta = _meta(ents)
        if size is not None and meta.get('size', '') != size:
            continue

        curve_pts = np.array([_pts(e)[0] for e in ents
                              if e['type'] == 'POINT' and e['layer'] == L_CURVE
                              and len(_pts(e))] or np.zeros((0, 2)))
        notches = np.array([_pts(e)[0] for e in ents
                            if e['type'] == 'POINT' and e['layer'] == L_NOTCH
                            and len(_pts(e))] or np.zeros((0, 2)))
        internal = [(e['layer'], _pts(e)) for e in ents
                    if e['layer'] in (L_INTERNAL, L_DRILL) and len(_pts(e)) >= 2]

        # grain -> rotation that brings it to +y
        grain_deg, rot = 90.0, np.eye(2)
        for e in ents:
            if e['type'] == 'LINE' and e['layer'] == L_GRAIN and len(_pts(e)) >= 2:
                v = _pts(e)[-1] - _pts(e)[0]
                grain_deg = float(np.degrees(np.arctan2(v[1], v[0])))
                break
        if upright:
            a = np.radians(90.0 - grain_deg)
            rot = np.array([[np.cos(a), -np.sin(a)],
                            [np.sin(a),  np.cos(a)]])

        # Grading is a DISPLACEMENT applied after smoothing, never before it.
        # The rule marks are matched to the raw stored vertices, but resampling a
        # graded outline changes its vertex count (146 -> 106 on one dress panel),
        # which would invalidate every hardcoded seam-junction index in
        # hyperdrop.py. Building the field on the raw loop and then displacing the
        # smoothed loop keeps the vertex count identical to the sample size's.
        field = ref = rule_marks = None
        if grade is not None:
            ref = _open_loop(sew if sew is not None else cut)
            field = _grade_field(ref, _grade_marks(ents, ref), grade)
            rule_marks = _rule_lookup(ents)

        def prep(loop):
            if loop is None:
                return None
            loop = _open_loop(loop)
            if smooth:
                loop = _smooth_boundary(loop, _mark_curves(loop, curve_pts))
            if field is not None:
                loop = _grade_loose(loop, ref, field)
            loop = (loop @ rot.T) * scale
            if _signed_area(loop) < 0:                # keep every piece CCW
                loop = loop[::-1]
            return loop

        def prep_pts(p):
            p = np.asarray(p, float).reshape(-1, 2)
            if not len(p):
                return np.zeros((0, 2))
            if field is not None:
                p = _grade_loose(p, ref, field, rule_marks, grade)
            return (p @ rot.T) * scale

        # Layer 6 is the mirror line: the edge the piece is cut on the fold,
        # which carries NO seam allowance because it is not a seam.
        fold = None
        for e in ents:
            if e['layer'] == L_MIRROR and len(_pts(e)) >= 2:
                fold = prep_pts(_pts(e)[[0, -1]])
                break

        p = Piece(
            block=blk['name'] or '',
            cut=prep(cut), sew=prep(sew), fold=fold,
            notches=prep_pts(notches),
            internal=[(l, prep_pts(v)) for l, v in internal],
            grain_deg=grain_deg,
            # cut as a left/right pair: Richpeace says Quantity 2 or '1,1',
            # Lectra says 'X2' somewhere in the name/category/annotation
            mirrored=_is_pair(meta),
            **{k: v for k, v in meta.items()},
        )
        # centre on the boundary's bbox so panels start at a sane local origin
        pieces.append(p)
    return pieces


# --------------------------------------------------------------------------- #
#  Grading  (.rul tables)
# --------------------------------------------------------------------------- #
def read_grade_rules(path):
    """Parse a Lectra/AAMA `.rul` grading table.

    Returns (size_labels, sample_size, {rule_number: [(dx, dy), ...]}) with one
    (dx, dy) per size, in the DXF's own drawing units (the deltas are applied to
    raw coordinates, before `_unit_scale`), in size-list order. The sample size's entry is
    (0, 0) -- it is the size the DXF geometry is drawn at.
    """
    sizes, sample, n_sizes, rules = [], None, None, {}
    cur, acc = None, []

    def flush():
        if cur is not None and acc:
            rules[cur] = acc[:len(sizes)] if sizes else acc[:]

    for line in open(path, errors='replace'):
        t = line.strip()
        if t.startswith('SIZE LIST:'):
            sizes = t.split(':', 1)[1].split()
        elif t.startswith('NUMBER OF SIZES:'):
            n_sizes = int(re.sub(r'\D', '', t) or 0)
        elif t.startswith('SAMPLE SIZE:'):
            sample = t.split(':', 1)[1].strip()
        elif t.startswith('RULE:'):
            flush()
            m = re.search(r'DELTA\s+(\d+)', t)
            cur, acc = (int(m.group(1)) if m else None), []
        elif cur is not None and t:
            # Two dialects: Lectra writes every size's pair on one line, PAD
            # System one pair per line. Accumulate until the size count is met.
            nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', t)]
            acc.extend(zip(nums[0::2], nums[1::2]))
            if n_sizes and len(acc) >= n_sizes:
                flush()
                cur, acc = None, []
    flush()
    _patch_grade_outliers(sizes, rules, path)
    return sizes, sample, rules


# Grade-rule outliers in the HDBAKIRA dress .rul (BOKE TECHNOLOGY export).
# The parse is not at fault: all 175 rows carry exactly one delta pair per size,
# every sample (M) column is exactly (0, 0), and the column order matches the
# file's own "SIZE LIST: XS S M L XL XXL". These specific rows have an L entry
# that no size run can produce, and each is replaced by the midpoint of its M and
# XL neighbours -- what an evenly stepped run implies:
#
#   78/79/88/89  XS -5.4  S -3.4  M 0  L +39.3  XL +14.7  XXL +23.0
#                L exceeds BOTH XL and XXL. Drives the back-side panel's
#                shoulder/armhole junction; left alone the size-L shoulder is
#                8.80cm against a 5.50cm spec. Repaired -> 5.50 exactly.
#   11/28        XS -45.3 S -23.4 M 0  L +68.7  XL +74.5  XXL +122.8
#                L is in range but crammed against XL (steps +21.9, +23.4,
#                +68.7, +5.8). Drives the front-centre panel width.
#
# Deliberately targeted rather than a general heuristic: a rule that also caught
# the 11/28 shape repaired 92 entries and broke the shoulder that 78 had fixed.
# Keyed by .rul stem so a rule number in one garment's table can never touch
# another's: pants rule 11 is a clean 0/+10/+20 and must be left alone.
# HDBAKIRA dress. Found by auditing every rule's dx column for an L value lying
# outside its own M/XL neighbours -- an ordering no size run can produce:
#     5, 22   M 0, L  -8.6, XL +45.4   (L negative between two positives)
#    78-80,
#    88-90   M 0, L +33.5..+39.3, XL +14.7..+18.5   (L exceeds XL and XXL)
#    11, 28   M 0, L +68.7, XL +74.5   (in range but crammed against XL)
# NOT included: 12/29 (L +55.9 vs XL +89.2) are disproportionate but ordered, and
# there is no spec measurement I can pin them against; and the sample column (M)
# is never touched -- it is (0, 0) by definition.
# 'midpoint'    -> L := halfway between M and XL. Spec-validated: it puts the
#                  size-L shoulder at exactly 5.50cm, the spec-sheet value.
# 'progression' -> L chosen so the row's STEP sequence increases linearly, i.e.
#                  d(M->L) + d(L->XL) = known span with the steps evenly ramped.
#                  Used where no spec measurement can validate a midpoint:
#                  rule 12 becomes +39.3, making its steps 27.2, 28.6, 39.3,
#                  49.9, 59.5. (Applying this method to rule 78 would give +6.0
#                  and a 5.35cm shoulder, so midpoint is kept where it is proven.)
_GRADE_L_OUTLIERS = {'32001145': {
    'midpoint': (5, 11, 22, 28, 78, 79, 80, 88, 89, 90),
    'progression': (12, 29),
}}


def _patch_grade_outliers(labels, rules, path=None):
    """Replace the L delta of known-bad rules with the M..XL midpoint."""
    stem = os.path.splitext(os.path.basename(str(path)))[0] if path else ''
    spec = _GRADE_L_OUTLIERS.get(stem) or {}
    rule_ids = tuple(spec.get('midpoint', ())) + tuple(spec.get('progression', ()))
    if not rule_ids or 'L' not in labels:
        return
    prog = set(spec.get('progression', ()))
    iL = labels.index('L')
    if iL == 0 or iL + 1 >= len(labels):
        return
    done = []
    for r in rule_ids:
        row = rules.get(r)
        if row is None:
            continue
        # Only touch a row that actually shows the defect. These rule NUMBERS
        # also exist in the pants and tee .rul files, where they are perfectly
        # even (pants 11: 0, +10, +20) -- patching those was a no-op only by
        # luck, because the midpoint equalled the original. The guard makes that
        # explicit: L must be out of its neighbours' range, or the M->L step must
        # swamp the L->XL step.
        # The guard only vets the 'midpoint' rules -- it exists to catch a rule
        # number reused in another garment's table. A 'progression' rule is an
        # explicit, file-scoped decision and is not second-guessed (12/29 sit at
        # a step ratio of 1.68, below the guard's threshold, so it would veto
        # exactly the correction we want).
        if r not in prog:
            m, l, xl = row[iL - 1][0], row[iL][0], row[iL + 1][0]
            lo, hi = (m, xl) if m <= xl else (xl, m)
            step_in, step_out = abs(l - m), abs(xl - l)
            if lo <= l <= hi and not (step_out > 1e-6
                                      and step_in > 2.0 * step_out):
                continue
        before = row[iL]
        if r in prog and iL >= 2:
            # steps ramp linearly: d1 = dprev + k, d2 = dprev + 2k, d1 + d2 = T
            vals = []
            for axis in (0, 1):
                dprev = row[iL - 1][axis] - row[iL - 2][axis]
                span = row[iL + 1][axis] - row[iL - 1][axis]
                k = (span - 2.0 * dprev) / 3.0
                vals.append(row[iL - 1][axis] + dprev + k)
            row[iL] = (vals[0], vals[1])
        else:
            row[iL] = ((row[iL - 1][0] + row[iL + 1][0]) / 2.0,
                       (row[iL - 1][1] + row[iL + 1][1]) / 2.0)
        done.append(f'{r}:{before[0]:+.1f}->{row[iL][0]:+.1f}')
    if done:
        print('  Grade rules: patched L outliers ' + ', '.join(done))


def _grade_marks(entities, poly, tol=0.01):
    """{vertex index: rule number} for a boundary, from its numbered TEXTs.

    Grade rules are keyed by the `# n` TEXT entities the exporter drops on each
    graded point. They are matched POSITIONALLY rather than by layer: Lectra puts
    them on the boundary's own layer, Richpeace on the turn/curve point layers,
    and the same rule number is reused across many points and pieces. The tight
    tolerance keeps cut-line marks (offset by the seam allowance) from being
    attached to the sew line.
    """
    marks = {}
    for e in entities:
        if e['type'] != 'TEXT' or not e['text']:
            continue
        t = e['text'].strip()
        if not t.startswith('#'):
            continue
        pts = _pts(e)
        if not len(pts):
            continue
        d = np.linalg.norm(poly - pts[0], axis=1)
        i = int(np.argmin(d))
        if d[i] <= tol:
            marks[i] = int(re.sub(r'\D', '', t) or 0)
    return marks


def _grade_field(poly, marks, delta_of):
    """Per-vertex (dx, dy) for a closed boundary.

    Graded vertices take their rule's delta outright. Everything between two
    graded vertices is blended along the boundary by arc length, which is how
    graded curve points actually move -- only the marked points carry rules.
    """
    n = len(poly)
    idx = sorted(marks)
    if not idx:
        return np.zeros((n, 2))
    step = np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    total = s[-1]
    out = np.zeros((n, 2))
    for a, b in zip(idx, idx[1:] + [idx[0] + n]):
        da = delta_of(marks[a])
        db = delta_of(marks[b % n])
        span = (s[b] if b < n else s[b - n] + total) - s[a]
        for k in range(a, b + 1):
            j = k % n
            pos = (s[k] if k < n else s[k - n] + total) - s[a]
            w = 0.0 if span <= 1e-9 else pos / span
            out[j] = (1 - w) * np.asarray(da) + w * np.asarray(db)
    return out


def _rule_lookup(entities):
    """[(position, rule number)] for every numbered TEXT in a block.

    Interior points -- notches, fold lines, tuck lines -- carry their own grade
    rules just as boundary points do; the tee has 74 numbered marks that are not
    on its outline. Using them is what keeps interior geometry correct under
    grading: displacing tuck lines by the nearest BOUNDARY vertex instead
    scrambled their spacing, and the tuck-pair detection then found a different
    number of tucks per size (11 strips at M, 4 at L).
    """
    out = []
    for e in entities:
        if e['type'] != 'TEXT' or not e['text']:
            continue
        t = e['text'].strip()
        if not t.startswith('#'):
            continue
        pts = _pts(e)
        if len(pts):
            out.append((pts[0], int(re.sub(r'\D', '', t) or 0)))
    return out


def _grade_loose(pts, poly, field, marks=None, delta_of=None, tol=0.01):
    """Displace points that are not on the boundary (notches, internal lines).

    A point sitting on a numbered mark takes that rule's delta outright; anything
    unmarked falls back to the displacement of the nearest graded boundary vertex.
    """
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts):
        return pts
    out = pts.copy()
    mpos = np.array([m[0] for m in marks]) if marks else None
    for i, p in enumerate(pts):
        if mpos is not None:
            d = np.linalg.norm(mpos - p, axis=1)
            j = int(np.argmin(d))
            if d[j] <= tol:
                out[i] = p + np.asarray(delta_of(marks[j][1]))
                continue
        out[i] = p + field[int(np.argmin(np.linalg.norm(poly - p, axis=1)))]
    return out


def grade_path(path):
    """The `.rul` belonging to a DXF, or None.

    Matched by STEM first. Taking the directory's first `.rul` is only safe when
    each garment has its own folder, which is how the Hyperdrop data is laid out
    but not the bonprix set -- five DXF/.rul pairs share one directory there, and
    the first-hit rule silently graded every one of them with the blouse's table
    (a run of '32 34 36 ...' where three of the styles grade '32/34 36/38 ...',
    so their pieces were then filtered out as a size that does not exist).
    The bare-directory fallback is kept for a lone `.rul` named differently from
    its DXF.
    """
    import glob
    stem = os.path.splitext(path)[0]
    for cand in (stem + '.rul', stem + '.RUL'):
        if os.path.exists(cand):
            return cand
    hits = [h for h in glob.glob(os.path.join(os.path.dirname(path), '*.rul'))]
    if len(hits) == 1:
        return hits[0]
    # Several tables, none matching this stem: pairing them up is guesswork.
    return None


@functools.lru_cache(maxsize=8)
def sizes_in(path):
    """Size labels present in the DXF.

    Cached: pieces_for_size() consults it before reading, and re-parsing a
    size-set DXF (the blazer's is 7.7 MB / 164 blocks) once per query was enough
    to exhaust memory when building several sizes in one process.
    """
    out = []
    for blk in _raw_blocks(path):
        s = _meta(blk['entities']).get('size')
        if s and s not in out:
            out.append(s)
    return out


def dedupe_identical(pieces, tol=1e-6):
    """Drop pieces whose boundary is geometrically identical to an earlier one.

    Some source files carry a stale copy of a panel under a second block name
    (the pants DXF has the back leg twice, as Pattern_4 and Pattern_4_1).
    """
    out, seen = [], []
    for p in pieces:
        b = p.boundary
        if any(o.shape == b.shape and np.abs(o - b).max() < tol for o in seen):
            continue
        seen.append(b)
        out.append(p)
    return out


# Which of several drafts of the same piece to keep: the one the marker says to
# CUT (QUANTITY > 0). Confirmed by the exporter for the jogger -- x2KN_1 is the
# back, x3KN_1 the front, both qty=1. Do not switch this back to "most vertices
# wins"; that rule silently built every jogger up to 2026-08-25 from a back-leg
# draft 2 cm too long.
PREFER_CUT_QUANTITY = True


def dedupe_near(pieces, tol_cm=2.5):
    """Drop re-cut copies of a piece that `dedupe_identical` cannot see.

    The bonprix markers each carry the style TWICE: a full numbered set plus a
    partial re-nest of the same pieces, and the two copies are not bit-identical
    -- the 8672609700 back bodice is 39.3 x 28.1 in one and 38.3 x 28.1 in the
    other, its back skirt has 208 vertices against 203. Same piece, redrawn.

    Two pieces are the same piece when they share an annotation (the exporter's
    real piece id -- the block name is uniquified per nest, the annotation is
    not) AND their bounding boxes agree to `tol_cm`. The bbox test is what keeps
    the rule safe where one annotation legitimately covers several pieces: the
    blouse writes '..._FRO_..._015' on both its 29 x 69 front panel and its
    4 x 46 cuff strip, which no tolerance confuses.

    The copy the marker says to CUT wins -- QUANTITY first, vertex count only as
    a tie-break. The jogger carries two back-leg drafts and three front-leg
    drafts under the same annotation, differing by real amounts (back
    40.0x100.1 vs 40.0x98.1, front 27.5x98.1 vs 27.5x97.1 -- 2 cm of leg
    length). The exporter confirmed x2KN_1 and x3KN_1, both qty=1, are the
    production pieces.

    Choosing by vertex count instead -- "the better-resolved outline" -- picked
    the qty=0 drafts, because they happen to carry more vertices AND come first
    in the file, so they won on both tests. Every jogger built before this used a
    back leg 2 cm too long.

    NOTE the qty=1 drafts grade worse with this .rul (7.4 mm at sizes 34/40
    against 2.8/1.3 for the discarded ones, while the sample size 38 is fine
    either way). That is a grading problem to fix, not a reason to build the
    wrong piece.

    2.5 cm rather than something tighter because the jogger carries two back-leg
    drafts differing by 2 cm of inseam. Widening past 2.5 changes nothing in any
    of the five files (checked to 4 cm), so the value sits inside a flat region
    rather than on an edge.
    """
    kept = []
    for p in pieces:
        w, h = p.size_cm()
        for i, q in enumerate(kept):
            qw, qh = q.size_cm()
            if (p.annotation == q.annotation
                    and abs(w - qw) <= tol_cm and abs(h - qh) <= tol_cm):
                def rank(x):
                    try:
                        qty = int(str(getattr(x, 'quantity', 0)).split(',')[0])
                    except (TypeError, ValueError):
                        qty = 0
                    return ((1 if qty > 0 else 0, len(x.boundary))
                            if PREFER_CUT_QUANTITY else (len(x.boundary),))
                if rank(p) > rank(q):
                    kept[i] = p
                break
        else:
            kept.append(p)
    return kept


def pieces_for_size(path, size, fabric_only=True, **kw):
    """Pieces at `size`: straight from the DXF if it holds that size, otherwise
    graded from the sample size using the `.rul` beside it.

    Lets a caller just ask for 'L' without caring whether the file is a size set
    (the blazer) or a single sample size (everything else).
    """
    reader = fabric_pieces if fabric_only else read_pieces
    if size in sizes_in(path):
        return reader(path, size=size, **kw)
    rul = grade_path(path)
    if rul is None:
        raise FileNotFoundError(
            f'{os.path.basename(path)} has no size {size!r} and no .rul to grade with')
    _, sample, _ = read_grade_rules(rul)
    return reader(path, size=sample, grade_to=size, **kw)


def fabric_pieces(path, size=None, dedupe=True, near_tol_cm=2.5, **kw):
    """Only the outer-shell fabric pieces (drops interfacing/lining/templates)."""
    out = [p for p in read_pieces(path, size=size, **kw) if p.kind == 'fabric']
    if not dedupe:
        return out
    out = dedupe_identical(out)
    return dedupe_near(out, near_tol_cm) if near_tol_cm else out




# =========================================================================== #
#  Section 2: DXF outline -> GarmentCode panel
#
#  The boundaries above are just polylines. To sew them we must cut each
#  outline exactly where one seam ends and the next begins (underarm, shoulder
#  tip, waist notch, ...) and fit each arc with cubic Beziers.
#
#  Curve fitting (`_seq_from_points`, above): a seam stitched to a DIFFERENTLY shaped
#  partner becomes ONE cubic edge (multi-segment seams get subdivided to
#  different vertex counts on each side and weld in a staircase), while a seam
#  whose partner is its own mirror image can keep the faithful piecewise fit.
# =========================================================================== #


# --------------------------------------------------------------------------- #
#  Outline analysis
# --------------------------------------------------------------------------- #
def turn_angles(loop):
    """Signed turn angle (deg) at every vertex of a closed polyline."""
    loop = np.asarray(loop, float)
    n = len(loop)
    prev = loop - np.roll(loop, 1, axis=0)
    nxt = np.roll(loop, -1, axis=0) - loop
    a1 = np.arctan2(prev[:, 1], prev[:, 0])
    a2 = np.arctan2(nxt[:, 1], nxt[:, 0])
    t = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
    ok = (np.linalg.norm(prev, axis=1) > 1e-9) & (np.linalg.norm(nxt, axis=1) > 1e-9)
    return np.where(ok, np.degrees(t), 0.0), n


def corners(loop, thr=25.0, window=None):
    """Indices of outline vertices that turn by more than `thr` degrees.

    On dense outlines a single physical corner is spread over several vertices;
    `window` (in cm) keeps only the sharpest vertex within that arc length.
    """
    ang, n = turn_angles(loop)
    cand = [i for i in range(n) if abs(ang[i]) > thr]
    if window is None or not cand:
        return cand
    loop = np.asarray(loop, float)
    step = np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]                # arc length at each vertex
    keep, used = [], []
    for i in sorted(cand, key=lambda i: -abs(ang[i])):
        if all(min(abs(s[i] - s[j]), s[-1] - abs(s[i] - s[j])) > window for j in used):
            keep.append(i)
            used.append(i)
    return sorted(keep)


def nearest_idx(loop, pt):
    return int(np.argmin(np.linalg.norm(np.asarray(loop, float) - np.asarray(pt, float), axis=1)))


def extreme_idx(loop, which):
    """Index of an extreme vertex: 'top' | 'bottom' | 'left' | 'right', with
    'top-left' style combinations resolved as lexicographic priority."""
    loop = np.asarray(loop, float)
    key = {
        'top': -loop[:, 1], 'bottom': loop[:, 1],
        'left': loop[:, 0], 'right': -loop[:, 0],
    }
    parts = which.split('-')
    score = sum(key[p] * (10.0 ** -i) for i, p in enumerate(parts))
    return int(np.argmin(score))


def arc(loop, a, b):
    """Boundary vertices walking forward from index a to index b (inclusive)."""
    n = len(loop)
    idx = []
    i = a
    while True:
        idx.append(i)
        if i == b:
            break
        i = (i + 1) % n
    return np.asarray(loop, float)[idx]


def split_arcs(loop, key_idx):
    """Split a closed outline at sorted key indices into consecutive arcs.

    Returns a list of point chains; arc k runs key_idx[k] -> key_idx[k+1], and
    the last wraps back to key_idx[0], so the arcs tile the whole outline.
    """
    key = sorted(set(int(i) for i in key_idx))
    if len(key) < 2:
        raise ValueError('need at least 2 split points')
    return [arc(loop, key[k], key[(k + 1) % len(key)]) for k in range(len(key))]


# --------------------------------------------------------------------------- #
#  Panel
# --------------------------------------------------------------------------- #
def simplify_loop(loop, keys, tol=0.12, min_edge=0.18):
    """Simplify a closed outline ONCE, keeping the junction vertices, and return
    (loop, keys) remapped.

    Doing this per-seam instead leaves collinear triples straddling a junction --
    each seam is clean on its own but the mesher still sees three nearly-collinear
    points and refuses to triangulate them ("Invalid edge lengths [0.342, 0.520,
    0.910]"). Simplifying the whole loop first sees across the junctions, and
    forcing the junctions to be kept means the seam split still lands where the
    pattern says.

    `tol` is the Douglas-Peucker bound (every dropped vertex is within it of the
    chord that replaces it, so the millimetre guarantee holds); `min_edge` then
    removes any vertex that would leave an edge too short to mesh, subject to the
    same bound.
    """
    P = np.asarray(loop, float)
    keys = [int(k) % len(P) for k in keys]
    n = len(P)
    protect = set(keys)
    # Douglas-Peucker on each junction-to-junction span, so junctions survive.
    order = sorted(protect)
    kept = []
    try:
        from shapely.geometry import LineString
        have_shapely = True
    except ImportError:
        have_shapely = False
    for a, b in zip(order, order[1:] + [order[0] + n]):
        span = np.array([P[i % n] for i in range(a, b + 1)])
        if have_shapely and len(span) > 2:
            span = np.asarray(
                LineString(span).simplify(tol, preserve_topology=False).coords,
                float)
        span = _thin_short(span, min_edge, tol)
        # Always contribute at least the junction itself. A span thinned down to
        # a single point contributes nothing, the running offset does not advance,
        # and the next junction inherits the same index -- two junctions collide,
        # a seam arrives with one point and the panel cannot be built.
        kept.append(span[:-1] if len(span) > 1 else span[:1])
    out = np.vstack(kept)
    # Junction positions in the simplified loop, keyed by their ORIGINAL index --
    # `order` is ascending, but the caller's `keys` are in ROLE order and must be
    # returned that way, or e.g. keys[2] stops being the crotch point.
    at, where = 0, {}
    for start, part in zip(order, kept):
        where[start % n] = at
        at += len(part)
    # The per-span passes never see the edge that LEAVES a junction, because each
    # span hands its last point to the next. Sweep the closed loop once more,
    # dropping non-junction vertices that leave an edge too short to mesh.
    prot = set(where.values())
    m = len(out)
    drop = set()
    prev = 0
    for i in range(1, m + 1):
        j = i % m
        if j in drop or j in prot:
            prev = j
            continue
        if np.linalg.norm(out[j] - out[prev]) >= min_edge:
            prev = j
            continue
        # Short edge -- but only drop the vertex if the chord that replaces it
        # still passes within `tol`. Dropping unconditionally here is what left
        # the jogger 1.17 mm off its DXF while every other pass measured clean.
        nxt = out[(j + 1) % m]
        seg = nxt - out[prev]
        L2 = float(np.dot(seg, seg))
        t = 0.0 if L2 < 1e-24 else float(
            np.clip(np.dot(out[j] - out[prev], seg) / L2, 0.0, 1.0))
        if float(np.linalg.norm(out[prev] + t * seg - out[j])) <= tol:
            drop.add(j)
        else:
            # Dropping would bend the seam, so MERGE the pair to its midpoint
            # instead: that removes the short edge at a cost of at most half its
            # length, where dropping costs the full offset. A sub-millimetre
            # boundary edge in a ~1 cm mesh makes boxmeshgen emit a sliver it
            # then refuses to triangulate.
            if j in prot and prev in prot:
                # Two junctions adjacent to each other: neither may move and
                # neither may go. Merging here dropped a PROTECTED vertex, which
                # both invalidated the junction map and left two edges sharing
                # endpoints -- svgpathtools then asserts on `self != other_seg`
                # inside is_self_intersecting.
                prev = j
            elif j not in prot and prev not in prot:
                mid = 0.5 * (out[prev] + out[j])
                out[prev] = mid
                out[j] = mid
                drop.add(j)
            else:
                keeper = prev if prev in prot else j
                goer = j if keeper is prev else prev
                out[goer] = out[keeper]
                drop.add(goer)
                prev = keeper
    if drop:
        surv = [i for i in range(m) if i not in drop]
        remap = {old_i: new_i for new_i, old_i in enumerate(surv)}
        out = out[surv]
        where = {k: remap[v] for k, v in where.items()}

    # Merging to midpoints can leave a vertex sitting ON the straight line
    # through its neighbours, and a collinear boundary triple triangulates to a
    # zero-area sliver the mesher rejects ("Invalid edge lengths [1.069, 0.515,
    # 0.515]" -- 0.515 + 0.515 < 1.069). Sweep once more for exact collinearity;
    # this only ever removes vertices that carry no shape at all.
    prot = set(where.values())
    m = len(out)
    flat = set()
    for j in range(m):
        if j in prot:
            continue
        a, b = out[(j - 1) % m], out[(j + 1) % m]
        seg = b - a
        L2 = float(np.dot(seg, seg))
        if L2 < 1e-24:
            continue
        t = float(np.clip(np.dot(out[j] - a, seg) / L2, 0.0, 1.0))
        if float(np.linalg.norm(a + t * seg - out[j])) < 1e-4:
            flat.add(j)
    if flat:
        surv = [i for i in range(m) if i not in flat]
        remap = {old_i: new_i for new_i, old_i in enumerate(surv)}
        out = out[surv]
        where = {k: remap[v] for k, v in where.items()}
    return out, [where[k] for k in keys]


def _at_arclen(chain, cum, at):
    """Point on a polyline at a given arc length (exact interpolation)."""
    at = float(np.clip(at, 0.0, cum[-1]))
    j = int(np.searchsorted(cum, at, 'right')) - 1
    j = max(0, min(j, len(chain) - 2))
    span = cum[j + 1] - cum[j]
    t = 0.0 if span <= 0 else (at - cum[j]) / span
    return chain[j] + t * (chain[j + 1] - chain[j])


def _seq_corner_fit(pts, label='', thr=20.0, min_seg=0.35, max_seg=8.0):
    """The seam as one cubic per SMOOTH RUN, split at its interior corners.

    The original cubic fitting with its one real defect removed. A cubic cannot
    cross a hard corner, so fitting a whole seam with one bowed every run that
    was part straight and part curved -- 14.7 mm off the DXF at worst, and
    drafted straight edges came out as curves. Split the chain wherever it turns
    sharply and each piece is either straight, which a cubic reproduces exactly,
    or one smooth arc, which it fits closely.

    Preferred over taking the polyline verbatim. That tracked the outline to
    0.3 mm but left 25 to 49 edges on a seam, and everything downstream assumes
    far fewer: `StitchingRule.match_interfaces` must reconcile two mismatched
    fraction sets and welds badly, the mesher builds slivers it then refuses to
    triangulate, and mapping junctions onto a sparse offset outline collapses
    them. Few edges of the right shape beat many edges in the right place.
    """
    P = _dedup(np.asarray(pts, float), tol=1e-3)
    if len(P) < 2:
        raise ValueError('seam has fewer than 2 distinct points')
    if len(P) < 4:
        # Too few points to fit anything: a cubic through 3 points can bow badly
        # (the blouse yoke's armhole is a 20 cm arc the DXF gives 3 points, and
        # one cubic put it 24 mm out), while straight edges through them
        # reproduce the source exactly. Nothing is lost -- those points ARE the
        # outline as drawn.
        return pyg.EdgeSequence(*[pyg.Edge(list(a), list(b), label=label)
                                  for a, b in zip(P[:-1], P[1:])])

    n = len(P)
    ang, _ = turn_angles(P)
    s_len = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]

    # Build the cut set as a SORTED SET of vertex indices: the two ends, every
    # interior corner, then extra cuts so no run exceeds `max_seg`. Built this
    # way rather than appended in passes -- doing it in passes let indices go out
    # of order, which sliced the chain wrongly and measured 100 mm out.
    cuts = {0, n - 1}
    for i in range(1, n - 1):
        if abs(ang[i]) > thr:
            cuts.add(i)
    if max_seg:
        for _ in range(12):                       # refine until every run fits
            ordered = sorted(cuts)
            added = False
            for a, b in zip(ordered[:-1], ordered[1:]):
                if s_len[b] - s_len[a] <= max_seg:
                    continue
                mid = float(0.5 * (s_len[a] + s_len[b]))
                j = a + int(np.argmin(np.abs(s_len[a:b + 1] - mid)))
                if a < j < b:
                    cuts.add(j)
                    added = True
            if not added:
                break
    # Drop cuts that would leave a run too short to fit, keeping the ends.
    ordered, keep = sorted(cuts), [0]
    for i in ordered[1:-1]:
        if s_len[i] - s_len[keep[-1]] > min_seg:
            keep.append(i)
    # The tail needs a LOOP, not one pop: the forward walk absorbs short runs
    # into their predecessor, but at the far end two consecutive short runs are
    # left behind by a single pop. That is how the jogger's leg waist kept a
    # 1.08 and a 1.09 cm edge after the leading 0.72/0.44 pair had been merged
    # away -- and those two, being under the mesher's 1 cm resolution, are the
    # welds that collapsed and threw the spike at the top of the side seam.
    while len(keep) > 1 and s_len[n - 1] - s_len[keep[-1]] <= min_seg:
        keep.pop()
    keep.append(n - 1)
    cuts = keep

    edges = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        run = P[a:b + 1]
        if len(run) < 4:
            # Straight edges along the run -- but its interior points are merged
            # up to `min_seg` first. Emitting one edge per point PAIR is what
            # kept sub-resolution edges alive no matter how high `min_seg` went:
            # the run passed the length test, its individual steps did not. A
            # 2.17 cm run of three points came out as 1.08 + 1.09 cm on the
            # jogger's leg waist, and those are the welds that collapsed and threw
            # the spike at the top of the side seam. Same cause as the hem's
            # 0.84/0.86 cm pair.
            sr = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(run, axis=0),
                                                    axis=1))]
            take = [0]
            for t in range(1, len(run) - 1):
                if (sr[t] - sr[take[-1]] > min_seg
                        and sr[-1] - sr[t] > min_seg):
                    take.append(t)
            take.append(len(run) - 1)
            edges.extend(pyg.Edge(list(run[u]), list(run[v]), label=label)
                         for u, v in zip(take[:-1], take[1:]))
        else:
            edges.extend(_seq_from_points(run, label=label, single=True))
    return pyg.EdgeSequence(*edges)


def _seq_verbatim(pts, label='', tol=0.03, min_edge=0.35):
    """The seam as STRAIGHT edges along the DXF polyline itself.

    Cubic fitting cannot hit the millimetre: one cubic per seam misses by up to
    14.7 mm, cutting a 10 cm cap off each brings that to 5.4 mm, and the adaptive
    piecewise fitter is no better -- a cubic simply cannot cross a hard corner,
    and these outlines are full of them. Measured against the source, both modes
    turned drafted straight edges into curves.

    So the outline is used as drawn. Douglas-Peucker at `tol` first, which is
    what makes this cheap AND exact in the ways that matter: it is a guaranteed
    bound (no retained point is further than `tol` from the source), a straight
    run of 70 vertices collapses to ONE straight edge, and a corner is never
    removed because removing it would exceed the bound.
    """
    P = _dedup(np.asarray(pts, float), tol=1e-3)
    if len(P) < 2:
        raise ValueError('seam has fewer than 2 distinct points')
    try:
        from shapely.geometry import LineString
        P = np.asarray(LineString(P).simplify(tol, preserve_topology=False).coords,
                       float)
    except ImportError:
        pass
    P = _thin_short(P, min_edge, tol)
    # pygarment compares edges with a 1e-2 tolerance, so two vertices closer than
    # that produce an edge indistinguishable from its neighbour and svgpathtools
    # asserts. Drop them here as a last line of defence.
    ends = (P[0].copy(), P[-1].copy())
    P = _dedup(P, tol=2e-2)
    if len(P) < 2:
        # The whole seam is shorter than the dedup tolerance. Keep its two
        # endpoints regardless: they are seam junctions, and dropping one detaches
        # the outline. Raising here instead killed the blouse at size 38.
        P = np.array(ends)
        if np.linalg.norm(P[1] - P[0]) < 1e-9:
            raise ValueError('seam has zero length')
    return pyg.EdgeSequence(*[pyg.Edge(list(a), list(b), label=label)
                              for a, b in zip(P[:-1], P[1:])])


def _thin_short(P, min_edge, tol):
    """Drop vertices that leave an edge shorter than `min_edge`, within `tol`.

    Simplification alone is not enough for the mesher. Douglas-Peucker keeps any
    vertex that carries shape, including pairs only 3 mm apart on a tight curve,
    and three nearly-collinear vertices that close together triangulate into a
    sliver -- boxmeshgen rejects them outright ("Invalid edge lengths
    [0.342, 0.520, 0.910]. Not possible to form a triangle", where 0.342 + 0.520
    is less than 0.910).

    A vertex is dropped only if the chord that replaces it still passes within
    `tol` of it, so the millimetre bound survives this pass; endpoints are never
    dropped, because they are seam junctions.
    """
    P = np.asarray(P, float)
    if len(P) < 3 or min_edge <= 0:
        return P
    def off_chord(pts, a, b):
        """Furthest of `pts` from the segment a-b."""
        seg = b - a
        L2 = float(np.dot(seg, seg))
        if L2 < 1e-24:
            return max((float(np.linalg.norm(q - a)) for q in pts), default=0.0)
        worst = 0.0
        for q in pts:
            t = float(np.clip(np.dot(q - a, seg) / L2, 0.0, 1.0))
            worst = max(worst, float(np.linalg.norm(a + t * seg - q)))
        return worst

    # Greedy, but every vertex dropped since the last kept one is re-checked
    # against the chord that will actually replace it. Checking only against the
    # immediate neighbours lets the error accumulate over a run of drops -- that
    # is what put the jogger 1.17 mm off its DXF while each individual drop
    # looked fine.
    keep, pending = [0], []
    for i in range(1, len(P) - 1):
        prev = P[keep[-1]]
        if np.linalg.norm(P[i] - prev) >= min_edge:
            keep.append(i)
            pending = []
            continue
        if off_chord(pending + [P[i]], prev, P[i + 1]) > tol:
            keep.append(i)
            pending = []
        else:
            pending.append(P[i])
    keep.append(len(P) - 1)
    # The two endpoints are seam junctions and are never dropped, so a short
    # FIRST or LAST edge has to be cured by removing its inner neighbour instead.
    while len(keep) > 2 and np.linalg.norm(P[keep[-1]] - P[keep[-2]]) < min_edge:
        keep.pop(-2)
    while len(keep) > 2 and np.linalg.norm(P[keep[1]] - P[keep[0]]) < min_edge:
        keep.pop(1)
    return P[keep]


class DxfPanel(pyg.Panel):
    """A GarmentCode panel whose outline comes verbatim from a DXF piece.

    `seams` is an ordered dict {name: point-chain}; consecutive chains must
    share endpoints and together close the loop. `single` names the seams to
    collapse to one cubic edge (those stitched to a differently-shaped panel).

    After construction `self.interfaces[name]` exists for every seam.
    """

    # Douglas-Peucker tolerance (cm) for `verbatim` seams. Every retained point
    # is within this of the source polyline, so it is a HARD bound on how far the
    # panel can differ from the DXF -- 0.03 cm = 0.3 mm.
    # Chosen for SEAM quality, not for the smallest number. 0.25 cm simplified
    # so hard that sparse outlines could no longer carry their own junctions --
    # the darted front has 11, and several collapsed onto one vertex. 0.12 keeps
    # enough of them while still cutting a seam from 25 edges to about 12. Tighter values track
    # the DXF better on paper -- 0.03 cm gives 0.3 mm -- but leave 25 to 49 edges
    # on a seam, and `StitchingRule.match_interfaces` then has to reconcile two
    # mismatched fraction sets, which welds badly (the staircase seam that
    # `single=True` existed to avoid) and at worst builds an impossible rest
    # triangle. At 0.25 cm the panels sit 2.4 mm from the DXF -- six times closer
    # than the single-cubic fitting this replaced, which was 14.7 mm out -- with
    # 8 edges on a seam instead of 25, so partners pair almost 1:1.
    # Turn angle (deg) that counts as a corner no cubic may cross.
    CORNER_THR = 20.0
    # Longest arc a single cubic may span (cm).
    CORNER_MAX_SEG = 10.0
    # Shortest arc a cut may leave (cm). Must stay ABOVE the box mesher's
    # resolution (1.0 cm): an edge shorter than one mesh cell gets too few
    # vertices to survive welding, its two ends merge into one, and the collapse
    # yanks that vertex out -- which is the spike that stood off the body at the
    # top of the jogger's side seam, where the leg waist's fitted edges ran 0.44
    # and 1.08 cm. `collapse_stitch_vertices` names those exact welds.
    #
    # Raising this alone did nothing for a long time, and the reason was in the
    # edge-building loop below, not here: a run of fewer than 4 points was
    # emitted as one straight edge per point PAIR, so a 2.17 cm run of three
    # points came out as 1.08 + 1.09 cm however high this went. With that fixed,
    # 1.1 removes every sub-resolution edge on the waist AND leaves fidelity
    # where it was (jogger 38/40 at 1.45/1.34 mm, unchanged); 1.6 over-smooths
    # and costs 1 mm for nothing.
    # Left at the conservative default: raising it globally to 1.1 bowed the
    # blouse's centre-back FOLD edge by 9.5 mm (a cut it needed was dropped),
    # which is the opposite of what this is for. Only the seams welded to a
    # straight band need the resolution floor, and they ask for it per seam.
    CORNER_MIN_SEG = 0.35
    VERBATIM_TOL = 0.12
    # A single cubic longer than this multiple of its own source chain is a
    # blown fit, not a curve -- see the fallback where `single` seams are built.
    OVERSHOOT = 1.5
    # Shortest edge kept. Deliberately small: the mesher's complaint was
    # COLLINEARITY, not shortness ("Invalid edge lengths [0.342, 0.520, 0.910]"
    # -- 0.342 + 0.520 < 0.910), and simplifying the whole loop already removes
    # collinear triples. Thinning harder than this costs real fidelity: at
    # 0.35 cm the accumulated drops put the jogger 2.00 mm off its DXF.
    VERBATIM_MIN_EDGE = 0.18

    def __init__(self, name, seams, single=(), reverse=False,
                 pivot=None, translation=None, rotation=None, label='',
                 edge_labels=None, verbatim=False, presimplified=False,
                 source=None, min_seg=None, parts=None):
        super().__init__(name, label=label)
        # Edge labels are the downstream SEGMENTATION labels, and boxmeshgen
        # rejects a stitch whose two edges carry different ones. Seam names here
        # are per-panel and differ across a stitch ('armhole_r' vs 'cap'), so
        # edges stay unlabelled unless a caller supplies a shared label.
        edge_labels = edge_labels or {}
        seams = {k: np.asarray(v, float) for k, v in seams.items()}

        # `parts` cuts named seams into sub-seams at given arc-length fractions,
        # IN PLACE in the dict so the outline still walks in order. The point is
        # to give the two sides of a seam the same number of parts at the same
        # fractions BEFORE they are stitched: `match_interfaces` then has nothing
        # to subdivide, and it is its subdivision that creates the sub-resolution
        # slivers which collapse when welded. Cutting both sides into N EQUAL
        # parts also makes the fraction set symmetric, so it holds whichever way
        # round each chain happens to run -- the direction is settled later, by
        # the stitch.
        def _apply_parts(chains):
            rebuilt = {}
            for k, v in chains.items():
                fr = parts.get(k)
                if fr is None:
                    rebuilt[k] = v
                    continue
                # An EMPTY fraction list still means "one part": the seam becomes
                # `<key>_0` and is forced to a single cubic. That is the whole
                # point when a seam is short enough to need no splitting -- both
                # sides then carry exactly one edge and match 1:1. Skipping it
                # would leave each side fitted to its own edge count again.
                d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(v, axis=0),
                                                        axis=1))]
                tot = d[-1]
                cuts = [0.0] + [f * tot for f in fr] + [tot]
                for i in range(len(cuts) - 1):
                    a, b = cuts[i], cuts[i + 1]
                    lo = np.searchsorted(d, a, 'right') - 1
                    hi = np.searchsorted(d, b, 'left')
                    inner = v[max(lo + 1, 0):hi]
                    pa = _at_arclen(v, d, a)
                    pb = _at_arclen(v, d, b)
                    chain = np.vstack([pa, inner, pb]) if len(inner) else \
                        np.vstack([pa, pb])
                    # drop duplicates the interpolation may have re-created
                    keep = [0]
                    for t in range(1, len(chain)):
                        if np.linalg.norm(chain[t] - chain[keep[-1]]) > 1e-9:
                            keep.append(t)
                    rebuilt[f'{k}_{i}'] = chain[keep]
            return rebuilt

        if parts:
            seams = _apply_parts(seams)
            # Each part becomes exactly ONE edge. That is the whole point of
            # `parts`: matching the two sides' PART counts is useless if a part
            # is then fitted into a variable number of edges -- the counts differ
            # again and `match_interfaces` subdivides anyway (the blouse went
            # from 63 stitches to 105 that way). One cubic per part is what the
            # jogger's leg seams do, and a part is short enough that it costs
            # nothing.
            single = tuple(single) + tuple(
                k for k in seams if k.rsplit('_', 1)[0] in parts)
            if source is not None:
                # The gate must compare each PART with its own stretch of the
                # source, not with the whole seam.
                source = _apply_parts(
                    {k: np.asarray(v, float) for k, v in source.items()})

        # Winding control. Two panels stitched along a single-edge seam must be
        # wound OPPOSITELY, so the shared edge is antiparallel and the default
        # end<->start weld is already correct. Reversing means reversing every
        # chain AND the assembly order -- that changes winding only, not shape.
        order = list(seams)
        if reverse:
            order = order[::-1]
            seams = {k: v[::-1] for k, v in seams.items()}

        sequences, edges = {}, []
        for key in order:
            if key in single:
                # Forced to ONE edge: used for the paired sub-seams of a shared
                # seam, where both sides must keep identical edge counts or
                # `match_interfaces` subdivides them (and swaps the halves).
                seq = _seq_from_points(seams[key],
                                       label=edge_labels.get(key, ''),
                                       single=True)
                # A cubic through 4 control points can leave the hull of its own
                # source chain: on 8242610411's armhole at size 40/42 one 4.28 cm
                # part came back 18.43 cm, a 4.3x overshoot, while every other
                # part on the same seam fitted to 0.01 cm. `check_fidelity`
                # would catch it, but the build dies first --
                # `match_interfaces` gets a NEGATIVE split fraction off the
                # over-long edge and raises inside `_subdivide`. A fit longer
                # than its own source is wrong by construction, so fall back to
                # the polyline, which cannot overshoot.
                src_len = float(np.sum(np.linalg.norm(
                    np.diff(np.asarray(seams[key], float), axis=0), axis=1)))
                fit_len = float(sum(e.length() for e in seq))
                if src_len > 1e-9 and fit_len > self.OVERSHOOT * src_len:
                    # Straight chord, not a polyline: this seam is `single`
                    # precisely because its edge COUNT has to stay 1, and a
                    # polyline fallback breaks that (match_interfaces then
                    # subdivides and fails the same way). A chord over a 4.28 cm
                    # arc costs a couple of mm of sagitta, which the fidelity
                    # gate judges on its own terms.
                    ch = np.asarray(seams[key], float)
                    print(f'{name}::WARNING::{key}: one cubic came out '
                          f'{fit_len:.2f} cm over a {src_len:.2f} cm source '
                          f'-- refitted as a straight chord')
                    seq = _seq_from_points(np.array([ch[0], ch[-1]]),
                                           label=edge_labels.get(key, ''),
                                           single=True)
            elif verbatim:
                # `min_seg` may be raised PER SEAM. It has to be above the box
                # mesher's resolution on any seam welded to a straight band: an
                # edge shorter than one mesh cell gets too few vertices to
                # survive welding, its two ends merge, and the collapse yanks
                # that vertex into a spike -- which is what stood off the body at
                # the top of the jogger's side seam, where the leg waist's fitted
                # edges ran 0.44 and 1.08 cm. Raising it EVERYWHERE instead cost
                # 1.45 -> 2.53 mm of fidelity, because it also smooths genuine
                # corners on seams that were never the problem.
                seq = _seq_corner_fit(seams[key],
                                      label=edge_labels.get(key, ''),
                                      thr=self.CORNER_THR,
                                      min_seg=(min_seg or {}).get(
                                          key, self.CORNER_MIN_SEG),
                                      max_seg=self.CORNER_MAX_SEG)
            else:
                seq = _seq_from_points(seams[key], label=edge_labels.get(key, ''),
                                       single=(key in single))
            sequences[key] = seq
            edges.extend(seq)
        # Force consecutive edges to share endpoints EXACTLY. The cubic fitter
        # de-duplicates its input first, and that can discard a chain's final
        # point when it sits within the dedup tolerance of its neighbour -- the
        # two seams then end a fraction of a mm apart, the panel loop never
        # closes, and triangulation returns an empty mesh.
        # ... but only a residue. A REAL gap means the seams were handed over in
        # the wrong order and do not tile the outline; snapping it shut silently
        # drags a whole edge across the panel, which every per-seam fidelity
        # check still passes because each seam matches its own source. That is
        # how the shirt-dress front got away with retracing its outline ten
        # times. Anything past a fitting residue is a hard error.
        SNAP_MAX = 0.2                                     # cm
        for i, (prev, nxt) in enumerate(zip(edges, edges[1:])):
            gap = float(np.hypot(*(np.asarray(nxt.start, float)
                                   - np.asarray(prev.end, float))))
            if gap > SNAP_MAX:
                raise ValueError(
                    f'{name}: seam chains do not join -- {gap:.2f} cm gap at '
                    f'edge {i}->{i + 1} (seam order {order}). The seams must be '
                    f'given in the order the outline walks them.')
            nxt.start = list(prev.end)

        # Edges are chained by VALUE, not identity (Panel.assembly() merges
        # coincident endpoints), so close the loop only when the seams leave a
        # real gap -- calling close_loop() on an already-closed outline asserts
        # on a zero-length edge.
        self.edges = pyg.EdgeSequence(*edges)
        gap = np.hypot(*(np.asarray(self.edges[-1].end, float)
                         - np.asarray(self.edges[0].start, float)))
        if gap > 1e-6:
            self.edges.append(pyg.Edge(self.edges[-1].end, self.edges[0].start))
        else:
            # Snap the closure EXACTLY. Panel.assembly() closes the loop with an
            # exact `==` on the first and last vertex, so a residual of even 1e-14
            # leaves the boundary open, the CDT returns no faces, and meshing dies
            # with an unrelated-looking IndexError. Seams that tile a whole outline
            # hit this whenever two of them derive the same corner by different
            # routes (a constant height vs an interpolated crossing).
            self.edges[-1].end[:] = list(self.edges[0].start)

        self.seam_names = order
        self.sequences = sequences
        # Keep the source point-chains so fit quality can be audited: a single
        # cubic forced onto a seam that is part straight and part curved will
        # cut the corner, and comparing arc lengths catches it.
        # The chains the fit is judged against. A caller that simplifies an
        # outline before handing it over MUST pass the originals here, or the
        # check compares the panel with itself and reports 0.00 mm.
        self.source_seams = {k: np.asarray(v, float)
                             for k, v in (source or seams).items()}
        self.interfaces = {k: pyg.Interface(self, v) for k, v in sequences.items()}
        if gap > 1e-6:
            self.interfaces['_closing'] = pyg.Interface(self, self.edges[-1])

        # Kept so a caller can put `source_seams` back into the panel's own
        # frame and check the fit against them; without it the two differ by the
        # pivot and every seam reads as ~75 cm out.
        self.pivot_2d = np.zeros(2)
        if pivot is not None:
            p = np.asarray(pivot, float)
            self.pivot_2d = p.copy()
            for v in self.edges.verts():
                v[0] -= float(p[0])
                v[1] -= float(p[1])
        if translation is not None:
            self.translation = np.asarray(translation, float)
        if rotation is not None:
            self.rotate_by(rotation)

    def autonorm(self):
        """Disabled for DXF panels -- use `face_to()` instead.

        `Panel.autonorm` is called implicitly by translate_by/mirror, but its
        normal estimate degenerates to NaN on axis-aligned outlines and its
        "outward" reference is the world origin (see `face_to`). Letting it run
        would emit a divide warning on every move and never actually decide
        anything, so DXF panels declare their facing explicitly.
        """
        return self

    def seam(self, *names):
        """Interface spanning several seams, in the given order."""
        return pyg.Interface.from_multiple(*(self.interfaces[n] for n in names))


def split_chain(chain, at_length):
    """Split a point chain at a given arc length, interpolating a new point.

    Seam junctions do not always land on an existing vertex -- the front-centre
    side of the dress is a single 82 cm straight segment that the waist notch
    cuts in the middle -- so snapping to the nearest stored vertex would move
    the join by tens of centimetres.
    """
    chain = np.asarray(chain, float)
    step = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    if at_length <= 0 or at_length >= s[-1]:
        raise ValueError(f'split at {at_length} outside chain of {s[-1]:.2f}')
    i = int(np.searchsorted(s, at_length)) - 1
    t = (at_length - s[i]) / max(step[i], 1e-9)
    cut = chain[i] + t * (chain[i + 1] - chain[i])
    return (np.vstack([chain[:i + 1], cut]), np.vstack([cut, chain[i + 1:]]))


def verify_panel(panel, samples=60):
    """Check a DxfPanel's fitted edges against the DXF chains they came from.

    Returns [(seam, source_len, fitted_len, max_deviation_cm)]. The deviation is
    the largest perpendicular distance from the fitted curve to the source
    polyline, which is the number that matters: arc length can match while a
    cubic bows off a straight run, and that bowing is exactly what turns a
    drafted straight edge into a curve.
    """
    out = []
    piv = getattr(panel, 'pivot_2d', np.zeros(2))
    for name, src in getattr(panel, 'source_seams', {}).items():
        chain = np.asarray(src, float) - piv
        pts = []
        for e in panel.sequences[name]:
            curve = e.as_curve()
            pts += [[curve.point(t).real, curve.point(t).imag]
                    for t in np.linspace(0.0, 1.0, samples)]
        pts = np.asarray(pts, float)
        dev = None
        # Measured against the source AND its x-mirror, keeping whichever fits.
        # `Component.mirror()` flips a panel's vertices without touching
        # `source_seams` -- hyperdrop's sleeve does exactly that for its right
        # side -- and shape identity up to reflection is what this check is for
        # anyway, so the mirror is not something to flag.
        for flip in (1.0, -1.0):
            c = chain * [flip, 1.0]
            a, b = c[:-1], c[1:]
            ab = b - a
            l2 = (ab * ab).sum(axis=1)
            l2[l2 == 0] = 1e-12
            d = []
            for q in pts:
                t = np.clip(((q - a) * ab).sum(axis=1) / l2, 0.0, 1.0)
                d.append(np.linalg.norm(a + t[:, None] * ab - q, axis=1).min())
            if dev is None or max(d) < max(dev):
                dev = d
        src_len = float(np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1)))
        out.append((name, src_len, float(panel.sequences[name].length()),
                    float(max(dev))))
    return out


def panel_diagram(panels, path, title='', labels=None, dpi=150):
    """Readable panel/seam diagram: one panel per column, every seam labelled
    with its edge ids, name and length.

    GarmentCode's own pattern.png puts panel names at the panel centre in a fixed
    size with no contrast against the fill, which is illegible on narrow panels.
    This draws the labels outside the outline on leader lines instead, so it can
    be used to agree construction with a pattern maker.

    `panels` -- {display name: DxfPanel};  `labels` -- optional {name: caption}.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    keys = list(panels)
    wide = [max(c[:, 0].ptp() for c in panels[k].source_seams.values()) for k in keys]
    tall = max(max(c[:, 1].ptp() for c in panels[k].source_seams.values())
               for k in keys)
    fig = plt.figure(figsize=(sum(w / 8 + 2.6 for w in wide), tall / 8 + 2.2))
    gs = fig.add_gridspec(1, len(keys),
                          width_ratios=[w / 8 + 2.6 for w in wide], wspace=0.05)
    for j, k in enumerate(keys):
        ax = fig.add_subplot(gs[0, j])
        p = panels[k]
        loop = np.vstack(list(p.source_seams.values()))
        ax.add_patch(Polygon(loop, closed=True, facecolor='#eef3f7',
                             edgecolor='none', zorder=0))
        centre = loop.mean(0)
        for nm, chain in p.source_seams.items():
            ids = ','.join(str(getattr(e, 'geometric_id', '?'))
                           for e in p.sequences[nm])
            L = float(np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1)))
            ax.plot(*chain.T, '-', color='#1f4e79', lw=2.4, zorder=2,
                    solid_capstyle='round')
            m = chain[len(chain) // 2]
            v = m - centre
            v = v / max(np.linalg.norm(v), 1e-9)
            ax.annotate(f'[{ids}] {nm}\n{L:.2f} cm', m,
                        xytext=m + v * max(wide[j], 10) * 0.16,
                        fontsize=8.5, ha='center', va='center', zorder=4,
                        bbox=dict(boxstyle='round,pad=0.32', fc='white',
                                  ec='#1f4e79', lw=0.8, alpha=0.97),
                        arrowprops=dict(arrowstyle='-', color='#1f4e79', lw=0.7))
        ax.set_title((labels or {}).get(k, k), fontsize=13, fontweight='bold',
                     color='#1f4e79', pad=14)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.margins(0.42)
    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.99)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def fit_report(panel, min_len=8.0):
    """Per-seam fit quality: source arc length vs the fitted curve's.

    A cubic forced onto a mixed straight/curved seam shortcuts it, so a fitted
    length materially below the source is the signature of an under-split seam
    (the pants outseam was 4.1 cm off before it was split at its notches).
    """
    rows = []
    for key, seq in panel.sequences.items():
        src = panel.source_seams.get(key)
        if src is None or len(src) < 2:
            continue
        s_len = float(np.sum(np.linalg.norm(np.diff(src, axis=0), axis=1)))
        if s_len < min_len:
            continue
        f_len = 0.0
        for e in seq:
            lin = e.linearize(150)
            pts = np.asarray(lin.verts() if hasattr(lin, 'verts')
                             else [e.start, e.end], float)
            f_len += float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        rows.append((key, s_len, f_len, f_len - s_len, len(seq)))
    return sorted(rows, key=lambda r: r[3])


def split_at_points(chain, points, tol=3.0, min_seg=1.0):
    """Split a seam chain wherever a marked point falls on it.

    AAMA notches are the pattern maker's own correspondence marks and are stored
    slightly OUTSIDE the cut line, so each is projected to its nearest position
    along the chain and the chain is cut there. Both sides of a seam carry the
    same notches, which is what makes the resulting sub-seams correspond.

    Fitting a long seam as one cubic is what makes this necessary: the pants
    outseam is dead straight for 78 cm and then curves over the hip, and a single
    cubic spanning both bows 4.1 cm where the source bows 2.2. Splitting at the
    notches lets the straight run stay straight.

    `tol` -- ignore points further than this from the chain (cm).
    `min_seg` -- do not create sub-segments shorter than this (cm).
    """
    chain = np.asarray(chain, float)
    if len(points) == 0:
        return [chain]
    step = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    # Project onto the POLYLINE, not onto its vertices: these chains are sparse
    # (the whole 78 cm straight run of the outseam is one segment), so snapping
    # to the nearest stored vertex would move the cut by tens of centimetres.
    seg = np.diff(chain, axis=0)
    seg_len2 = np.maximum(np.sum(seg * seg, axis=1), 1e-12)
    cuts = []
    for p in np.asarray(points, float).reshape(-1, 2):
        t = np.clip(np.sum((p - chain[:-1]) * seg, axis=1) / seg_len2, 0.0, 1.0)
        foot = chain[:-1] + t[:, None] * seg
        d = np.linalg.norm(foot - p, axis=1)
        i = int(np.argmin(d))
        if d[i] > tol:
            continue
        at = s[i] + t[i] * np.sqrt(seg_len2[i])
        if min_seg < at < s[-1] - min_seg:
            cuts.append(float(at))
    cuts = sorted(set(round(c, 6) for c in cuts))
    out, rest, base = [], chain, 0.0
    for c in cuts:
        if c - base < min_seg:
            continue
        head, rest = split_chain(rest, c - base)
        out.append(head)
        base = c
    out.append(rest)
    return out


def place_around(panel, theta_deg, y, rx, rz):
    """Place a panel tangent to an ellipse around the body's vertical axis.

    theta is measured from centre front (0) toward the wearer's left, so 90 is
    the side seam and 180 centre back. The panel is rotated to face outward and
    pushed out to the ellipse, which is the natural way to lay out a garment
    with many vertical panels: every panel starts roughly where it will end up,
    so zero-gravity stitching only has to close small gaps.
    """
    th = np.radians(theta_deg)
    outward = np.array([np.sin(th), 0.0, np.cos(th)])
    panel.rotate_by(R.from_euler('XYZ', [0, theta_deg, 0], degrees=True))
    panel.translate_to([rx * np.sin(th), y, rz * np.cos(th)])
    return face_to(panel, outward)


def face_to(panel, direction):
    """Reverse the panel's edge loop if its normal opposes `direction`.

    Replaces `Panel.autonorm()` for DXF panels. autonorm derives the normal
    from the 2D bbox corners, but `EdgeSequence.bbox()` returns them in an order
    that pairs DIAGONAL corners for an axis-aligned rectangle -- the cross
    product then vanishes and the normal comes out NaN, so autonorm silently
    does nothing. It also measures "outward" against the world origin, which is
    the floor centre, so for anything worn high on the body the vertical term
    dominates and the test is meaningless. Naming the outward direction per
    panel avoids both problems.
    """
    loop = np.array([v for e in panel.edges for v in (e.start,)], float)
    area = 0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1)
                        - np.roll(loop[:, 0], -1) * loop[:, 1])
    local_n = np.array([0.0, 0.0, 1.0 if area > 0 else -1.0])
    if np.dot(panel.rotation.apply(local_n), np.asarray(direction, float)) < 0:
        panel.edges.reverse()
    return panel


def seams_from_split(loop, key_idx, names):
    """Convenience: split `loop` at `key_idx` and label the arcs.

    `key_idx` is used in ascending order, so `names` must list the seams in
    outline order starting at the lowest index.
    """
    chains = split_arcs(loop, key_idx)
    if len(chains) != len(names):
        raise ValueError(f'{len(chains)} arcs but {len(names)} names')
    return dict(zip(names, chains))


def describe(loop, thr=25.0, window=2.0):
    """Debug helper: report detected corners with position and turn angle."""
    ang, _ = turn_angles(loop)
    out = []
    for i in corners(loop, thr=thr, window=window):
        out.append((i, float(loop[i][0]), float(loop[i][1]), float(ang[i])))
    return out


# --------------------------------------------------------------------------- #
#  Inspection CLI
# --------------------------------------------------------------------------- #
def _preview(pieces, path, title):
    """Upright preview of the extracted pieces (matplotlib imported lazily so
    the reader itself stays dependency-light)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(pieces)
    if not n:
        return
    cols = min(5, n)
    fig, axs = plt.subplots((n + cols - 1) // cols, cols,
                            figsize=(3.6 * cols, 4.4 * ((n + cols - 1) // cols)),
                            squeeze=False)
    axs = axs.ravel()
    for ax, p in zip(axs, pieces):
        if p.cut is not None:
            ax.plot(*np.vstack([p.cut, p.cut[:1]]).T, color='0.7', lw=0.8)
        b = p.boundary
        ax.plot(*np.vstack([b, b[:1]]).T, 'C0-', lw=1.3)
        for layer, seg in p.internal:
            ax.plot(*seg.T, color='C3' if layer == L_INTERNAL else 'C2', lw=0.6)
        if len(p.notches):
            ax.plot(*p.notches.T, 'kv', ms=3)
        w, h = p.size_cm()
        ax.set_title(f'{p.label}\n[{p.kind}] {w:.1f}x{h:.1f} cm\n'
                     f'{"sew" if p.has_sew_line else "CUT ONLY"}, n={len(b)}'
                     f'{", pair" if p.mirrored else ""}', fontsize=7)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
    for ax in axs[n:]:
        ax.axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def main():
    """Dump + preview the pieces of every DXF under a directory tree."""
    import argparse
    import glob
    import json
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='./hyperdrop_data')
    ap.add_argument('--size', default='M')
    ap.add_argument('--all-pieces', action='store_true',
                    help='keep interfacing/lining/template pieces too')
    # Inspection output is regenerable from the DXFs -- keep it out of the repo
    ap.add_argument('--out', default='/tmp/hyperdrop_extract')
    args = ap.parse_args()

    for dxf in sorted(glob.glob(os.path.join(args.data, '*', '*.dxf'))):
        garment = os.path.basename(os.path.dirname(dxf))
        avail = sizes_in(dxf)
        size = args.size if args.size in avail else (avail[0] if avail else None)
        pieces = read_pieces(dxf, size=size)
        if not args.all_pieces:
            pieces = dedupe_identical([p for p in pieces if p.kind == 'fabric'])

        odir = os.path.join(args.out, garment)
        os.makedirs(odir, exist_ok=True)
        _preview(pieces, os.path.join(odir, 'pieces.png'),
                 f'{garment}  |  {os.path.basename(dxf)}  |  size {size}')
        with open(os.path.join(odir, 'pieces.json'), 'w') as f:
            json.dump(dict(
                garment=garment, dxf=dxf, size=size, sizes_available=avail,
                pieces=[dict(
                    block=p.block, name=p.name, label=p.label, kind=p.kind,
                    quantity=p.quantity, category=p.category, fabric=p.fabric,
                    annotation=p.annotation, mirrored=p.mirrored,
                    grain_deg=p.grain_deg, has_sew_line=p.has_sew_line,
                    width_cm=round(p.size_cm()[0], 2),
                    height_cm=round(p.size_cm()[1], 2),
                    perimeter_cm=round(p.perimeter(), 2),
                    boundary=np.round(p.boundary, 3).tolist(),
                    cut=None if p.cut is None else np.round(p.cut, 3).tolist(),
                    notches=np.round(p.notches, 3).tolist(),
                    internal=[[l, np.round(v, 3).tolist()] for l, v in p.internal],
                ) for p in pieces]), f)

        print(f'\n{garment}  ({os.path.basename(dxf)})  sizes={avail} -> {size}')
        for p in pieces:
            w, h = p.size_cm()
            print(f'   {p.label:<44s} {p.kind:<11s} {w:6.1f} x {h:6.1f} cm  '
                  f'perim {p.perimeter():6.1f}  n={len(p.boundary):4d}  '
                  f'{"sew" if p.has_sew_line else "CUT"}'
                  f'{"  pair" if p.mirrored else ""}')
        print(f'   -> {odir}')


if __name__ == '__main__':
    main()
