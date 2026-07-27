"""
Custom Pants Generation & Simulation Pipeline
Uses the BUILT-IN GarmentCode parametric system (MetaGarment + Pants + StraightWB).

Production data: HM Wide High Waist Jeans
Body: global_women_size36_apose.obj
Simulation body mesh: global_women_size36_apose.obj (custom SMPL, 168cm)

Measurements from techpack:
  - Waist: "Waist top of WB"
  - Low Hip: "Seat"
  - Inseam: "Inseam"
  - Thigh: "Thigh"
  - Knee: "Knee"
  - Ankle: "Bottom leg"
  - Front rise: "Front rise"
  - Back rise: "Back rise"
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

from pathlib import Path
from datetime import datetime
import json
import xml.etree.ElementTree as ET

import numpy as np
import yaml
import cairosvg
import svgpathtools as svgpath
from scipy.spatial.transform import Rotation as Rotation3D

import pygarment as pyg
from pygarment.data_config import Properties
from production_to_design import ProductionToDesign


def normalize_body_mesh(obj_path):
    """Ensure body mesh has feet at Y=0.

    The simulation shifts both body and cloth by the same Y offset to place
    the body on the ground. If the body mesh doesn't start at Y=0, the cloth
    ends up displaced from the body. This normalizes the mesh in-place.
    """
    import trimesh
    obj_path = Path(obj_path)
    if not obj_path.exists():
        return
    mesh = trimesh.load(str(obj_path), process=False)
    min_y = mesh.vertices[:, 1].min()
    if abs(min_y) < 0.001:
        return
    print(f'  Normalizing body mesh {obj_path.name}: shifting Y by {-min_y:.4f}')
    mesh.vertices[:, 1] -= min_y
    mesh.export(str(obj_path))


# ============================================================
# Production measurements per garment size (full circumference, cm)
# ============================================================
GARMENT_SIZES = [32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54]
SIZES_TO_RUN  = [48, 50, 52]   # subset to generate/simulate; set to GARMENT_SIZES for all

PRODUCTION_DATA = {
    32: {
        'Waist': 64.8,
        'Low_Hip': 86.5,
        'Inseam': 76.9,
        'Thigh': 54.5,
        'Knee': 46.8,
        'Ankle': 51,
        'Front Rise': 22.6,
        'Back Rise': 30.2,
        'Hip_from_waist': 13.9,   # bottom of waistband → hip line (cm)
        'Knee_from_crotch': 32.1, # crotch (= thigh line) → knee line (cm)
    },
    34: {
        'Waist': 67.8,
        'Low_Hip': 90.5,
        'Inseam': 77.6,
        'Thigh': 57.3,
        'Knee': 48.7,
        'Ankle': 52.5,
        'Front Rise': 23.3,
        'Back Rise': 31.7,
        'Hip_from_waist': 14.1,
        'Knee_from_crotch': 32.4,
    },
    36: {
        'Waist': 71.5,
        'Low_Hip': 94.5,
        'Inseam': 78.3,
        'Thigh': 60.1,
        'Knee': 50.6,
        'Ankle': 54,
        'Front Rise': 24,
        'Back Rise': 33.2,
        'Hip_from_waist': 14.3,
        'Knee_from_crotch': 32.7,
    },
    38: {
        'Waist': 76,
        'Low_Hip': 98.5,
        'Inseam': 79,
        'Thigh': 62.8,
        'Knee': 52.5,
        'Ankle': 55.5,
        'Front Rise': 24.7,
        'Back Rise': 34.7,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 33.0,
    },
    40: {
        'Waist': 79.8,
        'Low_Hip': 101.5,
        'Inseam': 79,
        'Thigh': 64.8,
        'Knee': 53.9,
        'Ankle': 56.4,
        'Front Rise': 25.2,
        'Back Rise': 35.8,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 33.0,
    },
    42: {
        'Waist': 83.6,
        'Low_Hip': 104.5,
        'Inseam': 79,
        'Thigh': 66.8,
        'Knee': 55.3,
        'Ankle': 57.1,
        'Front Rise': 25.7,
        'Back Rise': 36.9,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 33.0,
    },
    44: {
        'Waist': 87.5,
        'Low_Hip': 107.5,
        'Inseam': 78.5,
        'Thigh': 68.8,
        'Knee': 56.7,
        'Ankle': 57.8,
        'Front Rise': 26.2,
        'Back Rise': 38,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 32.5,
    },
    46: {
        'Waist': 93.2,
        'Low_Hip': 110.5,
        'Inseam': 78,
        'Thigh': 70.8,
        'Knee': 57.9,
        'Ankle': 58.6,
        'Front Rise': 26.7,
        'Back Rise': 39.1,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 32.0,
    },
    48: {
        'Waist': 98.9,
        'Low_Hip': 115.1,
        'Inseam': 77.3,
        'Thigh': 73.5,
        'Knee': 58.9,
        'Ankle': 59.4,
        'Front Rise': 27.6,
        'Back Rise': 40.2,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 31.3,
    },
    50: {
        'Waist': 104.6,
        'Low_Hip': 119.7,
        'Inseam': 76.6,
        'Thigh': 76.1,
        'Knee': 59.9,
        'Ankle': 60.2,
        'Front Rise': 28.5,
        'Back Rise': 41.3,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 30.6,
    },
    52: {
        'Waist': 110.3,
        'Low_Hip': 124.3,
        'Inseam': 75.8,
        'Thigh': 78.8,
        'Knee': 61.5,
        'Ankle': 61,
        'Front Rise': 29.4,
        'Back Rise': 42.4,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 29.9,
    },
    54: {
        'Waist': 116,
        'Low_Hip': 128.9,
        'Inseam': 75,
        'Thigh': 81.4,
        'Knee': 63.1,
        'Ankle': 61.8,
        'Front Rise': 30.3,
        'Back Rise': 43.6,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 29.2,
    },
    56: {
        'Waist': 121.7,
        'Low_Hip': 133.5,
        'Inseam': 74.3,
        'Thigh': 83.9,
        'Knee': 64.7,
        'Ankle': 62.7,
        'Front Rise': 31.2,
        'Back Rise': 44.8,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 28.5,
    },
}

BODY_YAML = './assets/bodies/global_women_size36_apose.yaml'
BODY_NAME = 'global_women_size36_apose'


def _find_panel_edge(panel, stitches, role):
    """
    Find edge index in a panel by structural role.
    role:
      'wb_top'    – free edge at max-y (waistband top)
      'pant_hem'  – free edge at min-y (hem / ankle)
      'inseam'    – (unused; inseam edge found dynamically in annotate_pattern_svg)
      'f_rise'    – stitched to pant_b_l e11  (always e1 for pant_f_l)
      'b_rise'    – stitched to pant_f_l e1   (always e11 for pant_b_l)
    """
    panel_name = None
    for pn, pd in {'wb_front': panel, 'pant_f_l': panel, 'pant_b_l': panel}.items():
        pass  # just duck-type by using panel dict directly

    verts = panel['vertices']
    edges = panel['edges']
    ys = [v[1] for v in verts]
    min_y, max_y = min(ys), max(ys)

    # Build stitched set for this panel (passed via stitches list)
    stitched = {}  # edge_idx -> partner (panel_name, edge_idx)
    for s in stitches:
        if len(s) == 2:
            for a, b in [(s[0], s[1]), (s[1], s[0])]:
                stitched[(a['panel'], a['edge'])] = (b['panel'], b['edge'])

    if role == 'wb_top':
        # Free edge with both endpoints at max y
        for i, e in enumerate(edges):
            ep = e['endpoints']
            if (abs(verts[ep[0]][1] - max_y) < 0.5 and
                    abs(verts[ep[1]][1] - max_y) < 0.5):
                return i
        # Fallback: any free edge not at y=0
        for panel_name_key in ['wb_front', 'wb_back']:
            for i, e in enumerate(edges):
                ep = e['endpoints']
                if (abs(verts[ep[0]][1] - max_y) < 1.0 or
                        abs(verts[ep[1]][1] - max_y) < 1.0):
                    return i

    elif role == 'pant_hem':
        # Free edge with both endpoints at min y
        for i, e in enumerate(edges):
            ep = e['endpoints']
            if (abs(verts[ep[0]][1] - min_y) < 0.5 and
                    abs(verts[ep[1]][1] - min_y) < 0.5):
                return i

    elif role == 'f_rise':
        # Edge stitched to pant_b_l, with NO endpoint at min_y
        for i, e in enumerate(edges):
            partner = stitched.get(('pant_f_l', i))
            if partner and partner[0] == 'pant_b_l':
                ep = e['endpoints']
                if (abs(verts[ep[0]][1] - min_y) > 1.0 and
                        abs(verts[ep[1]][1] - min_y) > 1.0):
                    return i

    elif role == 'b_rise':
        # Edge on pant_b_l stitched to pant_f_l, with NO endpoint at min_y
        for i, e in enumerate(edges):
            partner = stitched.get(('pant_b_l', i))
            if partner and partner[0] == 'pant_f_l':
                ep = e['endpoints']
                if (abs(verts[ep[0]][1] - min_y) > 1.0 and
                        abs(verts[ep[1]][1] - min_y) > 1.0):
                    return i

    elif role == 'br_inner_rise':
        # Edge on pant_b_r stitched to pant_f_r, with NO endpoint at min_y
        for i, e in enumerate(edges):
            partner = stitched.get(('pant_b_r', i))
            if partner and partner[0] == 'pant_f_r':
                ep = e['endpoints']
                if (abs(verts[ep[0]][1] - min_y) > 1.0 and
                        abs(verts[ep[1]][1] - min_y) > 1.0):
                    return i

    return None


def _svg_line(parent, ns, p1, p2, color, width):
    line = ET.SubElement(parent, f'{{{ns}}}line')
    line.set('x1', f'{p1[0]:.3f}')
    line.set('y1', f'{p1[1]:.3f}')
    line.set('x2', f'{p2[0]:.3f}')
    line.set('y2', f'{p2[1]:.3f}')
    line.set('stroke', color)
    line.set('stroke-width', width)


def _panel_verts_to_svg(panel):
    """Transform panel local vertices to SVG coordinate space."""
    verts = np.array(panel['vertices'], dtype=float).copy()
    t = np.array(panel['translation'][:2], dtype=float)
    verts[:, 1] *= -1
    t[1] *= -1
    offset = verts.min(axis=0)
    verts -= offset
    t += offset
    rot3d = Rotation3D.from_euler('XYZ', panel['rotation'], degrees=True)
    res = rot3d.apply([0, 1, 0])
    v2 = res[:2]
    v2_len = np.linalg.norm(v2)
    if v2_len > 1e-6:
        v2 /= v2_len
    cos_a = float(np.clip(np.dot([0.0, 1.0], v2), -1.0, 1.0))
    cross = float(np.cross([0.0, 1.0], v2))
    flat_rot = -float(np.arccos(cos_a) * np.sign(cross) if abs(cross) > 1e-5 else 0.0)
    flat_rot = np.rad2deg(flat_rot)
    if abs(flat_rot) > 0.01:
        origin = verts[0].copy()
        c, s = np.cos(np.deg2rad(flat_rot)), np.sin(np.deg2rad(flat_rot))
        R_mat = np.array([[c, -s], [s, c]])
        verts = (R_mat @ (verts - origin).T).T + origin
    verts += t
    return verts


def _local_to_svg(panel, local_pt):
    """Convert a single local (x, y) panel point to SVG space."""
    verts_all = np.array(panel['vertices'], dtype=float).copy()
    pt = np.array([[local_pt[0], local_pt[1]]], dtype=float)
    t = np.array(panel['translation'][:2], dtype=float)
    # Flip Y
    verts_all[:, 1] *= -1
    pt[:, 1] *= -1
    t[1] *= -1
    offset = verts_all.min(axis=0)
    pt -= offset
    t += offset
    # Rotation
    rot3d = Rotation3D.from_euler('XYZ', panel['rotation'], degrees=True)
    res = rot3d.apply([0, 1, 0])
    v2 = res[:2]
    v2_len = np.linalg.norm(v2)
    if v2_len > 1e-6:
        v2 /= v2_len
    cos_a = float(np.clip(np.dot([0.0, 1.0], v2), -1.0, 1.0))
    cross = float(np.cross([0.0, 1.0], v2))
    flat_rot = -float(np.arccos(cos_a) * np.sign(cross) if abs(cross) > 1e-5 else 0.0)
    flat_rot = np.rad2deg(flat_rot)
    if abs(flat_rot) > 0.01:
        verts_for_origin = verts_all.copy()
        verts_for_origin -= offset
        origin = verts_for_origin[0]
        c, s = np.cos(np.deg2rad(flat_rot)), np.sin(np.deg2rad(flat_rot))
        R_mat = np.array([[c, -s], [s, c]])
        pt = (R_mat @ (pt - origin).T).T + origin
    pt += t
    return pt[0]


def _sample_edge_pts(v1, v2, edge, n=200):
    """Return n+1 points sampled along an edge (straight or curved)."""
    if 'curvature' not in edge:
        return np.array([v1 + (v2 - v1) * t for t in np.linspace(0, 1, n + 1)])
    curv = edge['curvature']
    ctype = curv['type']
    ts = np.linspace(0, 1, n + 1)
    if ctype == 'quadratic':
        cp = _rel_to_abs(v1, v2, curv['params'][0])
        return np.array([
            (1 - t)**2 * v1 + 2 * (1 - t) * t * cp + t**2 * v2
            for t in ts
        ])
    elif ctype == 'cubic':
        cp1 = _rel_to_abs(v1, v2, curv['params'][0])
        cp2 = _rel_to_abs(v1, v2, curv['params'][1])
        return np.array([
            (1 - t)**3 * v1 + 3*(1-t)**2*t * cp1 + 3*(1-t)*t**2 * cp2 + t**3 * v2
            for t in ts
        ])
    else:  # circle or unknown — sample linearly as approximation
        return np.array([v1 + (v2 - v1) * t for t in ts])


def _panel_x_range_at_y(panel, y_local):
    """Return (x_min, x_max) of the panel boundary at y=y_local (local space)."""
    verts = np.array(panel['vertices'], dtype=float)
    xs = []
    for edge in panel['edges']:
        ep = edge['endpoints']
        v1, v2 = verts[ep[0]], verts[ep[1]]
        pts = _sample_edge_pts(v1, v2, edge)
        for i in range(len(pts) - 1):
            ya, yb = pts[i][1], pts[i + 1][1]
            if min(ya, yb) <= y_local <= max(ya, yb) and abs(yb - ya) > 1e-9:
                t = (y_local - ya) / (yb - ya)
                xs.append(pts[i][0] + t * (pts[i + 1][0] - pts[i][0]))
    return (min(xs), max(xs)) if len(xs) >= 2 else None


def _cross_section_circumference(panels, y_local, above_crotch=False):
    """
    Measure garment circumference at height y_local (local panel coords, y=0 at hem).

    Below crotch (thigh, knee, ankle):
      One leg circumference = pant_f_l width + pant_b_l width at y.

    Above crotch (hip, waist):
      Full garment circumference = 2 × (pant_f_l + pant_b_l) width at y,
      because both left and right panels (mirror images) contribute equally.
    """
    total = 0.0
    for pname in ('pant_f_l', 'pant_b_l'):
        panel = panels.get(pname)
        if panel is None:
            continue
        xr = _panel_x_range_at_y(panel, y_local)
        if xr is not None:
            total += xr[1] - xr[0]
    return total * (2.0 if above_crotch else 1.0)


def _rise_seam_arc_length(panels, stitches, panel_name, partner_name):
    """Total arc length of the rise seam on panel_name stitched to partner_name."""
    stitched = {}
    for s in stitches:
        if len(s) == 2:
            for a, b in [(s[0], s[1]), (s[1], s[0])]:
                stitched[(a['panel'], a['edge'])] = (b['panel'], b['edge'])
    panel = panels[panel_name]
    verts = panel['vertices']
    min_y = min(v[1] for v in verts)
    total = 0.0
    for i, e in enumerate(panel['edges']):
        partner = stitched.get((panel_name, i))
        if partner and partner[0] == partner_name:
            ep = e['endpoints']
            if (abs(verts[ep[0]][1] - min_y) > 1.0 and
                    abs(verts[ep[1]][1] - min_y) > 1.0):
                total += edge_length(e, verts)
    return total


def _find_panel_rise_seam_endpoints(panels, stitches, panel_name, partner_name):
    """
    Find the top and bottom LOCAL-space points of the rise seam on `panel_name`
    that is stitched to `partner_name`, excluding edges that touch the hem (min_y).

    Used to span F.Rise (pant_f_l ↔ pant_f_r, center-front seam)
    and B.Rise (pant_b_l ↔ pant_b_r, center-back seam).

    Returns (top_local, bottom_local) as np.float64 arrays, or (None, None).
    """
    stitched = {}
    for s in stitches:
        if len(s) == 2:
            for a, b in [(s[0], s[1]), (s[1], s[0])]:
                stitched[(a['panel'], a['edge'])] = (b['panel'], b['edge'])

    panel = panels[panel_name]
    verts = np.array(panel['vertices'], dtype=float)
    min_y = verts[:, 1].min()

    pts = []
    for i, e in enumerate(panel['edges']):
        partner = stitched.get((panel_name, i))
        if partner and partner[0] == partner_name:
            ep = e['endpoints']
            if (abs(verts[ep[0]][1] - min_y) > 1.0 and
                    abs(verts[ep[1]][1] - min_y) > 1.0):
                pts.extend([verts[ep[0]], verts[ep[1]]])

    if not pts:
        return None, None
    pts = np.array(pts)
    top    = pts[pts[:, 1].argmax()]   # highest local y → waist area
    bottom = pts[pts[:, 1].argmin()]   # lowest  local y → crotch extension tip
    return top, bottom


def _compute_back_shift(panels):
    """Compute the x-shift applied to back panels in the SVG."""
    front_max_x = -np.inf
    back_min_x = np.inf
    for p in panels.values():
        sv = _panel_verts_to_svg(p)
        if p['translation'][2] >= 0:
            front_max_x = max(front_max_x, sv[:, 0].max())
        else:
            back_min_x = min(back_min_x, sv[:, 0].min())
    if front_max_x == -np.inf or back_min_x == np.inf:
        return 0.0
    return front_max_x - back_min_x + 10.0


def _draw_dim_annotation(root, ns, p1, p2, label, offset_dist, fontsize, color,
                         centroid=None, forced_normal=None):
    """Draw a dimension line annotation between two SVG points."""
    direction = p2 - p1
    dlen = np.linalg.norm(direction)
    if dlen < 0.1:
        return
    dir_unit = direction / dlen
    normal_ccw = np.array([-dir_unit[1], dir_unit[0]])
    mid = (p1 + p2) / 2

    if forced_normal is not None:
        normal = np.array(forced_normal, dtype=float)
        n_len = np.linalg.norm(normal)
        if n_len > 1e-9:
            normal /= n_len
    elif centroid is not None:
        outward = mid - centroid
        normal = normal_ccw if np.dot(normal_ccw, outward) >= 0 else -normal_ccw
    else:
        normal = normal_ccw

    a1 = p1 + normal * offset_dist
    a2 = p2 + normal * offset_dist
    a_mid = mid + normal * offset_dist
    tick = 2.5

    g = ET.SubElement(root, f'{{{ns}}}g')
    _svg_line(g, ns, a1, a2, color, '0.3')
    for ap in [a1, a2]:
        _svg_line(g, ns, ap - dir_unit * tick / 2, ap + dir_unit * tick / 2, color, '0.3')

    angle_deg = np.rad2deg(np.arctan2(direction[1], direction[0]))
    if angle_deg > 90 or angle_deg < -90:
        angle_deg += 180

    text = ET.SubElement(g, f'{{{ns}}}text')
    text.set('x', f'{a_mid[0]:.3f}')
    text.set('y', f'{a_mid[1]:.3f}')
    text.set('fill', color)
    text.set('font-size', str(fontsize))
    text.set('font-weight', 'bold')
    text.set('text-anchor', 'middle')
    text.set('dominant-baseline', 'middle')
    text.set('transform', f'rotate({angle_deg:.2f},{a_mid[0]:.3f},{a_mid[1]:.3f})')
    text.text = label


def annotate_pattern_svg(svg_path, spec_path, measurements, offset_dist=4.0, fontsize=3.5):
    """
    Add dimension line annotations for garment measurements to the pattern SVG.

    Edge annotations (production targets):
      wb_front top edge              → Waist
      pant_f_l 'left' edge (inseam) → Inseam (left side of front panel)
      pant_f_l hem edge             → Ankle (right side)
      pant_f_l↔pant_f_r seam chain → F.Rise (left side of front panel)
      pant_b_l↔pant_b_r seam chain → B.Rise (right side of back panel)
        Both rise annotations span from the waist-seam inner corner to the
        crotch-extension tip, matching the actual arc path and measurement point.

    Cross-section annotations (ACTUAL measured values from Bézier geometry):
      y = fl_max_y − Hip_from_waist → Hip  (×2, both legs)
      y = crotch_y                  → Thigh (one leg)
      y = crotch_y − Knee_from_crotch → Knee (one leg)
    """
    with open(spec_path) as f:
        spec = json.load(f)
    panels = spec['pattern']['panels']

    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    ET.register_namespace('ev', 'http://www.w3.org/2001/xml-events')
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    ns = 'http://www.w3.org/2000/svg'
    color = 'rgb(0,100,160)'

    # Expand SVG viewBox to accommodate annotations
    margin = offset_dist + fontsize + 3
    vb = root.get('viewBox').split()
    vb_x, vb_y, vb_w, vb_h = float(vb[0]), float(vb[1]), float(vb[2]), float(vb[3])
    vb_x -= margin;  vb_y -= margin;  vb_w += 2 * margin;  vb_h += 2 * margin
    root.set('viewBox', f'{vb_x} {vb_y} {vb_w} {vb_h}')
    root.set('width', f'{vb_w}cm')
    root.set('height', f'{vb_h}cm')

    back_shift = _compute_back_shift(panels)
    stitches = spec['pattern']['stitches']

    # ── Resolve edge indices dynamically (dart count varies by size) ─────────
    wb_top_idx = _find_panel_edge(panels['wb_front'], stitches, 'wb_top')
    hem_idx    = _find_panel_edge(panels['pant_f_l'], stitches, 'pant_hem')

    # Find inseam edge(s) on pant_f_l: the 'inside' interface in pants.py.
    # With subcurves this can be 3 edges (crotch→thigh, thigh→knee, knee→ankle).
    # Inseam edges are on the RIGHT side (high X); outseam on the LEFT (low X).
    _stitched_map = {}
    for _s in stitches:
        if len(_s) == 2:
            for _a, _b in [(_s[0], _s[1]), (_s[1], _s[0])]:
                _stitched_map[(_a['panel'], _a['edge'])] = (_b['panel'], _b['edge'])
    _fl = panels['pant_f_l']
    _fl_verts = _fl['vertices']
    _fl_centroid_x = np.mean([v[0] for v in _fl_verts])
    _inseam_edge_indices = []
    for _i, _e in enumerate(_fl['edges']):
        _partner = _stitched_map.get(('pant_f_l', _i))
        if _partner and _partner[0] == 'pant_b_l':
            _ep = _e['endpoints']
            _v1, _v2 = _fl_verts[_ep[0]], _fl_verts[_ep[1]]
            _mean_x = (_v1[0] + _v2[0]) / 2
            _dy = abs(_v1[1] - _v2[1])
            if _mean_x > _fl_centroid_x and _dy > 1.0:
                _inseam_edge_indices.append(_i)
    inseam_edge_idx = _inseam_edge_indices[0] if _inseam_edge_indices else 0

    # ── Waist / Inseam / Ankle edge annotations ──────────────────────────────
    # Waist and Ankle are annotated on BOTH front and back panels with actual
    # edge lengths (front ≠ back). Inseam is identical on both sides (stitched
    # edges always match), so only annotated once on pant_f_l.
    edge_annots = []

    # Waist: wb_front and wb_back top edges
    for wb_name in ('wb_front', 'wb_back'):
        wb_panel = panels.get(wb_name)
        if wb_panel is None:
            continue
        wb_idx = _find_panel_edge(wb_panel, stitches, 'wb_top')
        if wb_idx is not None:
            w = edge_length(wb_panel['edges'][wb_idx], wb_panel['vertices'])
            edge_annots.append((wb_name, wb_idx, f"Waist: {w:.1f}cm ({2*w:.1f}cm)"))

    # Inseam: annotation spanning full inseam chain on pant_f_l.
    # With subcurves this is multiple edges; draw from ankle to crotch endpoints.
    if len(_inseam_edge_indices) == 1:
        edge_annots.append(('pant_f_l', _inseam_edge_indices[0],
                            f"Inseam: {measurements['Inseam']:.1f}cm"))
    elif _inseam_edge_indices:
        # Find the overall span: lowest and highest Y inseam vertices
        _inseam_pts_y = []
        for _idx in _inseam_edge_indices:
            _ep = _fl['edges'][_idx]['endpoints']
            _inseam_pts_y.append((_fl_verts[_ep[0]][1], _ep[0]))
            _inseam_pts_y.append((_fl_verts[_ep[1]][1], _ep[1]))
        _bottom_vi = min(_inseam_pts_y, key=lambda x: x[0])[1]
        _top_vi = max(_inseam_pts_y, key=lambda x: x[0])[1]
        # Draw directly (bypassing edge_annots loop which expects single edge idx)
        _fl_svg_verts = _panel_verts_to_svg(_fl)
        _p1_is = _fl_svg_verts[_bottom_vi]
        _p2_is = _fl_svg_verts[_top_vi]
        _fl_centroid = _fl_svg_verts.mean(axis=0)
        _draw_dim_annotation(root, ns, _p1_is, _p2_is,
                             f"Inseam: {measurements['Inseam']:.1f}cm",
                             offset_dist, fontsize, color, centroid=_fl_centroid)

    # Ankle: pant_f_l and pant_b_l hem edges
    hem_b_idx = _find_panel_edge(panels['pant_b_l'], stitches, 'pant_hem')
    for hem_panel_name, h_idx in (('pant_f_l', hem_idx), ('pant_b_l', hem_b_idx)):
        if h_idx is None:
            continue
        p = panels[hem_panel_name]
        a = edge_length(p['edges'][h_idx], p['vertices'])
        edge_annots.append((hem_panel_name, h_idx, f"Ankle: {a:.1f}cm ({2*a:.1f}cm)"))

    for panel_name, edge_idx, label in edge_annots:
        panel = panels.get(panel_name)
        if panel is None:
            continue
        is_back = panel['translation'][2] < 0
        sx = back_shift if is_back else 0.0
        verts = _panel_verts_to_svg(panel)
        verts[:, 0] += sx
        edge = panel['edges'][edge_idx]
        i, j = edge['endpoints']
        p1, p2 = verts[i], verts[j]
        centroid = verts.mean(axis=0)
        _draw_dim_annotation(root, ns, p1, p2, label, offset_dist, fontsize, color,
                             centroid=centroid)

    # ── F.Rise: center-front seam (pant_f_l ↔ pant_f_r) ────────────────────
    # Spans from the waist-seam inner corner down to the crotch-extension tip.
    # Placed to the LEFT of the front panel.
    # Label shows ACTUAL arc length, not production target.
    fr_top, fr_bot = _find_panel_rise_seam_endpoints(
        panels, stitches, 'pant_f_l', 'pant_f_r')
    if fr_top is not None:
        actual_f_rise = _rise_seam_arc_length(panels, stitches, 'pant_f_l', 'pant_f_r')
        p1_svg = _local_to_svg(panels['pant_f_l'], fr_top)
        p2_svg = _local_to_svg(panels['pant_f_l'], fr_bot)
        fl_centroid = _panel_verts_to_svg(panels['pant_f_l']).mean(axis=0)
        _draw_dim_annotation(root, ns, p1_svg, p2_svg,
                             f"F.Rise: {actual_f_rise:.1f}cm",
                             offset_dist, fontsize, color, centroid=fl_centroid)

    # ── B.Rise: center-back seam (pant_b_l ↔ pant_b_r) ─────────────────────
    # Spans from the waist-seam inner corner down to the crotch-extension tip.
    # Placed to the RIGHT of the back panel.
    # Label shows ACTUAL arc length, not production target.
    br_top, br_bot = _find_panel_rise_seam_endpoints(
        panels, stitches, 'pant_b_l', 'pant_b_r')
    if br_top is not None:
        actual_b_rise = _rise_seam_arc_length(panels, stitches, 'pant_b_l', 'pant_b_r')
        p1_svg = _local_to_svg(panels['pant_b_l'], br_top)
        p1_svg[0] += back_shift
        p2_svg = _local_to_svg(panels['pant_b_l'], br_bot)
        p2_svg[0] += back_shift
        bl_verts_svg = _panel_verts_to_svg(panels['pant_b_l']).copy()
        bl_verts_svg[:, 0] += back_shift
        _draw_dim_annotation(root, ns, p1_svg, p2_svg,
                             f"B.Rise: {actual_b_rise:.1f}cm",
                             offset_dist, fontsize, color,
                                 centroid=bl_verts_svg.mean(axis=0))

    # ── Cross-section annotations (Hip, Thigh, Knee) on BOTH panels ─────────
    # Each panel is annotated with ITS OWN width at that height.
    # (Showing the total circumference on a single panel's span would be misleading.)
    #
    # Crotch height = U bottom (tip of center-front seam, fr_bot[1] = 78.3cm).
    fl_panel = panels.get('pant_f_l')
    section_heights = []   # list of (y_local, label_prefix)
    if fl_panel is not None:
        fl_verts_loc = np.array(fl_panel['vertices'], dtype=float)
        fl_min_y = fl_verts_loc[:, 1].min()
        fl_max_y = fl_verts_loc[:, 1].max()
        if fr_bot is not None:
            crotch_y = float(fr_bot[1])
        else:
            e0 = fl_panel['edges'][0]
            ep0 = e0['endpoints']
            crotch_y = max(fl_verts_loc[ep0[0]][1], fl_verts_loc[ep0[1]][1])

        section_heights.append((crotch_y, 'Thigh'))

        kfc = measurements.get('Knee_from_crotch')
        if kfc:
            knee_y = crotch_y - kfc
            if knee_y > fl_min_y:
                section_heights.append((knee_y, 'Knee'))

        hfw = measurements.get('Hip_from_waist')
        if hfw:
            hip_y = fl_max_y - hfw
            if hip_y > crotch_y:
                section_heights.append((hip_y, 'Hip'))

    # In SVG coords y increases downward (ankle ≈ +14, waist ≈ -93).
    # Force a consistent annotation side so front and back panels align visually:
    #   below-crotch (Thigh, Knee) → annotate toward ankle  = SVG (0, +1)
    #   above-crotch (Hip)         → annotate toward waist   = SVG (0, -1)
    _forced_normals = {'Thigh': (0, 1), 'Knee': (0, 1), 'Hip': (0, -1)}

    for y_local, prefix in section_heights:
        fnorm = _forced_normals.get(prefix)
        for panel_name in ('pant_f_l', 'pant_b_l'):
            panel = panels.get(panel_name)
            if panel is None:
                continue
            xr = _panel_x_range_at_y(panel, y_local)
            if xr is None:
                continue
            panel_width = xr[1] - xr[0]
            label = f"{prefix}: {panel_width:.1f}cm ({2 * panel_width:.1f}cm)"
            sx = back_shift if panel['translation'][2] < 0 else 0.0
            p1 = _local_to_svg(panel, [xr[0], y_local]).copy()
            p2 = _local_to_svg(panel, [xr[1], y_local]).copy()
            p1[0] += sx
            p2[0] += sx
            _draw_dim_annotation(root, ns, p1, p2, label, offset_dist, fontsize, color,
                                 forced_normal=fnorm)

    svg_bytes = ET.tostring(root, encoding='unicode', xml_declaration=False)
    with open(str(svg_path), 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
        f.write(svg_bytes)

    # Regenerate PNG from annotated SVG
    png_path = Path(str(svg_path).replace('_pattern.svg', '_pattern.png'))
    if png_path != Path(svg_path) and png_path.exists():
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=2.54 * 3)


def _fit_rise_parameters(prod, body_yaml_path, base_garment_measurements):
    """Numerically fit (rise_v, front_crotch_fraction) to match production F/B rise seam arcs.

    Joint 2D least-squares (trust-region-reflective) on the residual vector
    (F_arc - target_f, B_arc - target_b). Replaces an earlier alternating-1D-
    bisection scheme that oscillated when F and B targets coupled to the
    other parameter (e.g. wide-leg flared jeans where the F-rise target is
    only achievable at an interior rise_v ≠ either bisection endpoint).

    Returns: (rise_v, front_crotch_fraction)
    """
    from assets.garment_programs.meta_garment import MetaGarment
    from assets.bodies.body_params import BodyParameters
    from scipy.optimize import least_squares

    target_f = prod['Front Rise']
    target_b = prod['Back Rise']

    body_params = BodyParameters(body_yaml_path)
    mapper = ProductionToDesign(body_yaml_path)

    # FRAC_MIN: below this the front crotch extension goes ~0, which causes
    # Bézier curve fitting to degenerate.
    RISE_MIN, RISE_MAX = 0.2, 1.8
    FRAC_MIN, FRAC_MAX = 0.10, 0.99

    # Best-seen fallback: even with LSQ, return the closest iterate visited
    # (rather than the solver's final point) in case bounds clipping or a
    # max-eval cap stops us at a worse spot than something we already saw.
    best = {'score': float('inf')}

    def measure(rise_v, front_fraction, back_lift):
        gm = dict(base_garment_measurements)
        gm['rise'] = rise_v
        design = mapper.map_pants(gm)
        design['pants']['front_crotch_fraction'] = {'v': float(front_fraction)}
        design['pants']['back_rise_lift'] = {'v': float(back_lift)}
        try:
            garment = MetaGarment('_fit', body_params, design)
            pattern = garment.assembly()
        except (AssertionError, ValueError, ZeroDivisionError):
            # Degenerate panel construction (e.g. zero-length sub-edge in the
            # connector matching when body+garment proportions are extreme).
            # Return a large penalty so the LSQ optimizer steers away from this
            # parameter region while still converging on a feasible point.
            return target_f + 100.0, target_b + 100.0
        panels = pattern.pattern['panels']
        stitches = pattern.pattern['stitches']
        f = _rise_seam_arc_length(panels, stitches, 'pant_f_l', 'pant_f_r')
        b = _rise_seam_arc_length(panels, stitches, 'pant_b_l', 'pant_b_r')
        score = max(abs(f - target_f), abs(b - target_b))
        if score < best['score']:
            best.update(score=score, rise_v=rise_v, frac=front_fraction,
                        lift=back_lift, f=f, b=b)
        return f, b

    # Initial guess: rise_v from the average-rise formula, frac centered.
    body = mapper.body
    avg_rise = (target_f + target_b) / 2
    rise0 = float(np.clip((avg_rise - body['crotch_hip_diff']) / body['hips_line'],
                          RISE_MIN, RISE_MAX))
    frac0 = 0.55
    lift0 = 0.0

    # back_rise_lift (cm): a "back-rise scoop" — raises the center-back waist,
    # lengthening the back rise INDEPENDENTLY of the front. This is the 3rd DOF
    # that makes back-heavy splits (BR-FR > ~8cm) reachable; front_crotch_fraction
    # alone caps the spread at ~8cm because crotch_top is shared front/back.
    LIFT_MIN, LIFT_MAX = 0.0, 15.0
    # The system is 2 targets / 3 unknowns (underdetermined), so add a small
    # regularizer on the lift: among the solution family, prefer the smallest
    # scoop (lift→0 for pids already achievable in 2 DOF, leaving them unchanged).
    LIFT_REG = 0.03

    print(f'  Fitting rise: target F={target_f:.1f}cm, B={target_b:.1f}cm  '
          f'(start rise_v={rise0:.4f}, frac={frac0:.4f}, lift={lift0:.2f})')

    def residual(x):
        rv = float(np.clip(x[0], RISE_MIN, RISE_MAX))
        fr = float(np.clip(x[1], FRAC_MIN, FRAC_MAX))
        lf = float(np.clip(x[2], LIFT_MIN, LIFT_MAX))
        f, b = measure(rv, fr, lf)
        return np.array([f - target_f, b - target_b, LIFT_REG * lf])

    # diff_step=0.05 gives a ~5%-of-range FD step. The default ~1.5e-8 is far
    # too small for measure() — pattern build introduces tiny numerical jitter
    # that drowns the gradient signal at that scale, producing useless Jacobians.
    res = least_squares(
        residual, np.array([rise0, frac0, lift0]),
        method='trf',
        bounds=([RISE_MIN, FRAC_MIN, LIFT_MIN], [RISE_MAX, FRAC_MAX, LIFT_MAX]),
        diff_step=0.05,
        max_nfev=90,
        xtol=1e-6, ftol=1e-6,
    )

    rise_v = best['rise_v']
    front_fraction = best['frac']
    back_lift = best['lift']
    print(f'  Joint LSQ ({res.nfev} evals, status={res.status}): '
          f'rise_v={rise_v:.4f}, frac={front_fraction:.4f}, lift={back_lift:.2f}cm '
          f'→ F={best["f"]:.2f}cm (Δ{best["f"] - target_f:+.2f}), '
          f'B={best["b"]:.2f}cm (Δ{best["b"] - target_b:+.2f})')
    if best['score'] >= 0.05:
        print(f'  NOTE: max-error {best["score"]:.2f}cm above 0.05cm threshold — '
              f'targets may not be jointly achievable in this search space.')
    return rise_v, front_fraction, back_lift


def map_production_to_design(prod, body_yaml_path, elastic_waistband=False,
                             balloon_leg=False, cuff_inseam_fraction=None,
                             cuff_ease=1.0, elastic_waist_gather=False,
                             waist_match_body=False, front_slit=None):
    """Map production measurements to GarmentCode design parameters.

    Args:
        prod: dict of production measurements for a single size.
        body_yaml_path: path to the body YAML file.
        elastic_waistband: if True, clamp the waistband to be no larger than
            the body's waist. This ensures the gathered waistband always
            creates compression (elastic grip) against the body.
        balloon_leg: if True, keep the leg wide all the way down (leg opening
            ~ knee width, minimal taper) and add a gathered CuffBand that
            blouses the wide leg bottom into a narrow cuff pinned to the
            Ankle measurement. Models balloon/parachute trousers.
        cuff_inseam_fraction: if set (e.g. 0.15), add a turn-up hem cuff whose
            height is that fraction of the Inseam. The cuff is SUBTRACTED from
            the leg (the Inseam already includes the cuff fabric — a turn-up
            folds it back up), not added. Cuff circumference matches the leg
            opening (no gather). Mutually exclusive with balloon_leg.
    """
    mapper = ProductionToDesign(body_yaml_path)
    body = mapper.body

    # Panel length ≈ inseam + crotch_hip_diff
    panel_length = prod['Inseam'] + body['crotch_hip_diff']

    # For elastic waistbands: the production waist is the relaxed (unstretched)
    # measurement. When worn, the elastic stretches to grip the body but remains
    # under tension (pulling inward). We simulate this by setting the waistband
    # slightly smaller than body waist — the gathering creates compression that
    # mimics elastic tension. The ease factor controls how much tension:
    #   0.95 = 5% smaller than body → moderate elastic grip
    #   prod >= body → oversized, elastic not stretched, use prod waist (pants sag)
    ELASTIC_EASE = 0.95
    waist_circ = prod['Waist']
    if elastic_waistband and waist_circ < body['waist']:
        waist_circ = body['waist'] * ELASTIC_EASE
    # Fitted-to-body waist: override the config Waist with this body's waist
    # (adapts per height), e.g. a baggy-leg pant on a normal fitted waistband.
    if waist_match_body:
        waist_circ = body['waist']

    # Balloon leg: the leg stays wide to the bottom (≈ knee), and the Ankle
    # measurement is the gathered cuff circumference, NOT the leg opening.
    # So drive the leg opening from the knee width and gather to the ankle.
    if balloon_leg:
        leg_opening = prod.get('Knee') or prod['Ankle']
    else:
        leg_opening = prod['Ankle']

    garment_measurements = {
        'waist_circumference': waist_circ,
        'hip_circumference': prod['Low_Hip'],
        'length': panel_length,
        'leg_opening': leg_opening,
        'rise': 1.0,  # placeholder; overridden below
        'thigh_circumference': prod.get('Thigh'),
        'knee_circumference': prod.get('Knee'),
        'crotch_to_knee': prod.get('Knee_from_crotch'),
    }

    if 'Front Rise' in prod and 'Back Rise' in prod:
        # Fit rise_v, front_crotch_fraction, and back_rise_lift to match production
        # seam arc targets (3 DOF: total rise, F/B crotch split, back-rise scoop)
        rise_v, front_fraction, back_lift = _fit_rise_parameters(prod, body_yaml_path, garment_measurements)
        garment_measurements['rise'] = rise_v
        design = mapper.map_pants(garment_measurements)
        design['pants']['front_crotch_fraction'] = {'v': float(front_fraction)}
        design['pants']['back_rise_lift'] = {'v': float(back_lift)}
    else:
        # Fall back: use average of available rise values
        rises = [v for k, v in prod.items() if 'Rise' in k and v is not None]
        rise_v = (sum(rises) / len(rises) - body['crotch_hip_diff']) / body['hips_line'] if rises else 1.0
        garment_measurements['rise'] = float(np.clip(rise_v, 0.5, 1.5))
        design = mapper.map_pants(garment_measurements)

    # Balloon leg: attach a gathered cuff pinned to the Ankle circumference.
    # The wide leg bottom (driven by Knee above) blouses into this narrow band.
    if balloon_leg:
        cuff = design['pants']['cuff']
        cuff['type'] = {'v': 'CuffBand'}
        # Cuff band depth (× leg length): how far the gathered cuff extends
        # off the leg bottom. Kept minimal so the cuff reads as a tight
        # gather, not a tall band. ~0.02 × ~80cm leg ≈ 1.6 cm.
        cuff['cuff_len'] = {'v': 0.02}
        cuff['top_ruffle'] = {'v': 1.0}           # unused when target_width set
        cuff['target_width'] = {'v': float(prod['Ankle'])}

    # Turn-up hem cuff: cuff HEIGHT = fraction × Inseam, subtracted from the
    # leg (Inseam already includes the cuff fabric). Cuff circumference =
    # leg opening (top_ruffle 1.0, no gather). cuff_len is a fraction of
    # _leg_length, so convert: (frac × Inseam) / _leg_length.
    elif cuff_inseam_fraction:
        leg_length = body['_leg_length']
        cuff_len_v = float(cuff_inseam_fraction) * prod['Inseam'] / leg_length
        # cuff circumference = leg opening × cuff_ease (b_width = leg/top_ruffle,
        # so top_ruffle = 1/ease). ease > 1 makes the cuff slightly larger than
        # the leg opening, giving the turn-up a flared lip / visible depth.
        ease = float(cuff_ease) if cuff_ease else 1.0
        cuff = design['pants']['cuff']
        cuff['type'] = {'v': 'CuffBand'}
        cuff['cuff_len'] = {'v': cuff_len_v}
        cuff['top_ruffle'] = {'v': 1.0 / ease}

    # Elastic gathered waist: waistband fabric (= config Waist) bunches into an
    # elastic top edge cinched to body waist. Flag read by StraightWB.
    if elastic_waist_gather:
        design['waistband']['elastic_gather'] = {'v': True}

    # Front ankle split: cut an unstitched V-notch (this many cm tall) into the
    # front panel ankle edge at center-front. Read by PantPanel via PantsHalf.
    if front_slit:
        design['pants']['front_slit'] = {'v': float(front_slit)}

    return design


def _body_leg_x_centers(body_obj_path, y_lo, y_hi, lateral_min=0.5):
    """Return (left_x, right_x) in cm for the body in the world Y band [y_lo, y_hi].

    Body OBJ is in metres; values are converted to cm. Vertices are split by X
    sign (left X > +lateral_min, right X < -lateral_min) and means returned.
    """
    xs, ys = [], []
    with open(body_obj_path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                xs.append(float(p[1])); ys.append(float(p[2]))
    if not xs:
        return None, None
    xs_cm = [x * 100.0 for x in xs]
    ys_cm = [y * 100.0 for y in ys]
    left_xs  = [x for x, y in zip(xs_cm, ys_cm) if y_lo <= y <= y_hi and x >  lateral_min]
    right_xs = [x for x, y in zip(xs_cm, ys_cm) if y_lo <= y <= y_hi and x < -lateral_min]
    if not left_xs or not right_xs:
        return None, None
    return sum(left_xs) / len(left_xs), sum(right_xs) / len(right_xs)


def _apply_pose_x_rotation(folder, body_obj_path,
                           top_band_height=15.0,
                           top_band_offset_below_crotch=15.0,
                           bot_band_height=20.0,
                           bot_band_floor=0.0,
                           min_angle_deg=0.5,
                           back_rise_lift=0.0,
                           extra_x_sep=0.0):
    """Per-leg translation + Z rotation to align pant legs with diagonally
    asymmetric body legs (HM-style A-pose bodies). Each leg's two panels (front
    + back) translate and rotate as a rigid pair around their shared top-center
    pivot, so the waistband seam stays glued to the leg top while only the leg
    below tilts.

      pivot = mean of the leg's top-edge X corners (front + back), Y = leg top
      X shift = body_X_top  - pivot_X
      angle  = atan2(body_X_bot - body_X_top, leg_length)

    The waistband itself is not rotated; it absorbs the resulting top-edge tilt
    via stitching during sim.
    """
    import math
    spec_path = next(Path(folder).glob('*_specification.json'))
    with open(spec_path) as f:
        spec = json.load(f)

    panels = spec['pattern']['panels']
    needed = ['pant_f_l', 'pant_b_l', 'pant_f_r', 'pant_b_r']
    if not all(n in panels for n in needed):
        print('  Pose-X-Rot skipped (missing pant panels)')
        return

    # 3D Y extents per panel (assumes rotation is currently [0,0,0])
    panel_y_max = {n: panels[n]['translation'][1] + max(v[1] for v in panels[n]['vertices'])
                   for n in needed}
    panel_y_min = {n: panels[n]['translation'][1] + min(v[1] for v in panels[n]['vertices'])
                   for n in needed}
    # Use the UN-scooped waist level as the leg top: the back-rise scoop raises
    # the (always-highest) center-back corner by back_rise_lift, so the raw max
    # would push the body-leg sampling band / pivot up by the lift and shift the
    # pose-X correction (up to ~2.8cm at large lifts). Subtracting back_rise_lift
    # recovers the no-scoop reference; no-op when back_rise_lift=0.
    leg_top_y = max(panel_y_max.values()) - back_rise_lift
    leg_bot_y = min(panel_y_min.values())
    leg_length = leg_top_y - leg_bot_y

    top_band = (leg_top_y - top_band_offset_below_crotch - top_band_height,
                leg_top_y - top_band_offset_below_crotch)
    # Sample the body at a fixed floor-relative SHIN band, NOT tracking the
    # garment hem. When a garment is too long for a short body, its hem drops
    # below the floor, which used to push this band down onto the FEET -- and
    # this A-pose body's feet splay ~8-9cm outboard of the shin, so the leg got
    # aimed at the splayed foot and (especially for skinny cuts) wrapped outside
    # the body leg. A fixed shin band tracks the actual lower leg for every
    # body/garment, so theta aims the leg at the leg, not the foot.
    bot_band = (bot_band_floor, bot_band_floor + bot_band_height)

    body_top_L, body_top_R = _body_leg_x_centers(body_obj_path, *top_band)
    body_bot_L, body_bot_R = _body_leg_x_centers(body_obj_path, *bot_band)
    if any(v is None for v in (body_top_L, body_top_R, body_bot_L, body_bot_R)):
        print(f'  Pose-X-Rot skipped (no body leg samples; top band={top_band}, bot band={bot_band})')
        return

    # print(f'  Pose-X-Rot: leg_top_y={leg_top_y:.2f} leg_bot_y={leg_bot_y:.2f} leg_len={leg_length:.2f}')
    # print(f'    body top band Y=[{top_band[0]:.2f}, {top_band[1]:.2f}]  L={body_top_L:+.2f} R={body_top_R:+.2f}')
    # print(f'    body bot band Y=[{bot_band[0]:.2f}, {bot_band[1]:.2f}]  L={body_bot_L:+.2f} R={body_bot_R:+.2f}')

    def pivot_x_for_leg(panel_names):
        # Average X over the panel's full top (waist) edge. The back-rise scoop
        # raises ONLY the center-back corner by `back_rise_lift`, tilting the back
        # waist edge; an exact `v[1] == ymax` test would then pick only that
        # raised corner and skew the pivot ~half-a-waist toward center-back,
        # splaying the legs. A Y tolerance spanning the tilt averages the whole
        # waist edge, giving the true waist-center X — invariant to the scoop
        # (the lift moves the corner in Y only, never X). tol→0.5 with no scoop,
        # reproducing the original flat-waist behaviour exactly.
        xs = []
        for n in panel_names:
            p = panels[n]
            # Only the BACK panel carries the scoop (its waist edge tilts up to
            # center-back), so only it needs a tolerance spanning the lift. The
            # FRONT waist is flat — giving it the large tolerance would wrongly
            # pull in side-seam vertices below the waist and skew the pivot
            # (seen as a ~2.8cm placement shift at large lifts). Keep front tight.
            tol = (back_rise_lift + 0.5) if n.startswith('pant_b') else 0.5
            ymax = max(v[1] for v in p['vertices'])
            xs.extend(p['translation'][0] + v[0]
                      for v in p['vertices'] if v[1] >= ymax - tol)
        return sum(xs) / len(xs) if xs else None

    pivot_L_x = pivot_x_for_leg(['pant_f_l', 'pant_b_l'])
    pivot_R_x = pivot_x_for_leg(['pant_f_r', 'pant_b_r'])
    pivot_y = leg_top_y

    shift_L = body_top_L - pivot_L_x
    shift_R = body_top_R - pivot_R_x
    theta_L = math.atan2(body_bot_L - body_top_L, leg_length)
    theta_R = math.atan2(body_bot_R - body_top_R, leg_length)

    # Adaptive outward X separation. The pose alignment glues each leg's waist
    # pivot to its body-leg center; for a deep DXF front-crotch hook this leaves
    # the two front panels' fly edges (and the back panels' CB edges) overlapping
    # at init -> a self-contact mess in the boxmesh. Measure that overlap on THIS
    # body+size -- the front/back panels' inward x-reach AFTER the body-align
    # shift+rotation -- and push each leg out by half of it plus a target gap, so
    # the hooks clear regardless of body leg spacing or garment size. The fly/CB
    # stitches close the gap during zero-gravity, so the final drape is unchanged.
    # `extra_x_sep` is the target clearance gap; 0.0 -> feature off (no-op for
    # every existing config; only the next.dxf PantsCLO path enables it).
    if extra_x_sep:
        def _post_x_extent(panel, shift_x, theta, pivot_x):
            # world x-extent of the panel after apply_rigid(shift_x, theta)
            c, s = math.cos(theta), math.sin(theta)
            tx, ty = panel['translation'][0], panel['translation'][1]
            new_tx = (pivot_x + shift_x) + (tx - pivot_x) * c - (ty - pivot_y) * s
            ang = math.radians(panel['rotation'][2]) + theta
            ca, sa = math.cos(ang), math.sin(ang)
            xs = [v[0] * ca - v[1] * sa + new_tx for v in panel['vertices']]
            return min(xs), max(xs)

        def _fly_overlap(rname, lname):
            rmin, rmax = _post_x_extent(panels[rname], shift_R, theta_R, pivot_R_x)
            lmin, lmax = _post_x_extent(panels[lname], shift_L, theta_L, pivot_L_x)
            # positive == the two panels cross the centerline into each other
            return (rmax - lmin) if body_top_L > body_top_R else (lmax - rmin)

        overlap = max(_fly_overlap('pant_f_r', 'pant_f_l'),
                      _fly_overlap('pant_b_r', 'pant_b_l'))
        sep = max(0.0, (overlap + extra_x_sep) / 2.0)
        shift_L += math.copysign(sep, body_top_L)
        shift_R += math.copysign(sep, body_top_R)
    # print(f'    adaptive x-sep: max fly/CB overlap={overlap:+.2f} '
    # f'-> sep=±{sep:.2f} (target gap {extra_x_sep})')

    # print(f'    garment pivots L={pivot_L_x:+.2f} R={pivot_R_x:+.2f}  pivot_y={pivot_y:.2f}')
    # print(f'    shifts L={shift_L:+.2f} R={shift_R:+.2f}')
    # print(f'    angles L={math.degrees(theta_L):+.2f}° R={math.degrees(theta_R):+.2f}°')

    if not extra_x_sep \
            and abs(math.degrees(theta_L)) < min_angle_deg and abs(math.degrees(theta_R)) < min_angle_deg \
            and abs(shift_L) < 0.5 and abs(shift_R) < 0.5:
    # print('    below min thresholds; no transform applied')
        return

    def apply_rigid(panel, shift_x, theta, pivot_x_world):
        # Pivot in world after the X shift
        rot_pivot_x = pivot_x_world + shift_x
        tx, ty, tz = panel['translation']
        tx += shift_x
        dx, dy = tx - rot_pivot_x, ty - pivot_y
        c, s = math.cos(theta), math.sin(theta)
        panel['translation'] = [rot_pivot_x + dx * c - dy * s,
                                pivot_y + dx * s + dy * c,
                                tz]
        rx, ry, rz = panel['rotation']
        panel['rotation'] = [rx, ry, rz + math.degrees(theta)]

    # Move each leg AND its cuff panels together so the cuff stays coaxial
    # with the leg after the per-leg shift/rotation. (Cuff panels are named
    # pant_l_cuff_* / pant_r_cuff_*; absent when there is no cuff.)
    left_grp = ['pant_f_l', 'pant_b_l'] + [n for n in panels if 'l_cuff' in n]
    right_grp = ['pant_f_r', 'pant_b_r'] + [n for n in panels if 'r_cuff' in n]
    for n in left_grp:
        apply_rigid(panels[n], shift_L, theta_L, pivot_L_x)
    for n in right_grp:
        apply_rigid(panels[n], shift_R, theta_R, pivot_R_x)

    with open(spec_path, 'w') as f:
        json.dump(spec, f, indent=2)


CROTCH_TARGET_OFFSET = 0.5  # garment crotch placed 0.5cm above body crotch
# NOTE: the garment crotch is placed at exactly body_crotch_Y + CROTCH_TARGET_OFFSET.
# body_crotch_Y now comes from the SMPL crotch landmark vertex (see _body_crotch_Y)
# when the body mesh is SMPL, so offset 0.5 lands the garment 0.5cm above the real
# crotch (the as-built crotch is measured from the pattern, not assumed; the old
# assumption was ~10cm too high). A negative offset would place the garment
# that many cm below the body crotch; a positive one, above it.


def _garment_crotch_Y(garment):
    """Return the as-built 3D Y of the garment crotch (min Y of the pants
    crotch interfaces), or None if no pants component is present.

    Used to place the crotch at an absolute body-relative target before sim,
    instead of assuming a fixed offset from the panels' build position.
    """
    ys = []
    for sub in getattr(garment, 'subs', []):
        right = getattr(sub, 'right', None)
        if right is None or 'crotch_f' not in getattr(right, 'interfaces', {}):
            continue
        for half in (sub.right, sub.left):
            for key in ('crotch_f', 'crotch_b'):
                if key in half.interfaces:
                    bb = half.interfaces[key].bbox_3d()
                    ys.append(bb[0][1])   # min-Y corner = lowest crotch point
    return min(ys) if ys else None


SMPL_CROTCH_VERTEX = 1210  # SMPL crotch landmark (pose-invariant, fixed topology)

def _body_crotch_Y(body, body_yaml_path):
    """Body crotch height (cm above floor).

    If the body mesh is SMPL (6890 verts) use the crotch landmark vertex 1210 --
    pose-invariant and accurate. Otherwise fall back to the measurement formula,
    which underestimates the true SMPL crotch by a height-dependent ~3-4.5cm.
    (The formula is evaluated lazily so SMPL bodies work without a yaml.)
    """
    obj = Path(str(body_yaml_path).replace('.yaml', '.obj'))
    if obj.exists():
        try:
            V = np.array([[float(x) for x in ln.split()[1:4]]
                          for ln in open(obj, errors='replace') if ln.startswith('v ')])
            if len(V) == 6890:  # SMPL topology
                span = V[:, 1].max() - V[:, 1].min()
                scale = 100.0 if span < 10 else 1.0  # body OBJ is in meters
                return float((V[SMPL_CROTCH_VERTEX, 1] - V[:, 1].min()) * scale)
        except Exception:
            pass
    return (body['height'] - body['head_l'] - body['waist_line']
            - body['hips_line']) - body['crotch_hip_diff']


def generate_pattern(size, design, body_yaml_path, output_base, name_prefix='',
                     garment_prefix='hm_pants',
                     ankle_clearance_pct=0.0,
                     anchor_mode='crotch_zero'):
    """Generate pattern using built-in MetaGarment system.

    Placement: 'crotch_zero' is the ONLY supported anchor mode. The whole
    MetaGarment (pants panels + waistband) is lifted uniformly in Y so the
    garment's as-built crotch sits at body_crotch_Y + CROTCH_TARGET_OFFSET
    (CROTCH_TARGET_OFFSET is currently 0.5, i.e. 0.5cm above the body crotch).
    There is no floor safety here — if the hem then sits below the floor, the
    sim's auto pre-lift compression (garment.py) compresses the lower garment
    so the hem ends ~2cm above the floor.

    `ankle_clearance_pct` and `anchor_mode` are kept in the signature only for
    call-site compatibility; any anchor_mode other than 'crotch_zero' is an
    error.
    """
    from assets.garment_programs.meta_garment import MetaGarment
    from assets.bodies.body_params import BodyParameters

    if anchor_mode != 'crotch_zero':
        raise ValueError(
            f"anchor_mode={anchor_mode!r} is no longer supported; "
            f"'crotch_zero' is the only mode")

    garment_name = f'{name_prefix}{garment_prefix}_size{size}'

    body = BodyParameters(body_yaml_path)
    # Ensure the body mesh has its feet at Y=0 (SMPL fits come out
    # pelvis-centered). Both the pose-X leg sampling below and the sim's body
    # collision assume the feet-at-0 convention; without this a pelvis-centered
    # body fails leg sampling (pose-X + x-sep skipped) and drapes misaligned.
    # In-place and idempotent (no-op when already normalized, e.g. HM bodies).
    normalize_body_mesh(str(body_yaml_path).replace('.yaml', '.obj'))
    garment = MetaGarment(garment_name, body, design)

    band_width = (design.get('waistband', {}).get('width', {}).get('v', 0.0)
                  * body['hips_line'])

    # Canonical, body-agnostic placement: serialize with the garment crotch at
    # exactly Y=0. The body-specific lift (crotch height + offset) and the
    # pose-X leg alignment are deferred to sim time, applied against the
    # actual sim body (_place_pattern_for_body), so one generated pattern
    # drapes onto any body.
    as_built_crotch_Y = _garment_crotch_Y(garment)
    body_agnostic = as_built_crotch_Y is not None
    if body_agnostic:
        lift = -as_built_crotch_Y
    else:
        # Legacy body-baked fallback when the crotch interface can't be
        # located (e.g. non-pants lower garments).
        lift = band_width + 5.0 + CROTCH_TARGET_OFFSET

    print(f'  anchor_mode=crotch_zero  band_width={band_width:.2f}  '
          f'applied_lift={lift:.2f}  body_agnostic={body_agnostic}')
    if lift != 0.0:
        garment.translate_by([0, lift, 0])

    pattern = garment.assembly()

    if garment.is_self_intersecting():
        print(f'  WARNING: {garment_name} has self-intersecting panels')

    folder = pattern.serialize(
        output_base,
        tag='_' + datetime.now().strftime("%y%m%d-%H-%M-%S"),
        to_subfolder=True,
        with_3d=False,
        with_text=False,
        view_ids=False,
        with_printable=True
    )

    # Pose hints consumed by the sim-time pose-X leg alignment.
    # Extra leg X-separation for the next.dxf PantsCLO path: its DXF front-crotch
    # hook is deep enough that the two legs' fly edges overlap ~11cm at init.
    # `clo_dxf.x_sep` (a profile-supplied half-distance) clears it; absent/zero
    # for every other config, so this is a no-op for the whole HM_Jeans batch.
    _xsep = 0.0
    _clo = design.get('clo_dxf')
    if isinstance(_clo, dict):
        _dxf_path = _clo.get('path', {}).get('v') if isinstance(_clo.get('path'), dict) else _clo.get('path')
        if _dxf_path:
            from assets.garment_programs.pants_clo import _profile
            _xsep = float(_profile(_dxf_path).get('x_sep', 0.0) or 0.0)
    _brl = (design['pants']['back_rise_lift']['v']
            if isinstance(design.get('pants', {}).get('back_rise_lift'), dict)
            and design['pants']['back_rise_lift'].get('v') is not None
            else 0.0)

    if body_agnostic:
        # Stamp the spec as canonically placed. Sim-time placement
        # (_place_pattern_for_body) reads the anchor + pose hints and bakes
        # the body-specific lift and leg alignment against the sim body.
        spec_path = next(Path(folder).glob('*_specification.json'))
        with open(spec_path) as f:
            spec = json.load(f)
        spec.setdefault('properties', {})['placement'] = {
            'anchor': 'crotch_at_zero',
            'back_rise_lift': _brl,
            'extra_x_sep': _xsep,
        }
        with open(spec_path, 'w') as f:
            json.dump(spec, f, indent=2)
    else:
        # Legacy path: bake pose-X against the drafting body at generation.
        body_obj = Path(str(body_yaml_path).replace('.yaml', '.obj'))
        if body_obj.exists():
            _apply_pose_x_rotation(folder, body_obj, back_rise_lift=_brl, extra_x_sep=_xsep)
        else:
            print(f'  Pose-X correction skipped (no body OBJ at {body_obj})')

    print(f'  Pattern generated: {garment_name} -> {folder}')
    return Path(folder), garment_name


def _place_pattern_for_body(spec_file, body_name, output_base, placement):
    """Bake body-specific placement into a copy of a body-agnostic spec.

    Canonical specs (generate_pattern) are serialized with the garment crotch
    at exactly Y=0 and carry a properties.placement marker. This lifts all
    panels so the crotch sits at body_crotch_Y + CROTCH_TARGET_OFFSET for THIS
    body, then aligns the pant legs to the body legs (pose-X). The placed spec
    is written to a per-run folder; the input pattern is never mutated.
    Assumes the body OBJ has feet at Y=0 (normalize_body_mesh).
    """
    bodies = Path('./assets/bodies')
    body_obj = bodies / f'{body_name}.obj'
    body_yaml = bodies / f'{body_name}.yaml'
    body = {}
    if body_yaml.exists():
        with open(body_yaml) as f:
            body = yaml.safe_load(f)['body']
    try:
        body_crotch = _body_crotch_Y(body, body_yaml)
    except KeyError:
        raise RuntimeError(
            f'Cannot place pattern: body {body_name} is not SMPL topology '
            'and has no measurements yaml for the crotch formula')

    spec_file = Path(spec_file)
    in_name = spec_file.stem.replace('_specification', '')
    placed_folder = Path(output_base) / f'{in_name}_placed'
    placed_folder.mkdir(parents=True, exist_ok=True)

    with open(spec_file) as f:
        spec = json.load(f)
    lift = body_crotch + CROTCH_TARGET_OFFSET
    for panel in spec['pattern']['panels'].values():
        panel['translation'][1] += lift
    # Mark placement as baked so this spec (and the copy BoxMesh writes into
    # the sim output folder) is never placed a second time when re-simulated.
    spec['properties']['placement'] = dict(
        placement, anchor='placed', placed_for_body=str(body_name))
    with open(placed_folder / spec_file.name, 'w') as f:
        json.dump(spec, f, indent=2)
    print(f'  Placement: crotch lift {lift:.2f} for body {body_name} '
          f'-> {placed_folder}')

    if body_obj.exists():
        _apply_pose_x_rotation(
            placed_folder, body_obj,
            back_rise_lift=float(placement.get('back_rise_lift', 0.0) or 0.0),
            extra_x_sep=float(placement.get('extra_x_sep', 0.0) or 0.0))
    else:
        print(f'  Pose-X skipped (no body OBJ at {body_obj})')
    return placed_folder


def simulate_pattern(pattern_folder, garment_name, output_base,
                     body_name=None, sim_props=None):
    """Run physics simulation on A-pose body.

    Args:
        sim_props: dict with sim/render keys (inlined sim props from garment config),
                   or a file path string for backward compatibility.
    """
    if body_name is None:
        body_name = BODY_NAME

    from pygarment.meshgen.boxmeshgen import BoxMesh
    from pygarment.meshgen.simulation import run_sim
    from pygarment.meshgen.sim_config import PathCofig

    if sim_props is None:
        props = Properties('./assets/Sim_props/default_sim_props.yaml')
    elif isinstance(sim_props, str):
        props = Properties(sim_props)
    else:
        props = Properties()
        props.properties = dict(sim_props)
    props.set_section_stats(
        'sim', fails={}, sim_time={}, spf={},
        fin_frame={}, body_collisions={}, self_collisions={}
    )
    props.set_section_stats('render', render_time={})

    spec_files = list(pattern_folder.glob('*_specification.json'))
    if not spec_files:
        print(f'  ERROR: No specification file found in {pattern_folder}')
        return None
    spec_file = spec_files[0]
    in_name = spec_file.stem.replace('_specification', '')

    # Body-agnostic (canonical) specs defer placement to sim time: bake the
    # crotch lift + pose-X leg alignment for THIS body into a per-run copy.
    # Baked specs (no marker) simulate as-is, exactly as before.
    try:
        with open(spec_file) as f:
            placement = json.load(f).get('properties', {}).get('placement')
    except (OSError, json.JSONDecodeError):
        placement = None
    placed_folder = None
    if placement and placement.get('anchor') == 'crotch_at_zero':
        placed_folder = _place_pattern_for_body(
            spec_file, body_name, output_base, placement)
        pattern_folder = placed_folder
        spec_file = pattern_folder / spec_file.name

    paths = PathCofig(
        in_element_path=pattern_folder,
        out_path=output_base,
        in_name=in_name,
        body_name=body_name,
        smpl_body=True,
        add_timestamp=True
    )

    print(f'  Generating box mesh for {in_name}...')
    resolution_scale = props['sim']['config']['resolution_scale']
    garment_box_mesh = BoxMesh(paths.in_g_spec, resolution_scale)
    garment_box_mesh.load()
    garment_box_mesh.serialize(
        paths, store_panels=False,
        uv_config=props['render']['config']['uv_texture']
    )

    # The placed copy has served its purpose: BoxMesh.serialize wrote the
    # placed spec into the sim output folder (paths.g_specs), which is what
    # the sim reads from here on.
    if placed_folder is not None:
        import shutil
        shutil.rmtree(placed_folder, ignore_errors=True)

    props.serialize(paths.element_sim_props)

    print(f'  Running simulation for {in_name}...')
    # Allow configs to opt out of per-frame video rendering (EGL/permission
    # constrained hosts). Default True keeps prior behaviour for all configs
    # that don't set the flag.
    save_sim_video = props['sim']['config'].get('save_sim_video', True)
    run_sim(
        garment_box_mesh.name,
        props,
        paths,
        save_v_norms=False,
        store_usd=False,
        optimize_storage=False,
        verbose=False,
        save_sim_video=save_sim_video,
        video_frame_interval=10,
        video_fps=30,
    )

    props.serialize(paths.element_sim_props)
    # Always emit the combined body+garment mesh (combined.obj + combined.glb),
    # so direct simulate_pattern callers get it too -- run_garment.py's Step 4
    # only covers its own flow. Uses the exact-frame _body_final.obj written by
    # the sim, so no rescaling/repose handling is needed here.
    try:
        save_combined_mesh(paths.out_el, body_name=body_name)
    except Exception as e:
        print(f'  Combined mesh skipped: {e}')
    #print(f'  Simulation complete: {in_name} -> {paths.out_el}')
    return paths.out_el


# ============================================================
# Measurement verification from spec JSON
# ============================================================

def _rel_to_abs(start, end, rel_pt):
    """Convert relative Bezier control point to absolute 2D coordinates."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    edge = end - start
    edge_perp = np.array([-edge[1], edge[0]])
    return start + rel_pt[0] * edge + rel_pt[1] * edge_perp


def edge_length(edge_dict, verts):
    """Return the arc-length of a single edge from a panel JSON dict."""
    ep = edge_dict['endpoints']
    v1 = verts[ep[0]]
    v2 = verts[ep[1]]

    if 'curvature' not in edge_dict:
        return float(np.linalg.norm(np.array(v2) - np.array(v1)))

    curv = edge_dict['curvature']
    ctype = curv['type']

    if ctype == 'circle':
        radius, large_arc, sweep = curv['params']
        arc = svgpath.Arc(
            complex(v1[0], v1[1]),
            complex(radius, radius),
            0, int(large_arc), int(sweep),
            complex(v2[0], v2[1]),
        )
        return abs(arc.length())
    elif ctype == 'cubic':
        cp1 = _rel_to_abs(v1, v2, curv['params'][0])
        cp2 = _rel_to_abs(v1, v2, curv['params'][1])
        curve = svgpath.CubicBezier(
            complex(v1[0], v1[1]),
            complex(cp1[0], cp1[1]),
            complex(cp2[0], cp2[1]),
            complex(v2[0], v2[1]),
        )
        return abs(curve.length())
    elif ctype == 'quadratic':
        cp1 = _rel_to_abs(v1, v2, curv['params'][0])
        curve = svgpath.QuadraticBezier(
            complex(v1[0], v1[1]),
            complex(cp1[0], cp1[1]),
            complex(v2[0], v2[1]),
        )
        return abs(curve.length())
    else:
        return float(np.linalg.norm(np.array(v2) - np.array(v1)))


def _panel_edge_lengths(panel):
    """Return list of arc-lengths for every edge of a panel."""
    verts = panel['vertices']
    return [edge_length(e, verts) for e in panel['edges']]


def _free_edges(panels, stitches, panel_name):
    """Return indices and edge dicts of free (unstitched) edges."""
    stitched = set()
    for stitch in stitches:
        for side in stitch:
            stitched.add((side['panel'], side['edge']))
    return [
        (i, panels[panel_name]['edges'][i])
        for i in range(len(panels[panel_name]['edges']))
        if (panel_name, i) not in stitched
    ]


def verify_measurements(spec_path, size, prod, body_yaml=None, elastic_waistband=False):
    """Extract pants measurements from spec JSON and compare to production targets.

    Panel structure (from stitching analysis):
      pant_f_r: e0=bottom(hem), e1=inseam, e2=crotch_bottom, e3=crotch_top,
                e4=top(waist), e5=outside_top, e6=outside_bottom
      pant_b_r: e0=outside_bottom, e1=outside_top, e2..e9=top(with darts),
                e10=crotch_top, e11=inseam, e12=bottom(hem)
      wb_front: lower_interface on e3 (top edge)
      wb_back:  lower_interface on e1 (top edge)
    """
    with open(spec_path, 'r') as f:
        spec = json.load(f)

    panels = spec['pattern']['panels']
    stitches = spec['pattern']['stitches']

    # Build stitching map for edge identification
    stitched = set()
    stitch_map = {}  # (panel, edge) -> (other_panel, other_edge)
    for stitch in stitches:
        # a stitch is [sideA, sideB] and may carry a trailing 'right_wrong' str
        sides = [s for s in stitch if isinstance(s, dict)]
        for side in sides:
            stitched.add((side['panel'], side['edge']))
        if len(sides) == 2:
            a, b = sides
            stitch_map[(a['panel'], a['edge'])] = (b['panel'], b['edge'])
            stitch_map[(b['panel'], b['edge'])] = (a['panel'], a['edge'])

    results = {}

    # --- Waist circumference ---
    # Find edges with 'lower_interface' label on waistband panels
    waist_circ = 0.0
    for pname in ['wb_front', 'wb_back']:
        panel = panels.get(pname)
        if not panel:
            continue
        for e in panel['edges']:
            if e.get('label') == 'lower_interface':
                waist_circ += edge_length(e, panel['vertices'])
    results['Waist'] = waist_circ if waist_circ > 0 else None

    # --- Leg opening (ankle circumference) ---
    # Free (unstitched) edges on pant panels at y ≈ 0 (hem level)
    # For right leg: pant_f_r and pant_b_r free edges at bottom
    leg_opening = 0.0
    for pname in ['pant_f_r', 'pant_b_r']:
        panel = panels.get(pname)
        if not panel:
            continue
        verts = panel['vertices']
        for i, e in enumerate(panel['edges']):
            if (pname, i) in stitched:
                continue
            ep = e['endpoints']
            v1, v2 = verts[ep[0]], verts[ep[1]]
            # Bottom edge: both endpoints near y = min_y of panel
            min_y = min(v[1] for v in verts)
            if abs(v1[1] - min_y) < 1.0 and abs(v2[1] - min_y) < 1.0:
                leg_opening += edge_length(e, verts)
    # front_bottom + back_bottom = full circumference of one leg opening (no ×2)
    results['Ankle'] = leg_opening if leg_opening > 0 else None

    # --- Inseam ---
    # The inseam is the stitched edge(s) between front and back panels on the inside.
    # With subcurves this can be 3 edges (crotch→thigh, thigh→knee, knee→ankle).
    # Inseam edges are on the RIGHT side (high X) of the panel; outseam on the LEFT
    # (low X). The crotch U-curve is also stitched but is horizontal at crotch level.
    # We identify inseam by: stitched to pant_b_r, mean X > panel centroid X,
    # and NOT horizontal (i.e., has vertical extent — excludes crotch curve).
    inseam = 0.0
    pant_fr = panels.get('pant_f_r')
    pant_br = panels.get('pant_b_r')
    if pant_fr and pant_br:
        fr_verts = pant_fr['vertices']
        centroid_x = np.mean([v[0] for v in fr_verts])
        for i, e in enumerate(pant_fr['edges']):
            partner = stitch_map.get(('pant_f_r', i))
            if partner and partner[0] == 'pant_b_r':
                ep = e['endpoints']
                v1, v2 = fr_verts[ep[0]], fr_verts[ep[1]]
                mean_x = (v1[0] + v2[0]) / 2
                dy = abs(v1[1] - v2[1])
                # Inseam: on the right (high-X) side and has vertical extent
                if mean_x > centroid_x and dy > 1.0:
                    inseam += edge_length(e, fr_verts)
    results['Inseam'] = inseam if inseam > 0 else None

    # --- Cross-section circumferences (Thigh, Knee, Hip) ---
    # Crotch height = U bottom = tip of center-front seam (pant_f_l ↔ pant_f_r).
    # This is the assembly crotch point, crotch_hip_diff (8cm) BELOW the inseam
    # endpoint in the flat pattern.
    fl_panel = panels.get('pant_f_l')
    if fl_panel:
        fl_verts = fl_panel['vertices']
        fl_min_y = min(v[1] for v in fl_verts)
        fl_max_y = max(v[1] for v in fl_verts)
        # Crotch y: bottom of center-front seam = actual crotch seam tip
        _fr_top, _fr_bot = _find_panel_rise_seam_endpoints(panels, stitches, 'pant_f_l', 'pant_f_r')
        if _fr_bot is not None:
            crotch_y = float(_fr_bot[1])
        else:
            e0 = fl_panel['edges'][0]
            ep0 = e0['endpoints']
            crotch_y = max(fl_verts[ep0[0]][1], fl_verts[ep0[1]][1])

        # Thigh: at crotch level (one leg circumference)
        results['Thigh'] = _cross_section_circumference(panels, crotch_y, above_crotch=False)

        # Knee: crotch_y minus the thigh-to-knee distance
        if prod.get('Knee_from_crotch'):
            knee_y = crotch_y - prod['Knee_from_crotch']
            if knee_y > fl_min_y:
                results['Knee'] = _cross_section_circumference(panels, knee_y, above_crotch=False)

        # Hip: fl_max_y (waist seam = bottom of waistband) minus hip_from_waist
        if prod.get('Hip_from_waist'):
            hip_y = fl_max_y - prod['Hip_from_waist']
            if hip_y > crotch_y:
                results['Hip'] = _cross_section_circumference(panels, hip_y, above_crotch=True)

    # --- Rise arc lengths ---
    results['Front Rise'] = _rise_seam_arc_length(panels, stitches, 'pant_f_l', 'pant_f_r')
    results['Back Rise']  = _rise_seam_arc_length(panels, stitches, 'pant_b_l', 'pant_b_r')

    if body_yaml is None:
        body_yaml = BODY_YAML
    with open(body_yaml) as f:
        body_data = yaml.safe_load(f)['body']

    # --- Print results ---
    print(f'\n  {"="*60}')
    print(f'  Measurement Verification – Size {size}')
    print(f'  {"="*60}')
    print(f'  {"Measurement":<25} {"Target":>10} {"Measured":>10} {"Delta":>10}')
    print(f'  {"-"*60}')

    hfw = prod.get('Hip_from_waist')
    kfc = prod.get('Knee_from_crotch')
    checks = [
        ('Waist',      'Waist',      'WB lower_interface edge length'),
        ('Hip',        'Low_Hip',    f'Cross-section at {hfw:.1f}cm below waist seam (×2)' if hfw else 'Cross-section at hip height (×2)'),
        ('Thigh',      'Thigh',      'Cross-section at crotch level (one leg)'),
        ('Knee',       'Knee',       f'Cross-section at {kfc:.1f}cm below crotch (one leg)' if kfc else 'Cross-section at knee height (one leg)'),
        ('Inseam',     'Inseam',     'Inseam edge arc length'),
        ('Ankle',      'Ankle',      'Free hem edges (one leg circumference)'),
        ('Front Rise', 'Front Rise', 'Center-front seam arc length'),
        ('Back Rise',  'Back Rise',  'Center-back seam arc length'),
    ]

    for meas_key, prod_key, note in checks:
        measured = results.get(meas_key)
        target = prod.get(prod_key)
        # For elastic waistband: when prod < body, pattern uses body waist * ease
        if elastic_waistband and meas_key == 'Waist' and prod['Waist'] < body_data.get('waist', 0):
            target = body_data['waist'] * 0.95
        if target is None:
            continue
        if measured is not None:
            delta = measured - target
            flag = '  <-- CHECK' if abs(delta) > 1.5 else ''
            print(f'  {meas_key:<25} {target:>10.1f} {measured:>10.1f} {delta:>+10.1f}{flag}')
            print(f'    {note}')
        else:
            print(f'  {meas_key:<25} {target:>10.1f} {"N/A":>10}')
            print(f'    {note}')

    waist_body = body_data.get('waist', '?')
    if elastic_waistband and prod['Waist'] < float(waist_body):
        elastic_waist = float(waist_body) * 0.95
        print(f'\n  Note: Elastic waistband — pattern waist {elastic_waist:.1f}cm '
              f'(body {waist_body}cm × 0.95 ease). '
              f'Production relaxed waist is {prod["Waist"]}cm.')
    elif elastic_waistband:
        print(f'\n  Note: Elastic waistband — production waist ({prod["Waist"]}cm) >= body waist '
              f'({waist_body}cm), oversized fit (elastic not stretched).')
    else:
        print(f'\n  Note: Waist target ({prod["Waist"]}) is production spec; '
              f'body waist is {waist_body}cm (elastic/stretch fit).')

    return results


def save_combined_mesh(sim_folder, body_obj_path=None, body_name=None):
    """Create combined body + garment mesh."""
    if body_name is None:
        body_name = BODY_NAME

    import trimesh

    sim_folder = Path(sim_folder)
    garment_files = list(sim_folder.glob('*_sim.obj'))
    if not garment_files:
        print(f'  No garment mesh found in {sim_folder}')
        return

    garment_path = garment_files[0]
    garment = trimesh.load(str(garment_path), process=False)

    # Prefer the sim's exported final-pose body (it's in the cloth's EXACT frame,
    # so no scaling/shift needed) — this matters when the body was animated to a
    # new pose during the sim. Skip it if an explicit body_obj_path was given.
    final_body = list(sim_folder.glob('*_body_final.obj'))
    if final_body and body_obj_path is None:
        body = trimesh.load(str(final_body[0]), process=False)
    else:
        if body_obj_path is None:
            body_obj_path = f'./assets/bodies/{body_name}.obj'
        body = trimesh.load(str(body_obj_path), process=False)
        if body.vertices.max() < 3.0:
            body.vertices = body.vertices * 100.0
        min_y = body.vertices[:, 1].min()
        if min_y < 0:
            body.vertices[:, 1] += abs(min_y)

    body_v = np.array(body.vertices)
    body_f = np.array(body.faces)
    garm_v = np.array(garment.vertices)
    garm_f = np.array(garment.faces)
    garm_f_offset = garm_f + len(body_v)

    output_path = sim_folder / 'combined.obj'
    with open(output_path, 'w') as f:
        f.write("# Combined body + garment mesh\n\n")
        for v in body_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for v in garm_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        f.write("\ng body\n")
        for face in body_f:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        f.write("\ng garment\n")
        for face in garm_f_offset:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    #print(f'  Combined mesh saved to {output_path}')

    body.visual = trimesh.visual.ColorVisuals(
        mesh=body, face_colors=np.tile([80, 80, 80, 255], (len(body.faces), 1)))
    garment.visual = trimesh.visual.ColorVisuals(
        mesh=garment, face_colors=np.tile([220, 220, 220, 255], (len(garment.faces), 1)))
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name='body')
    scene.add_geometry(garment, node_name='garment')
    glb_path = sim_folder / 'combined.glb'
    scene.export(str(glb_path))
    #print(f'  Combined GLB saved to {glb_path}')


# ============================================================
# Main pipeline
# ============================================================

if __name__ == '__main__':
    sys_props = Properties('./system.json')
    output_base = sys_props['output']

    print("=" * 60)
    print("  Custom Pants Pipeline (Built-in Parametric System)")
    print(f"  Simulation body: {BODY_NAME} (A-pose)")
    print(f"  GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 60)

    normalize_body_mesh(f'./assets/bodies/{BODY_NAME}.obj')

    # Step 1: Generate patterns for all sizes
    print("\n--- Step 1: Generating patterns ---")
    generated = []
    for size in SIZES_TO_RUN:
        prod = PRODUCTION_DATA[size]
        print(f'\nSize {size}: Waist={prod["Waist"]}, Hip={prod["Low_Hip"]}, '
              f'Inseam={prod["Inseam"]}, Ankle={prod["Ankle"]}')
        design = map_production_to_design(prod, BODY_YAML)
        print(f'  Design: width={design["pants"]["width"]["v"]:.3f}, '
              f'length={design["pants"]["length"]["v"]:.3f}, '
              f'flare={design["pants"]["flare"]["v"]:.3f}, '
              f'thigh={design["pants"]["thigh"]["v"]:.3f}, '
              f'knee={design["pants"]["knee"]["v"]:.3f}')
        folder, name = generate_pattern(size, design, BODY_YAML, output_base)
        generated.append((folder, name, size))

        # Annotate pattern SVG with key measurements
        svg_files = [f for f in folder.glob('*_pattern.svg') if '_print_' not in f.name]
        spec_files = list(folder.glob('*_specification.json'))
        if svg_files and spec_files:
            annot_meas = {
                'Waist': prod['Waist'],
                'Inseam': prod['Inseam'],
                'Ankle': prod['Ankle'],
                'Hip': prod['Low_Hip'],
                'Front Rise': prod['Front Rise'],
                'Back Rise': prod['Back Rise'],
            }
            # Optional techpack fields — only included when the product provides them.
            # Thigh/Knee are omitted by some garment configs (e.g. pleated or
            # multi-segment leg styles). Hip_from_waist/Knee_from_crotch are
            # vertical position data that some techpacks don't supply.
            for opt_key in ('Thigh', 'Knee', 'Hip_from_waist', 'Knee_from_crotch'):
                if opt_key in prod:
                    annot_meas[opt_key] = prod[opt_key]
            annotate_pattern_svg(svg_files[0], spec_files[0], annot_meas)

    # Step 2: Verify measurements from spec JSONs
    print("\n--- Step 2: Verifying pattern measurements ---")
    for folder, name, size in generated:
        spec_files = list(folder.glob('*_specification.json'))
        if spec_files:
            verify_measurements(spec_files[0], size, PRODUCTION_DATA[size])
        else:
            print(f'  No spec file found for size {size}')

    # Step 3: Simulate each pattern on A-pose body
    print("\n--- Step 3: Running simulations (A-pose) ---")
    sim_results = []
    for folder, name, size in generated:
        print(f'\nSimulating size {size}...')
        try:
            sim_folder = simulate_pattern(folder, name, output_base)
            if sim_folder:
                sim_results.append((sim_folder, size))
        except Exception as e:
            print(f'  Simulation failed for size {size}: {e}')
            import traceback
            traceback.print_exc()

    # Step 4: Create combined meshes
    print("\n--- Step 4: Creating combined body+garment meshes ---")
    for sim_folder, size in sim_results:
        print(f'\nCombining size {size}...')
        try:
            save_combined_mesh(sim_folder)
        except Exception as e:
            print(f'  Combined mesh failed for size {size}: {e}')
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print("\nGenerated patterns:")
    for folder, name, size in generated:
        print(f"  Size {size}: {folder}")
    print("\nSimulation results:")
    for sim_folder, size in sim_results:
        print(f"  Size {size}: {sim_folder}")
    print(f"\nAll output in: {output_base}")
