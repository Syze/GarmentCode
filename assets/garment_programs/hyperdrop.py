"""Hyperdrop garments built directly from their AAMA DXF patterns.

One builder per style. Each reads the shell pieces via `aama_dxf`, cuts every
outline into named seams, wraps them in `DxfPanel`s, places those around the
body and declares the stitches -- the DXFs carry no connectivity at all, so the
seam map is authored here.

Outer shell only: pleats, gathers, welt pockets, collar build-up, facings,
belt loops and appliques are deliberately not modelled.
"""
from __future__ import annotations

import numpy as np
import pygarment as pyg
from scipy.spatial.transform import Rotation as R

from assets.garment_programs.aama_dxf import (
    _auto_rw,
    L_INTERNAL, DxfPanel, arc, face_to, nearest_idx, pieces_for_size,
    place_around, split_at_points, split_chain,
)

DATA = './hyperdrop_data'

# Start-of-sim Z for the pant legs, copied verbatim from the parametric pants
# (pants.py: front `translate_by([..., 25])`, back `translate_by([..., -20])`).
# The asymmetry is deliberate there and the 45 cm total gap is what gives the
# outseam/inseam welds enough dz to wrap the panels around the body instead of
# fighting body_friction. The ±12 this used before was half that.
Z_FRONT, Z_BACK = 25.0, -20.0


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _normalised(piece):
    """Boundary + notches with x centred on the bbox and y=0 at the lowest point."""
    b = piece.boundary
    c = np.array([(b[:, 0].min() + b[:, 0].max()) / 2, b[:, 1].min()])
    notches = piece.notches - c if len(piece.notches) else np.zeros((0, 2))
    return b - c, notches


def _by_keyword(pieces, **keywords):
    """Pick pieces by a substring of their (Chinese or Latin) piece name."""
    out = {}
    for key, needle in keywords.items():
        hits = [p for p in pieces if needle in p.name]
        if not hits:
            raise KeyError(f'no piece matching {needle!r}; have '
                           f'{[p.name for p in pieces]}')
        out[key] = hits[0]
    return out


def _stitch(a, b):
    """Stitch pair with the weld direction decided from the real 3D geometry.

    Reused from the CLO pants path: pairing by edge direction is unreliable
    where seams meet at sharp angles, but endpoint proximity is not.
    """
    return (a, _auto_rw(a, b))


# --------------------------------------------------------------------------- #
#  HDORIA -- short-sleeve tee (front, back, 2 sleeves)
# --------------------------------------------------------------------------- #
class HyperdropTee(pyg.Component):
    """HDORIA SS pleated t-shirt, shell only. Real pintucks, band only.

    The waist pintucks ARE the shape of this garment: the DXF panel is a straight
    56.6 cm rectangle, ten 1.769 cm tucks (layer 11, 10 cm tall, evenly spaced)
    pull it to 39 cm, and it flares back to a 57 cm hem -- that flare is the peplum.

    Each face is cut into a yoke, ELEVEN band strips and a peplum. Every strip
    edge that sits on a tuck line pulls in by half the tuck's width, tapered by a
    smoothstep, so the take-up is ZERO at the band's top and bottom and full
    across the stitched height. Consequences:
      * the narrowing is distributed over the ten tuck seams, exactly where the
        garment puts it, instead of being taken off the two side edges,
      * the yoke and peplum seams are length-matched, so there is NO gather --
        an earlier version gathered the whole width onto a uniform band and read
        as heavy ruching, an abrupt band, then heavy ruching again,
      * the width changes continuously through the band, and the ten seams land
        on the tuck lines so they are visible.

    Only the band is stripped. Cutting the whole face into strips would also slice
    the neckline, both shoulders and both armholes, forcing matching cuts in the
    sleeve cap.

    Front and back are full-width (the DXF does not cut them on the fold), so
    there is no CF/CB seam -- only shoulders, sides and armholes.
    """
    DXF = f'{DATA}/HDORIA Tee/32000928.dxf'

    FRONT_KEYS = [59, 401, 515, 731, 845, 1187, 1247, 1535]
    BACK_KEYS = [59, 354, 444, 852, 942, 1237, 1297, 1299]
    TORSO_SEAMS = ['armhole_r', 'shoulder_r', 'neck', 'shoulder_l',
                   'armhole_l', 'side_l', 'hem', 'side_r']
    N_SAMPLE = 26           # y samples describing a strip's tapered side
    STRIP_STAGGER = 0.75    # alternate strips offset in z so none start coplanar
    # Fraction of each layer-11 line pair's spacing that the sewn tuck actually
    # takes out of the flat width. Treating the full spacing as take-up made the
    # finished 1/2 waist 4.14cm narrow against the spec sheet at EVERY size
    # (33.36/35.36/38.36 vs 37.50/39.50/42.50). The ratio spec-take-up /
    # measured-take-up is 0.805 for S, M and L alike, so one factor corrects the
    # whole size run without touching the grading.
    TUCK_TAKEUP = 0.805

    def __init__(self, body, design=None, size='M') -> None:
        super().__init__('hyperdrop_tee')
        pieces = pieces_for_size(self.DXF, size)
        p = _by_keyword(pieces, front='前片', back='后片', sleeve='袖子')
        hps = body['height'] - body['head_l']

        # One band shared by both faces: their marked tuck bands differ (the
        # front's are staggered to follow its dipped hem) and the cut heights must
        # agree or the side seams will not pair.
        bands = [self._tucks(p[k]) for k in ('front', 'back')]
        # The stripped band is exactly the MARKED tuck extent, and the fan-out
        # taper lives inside it. Putting the taper outside instead made the band
        # 23.4 cm tall against a 13.4 cm marked extent, so the tuck seams ran more
        # than twice the length of the real 10 cm tucks.
        self.y0 = min(b[2] for b in bands)
        self.y1 = max(b[3] for b in bands)
        self.cuts = np.mean([b[0] for b in bands], axis=0)      # 10 tuck lines
        self.gaps = np.mean([b[1] for b in bands], axis=0) * self.TUCK_TAKEUP
        # Full take-up holds over the tucks' own height; the rest of the band is
        # the fan-out at each end.
        tuck_h = float(np.mean([b[4] for b in bands]))
        self.taper = max(0.5, (self.y1 - self.y0 - tuck_h) / 2)

        self.faces = {}
        for name, piece, keys, z in (('f', p['front'], self.FRONT_KEYS, 16.0),
                                     ('b', p['back'], self.BACK_KEYS, -16.0)):
            self.faces[name] = self._face(f'tee_{name}', piece, keys, hps, z)
        self.sleeve_r = HyperdropSleeve('tee_sleeve_r', p['sleeve'], body, side=+1)
        self.sleeve_l = HyperdropSleeve('tee_sleeve_l', p['sleeve'], body, side=-1)
        self.subs = [pan for f in self.faces.values() for pan in f.values()]

        F, B = self.faces['f'], self.faces['b']
        n = len(self.cuts) + 1
        rules = [
            _stitch(F['yoke'].interfaces['shoulder_r'], B['yoke'].interfaces['shoulder_r']),
            _stitch(F['yoke'].interfaces['shoulder_l'], B['yoke'].interfaces['shoulder_l']),
            _stitch(F['yoke'].interfaces['side_r'], B['yoke'].interfaces['side_r']),
            _stitch(F['yoke'].interfaces['side_l'], B['yoke'].interfaces['side_l']),
            _stitch(F['peplum'].interfaces['side_r'], B['peplum'].interfaces['side_r']),
            _stitch(F['peplum'].interfaces['side_l'], B['peplum'].interfaces['side_l']),
            # the outermost strips carry the band's share of the side seams
            _stitch(F[f'strip{n - 1}'].interfaces['right'],
                    B[f'strip{n - 1}'].interfaces['right']),
            _stitch(F['strip0'].interfaces['left'], B['strip0'].interfaces['left']),
        ]
        for face in (F, B):
            for k in range(n):
                rules += [
                    _stitch(face['yoke'].interfaces[f'bot_{k}'],
                            face[f'strip{k}'].interfaces['top']),
                    _stitch(face['peplum'].interfaces[f'top_{k}'],
                            face[f'strip{k}'].interfaces['bottom']),
                ]
            # the tuck seams themselves
            for k in range(n - 1):
                rules.append(_stitch(face[f'strip{k}'].interfaces['right'],
                                     face[f'strip{k + 1}'].interfaces['left']))
        for tag, sl in (('r', self.sleeve_r), ('l', self.sleeve_l)):
            rules += [
                _stitch(sl.front.interfaces['top'], sl.back.interfaces['top']),
                _stitch(sl.front.interfaces['underarm'], sl.back.interfaces['underarm']),
                _stitch(F['yoke'].interfaces[f'armhole_{tag}'], sl.front.interfaces['cap']),
                _stitch(B['yoke'].interfaces[f'armhole_{tag}'], sl.back.interfaces['cap']),
            ]
        self.stitching_rules = pyg.Stitches(*rules)
        self.interfaces = {'f_bottom': F['peplum'].interfaces['hem'],
                           'b_bottom': B['peplum'].interfaces['hem']}
        # Panel labels drive the body-collision filters in meshgen/garment.py:
        # only 'arm' panels keep colliding with the arms, everything else has the
        # arms filtered out. Sleeves first, then overwrite=False for the rest.
        self.sleeve_r.set_panel_label('arm')
        self.sleeve_l.set_panel_label('arm')
        self.set_panel_label('body', overwrite=False)

    # ---- tuck geometry ----------------------------------------------------
    @staticmethod
    def _tucks(piece):
        """(cut positions, take-up per cut, band lo, band hi) from DXF layer 11.

        Tucks are marked as pairs of near-vertical lines; the pair's midpoint is
        the tuck line and its gap is the width sewing that tuck removes.
        """
        b = piece.boundary
        c = np.array([(b[:, 0].min() + b[:, 0].max()) / 2, b[:, 1].min()])
        lines = [v - c for layer, v in piece.internal
                 if layer == '11' and abs(v[0, 0] - v[-1, 0]) < 0.2
                 and 9.5 < abs(v[0, 1] - v[-1, 1]) < 10.5]
        xs = np.array(sorted(v[0, 0] for v in lines))
        pairs = [(xs[i], xs[i + 1]) for i in range(len(xs) - 1)
                 if 1.5 < xs[i + 1] - xs[i] < 2.1]
        return (np.array([(a + b_) / 2 for a, b_ in pairs]),
                np.array([b_ - a for a, b_ in pairs]),
                min(min(v[0, 1], v[-1, 1]) for v in lines),
                max(max(v[0, 1], v[-1, 1]) for v in lines),
                float(np.mean([abs(v[0, 1] - v[-1, 1]) for v in lines])))

    def _profile(self, y):
        """Fraction of the tuck take-up in effect at height y (0 at band edges)."""
        t = self.taper
        f = np.minimum(np.clip((np.asarray(y, float) - self.y0) / t, 0, 1),
                       np.clip((self.y1 - np.asarray(y, float)) / t, 0, 1))
        return f * f * (3.0 - 2.0 * f)

    # ---- face construction -----------------------------------------------
    def _face(self, name, piece, keys, hps, z):
        loop, _ = _normalised(piece)
        chains = dict(zip(self.TORSO_SEAMS,
                          [arc(loop, keys[i], keys[(i + 1) % len(keys)])
                           for i in range(len(keys))]))
        translation = [0, hps - loop[:, 1].max(), z]
        facing = [0, 0, np.sign(z)]

        # side_r runs hem -> underarm so it crosses y0 then y1; side_l the reverse
        sr = _split_at_arclens(chains['side_r'],
                               _arc_at_ys(chains['side_r'], (self.y0, self.y1)))
        sl = _split_at_arclens(chains['side_l'],
                               _arc_at_ys(chains['side_l'], (self.y1, self.y0)))
        # The outermost strips inherit the REAL side-seam curve over the band, and
        # the two chords take their outer ends from the outline at their own
        # height. Using one mean x for both instead left the yoke's chord ending
        # off its side seam, and the closing edge jogged outward across it -- a
        # self-intersecting panel, which crashes triangulation with no message.
        band_r = sr[1]                       # side_r over the band, y0 -> y1
        band_l = sl[1][::-1]                 # side_l over the band, y0 -> y1
        ys = np.linspace(self.y0, self.y1, self.N_SAMPLE)
        prof = self._profile(ys)
        n = len(self.cuts) + 1

        def cut_edge(x, half):
            """A tuck line pulled in by `half` of its take-up."""
            return np.column_stack([x - half * prof, ys])

        out = {}
        for k in range(n):
            if k == 0:
                left = band_l
            else:
                left = cut_edge(self.cuts[k - 1], -self.gaps[k - 1] / 2)
            if k == n - 1:
                right = band_r
            else:
                right = cut_edge(self.cuts[k], self.gaps[k] / 2)
            # Adjacent strips meet with zero gap at the band's top and bottom
            # (the take-up tapers to nothing there). Coplanar panels touching
            # along an edge produced 46903 edge-edge contacts at frame 1 and the
            # watchdog killed the run, so alternate strips are nudged in z. Each
            # panel is still flat -- all of its vertices share one z.
            z_off = self.STRIP_STAGGER * (1 if k % 2 else -1) * np.sign(z)
            strip = DxfPanel(f'{name}_strip{k}', {
                'bottom': np.array([left[0], right[0]]),
                'right': right,
                'top': np.array([right[-1], left[-1]]),
                'left': left[::-1],
            # All four edges single-cubic. Piecewise-fitting the tapered sides
            # splits them into 3, and because strip k's right runs y0->y1 while
            # strip k+1's left runs y1->y0 the two sets come out in OPPOSITE
            # order -- edge 0 then pairs with edge 0 across opposite ends of the
            # seam, which invalidated 112 of the stitches. One cubic per side
            # costs 0.24 cm of length and 0.39 cm of deviation, and adjacent
            # strips are exact mirrors so they still match each other exactly.
            }, single=('bottom', 'right', 'top', 'left'),
                translation=[translation[0], translation[1],
                             translation[2] + z_off])
            out[f'strip{k}'] = face_to(strip, facing)

        bounds_hi = [band_l[-1][0], *self.cuts, band_r[-1][0]]
        bounds_lo = [band_l[0][0], *self.cuts, band_r[0][0]]

        # yoke: everything above y1, closed by a chord split at the strip bounds
        chord_top = self._chord(self.y1, bounds_hi)
        yoke = {'side_r': sr[2], 'armhole_r': chains['armhole_r'],
                'shoulder_r': chains['shoulder_r'], 'neck': chains['neck'],
                'shoulder_l': chains['shoulder_l'],
                'armhole_l': chains['armhole_l'], 'side_l': sl[0]}
        yoke.update({f'bot_{k}': seg for k, seg in enumerate(chord_top)})
        out['yoke'] = face_to(DxfPanel(
            f'{name}_yoke', yoke,
            single=tuple(k for k in yoke if k not in ('neck',)),
            translation=translation), facing)

        # peplum: everything below y0, chord split the same way but reversed to
        # keep the loop counter-clockwise
        # Each chord segment must run right-to-left here, not just be listed in
        # reverse: DxfPanel welds consecutive endpoints, so left-to-right segments
        # in a right-to-left loop get dragged onto each other and the chord comes
        # out 5 cm short.
        chord_bot = self._chord(self.y0, bounds_lo)
        peplum = {'hem': chains['hem'], 'side_r': sr[0]}
        peplum.update({f'top_{k}': seg[::-1] for k, seg in
                       reversed(list(enumerate(chord_bot)))})
        peplum['side_l'] = sl[2]
        out['peplum'] = face_to(DxfPanel(
            f'{name}_peplum', peplum,
            single=tuple(k for k in peplum if k != 'hem'),
            translation=translation), facing)
        return out

    @staticmethod
    def _chord(y, bounds):
        """Horizontal chord at height y, split at each strip boundary."""
        return [np.array([[bounds[k], y], [bounds[k + 1], y]])
                for k in range(len(bounds) - 1)]


class HyperdropSleeve(pyg.Component):
    """One short sleeve, split at the shoulder notch into front/back halves.

    The DXF stores the sleeve as a single flat piece (cap arc, two underarm
    seams, hem). Splitting it at the cap apex and the hem midpoint gives the
    same two-panel layout `sleeves.Sleeve` uses, which drapes far more reliably
    than trying to place one flat piece around the arm.

    Local frame per half, matching `SleevePanel`: shoulder point at the origin,
    the fold line running -x along the arm, and the sleeve width spanning -y for
    BOTH halves, so stitching fold<->fold and underarm<->underarm closes the
    tube.
    """

    def __init__(self, name, piece, body, side=+1):
        super().__init__(name)
        loop, notches = _normalised(piece)
        # Shoulder split: prefer the cap notch the pattern maker marked over the
        # geometric apex (they differ by ~0.7 cm here).
        i_apex = (nearest_idx(loop, max(notches, key=lambda q: q[1]))
                  if len(notches) else int(np.argmax(loop[:, 1])))
        apex = loop[i_apex].copy()
        hem_y = loop[:, 1].min()
        hem = np.where(loop[:, 1] < hem_y + 0.05)[0]
        i_hem_p, i_hem_m = int(hem[np.argmax(loop[hem, 0])]), int(hem[np.argmin(loop[hem, 0])])
        i_hem_mid = nearest_idx(loop, [apex[0], hem_y])
        # underarm corners = the extreme-x vertices above the hem
        above = loop[:, 1] > hem_y + 0.5
        i_ua_p = int(np.argmax(np.where(above, loop[:, 0], -1e9)))
        i_ua_m = int(np.argmin(np.where(above, loop[:, 0], +1e9)))

        # Both halves are plain consecutive arcs of the CCW outline; the fold
        # line is supplied by close_loop().
        #
        # The -x cap half matches the FRONT armhole (26.87 vs 26.55 cm) and the
        # +x half the BACK (27.35 vs 27.81); the other pairing is ~1 cm worse on
        # both. The single cap notch also sits on the -x half, which is where a
        # front sleeve notch belongs.
        halves = {
            'f': (['cap', 'underarm', 'hem'],
                  [(i_apex, i_ua_m), (i_ua_m, i_hem_m), (i_hem_m, i_hem_mid)]),
            'b': (['hem', 'underarm', 'cap'],
                  [(i_hem_mid, i_hem_p), (i_hem_p, i_ua_p), (i_ua_p, i_apex)]),
        }
        rot = np.array([[0.0, 1.0], [-1.0, 0.0]])       # -90 deg about Z
        for role, (names, spans) in halves.items():
            seams = {n: (arc(loop, a, b) - apex) @ rot.T
                     for n, (a, b) in zip(names, spans)}
            if role == 'f':
                # the -x half comes out spanning +y; flip so both halves hang
                # the same way off the shared fold line
                seams = {k: v * [1.0, -1.0] for k, v in seams.items()}
            panel = DxfPanel(f'{name}_{role}', seams,
                             single=('cap', 'underarm'),
                             translation=[-body['shoulder_w'] / 2, 0,
                                          15.0 if role == 'f' else -15.0])
            panel.interfaces['top'] = panel.interfaces.pop('_closing')
            setattr(self, 'front' if role == 'f' else 'back', panel)

        self.subs = [self.front, self.back]
        self.translate_by([0, body['height'] - body['head_l'], 0])
        # Rotate each PANEL about its own origin rather than calling
        # Component.rotate_by: the component rotates about its bbox pivot, which
        # is not the shoulder and drags the sleeve ~8 cm up and 3 cm out. Every
        # half already has the shoulder point at its local (0, 0), so a plain
        # panel rotation swings the sleeve down the arm and leaves the shoulder
        # exactly where it was placed.
        arm = R.from_euler('XYZ', [0, 0, body['arm_pose_angle']], degrees=True)
        for panel in self.subs:
            panel.rotate_by(arm)
        if side > 0:
            self.mirror()
        # each half wraps the arm from its own side
        face_to(self.front, [0, 0, 1])
        face_to(self.back, [0, 0, -1])


# --------------------------------------------------------------------------- #
#  HDCLARA -- mid-waist wide-leg pants (2 legs + waistband)
# --------------------------------------------------------------------------- #
def _rect_seams(width, height, names=('bottom_l', 'bottom_r', 'side_r',
                                      'top_r', 'top_l', 'side_l')):
    """CCW rectangle centred on x, split at x=0 top and bottom.

    Splitting the band's long edges in half turns the band<->legs joins into
    clean 1:1 edge pairs (one per leg), which keeps the per-edge weld direction
    decidable; a single long band edge against two leg waists is an n-to-m match
    whose orientation has to be hand-tuned instead.
    """
    w, h = width / 2, height
    return dict(zip(names, [
        np.array([[-w, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [w, 0.0]]),
        np.array([[w, 0.0], [w, h]]),
        np.array([[w, h], [0.0, h]]),
        np.array([[0.0, h], [-w, h]]),
        np.array([[-w, h], [-w, 0.0]]),
    ]))


class HyperdropPantsLeg(pyg.Component):
    """One leg of the HDCLARA pants: front + back panel.

    Both DXF legs are stored with the crotch on +x and the outseam on -x, so an
    unbuilt leg occupies negative x. The other side is built from a REFLECTED
    outline (`reflect=True`) rather than by mirroring the assembled leg: mirroring
    flips each panel's winding, and because DxfPanel disables autonorm the facing
    is then never corrected, so one leg ends up inside out. Mirroring afterwards
    also invalidates this component's own outseam/inseam welds, which were decided
    from the geometry before the mirror. Building from reflected input keeps both
    correct. (`pants_clo` carries the same `reflect` path for the same reason.)

    Every stitched seam is a SINGLE cubic here, splitting the outseam and rise at
    their notches where needed. Multi-edge interfaces are a hazard across a
    reflection: reflecting flips the winding, `face_to` then reverses the panel's
    edge list, and an interface's edges end up in the opposite order from its
    partner's, so edge i no longer pairs with edge i.
    """
    FRONT_KEYS = [25, 26, 27, 94, 118]
    BACK_KEYS = [30, 31, 32, 135, 201]
    SEAMS = ['hem', 'inseam', 'rise', 'waist', 'outseam']

    def __init__(self, tag, front_piece, back_piece, crotch_y,
                 z_front=Z_FRONT, z_back=Z_BACK, reflect=False):
        super().__init__(f'pant_{tag}')
        # Panels take the PARAMETRIC pants' names (pant_f_l / pant_b_l / ...).
        # run_custom_pants' sim-time placement looks the four leg panels up by
        # exactly those names, so matching them is what lets the DXF pants reuse
        # the crotch lift + pose-X leg alignment instead of being skipped.
        x_off = 0.5 if reflect else -0.5   # crotch tips 1 cm apart about centre
        self.front, self.f_waist_y = self._leg(
            f'pant_f_{tag}', front_piece, self.FRONT_KEYS, crotch_y, z_front,
            x_off, reflect)
        self.back, self.b_waist_y = self._leg(
            f'pant_b_{tag}', back_piece, self.BACK_KEYS, crotch_y, z_back,
            x_off, reflect)
        self.subs = [self.front, self.back]

        n = min(self.front.n_outseam, self.back.n_outseam)
        self.stitching_rules = pyg.Stitches(
            *[_stitch(self.front.interfaces[f'outseam_{i}'],
                      self.back.interfaces[f'outseam_{i}']) for i in range(n)],
            _stitch(self.front.interfaces['inseam'], self.back.interfaces['inseam']),
        )
        self.n_rise = min(self.front.n_rise, self.back.n_rise)
        self.interfaces = {
            'waist_f': self.front.interfaces['waist'],
            'waist_b': self.back.interfaces['waist'],
            **{f'rise_f_{i}': self.front.interfaces[f'rise_{i}']
               for i in range(self.front.n_rise)},
            **{f'rise_b_{i}': self.back.interfaces[f'rise_{i}']
               for i in range(self.back.n_rise)},
        }

    def _leg(self, name, piece, keys, crotch_y, z, x_off, reflect=False):
        loop, notches = _normalised(piece)
        if reflect:
            loop = loop * [-1.0, 1.0]
            notches = notches * [-1.0, 1.0] if len(notches) else notches
        seams = {s: arc(loop, keys[i], keys[(i + 1) % len(keys)])
                 for i, s in enumerate(self.SEAMS)}
        # Split the outseam and the rise at the DXF's own match notches. The
        # outseam is dead straight for 78 cm and then curves over the hip; as one
        # cubic it bowed 4.1 cm where the source bows 2.2. Both legs carry the
        # same notches, which is what makes the sub-seams correspond.
        parts = {}
        for key in ('outseam', 'rise'):
            chunks = split_at_points(seams.pop(key), notches)
            for i, chunk in enumerate(chunks):
                parts[f'{key}_{i}'] = chunk
            parts[f'n_{key}'] = len(chunks)
        counts = {k: parts.pop(k) for k in ('n_outseam', 'n_rise')}
        seams = {'hem': seams['hem'], 'inseam': seams['inseam'],
                 **{k: v for k, v in parts.items() if k.startswith('rise')},
                 'waist': seams['waist'],
                 **{k: v for k, v in parts.items() if k.startswith('outseam')}}
        # Everything that gets stitched is a single cubic; only the free hem is
        # fitted piecewise. See the class docstring on multi-edge interfaces.
        single = tuple(k for k in seams if k != 'hem')
        panel = DxfPanel(name, seams, single=single,
                         pivot=loop[keys[2]],          # crotch point -> origin
                         translation=[x_off, crotch_y, z])
        panel.n_outseam, panel.n_rise = counts['n_outseam'], counts['n_rise']
        face_to(panel, [0, 0, np.sign(z)])
        waist_y = float(seams['waist'][:, 1].mean() - loop[keys[2]][1])
        return panel, waist_y


class HyperdropPants(pyg.Component):
    """HDCLARA MW wide pants, shell only.

    Dropped from the DXF: the belt loop (KOPRU) and the bias strip for the rose
    applique (GUL). The back waist dart is not sewn as a dart -- its width is
    taken out of the waistband split instead, so the back waist eases onto the
    band by the same amount the dart would have removed.
    """
    DXF = f'{DATA}/HDCLARA Pants/32000645.dxf'
    # Faced double-ply band (DXF: KEMER X2 TELA X2). OFF: two full-circumference
    # plies generate ~17k persistent edge-edge contacts, which overflows the
    # contact buffer at default and needs >30s/frame even with it enlarged.
    FACED_BAND = False
    # Z between the two plies at frame 0. Must clear BOTH plies' collision
    # thickness (2 x fabric_thickness = 1.0cm) with margin: at 1.0 the surfaces
    # touch, every band edge pair registers a contact, and the edge-edge buffer
    # overflows (17k contacts vs a 250 limit) and the watchdog kills frame 1.
    # The top-edge fold seam closes the gap during the zero-gravity steps.
    FACING_GAP = 3.0

    def __init__(self, body, design=None, size='M') -> None:
        super().__init__('hyperdrop_pants')
        pieces = pieces_for_size(self.DXF, size)
        by_block = {p.block.split('-')[0]: p for p in pieces}
        front, back, band = (by_block['HDCLARA_3'], by_block['Pattern_4'],
                             by_block['Pattern_5'])

        # Canonical, body-agnostic placement, exactly as the parametric pants:
        # build with the garment crotch at Y=0 and let sim-time placement
        # (_place_pattern_for_body) apply the body's crotch lift and the pose-X
        # leg alignment. generate_pattern stamps the properties.placement marker
        # that path keys off.
        crotch_y = 0.0

        self.right = HyperdropPantsLeg('r', front, back, crotch_y)
        self.left = HyperdropPantsLeg('l', front, back, crotch_y,
                                      reflect=True)

        # --- waistband split -------------------------------------------------
        wf = self.right.interfaces['waist_f'].edges.length()
        wb = self.right.interfaces['waist_b'].edges.length()
        dart = _waist_dart(back)
        band_len = float(band.boundary[:, 0].ptp())
        band_h = float(band.boundary[:, 1].ptp())
        # The band is the constraint: 2*wf at the front and 2*(wb - dart) at the
        # back are what actually has to fit onto it once the darts are folded
        # out, so divide the band in that ratio and let the leftover become ease.
        eff_f, eff_b = 2 * wf, 2 * max(wb - dart, 1.0)
        band_f_len = band_len * eff_f / (eff_f + eff_b)
        band_b_len = band_len - band_f_len

        self.band_f = self._band('wb_front', band_f_len, band_h,
                                 crotch_y + self.right.f_waist_y, Z_FRONT)
        self.band_b = self._band('wb_back', band_b_len, band_h,
                                 crotch_y + self.right.b_waist_y, Z_BACK)
        # Faced band: the DXF annotates the band piece 'KEMER X2 TELA X2' -- cut
        # twice in shell plus twice in interfacing -- so the real band is a folded
        # double ply, not the single strip we had. The facing OVERLAPS the outer
        # band (same Y range, same length); only the Z differs, by FACING_GAP, so
        # the two plies don't start interpenetrating. The top edges are stitched:
        # that seam is the fold, and it makes the facing a second hoop-carrying
        # ring, which is what stops the band stretching 13-18% and walking over
        # the hip. The facing's bottom edge is left free, as it is in the garment.
        self.facing_f = self.facing_b = None
        if self.FACED_BAND:
            self.facing_f = self._band(
                'wb_front_facing', band_f_len, band_h,
                crotch_y + self.right.f_waist_y, Z_FRONT - self.FACING_GAP)
            self.facing_b = self._band(
                'wb_back_facing', band_b_len, band_h,
                crotch_y + self.right.b_waist_y, Z_BACK + self.FACING_GAP)
        # Gather goes on the LONGER side -- the leg waists ease onto the band.
        for leg in (self.right, self.left):
            _set_ruffle(leg.interfaces['waist_f'], 2 * wf / band_f_len)
            _set_ruffle(leg.interfaces['waist_b'], 2 * wb / band_b_len)

        self.stitching_rules = pyg.Stitches(
            # centre front and centre back seams, leg to leg
            *[_stitch(self.right.interfaces[f'rise_{s}_{i}'],
                      self.left.interfaces[f'rise_{s}_{i}'])
              for s in ('f', 'b')
              for i in range(min(self.right.n_rise, self.left.n_rise))],
            # band side seams
            _stitch(self.band_f.interfaces['side_r'], self.band_b.interfaces['side_r']),
            _stitch(self.band_f.interfaces['side_l'], self.band_b.interfaces['side_l']),
            # band to legs: the unmirrored leg sits at -x, the mirrored one at +x
            _stitch(self.band_f.interfaces['bottom_l'], self.right.interfaces['waist_f']),
            _stitch(self.band_f.interfaces['bottom_r'], self.left.interfaces['waist_f']),
            _stitch(self.band_b.interfaces['bottom_l'], self.right.interfaces['waist_b']),
            _stitch(self.band_b.interfaces['bottom_r'], self.left.interfaces['waist_b']),
            # facing: its own side seams close it into a ring, then the top edges
            # weld to the outer band's -- that seam is the fold at the band's top.
            *([_stitch(self.facing_f.interfaces['side_r'], self.facing_b.interfaces['side_r']),
               _stitch(self.facing_f.interfaces['side_l'], self.facing_b.interfaces['side_l']),
               _stitch(self.band_f.interfaces['top_l'], self.facing_f.interfaces['top_l']),
               _stitch(self.band_f.interfaces['top_r'], self.facing_f.interfaces['top_r']),
               _stitch(self.band_b.interfaces['top_l'], self.facing_b.interfaces['top_l']),
               _stitch(self.band_b.interfaces['top_r'], self.facing_b.interfaces['top_r'])]
              if self.FACED_BAND else []),
        )
        self.interfaces = {
            # NOTE: with FACED_BAND these top edges are also the fold seam to the
            # facing, so there is no free edge at the top of the garment any more.
            # Fine for the pants alone; a layered outfit that stitches 'top' to an
            # upper garment would be putting a second stitch on the same edges.
            'top': pyg.Interface.from_multiple(
                self.band_f.interfaces['top_l'], self.band_f.interfaces['top_r'],
                self.band_b.interfaces['top_l'], self.band_b.interfaces['top_r']),
        }
        # 'leg' is what marks this as a lower-body garment for the body-collision
        # filters (meshgen/garment.py): it filters the arms out for every particle,
        # so the hands can't push the pants around. The waistband stays 'body'.
        self.left.set_panel_label('leg')
        self.right.set_panel_label('leg')
        self.set_panel_label('body', overwrite=False)

    def _band(self, name, length, height, y, z):
        panel = DxfPanel(name, _rect_seams(length, height),
                         translation=[0, y, z])
        return face_to(panel, [0, 0, np.sign(z)])


def _fold_about(panel, pivot, axis, deg):
    """Rotate a already-placed panel about the world line (pivot, axis).

    `Panel.rotate_by` turns about the panel's own origin, which would swing the
    lapel away from the roll line instead of hinging on it, so compose the
    rotation into both the panel's rotation AND its translation here.
    """
    axis = np.asarray(axis, float)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    Rf = R.from_rotvec(axis * np.deg2rad(deg))
    pivot = np.asarray(pivot, float)
    t = np.asarray(panel.translation, float)
    panel.rotation = Rf * panel.rotation
    panel.translation = (pivot + Rf.apply(t - pivot)).tolist()


def _zero_pivot(panel):
    """Move the panel's local origin onto its first vertex, exactly.

    Panel.assembly() runs set_pivot(edges[0].start, replicate_placement=True),
    which shifts vertices by int() of that point while setting the translation
    from its exact value -- displacing the panel by the fractional part. If the
    first vertex is already (0, 0) the whole step is a no-op.
    """
    v0 = np.array(list(panel.edges[0].start), float)
    if np.allclose(v0, 0.0):
        return
    for v in panel.edges.verts():
        v[0] -= v0[0]
        v[1] -= v0[1]
    panel.translation = (np.asarray(panel.translation, float)
                         + panel.rotation.apply([v0[0], v0[1], 0.0])).tolist()


def _rot_for_meshgen(panel):
    """Re-encode the panel's rotation into the convention boxmeshgen reads.

    Panel.assembly() writes self.rotation.as_euler('XYZ') -- scipy INTRINSIC,
    i.e. Rx.Ry.Rz. boxmeshgen feeds those angles to rotation_tools.euler_xyz_to_R,
    which builds Rz.Ry.Rx (scipy EXTRINSIC 'xyz'). The two agree only for a
    single-axis rotation, which is every other panel in this codebase -- so a
    genuinely 3-axis rotation like a folded lapel lands somewhere else entirely
    in the simulated mesh (measured: x[-4.25, 11.50] intended vs x[-27.72, -4.79]
    actual). Store the rotation whose 'XYZ' angles ARE the extrinsic ones.

    Call LAST: afterwards panel.rotation no longer describes the true orientation,
    only what serializes correctly.
    """
    panel.rotation = R.from_euler(
        'XYZ', panel.rotation.as_euler('xyz', degrees=True), degrees=True)


def _int_align(panel):
    """Make the panel's first local vertex integral, compensating in translation.

    Panel.assembly() runs set_pivot(edges[0].start, replicate_placement=True),
    which sets `translation` to the EXACT world position of that point but shifts
    the vertices by int() of it -- so the panel is displaced by the fractional
    part. For an unrotated panel that is a constant offset nobody notices; rotate
    the panel and the offset rotates with it, so a hinged lapel drifts off its
    seam (0.23cm at fold 0 became 1.22cm at fold 150). Zeroing the fraction makes
    the truncation exact and the displacement vanish. World geometry is unchanged
    here: the vertices move by -frac and the translation by +rotation*frac.
    """
    v0 = np.array(list(panel.edges[0].start), float)
    frac = v0 - np.floor(v0)
    if np.allclose(frac, 0.0):
        return
    for v in panel.edges.verts():
        v[0] -= frac[0]
        v[1] -= frac[1]
    panel.translation = (np.asarray(panel.translation, float)
                         + panel.rotation.apply([frac[0], frac[1], 0.0])).tolist()


def _world_pt(panel, pt2d):
    """A panel-local 2D point in world coordinates, after placement."""
    return (np.asarray(panel.translation, float)
            + panel.rotation.apply([pt2d[0], pt2d[1], 0.0]))


def _tilt_run(chain, max_tilt=10.0):
    """Arc-length fractions (f0, f1) of the chain's longest near-VERTICAL run.

    The closure belongs on the straight vertical part of the front edge, not on
    the angled sweep at the hem. Centring the band by arc length put it at
    23-31 deg off vertical; the vertical run on HDRITA sits just below the lapel
    break at 0.7 deg. Returning fractions (not cm) lets the same cut be applied
    to the under front's placket, whose total length differs.
    """
    P = np.asarray(chain, float)
    if len(P) < 3:
        return 0.35, 0.65
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    ang = np.degrees(np.arctan2(np.abs(np.diff(P[:, 0])), np.abs(np.diff(P[:, 1]))))
    ok = ang < max_tilt
    best = (0, 0)
    i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if arc[j + 1] - arc[i] > arc[best[1]] - arc[best[0]]:
                best = (i, j + 1)
            i = j + 1
        else:
            i += 1
    if best[1] == best[0] or arc[-1] <= 0:
        return 0.35, 0.65
    return arc[best[0]] / arc[-1], arc[best[1]] / arc[-1]


def _split_frac(chain, f0, f1):
    """Split into (lower, middle, upper) at two arc-length FRACTIONS."""
    total = _arclen(chain)
    lo, rest = split_chain(chain, min(max(f0 * total, 0.5), total - 1.0))
    mid, hi = split_chain(rest, min(max((f1 - f0) * total, 0.5),
                                    _arclen(rest) - 0.5))
    return lo, mid, hi


def _mid_split(chain, span):
    """Split `chain` into (lower, middle, upper) with the middle `span` long and
    centred on the chain -- the band a closure holds, with both ends left free."""
    total = _arclen(chain)
    span = min(span, total - 1.0)
    lo, rest = split_chain(chain, max((total - span) / 2.0, 0.5))
    mid, hi = split_chain(rest, span)
    return lo, mid, hi


def _weld(a, b, right_wrong):
    """Stitch pair with an explicit weld direction.

    `right_wrong=None` falls back to deciding from the geometry (_stitch).
    """
    if right_wrong is None:
        return _stitch(a, b)
    out = pyg.Interface.from_multiple(b)
    out.right_wrong = [bool(right_wrong)] * len(out.edges)
    return (a, out)


def _arc_at_ys(chain, ys):
    """Arc lengths along `chain` where it crosses each height in `ys`."""
    chain = np.asarray(chain, float)
    step = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    out = []
    for yv in ys:
        for i in range(len(chain) - 1):
            a, b = chain[i, 1], chain[i + 1, 1]
            if (a - yv) * (b - yv) <= 0 and abs(b - a) > 1e-9:
                out.append(float(s[i] + (yv - a) / (b - a) * step[i]))
                break
    return sorted(out)


def _split_at_arclens(chain, offsets, min_seg=1.0):
    """Split a chain at the given arc-length offsets from its start.

    Seam correspondence must be measured along the seam, not by height: the tee's
    front hem dips 3.06 cm below the back's, so its side seam starts 3 cm higher
    even though the two are sewn from the same corner. Splitting both by y put
    the cuts 3 cm out of step; splitting both at the same arc length from the hem
    keeps every sub-seam paired.
    """
    chain = np.asarray(chain, float)
    total = float(np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1)))
    cuts = sorted(c for c in offsets if min_seg < c < total - min_seg)
    out, rest, base = [], chain, 0.0
    for c in cuts:
        if c - base < min_seg:
            continue
        head, rest = split_chain(rest, c - base)
        out.append(head)
        base = c
    out.append(rest)
    return out


def _poly_world(panel):
    """Panel outline in world XY (rotation is identity for flat placement)."""
    return np.array([panel.point_to_3D(list(e.start))[:2] for e in panel.edges])


def _x_span_at(poly, y):
    """(min_x, max_x) where a closed polygon crosses height y, or None."""
    xs = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (a[1] - y) * (b[1] - y) <= 0 and abs(b[1] - a[1]) > 1e-9:
            xs.append(a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]))
    return (min(xs), max(xs)) if len(xs) >= 2 else None


def _clear_right(local_poly, refs, gap, min_x=None, step=0.5):
    """Smallest x offset that keeps `local_poly` clear of every ref polygon.

    Panels in one z layer start coplanar, so any x overlap is a real
    self-intersection at frame 0 -- the solver has to push them apart before it
    can stitch, which is what tangles a drape. Butting mating edges together is
    not enough because those edges are slanted; this walks the shared height
    range and shifts until the outlines genuinely separate.
    """
    need = 0.0 if min_x is None else min_x - local_poly[:, 0].min()
    # Seed from the polygon's GLOBAL min x as well as the per-height scan: the
    # scan samples at `step` and can step over the true extreme (centre back
    # overshot x=0 by 0.44 cm that way).
    ys = np.arange(local_poly[:, 1].min(), local_poly[:, 1].max(), step)
    for y in ys:
        cand = _x_span_at(local_poly, y)
        if cand is None:
            continue
        if min_x is not None:
            need = max(need, min_x - cand[0])
        for ref in refs:
            r = _x_span_at(ref, y)
            if r is not None:
                need = max(need, r[1] - cand[0] + gap)
    return max(need, 0.0)


def _place_flat(panel, x, y, z, mirror=False):
    """Place a panel flat -- no rotation, every vertex at one z.

    The stitches then wrap it around the body, which is how the pants panels are
    placed. `mirror=True` reflects the panel and negates the translation, so pass
    the +x partner's offset.
    """
    panel.translate_to([x, y, z])
    if mirror:
        panel.mirror()
    return face_to(panel, [0.0, 0.0, np.sign(z)])


def _mirrored(panel, theta, y, rx, rz):
    """Place a panel at +theta, then mirror it onto the -theta side.

    mirror() negates the x translation and the Y/Z euler angles, so a panel
    placed at +theta lands exactly at -theta with mirrored geometry. Facing has
    to be restated afterwards because DxfPanel disables autonorm.
    """
    place_around(panel, theta, y, rx, rz)
    panel.mirror()
    th = np.radians(-theta)
    return face_to(panel, [np.sin(th), 0.0, np.cos(th)])


def _set_ruffle(interface, coeff):
    """Interface.ruffle is a list of {coeff, sec} sections, not a bare float."""
    interface.ruffle = [dict(coeff=float(coeff), sec=[0, len(interface.edges)])]


def _waist_dart(back_piece, band=2.5):
    """Width of the back waist dart, read off the pair of notches the pattern
    marks on the waist edge. Returns 0 when no such pair exists."""
    loop, notches = _normalised(back_piece)
    if len(notches) < 2:
        return 0.0
    top = notches[notches[:, 1] > loop[:, 1].max() - band]
    if len(top) < 2:
        return 0.0
    return float(max(np.linalg.norm(a - b)
                     for i, a in enumerate(top) for b in top[i + 1:]))


# --------------------------------------------------------------------------- #
#  HDBAKIRA -- sleeveless boatneck midi dress
# --------------------------------------------------------------------------- #
class HyperdropDress(pyg.Component):
    """HDBAKIRA SL boatneck midi dress, shell only.

    Ten panels: a full-length centre-front, front/back side bodices, a centre
    back band, and four gathered skirt panels.

    The DXF has **no centre-back bodice panel** -- only two smocking templates,
    a 23.43 x 11.50 cm piece and an 11.72 x 11.50 one (exactly 2:1). The band is
    rebuilt from the smaller, FINISHED one -- 11.72 x 11.50 -- on the reading that
    23.43 is the same band before smocking. The smocking itself is not modelled.

    Panels are placed FLAT -- no rotation, every vertex of a panel at one z, front
    group in front of the body and back group behind it -- and the stitches wrap
    them around, exactly as the pants panels do. Laying them out tangent to an
    ellipse instead only helped the initial guess and made the required flip
    direction depend on theta.

    Every seam pairs to within 0.5 cm; see the arc lengths in the seam map below.
    """
    DXF = f'{DATA}/HDBAKIRA Dress/32001145.dxf'

    KEYS = {
        'fc': ([7, 35, 49, 50, 102, 103, 117, 145],
               ['side_r', 'armhole_r', 'shoulder_r', 'neck',
                'shoulder_l', 'armhole_l', 'side_l', 'hem']),
        'fs': ([19, 28, 29, 49], ['armhole', 'sideseam', 'waist', 'princess']),
        'bs': ([0, 4, 5, 54, 55, 95],
               ['waist', 'sideseam', 'armhole', 'shoulder', 'scoop', 'band']),
        'bk': ([0, 115, 116, 182], ['hem', 'side', 'top', 'cb']),
        'fk': ([19, 20, 36, 37], ['princess', 'top', 'side', 'hem']),
    }
    FREE = ('neck', 'hem', 'armhole', 'armhole_r', 'armhole_l', 'scoop')
    # A panel placed at +x with no rotation must carry the edge that faces x=0 on
    # its local -x side. For the front group that edge is the centre-front-facing
    # one (fs/fk princess, stored on +x -> flip); for the back group it is the
    # centre-BACK-facing one (bs band, bk cb), which the DXF already stores on -x.
    FLIP = ('fs', 'fk')
    # centre-front side edge: skirt below the waist notch, bodice above
    FC_SKIRT_LEN = 75.25
    FRONT_Z, BACK_Z = 18.0, -18.0
    LAYER = 4.0         # skirt sits this much further out than the bodice
    PANEL_GAP = 1.5     # clearance between coplanar panels at frame 0
    # Everything hangs off HPS: only fc and bs are pinned to the body (their top
    # vertex sits at HPS), the other four panels chain off those two. So this one
    # offset raises the whole garment.
    HPS_LIFT = 5.0      # cm above the body's high point shoulder
    # Shift every panel back (negative z) at frame 0. NOTE: nothing pins this
    # garment -- no attachment constraint, no panel springs, empty vertex_labels
    # -- so initial placement only decides where the sim STARTS. A 5cm HPS lift
    # measurably changed nothing in the settled mesh (fc top 142.28 -> 142.33).
    Z_SHIFT = -5.0
    # Labelling the skirt 'leg' makes garment.py's is_lower_body test fire (leg
    # panels present, no arm panels), which filters the arms out of body
    # collisions -- and not just for the skirt: the sweep at garment.py:351-356
    # puts every remaining unfiltered particle in the arm filter too, so the
    # whole dress passes through the arms. On the avatar that was the single
    # biggest drape win of the session; on the SMPL A-pose body it let the
    # bodice sink 3.15cm into the arms (vs 1.39cm with it off), so it is off by
    # default. Config override: `hyperdrop: {skirt_as_leg: 1}`.
    SKIRT_AS_LEG = False
    FREE_EDGE_LIFT = {}  # {seam name: cm} -- raise a free edge's interior
    FREE_EDGE_LIFT_POW = 2.0  # lift profile width: lower = flatter/broader
    SHOULDER_TRIM = 0.0  # cm off each shoulder seam at the armhole end
    FREE_EDGE_FLATTEN = {}   # {seam name: 0..1} pull a free edge toward its chord
    BACK_SCOOP_RAISE = 0.0   # cm; raise the bs scoop/band corner (back neck)

    def __init__(self, body, design=None, size='M') -> None:
        super().__init__('hyperdrop_dress')
        pieces = pieces_for_size(self.DXF, size, fabric_only=False)
        get = lambda zh: next(p for p in pieces if zh in p.name)
        raw = {'fc': get('前中'), 'fs': get('前侧'), 'bs': get('后侧拼'),
               'bk': get('后裙片'), 'fk': get('前裙侧')}
        band_piece = get('司马克毛样')        # smocking template, 23.43 x 11.50
        loops = {k: _normalised(v)[0] for k, v in raw.items()}

        hps = body['height'] - body['head_l'] + self.HPS_LIFT
        fc_y = hps - loops['fc'][:, 1].max()

        # ---- panels -------------------------------------------------------
        self.fc = self._panel('dress_fc', 'fc', loops['fc'], split_fc=True)
        _place_flat(self.fc, 0.0, fc_y, self.FRONT_Z)

        # front side: its top corner is the same point as the centre front's
        # underarm, which fixes its height
        fs_y = fc_y + float(loops['fc'][35][1]) - loops['fs'][:, 1].max()
        bs_y = hps - loops['bs'][:, 1].max()
        fk_y = fc_y + self._fc_waist_y - float(loops['fk'][20][1])

        # centre-back band, full width (see the class docstring)
        bw = float(band_piece.boundary[:, 0].ptp()) / 2.0    # 23.43 -> 11.72 cm
        bh = float(band_piece.boundary[:, 1].ptp())
        self.band_w = bw
        band_y = bs_y + float(loops['bs'][95][1]) - bh
        # The lower back block is translated, not stretched, so the band seam keeps
        # its length and cb keeps its size -- it just rides up with the block.
        band_y += float(self.BACK_SCOOP_RAISE)
        self.cb = DxfPanel('dress_cb', _rect_seams(bw, bh),
                           single=('bottom_l', 'bottom_r', 'side_l', 'side_r'))
        _place_flat(self.cb, 0.0, band_y, self.BACK_Z)

        bk_y = fk_y + float(loops['fk'][36][1]) - float(loops['bk'][116][1])
        w_bs_src = _arclen(arc(loops['bs'], *self.KEYS['bs'][0][0:2]))
        self.bk_top_outer = w_bs_src / (w_bs_src + bw / 2)
        # bk's gathered top feeds the back-side waist (outer) and half the band
        # (inner); the split follows those two lengths, so it moves with the band
        # width. _panel reads this when it cuts bk, hence set before building it.

        # x offsets: pushed out until the outlines genuinely clear each other.
        # Bodice and skirt sit on separate z layers (18/22 and -18/-22) so a
        # bodice panel can never intersect a skirt one; within a layer only two
        # or three panels have to be kept apart in x. Each panel is still flat --
        # every one of its vertices shares a single z.
        GAP = self.PANEL_GAP
        fc_poly = _poly_world(self.fc)
        cb_poly = _poly_world(self.cb)

        def flat(name, kind, y, z, refs, min_x=None, **kw):
            p = self._panel(name, kind, loops[kind], **kw)
            # Lift the candidate into world Y before comparing: the refs are
            # already placed, so matching their height range is what makes the
            # clearance test see the real overlap at all.
            probe = np.array([list(e.start) for e in p.edges])
            probe[:, 1] += y
            return _place_flat(p, _clear_right(probe, refs, GAP, min_x), y, z)

        self.fs_r = flat('dress_fs_r', 'fs', fs_y, self.FRONT_Z, [fc_poly])
        self.bs_r = flat('dress_bs_r', 'bs', bs_y, self.BACK_Z, [cb_poly])
        self.fk_r = flat('dress_fk_r', 'fk', fk_y, self.FRONT_Z + self.LAYER,
                         [fc_poly])
        self.bk_r = flat('dress_bk_r', 'bk', bk_y, self.BACK_Z - self.LAYER,
                         [], min_x=GAP / 2, split_bk=True)

        # Mirrors: same offset, then mirror, which negates the translation.
        self.fs_l = _place_flat(self._panel('dress_fs_l', 'fs', loops['fs']),
                                self.fs_r.translation[0], fs_y, self.FRONT_Z, True)
        self.bs_l = _place_flat(self._panel('dress_bs_l', 'bs', loops['bs']),
                                self.bs_r.translation[0], bs_y, self.BACK_Z, True)
        self.fk_l = _place_flat(self._panel('dress_fk_l', 'fk', loops['fk']),
                                self.fk_r.translation[0], fk_y,
                                self.FRONT_Z + self.LAYER, True)
        self.bk_l = _place_flat(self._panel('dress_bk_l', 'bk', loops['bk'],
                                            split_bk=True),
                                self.bk_r.translation[0], bk_y,
                                self.BACK_Z - self.LAYER, True)

        # ---- gathers: the skirt eases onto the bodice ----------------------
        for fk in (self.fk_r, self.fk_l):
            _set_ruffle(fk.interfaces['top'],
                        fk.interfaces['top'].edges.length()
                        / self.fs_r.interfaces['waist'].edges.length())
        w_bs = self.bs_r.interfaces['waist'].edges.length()
        for bk in (self.bk_r, self.bk_l):
            _set_ruffle(bk.interfaces['top_outer'],
                        bk.interfaces['top_outer'].edges.length() / w_bs)
            _set_ruffle(bk.interfaces['top_inner'],
                        bk.interfaces['top_inner'].edges.length() / (bw / 2))

        # ---- seams ---------------------------------------------------------
        rules = [
            _stitch(self.fc.interfaces['side_bodice_r'], self.fs_r.interfaces['princess']),
            _stitch(self.fc.interfaces['side_bodice_l'], self.fs_l.interfaces['princess']),
            _stitch(self.fc.interfaces['side_skirt_r'], self.fk_r.interfaces['princess']),
            _stitch(self.fc.interfaces['side_skirt_l'], self.fk_l.interfaces['princess']),
            _stitch(self.fc.interfaces['shoulder_r'], self.bs_r.interfaces['shoulder']),
            _stitch(self.fc.interfaces['shoulder_l'], self.bs_l.interfaces['shoulder']),
            _stitch(self.fs_r.interfaces['sideseam'], self.bs_r.interfaces['sideseam']),
            _stitch(self.fs_l.interfaces['sideseam'], self.bs_l.interfaces['sideseam']),
            # No rotation now, so the band's local -x edge is on the world -x
            # side and pairs with the left-hand panels directly.
            _stitch(self.bs_r.interfaces['band'], self.cb.interfaces['side_r']),
            _stitch(self.bs_l.interfaces['band'], self.cb.interfaces['side_l']),
            _stitch(self.fk_r.interfaces['side'], self.bk_r.interfaces['side']),
            _stitch(self.fk_l.interfaces['side'], self.bk_l.interfaces['side']),
            _stitch(self.bk_r.interfaces['cb'], self.bk_l.interfaces['cb']),
            _stitch(self.fs_r.interfaces['waist'], self.fk_r.interfaces['top']),
            _stitch(self.fs_l.interfaces['waist'], self.fk_l.interfaces['top']),
            _stitch(self.bs_r.interfaces['waist'], self.bk_r.interfaces['top_outer']),
            _stitch(self.bs_l.interfaces['waist'], self.bk_l.interfaces['top_outer']),
            _stitch(self.cb.interfaces['bottom_r'], self.bk_r.interfaces['top_inner']),
            _stitch(self.cb.interfaces['bottom_l'], self.bk_l.interfaces['top_inner']),
        ]
        self.stitching_rules = pyg.Stitches(*rules)
        # Arm collision. A sleeveless upper garment trips NEITHER of the two
        # conditions meshgen/garment.py uses to filter the arms out
        # (is_lower_body needs 'leg' panels, has_arm_panels needs 'arm' ones), so
        # the dress collides with the arms and wraps around them. Labelling the
        # skirt 'leg' -- which is what it physically covers -- makes is_lower_body
        # true, and the existing logic then filters the arms for every particle.
        # No change to pygarment required.
        if self.SKIRT_AS_LEG:
            for n in ('fk_r', 'fk_l', 'bk_r', 'bk_l'):
                getattr(self, n).set_panel_label('leg')

        if self.Z_SHIFT:
            for sub in self._get_subcomponents():
                if hasattr(sub, 'edges'):
                    sub.translate_by([0.0, 0.0, self.Z_SHIFT])

        self.set_panel_label('body', overwrite=False)


    # (shoulder chain, armhole chain) pairs -- the shoulder seam is stitched, the
    # armhole is free, so trimming the shoulder at its armhole end and dragging
    # the armhole along narrows the strap without breaking any seam match.
    SHOULDER_PAIRS = (('shoulder_r', 'armhole_r'), ('shoulder_l', 'armhole_l'),
                      ('shoulder', 'armhole'))

    # bs chains below the scoop corner, moved as one rigid block, and the two
    # free chains that bridge the block to the fixed strap above it.
    BACK_BLOCK = ('band', 'waist', 'sideseam')
    BACK_BRIDGE = (('scoop', False), ('armhole', True))   # (chain, dy at start?)

    def _raise_back_scoop(self, seams, dy):
        """Raise the whole lower back block by dy, shortening the strap above it.

        The corner where the scoop meets the centre-back band IS the bottom of
        the back neck opening. Rather than stretch the band edge to reach a
        higher corner -- which would make the centre-back panel taller -- the
        band/waist/sideseam block is TRANSLATED bodily upwards, so every one of
        its dimensions is preserved and the cb rectangle keeps its size (__init__
        just places it dy higher).

        Only the two free chains bridging the block to the strap absorb the
        move: `scoop` at its band end and `armhole` at its underarm end, each
        ramped to 0 at the shoulder so that corner does not budge. Both shorten,
        so this also takes a little more out of the armhole ring.
        """
        for key in self.BACK_BLOCK:
            if key in seams:
                c = np.array(seams[key], float)
                c[:, 1] += dy
                seams[key] = c
        for key, at_start in self.BACK_BRIDGE:
            if key not in seams:
                continue
            c = np.array(seams[key], float)
            seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
            t = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
            c[:, 1] += dy * ((1.0 - t) if at_start else t)
            seams[key] = c

    def _trim_shoulder(self, seams, trim):
        """Shorten each shoulder seam by `trim` cm at its armhole end.

        The shoulder is trimmed identically on the front and back panels, so the
        pair still matches. The armhole chain is translated by a ramp -- full
        delta at the shoulder corner, zero at the underarm -- which keeps the
        underarm corner on the side seam and leaves the outline closed.
        """
        for sh, ah in self.SHOULDER_PAIRS:
            if sh not in seams or ah not in seams:
                continue
            S = np.array(seams[sh], float)
            A = np.array(seams[ah], float)
            # find which ends of the two chains are the shared corner
            si, ai = min(((i, j) for i in (0, -1) for j in (0, -1)),
                         key=lambda p: np.linalg.norm(S[p[0]] - A[p[1]]))
            old = S[si].copy()
            # walk `trim` cm inward along the shoulder from the shared corner
            order = S[::-1] if si == -1 else S
            seg = np.linalg.norm(np.diff(order, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            if trim >= cum[-1]:
                raise ValueError(f'shoulder_trim {trim} exceeds {sh} length {cum[-1]:.2f}')
            k = int(np.searchsorted(cum, trim))
            f = (trim - cum[k - 1]) / max(seg[k - 1], 1e-9)
            new = order[k - 1] + f * (order[k] - order[k - 1])
            kept = np.vstack([[new], order[k:]])
            seams[sh] = kept[::-1] if si == -1 else kept
            # drag the armhole's shared end to the new corner, tapering to 0
            d = new - old
            aseg = np.linalg.norm(np.diff(A, axis=0), axis=1)
            t = np.concatenate([[0.0], np.cumsum(aseg)]) / max(aseg.sum(), 1e-9)
            w = (1.0 - t) if ai == 0 else t
            seams[ah] = A + w[:, None] * d

    def _panel(self, name, kind, loop, split_fc=False, split_bk=False):
        if kind in self.FLIP:
            loop = np.asarray(loop, float) * [-1.0, 1.0]
        keys, names = self.KEYS[kind]
        seams = {n: arc(loop, keys[i], keys[(i + 1) % len(keys)])
                 for i, n in enumerate(names)}
        # Re-cut named FREE edges by lifting their interior. The endpoints stay
        # put, so every seam this panel shares is untouched -- only the open edge
        # changes shape. Used to raise the necklines to match a reference drape.
        for key, lift in (self.FREE_EDGE_LIFT or {}).items():
            if key not in seams or not lift:
                continue
            c = np.array(seams[key], float)
            seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
            t = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
            # pow 2 is a centre bump; pow<1 broadens it into a flat boat-neck
            # lift that still tapers to 0 at both shared corners.
            # clip: sin(pi*t) lands a hair below 0 at t=1, and a fractional
            # power of a negative is nan
            c[:, 1] += lift * np.clip(np.sin(np.pi * t), 0.0, None) ** self.FREE_EDGE_LIFT_POW
            seams[key] = c
        # Pull a free edge toward the straight chord between its own endpoints.
        # Both ends stay put, so seams are untouched; the arc gets shorter, which
        # on a concave armhole means filling the scoop in -- a smaller opening.
        for key, frac in (self.FREE_EDGE_FLATTEN or {}).items():
            if key not in seams or not frac:
                continue
            c = np.array(seams[key], float)
            seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
            t = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
            chord = c[0] + t[:, None] * (c[-1] - c[0])
            seams[key] = c + float(frac) * (chord - c)
        if self.BACK_SCOOP_RAISE and kind == 'bs':
            self._raise_back_scoop(seams, float(self.BACK_SCOOP_RAISE))
        if self.SHOULDER_TRIM:
            self._trim_shoulder(seams, float(self.SHOULDER_TRIM))
        if split_fc:
            # side_r runs hem -> underarm, side_l runs underarm -> hem, so the
            # skirt portion is at opposite ends of the two chains
            sk_r, bd_r = split_chain(seams.pop('side_r'), self.FC_SKIRT_LEN)
            tot_l = float(np.sum(np.linalg.norm(
                np.diff(seams['side_l'], axis=0), axis=1)))
            bd_l, sk_l = split_chain(seams.pop('side_l'), tot_l - self.FC_SKIRT_LEN)
            seams = {'side_skirt_r': sk_r, 'side_bodice_r': bd_r,
                     **seams, 'side_bodice_l': bd_l, 'side_skirt_l': sk_l}
            # reorder to keep the outline order: hem is last in KEYS
            order = ['side_skirt_r', 'side_bodice_r', 'armhole_r', 'shoulder_r',
                     'neck', 'shoulder_l', 'armhole_l', 'side_bodice_l',
                     'side_skirt_l', 'hem']
            seams = {k: seams[k] for k in order}
            self._fc_waist_y = float(sk_r[-1][1])
        if split_bk:
            # The back skirt's gathered top feeds two partners: the back-side
            # waist (outer) and half the centre-back band (inner). Cut it in the
            # ratio of those two lengths so both halves gather by the SAME amount
            # -- an even cut instead concentrates all the fullness at centre back.
            top = seams.pop('top')
            tot = _arclen(top)
            outer, inner = split_chain(top, tot * self.bk_top_outer)
            seams = {'hem': seams['hem'], 'side': seams['side'],
                     'top_outer': outer, 'top_inner': inner, 'cb': seams['cb']}
        single = tuple(k for k in seams if k not in self.FREE)
        return DxfPanel(name, seams, single=single)



# --------------------------------------------------------------------------- #
#  HDRITA -- long-sleeve fitted blazer
# --------------------------------------------------------------------------- #
def _arclen(pts):
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(pts, float), axis=0), axis=1)))


class HyperdropBlazer(pyg.Component):
    """HDRITA LS fitted blazer, shell only.

    Twelve panels: front centre (carrying the lapel), front side, back centre,
    back side and a two-piece tailored sleeve, all mirrored. Dropped from the
    DXF: linings, fusibles, the front facing, back neck facing and loop, welt
    pockets and pocket bags, and the collar -- the neckline/gorge is left free,
    so the lapel is present but the collar that folds over it is not.

    Seam pairings all close: side seam 38.82/38.86, back princess 51.60/52.37,
    front princess 50.55/49.57, shoulder 13.44/13.77, sleeve seams 46.73/46.82
    and 54.46/54.46, sleeve cap 54.27 against a 52.90 armhole (1.4 cm ease).
    """
    DXF = f'{DATA}/HDRITA Blazer/32001051-size set.dxf'

    KEYS = {
        'fc': ([59, 503, 599, 671, 749, 923, 1361],
               ['front_edge', 'lapel', 'gorge', 'shoulder', 'armhole',
                'princess', 'hem']),
        'fs': ([83, 539, 791, 1079], ['princess', 'armhole', 'sideseam', 'hem']),
        'bc': ([131, 401, 461, 723, 813, 867],
               ['cb', 'hem', 'princess', 'armhole', 'shoulder', 'neck']),
        'bs': ([419, 509, 803, 1103], ['hem', 'sideseam', 'armhole', 'princess']),
        'us': ([71, 665, 863], ['cap', 'seam_b', 'hem']),
        'un': ([41, 239, 443, 677], ['seam_b', 'cap', 'seam_a', 'hem']),
        # C_3: the collar band. Its SHORTER long edge (36.62 cm) is the neckline
        # edge -- it meets 2x gorge + 2x back neck = 37.37 cm -- and the longer
        # one (42.96) is the free outer style edge.
        'co': ([47, 71, 371, 395, 443, 887],
               ['end_l', 'outer', 'end_r', 'notch_r', 'neck', 'notch_l']),
    }
    FREE = ('hem', 'front_edge', 'lapel', 'gorge', 'neck',
            'outer', 'end_l', 'end_r', 'notch_r', 'notch_l',
            'neck_0', 'neck_3')
    # The upper sleeve's long 71->665 edge is the hindarm seam followed by the
    # cap; the split is at the length of the matching under-sleeve edge.
    US_SEAM_A = 54.46
    RX, RZ = 24.0, 17.0
    # Panels are placed FLAT -- no rotation, every vertex of a panel at one z --
    # and the stitches wrap them, as the pants and dress do. A panel at +x must
    # carry its x=0-facing seam on local -x: for the front group that is the
    # front edge / princess (stored on +x, so flipped), for the back group the
    # centre-back-facing princess / cb, which the DXF already stores on -x.
    FLIP = ('fc', 'fs')
    FRONT_Z, BACK_Z = 18.0, -18.0
    LAYER = 4.0          # collar sits this much further back than the bodice
    PANEL_GAP = 1.5      # clearance between coplanar panels at frame 0
    HALF_GAP = 0.75      # centre seams start this far from x=0
    # SR_1/SL_1 (the upper sleeves) are the widest sleeve panels and sit only 8 cm
    # in z from the front bodice, so they start heavily overlapped with it in x.
    # Push them outboard; the cap stitches pull them back in.
    # 10 / 0 is the best found. Pushing further out (18 / 8) leaves the cap
    # stitches too far to close: the run then never settles (5000-frame cap,
    # 4352 vertices still moving) even though the panels overlap less at frame 0.
    UPPER_X_GAP = 10.0
    UNDER_X_GAP = 0.0
    # Experiment: drop SR_1/SL_1 (upper sleeves). The under sleeve then hangs from
    # the front-side armhole plus the lower back-side stretch, and the rest of the
    # armhole (FMR_2, B_3, upper B_1) is left as a free edge.
    UPPER_SLEEVE = False
    # Second experiment: drop the under sleeves (SR/SL) too, leaving the body
    # shell alone. Every armhole edge then becomes a free edge.
    SLEEVES = False
    # FMR_2 and FML_2 do not meet at centre front -- they OVERLAP to form the
    # button stand. The DXF has no CF line to read the stand from, so this is a
    # 8.5 cm total, measured on the garment by the user (double-breasted lap). Each front moves as a unit with its
    # lapel (they share the roll-line seam) so that seam is not stretched.
    # FRONT_Z_STEP must clear 2 x fabric_thickness with margin: the faced
    # waistband showed that plies a thickness apart put every edge pair in
    # contact at frame 0 and overflow the edge-edge buffer.
    FRONT_OVERLAP = 8.5   # measured on the garment: FMR_2 over FML_2
    # 0 = off. Truncating the front to fake a placket is the WRONG construction:
    # the DXF ships a real placket piece (门襟, 20.0 x 69.8 cm). Left off until
    # that piece is built; the invented cut self-intersected fc_l.
    BUTTON_LINE = 8.5
    # Experiment: drop the lapels. The front is then ONE panel with the lapel
    # region left flat and unfolded, and the roll-line hinge seam disappears.
    LAPELS = True
    LAPEL_FOLD = 150.0     # degrees the lapel hinges out about the roll line
    LAPEL_WELD = False     # weld direction for the lapel roll seam
    # Panel-to-panel springs holding the lapel's gorge edge to the front's, so
    # the pre-built fold survives the initial seam pulls. A STITCH here collapses
    # the two edges together and flattens the fold; a spring keeps their frame-0
    # separation. (ke, kd) or None to disable.
    LAPEL_TACK = (8000.0, 20.0)
    # The closure holds a BUTTON_SPAN band in the MIDDLE of the front edge.
    # Both ends stay free: seaming to the hem welded the fronts shut, and holding
    # only the top let the loose lower fronts swing across the body.
    BUTTON_SPAN = 18.0
    # Must exceed the UNDER front's lapel projection: that lapel folds outward
    # into the gap, and the over front then has to lie in front of it. Too small
    # and the over panel starts inside the under lapel.
    FRONT_Z_STEP = 10.0
    # Angular spans from the hem widths (fc 9.33, fs 13.69, bs 15.54, bc 13.44;
    # 52.0 cm per side). Negative, because place_around points a panel's local
    # +x toward increasing theta and every piece stores its centre-front-most
    # edge on +x; the other side is then a plain mirror.
    THETA = {'fc': -16.2, 'fs': -56.0, 'bs': -106.6, 'bc': -156.7}
    # Weld direction for the five cap seams (bs_hi, bc, fc, fs, bs_lo).
    # The sleeve starts ~50 cm from the armhole, so endpoint proximity cannot
    # tell the two pairings apart (59.9 vs 59.3 cm on the worst of them).
    # Solved instead by enumerating all 32 combinations against boxmeshgen's
    # stitch-validity check: this one is the unique combination scoring 0 bad
    # stitches (next best is 10). The three upper-cap seams run opposite to
    # their armhole partners and take the default weld; the two under-cap seams
    # run with theirs and need start<->start.
    CAP_WELD = (False, False, False, True, True)

    def __init__(self, body, design=None, size='M') -> None:
        super().__init__('hyperdrop_blazer')
        pieces = pieces_for_size(self.DXF, size, fabric_only=False)
        want = {'fc': '前中', 'fs': '前侧', 'bc': '后中', 'bs': '后侧',
                'us': '大袖面', 'un': '小袖面', 'co': '领面面'}
        raw = {k: next(p for p in pieces if p.kind == 'fabric' and v in p.name)
               for k, v in want.items()}
        loops = {k: _normalised(v)[0] for k, v in raw.items()}

        # ---- armhole budget, measured on the raw outlines --------------------
        a = {'fc': _arclen(arc(loops['fc'], 749, 923)),
             'bc': _arclen(arc(loops['bc'], 723, 813)),
             'bs': _arclen(arc(loops['bs'], 803, 1103)),
             'fs': _arclen(arc(loops['fs'], 539, 791))}
        un_cap = _arclen(arc(loops['un'], 239, 443))
        us_cap = _arclen(arc(loops['us'], 71, 665)) - self.US_SEAM_A
        # The under sleeve spans the underarm: the whole front-side armhole plus
        # the first stretch of the back-side one. The upper sleeve takes the
        # rest, divided in proportion to the three arcs it has to cover.
        self.bs_lo = un_cap - a['fs']
        bs_hi = a['bs'] - self.bs_lo
        share = us_cap / (bs_hi + a['bc'] + a['fc'])
        cap_split = (bs_hi * share, a['bc'] * share)      # cumulative below

        hps = body['height'] - body['head_l']
        y = {'fc': hps - loops['fc'][:, 1].max(),
             'bc': hps - loops['bc'][:, 1].max()}
        # side panels hang off their princess partner's top corner
        y['fs'] = y['fc'] + float(loops['fc'][923][1]) - float(loops['fs'][539][1])
        y['bs'] = y['bc'] + float(loops['bc'][723][1]) - float(loops['bs'][1103][1])

        # C_3 collar: one panel at centre back, its neckline edge cut into the
        # four arcs it meets (right gorge, right back neck, left back neck, left
        # gorge) so each is a 1:1 seam. One many-to-many stitch against a 4-edge
        # neckline is what left the armhole undetermined earlier.
        # C_3 attaches only to the TOP OF THE BACK PANELS. Its neckline edge is
        # 36.62 cm against 2 x 10.28 cm of back neck, so the middle 20.56 is
        # stitched and ~8 cm at each end is left free (that stretch runs down the
        # gorge in the real jacket). Lengths match, so there is no forced gather.
        neck_parts = [_arclen(arc(loops['bc'], 867, 131))]
        self.collar = self._panel('blazer_collar', 'co', loops['co'],
                                  neck_split=neck_parts)
        _place_flat(self.collar, 0.0, hps - loops['co'][:, 1].max(),
                    self.BACK_Z - self.LAYER)

        # Flat placement: front group in front of the body, back group behind,
        # each panel pushed out in x until its outline clears its already-placed
        # neighbours. The right-hand set is built at +x and the left is its mirror.
        GAP = self.PANEL_GAP
        Z = {'fc': self.FRONT_Z, 'fs': self.FRONT_Z,
             'bs': self.BACK_Z, 'bc': self.BACK_Z}
        refs = {self.FRONT_Z: [], self.BACK_Z: []}

        xoff = {}

        def flat(name, kind, mirror=False):
            p = self._panel(name, kind, loops[kind])
            if mirror:                        # mirror side: reuse the +x offset
                x = xoff[kind]
            else:
                probe = np.array([list(e.start) for e in p.edges])
                probe[:, 1] += y[kind]
                x = _clear_right(probe, refs[Z[kind]], GAP, min_x=self.HALF_GAP)
                xoff[kind] = x
            _place_flat(p, x, y[kind], Z[kind], mirror)
            if not mirror:
                # AFTER placing: recording the polygon first stored it at x~0, so
                # every later clearance test was measured against the wrong
                # position (B_1 ended up 14 cm inside B_3).
                refs[Z[kind]].append(_poly_world(p))
            return p

        # centre panels first, then the side panels clear of them
        front_r = self._front_panels(
            'r', loops['fc'], raw['fc'], y['fc'], self.FRONT_Z,
            refs[self.FRONT_Z], GAP)
        if len(front_r) == 2:
            (self.lapel_r, x_lap), (self.fc_r, x_fc) = front_r
        else:
            self.lapel_r, (self.fc_r, x_fc) = None, front_r[0]
            x_lap = x_fc
        refs[self.FRONT_Z].append(_poly_world(self.fc_r))
        if self.lapel_r is not None:
            refs[self.FRONT_Z].append(_poly_world(self.lapel_r))
        self.bc_r = flat('blazer_bc_r', 'bc')
        self.fs_r = flat('blazer_fs_r', 'fs')
        self.bs_r = flat('blazer_bs_r', 'bs')

        front_l = self._front_panels(
            'l', loops['fc'], raw['fc'], y['fc'], self.FRONT_Z,
            refs[self.FRONT_Z], GAP, offsets=(x_lap, x_fc),
            inset=self.BUTTON_LINE)
        if len(front_l) == 2:
            (self.lapel_l, _), (self.fc_l, _) = front_l
        else:
            self.lapel_l, (self.fc_l, _) = None, front_l[0]
        self.bc_l = flat('blazer_bc_l', 'bc', mirror=True)
        self.fs_l = flat('blazer_fs_l', 'fs', mirror=True)
        self.bs_l = flat('blazer_bs_l', 'bs', mirror=True)

        for side, tag in ((-1, 'l'), (+1, 'r')):
            if not self.SLEEVES:
                setattr(self, f'sleeve_{tag}', None)
                continue
            setattr(self, f'sleeve_{tag}', HyperdropTwoPieceSleeve(
                f'blazer_sleeve_{tag}', loops['us'], loops['un'], body,
                self.US_SEAM_A, un_cap - self.bs_lo, cap_split, side=side,
                upper_x_gap=self.UPPER_X_GAP, under_x_gap=self.UNDER_X_GAP,
                include_upper=self.UPPER_SLEEVE))

        rules = []
        for tag in ('l', 'r'):
            fc, fs = getattr(self, f'fc_{tag}'), getattr(self, f'fs_{tag}')
            bs, bc = getattr(self, f'bs_{tag}'), getattr(self, f'bc_{tag}')
            sl = getattr(self, f'sleeve_{tag}')
            rules += [
                _stitch(fc.interfaces['princess'], fs.interfaces['princess']),
                _stitch(fs.interfaces['sideseam'], bs.interfaces['sideseam']),
                _stitch(bs.interfaces['princess'], bc.interfaces['princess']),
                # Shoulder uses the DEFAULT end<->start weld rather than
                # _auto_rw: the seam folds over the top of the shoulder, so its
                # two halves start ~55 cm apart and the endpoint-proximity
                # heuristic cannot separate the pairings (61.0 vs 55.4 cm). The
                # orientation is known instead -- the front shoulder runs
                # neck->armhole and the back one armhole->neck, i.e.
                # antiparallel, which is exactly what the default does.
                (fc.interfaces['shoulder'], bc.interfaces['shoulder']),
            ]
            if sl is not None:
                rules += [
                    _weld(fs.interfaces['armhole'], sl.under.interfaces['cap_fs'], self.CAP_WELD[3]),
                    _weld(bs.interfaces['armhole_lo'], sl.under.interfaces['cap_bs'], self.CAP_WELD[4]),
                ]
            if sl is not None and sl.upper is not None:
                rules += [
                    _stitch(sl.upper.interfaces['seam_a'], sl.under.interfaces['seam_a']),
                    _stitch(sl.upper.interfaces['seam_b'], sl.under.interfaces['seam_b']),
                    _weld(bs.interfaces['armhole_hi'], sl.upper.interfaces['cap_bs'], self.CAP_WELD[0]),
                    _weld(bc.interfaces['armhole'], sl.upper.interfaces['cap_bc'], self.CAP_WELD[1]),
                    _weld(fc.interfaces['armhole'], sl.upper.interfaces['cap_fc'], self.CAP_WELD[2]),
                ]
        # Button line: the over front (right, z=19.5) seams to the under front's
        # placket. Without this the fronts are free edges and gravity opens them.
        if self.BUTTON_LINE:
            r_key = ('front_edge_mid' if 'front_edge_mid' in self.fc_r.interfaces
                     else 'front_edge')
            l_key = ('placket_mid' if 'placket_mid' in self.fc_l.interfaces
                     else 'placket')
            rules.append(_stitch(self.fc_r.interfaces[r_key],
                                 self.fc_l.interfaces[l_key]))
        rules.append(_stitch(self.bc_l.interfaces['cb'], self.bc_r.interfaces['cb']))
        # collar to neckline, walking right gorge -> right back -> left back -> left gorge
        # neck_0 and neck_3 are the free ends; only the middle two are stitched
        rules.append(_stitch(self.bc_r.interfaces['neck'],
                             self.collar.interfaces['neck_1']))
        rules.append(_stitch(self.bc_l.interfaces['neck'],
                             self.collar.interfaces['neck_2']))
        # lapel folds back along the roll line -- a real hinge, one seam per side
        for tag in ('l', 'r'):
            if getattr(self, f'lapel_{tag}') is not None:
                # Explicit weld direction. _stitch/_auto_rw got this wrong: it
                # paired the two roll edges start-to-start, joining the lapel with
                # a 180 deg twist, so the merged seam ran the opposite way from the
                # roll line and swung with the fold (11.26cm of seam movement
                # between fold 150 and 110 in the boxmesh).
                rules.append(_weld(getattr(self, f'fc_{tag}').interfaces['roll'],
                                   getattr(self, f'lapel_{tag}').interfaces['roll'],
                                   self.LAPEL_WELD))
        # Lap the fronts over each other for the button stand. The overlap is
        # measured on FMR_2/FML_2 (the `fc` panels) themselves, NOT on the
        # lapels: fc's centre-front edge has to cross the centreline by half the
        # lap. The shift is derived from where that edge actually sits so it
        # holds across sizes. Each lapel moves with its own fc -- they share the
        # roll-line seam, so shifting one without the other would stretch it.
        if self.FRONT_OVERLAP:
            half = self.FRONT_OVERLAP / 2
            # The step goes entirely OUTWARD (away from the body) onto the OVER
            # front. Splitting it symmetrically pushed the under front to z=13,
            # inside the body's front surface (~z=14), so it began the sim buried.
            for tag, sgn, dz in (('r', -1.0, self.FRONT_Z_STEP), ('l', +1.0, 0.0)):
                fc = getattr(self, f'fc_{tag}')
                xs = [v[0] + fc.translation[0] for e in fc.edges for v in (e.start, e.end)]
                cf_edge = min(xs) if sgn < 0 else max(xs)   # the CF-side edge
                dx = (sgn * half) - cf_edge                 # land it past centre
                for attr in (f'fc_{tag}', f'lapel_{tag}'):
                    pan = getattr(self, attr)
                    if pan is not None:
                        pan.translate_by([dx, 0, dz])

        # Fold the lapels LAST -- after every placement and translation. Doing it
        # inside _front_panels meant the later overlap translation was applied on
        # top of a rotation computed in the pre-translation frame, which shifted
        # the hinge itself: not one lapel vertex stayed put (min 1.22 cm). The axis
        # is taken from the BODY panel's roll seam, so the hinge is exactly the
        # shared edge, and the two roll endpoints are then provably fixed.
        if self.LAPELS:
            for tag in ('l', 'r'):
                lapel = getattr(self, f'lapel_{tag}')
                if lapel is None:
                    continue
                if not self.LAPEL_FOLD:
                    _int_align(getattr(self, f'fc_{tag}'))
                    _int_align(lapel)
                    continue
                # Hinge in the panel's OWN LOCAL frame. Computing a world-space
                # axis does not work: assembly() re-centres each panel's local
                # coordinates and rewrites its translation, so a pre-assembly
                # world axis ends up ~0.65cm off the real seam, and a 150 deg
                # turn about it displaced the hinge 1.22cm -- the lapel visibly
                # detached. set_pivot puts the panel origin ON the roll start, so
                # rotate_by (which turns about the origin) fixes that point by
                # construction, whatever assembly does afterwards.
                re = lapel.interfaces['roll'].edges
                pa = _world_pt(lapel, re[0].start)
                pb = _world_pt(lapel, re[-1].end)
                _fold_about(lapel, pa, pb - pa, self.LAPEL_FOLD)
                # Kill the assembly() pivot-truncation drift on BOTH panels, or
                # the hinge separates by the difference of their fractions.
                _int_align(getattr(self, f'fc_{tag}'))
                _zero_pivot(lapel)
                _rot_for_meshgen(lapel)

        self.stitching_rules = pyg.Stitches(*rules)
        # Sleeves keep colliding with the arms; the body panels get the arms
        # filtered out. See meshgen/garment.py's body-collision filters.
        self.panel_springs = []
        if self.LAPELS and self.LAPEL_TACK:
            ke, kd = self.LAPEL_TACK
            for tag in ('l', 'r'):
                if getattr(self, f'lapel_{tag}') is not None:
                    self.panel_springs.append(
                        [f'tack_lapel_{tag}', f'blazer_fc_{tag}', ke, kd])
        for tag in ('l', 'r'):
            sl = getattr(self, f'sleeve_{tag}')
            if sl is not None:
                sl.set_panel_label('arm')
        self.set_panel_label('body', overwrite=False)

    def _panel(self, name, kind, loop, neck_split=None):
        if kind in self.FLIP:
            loop = np.asarray(loop, float) * [-1.0, 1.0]
        keys, names = self.KEYS[kind]
        seams = {n: arc(loop, keys[i], keys[(i + 1) % len(keys)])
                 for i, n in enumerate(names)}
        if neck_split:
            chain = seams.pop('neck')
            back = neck_split[0]
            free = max((_arclen(chain) - 2 * back) / 2, 0.5)
            parts, rest = {}, chain
            for i, w in enumerate((free, back, back)):
                head, rest = split_chain(rest, w)
                parts[f'neck_{i}'] = head
            parts['neck_3'] = rest
            seams = {'end_l': seams['end_l'], 'outer': seams['outer'],
                     'end_r': seams['end_r'], 'notch_r': seams['notch_r'],
                     **parts, 'notch_l': seams['notch_l']}
        if kind == 'bs':
            lo, hi = split_chain(seams.pop('armhole'), self.bs_lo)
            seams = {'hem': seams['hem'], 'sideseam': seams['sideseam'],
                     'armhole_lo': lo, 'armhole_hi': hi,
                     'princess': seams['princess']}
        single = tuple(k for k in seams if k not in self.FREE)
        return DxfPanel(name, seams, single=single)


    def _roll_line(self, loop, piece, seams):
        """The lapel roll line: the layer-8 internal segment whose two ends sit on
        the boundary, one on the lapel edge and one on the front edge.

        On FMR_2 it runs (5.35, 61.84) -> (13.65, 32.00), 30.97 cm; the second
        point is the break point where the lapel stops folding back.
        """
        b = piece.boundary
        c = np.array([(b[:, 0].min() + b[:, 0].max()) / 2, b[:, 1].min()])
        best = None
        for layer, v in piece.internal:
            if layer != L_INTERNAL:
                continue
            w = np.asarray(v, float) - c
            if self.FLIP and 'fc' in self.FLIP:
                w = w * [-1.0, 1.0]
            if len(w) < 2 or _arclen(w) < 20.0:
                continue
            for chain, other in ((w, w[::-1]),):
                d_lap = np.min(np.linalg.norm(seams['lapel'] - chain[0], axis=1))
                d_fe = np.min(np.linalg.norm(seams['front_edge'] - chain[-1], axis=1))
                if d_lap < 0.4 and d_fe < 0.4:
                    if best is None or _arclen(chain) > _arclen(best):
                        best = chain
        return best

    def _front_panels(self, tag, loop, piece, y, z, refs, gap, offsets=None,
                      inset=None):
        """FMR_2/FML_2 split into body + lapel along the roll line.

        Two panels stitched at the roll line give the sim a real hinge there;
        one panel with the roll line merely drawn on it cannot fold.

        `inset` truncates this front at the button line, `inset` cm in from its
        own front edge, replacing that edge with a straight 'placket' seam. The
        OVER front's front_edge stitches to it, which is what stops the jacket
        opening. Only the under front gets this. A pairwise edge stitch pulls two
        edges into coincidence, so there is no way to hold a lap AND keep both
        fronts full width -- the under front gives up its `inset` extension and
        the over front's own extension provides the visible lap.
        """
        keys, names = self.KEYS['fc']
        flip = np.array([-1.0, 1.0]) if 'fc' in self.FLIP else np.array([1.0, 1.0])
        lp = np.asarray(loop, float) * flip
        seams = {n: arc(lp, keys[i], keys[(i + 1) % len(keys)])
                 for i, n in enumerate(names)}
        if not self.LAPELS:
            f0, f1 = _tilt_run(seams['front_edge'])
            fe = _split_frac(seams['front_edge'], f0, f1)
            body_seams = {
                'front_edge_lo': fe[0], 'front_edge_mid': fe[1],
                'front_edge_hi': fe[2], 'lapel': seams['lapel'],
                'gorge': seams['gorge'], 'shoulder': seams['shoulder'],
                'armhole': seams['armhole'], 'princess': seams['princess'],
                'hem': seams['hem'],
            }
            if inset:
                # The cut on the UNDER front is the OVER front's edge as it lies
                # across it. The two fronts are mirrors built from the same local
                # loop, so that curve is just this panel's own front edge shifted
                # inward by the overlap -- same shape, offset. Its ends are then
                # snapped onto the real hem and lapel chains so the loop closes on
                # actual boundary points (a straight chord between invented end
                # points crossed the outline and self-intersected the panel).
                fe = np.asarray(seams['front_edge'], float)
                hem = np.asarray(seams['hem'], float)
                lap = np.asarray(seams['lapel'], float)
                cx = np.mean(np.vstack([fe, hem, lap])[:, 0])
                sign = 1.0 if np.mean(fe[:, 0]) < cx else -1.0
                cut = fe + np.array([sign * inset, 0.0])
                i_hem = int(np.argmin(np.linalg.norm(hem - cut[0], axis=1)))
                i_lap = int(np.argmin(np.linalg.norm(lap - cut[-1], axis=1)))
                hem_rest = hem[:i_hem + 1]
                lap_rest = lap[i_lap:]
                if len(hem_rest) < 2 or len(lap_rest) < 2:
                    raise ValueError('button-line inset too deep for this front')
                cut[0], cut[-1] = hem_rest[-1], lap_rest[0]
                body_seams['hem'] = hem_rest
                body_seams['lapel'] = lap_rest
                body_seams.pop('front_edge_lo'); body_seams.pop('front_edge_mid')
                body_seams.pop('front_edge_hi')
                parts = _split_frac(cut, f0, f1)
                body_seams = {'placket_lo': parts[0], 'placket_mid': parts[1],
                              'placket_hi': parts[2], **body_seams}
            body = DxfPanel(f'blazer_fc_{tag}', body_seams,
                            single=tuple(k for k in ('shoulder', 'armhole',
                                                     'princess')
                                         if k in body_seams))
            x = (offsets[0] if offsets is not None else
                 _clear_right(np.array([list(e.start) for e in body.edges])
                              + np.array([0.0, y]), list(refs), gap,
                              min_x=self.HALF_GAP))
            _place_flat(body, x, y, z, mirror=offsets is not None)
            return [(body, x)]

        roll = self._roll_line(loop, piece, seams)
        if roll is None:
            raise ValueError('no lapel roll line found on the front centre panel')
        # Orient from GEOMETRY, not from how the DXF happens to store the segment:
        # the end nearer the lapel edge is the neckline end (A), the other is the
        # break point on the front edge (B). Taking the stored order on trust cut
        # each chain at the wrong end and produced an upside-down lapel.
        d_lap = [np.min(np.linalg.norm(seams['lapel'] - q, axis=1))
                 for q in (roll[0], roll[-1])]
        if d_lap[1] < d_lap[0]:
            roll = roll[::-1]
        A, B = roll[0], roll[-1]

        def cut(chain, pt):
            d = np.linalg.norm(chain - pt, axis=1)
            at = _arclen(chain[:int(np.argmin(d)) + 1])
            return split_chain(chain, min(max(at, 0.5), _arclen(chain) - 0.5))

        fe_lo, fe_hi = cut(seams['front_edge'], B)   # hem->B, B->lapel point
        lp_hi, lp_lo = cut(seams['lapel'], A)        # lapel point->A, A->gorge

        f0, f1 = _tilt_run(fe_lo)
        fe_parts = _split_frac(fe_lo, f0, f1)
        body_seams = {
            'front_edge_lo': fe_parts[0], 'front_edge_mid': fe_parts[1],
            'front_edge_hi': fe_parts[2], 'roll': roll[::-1], 'lapel': lp_lo,
            'gorge': seams['gorge'], 'shoulder': seams['shoulder'],
            'armhole': seams['armhole'], 'princess': seams['princess'],
            'hem': seams['hem'],
        }
        if inset:
            # The cut on the UNDER front is the OVER front's front edge as it lies
            # across it: this panel's own fe_lo shifted inward by the overlap. The
            # offset TAPERS to zero at B, because that is where the front edge and
            # the roll line meet on the garment -- the button stand exists only
            # below the lapel break. Snapping the bottom end onto the real hem
            # keeps the loop closing on a boundary point, and leaving `roll`
            # untouched keeps the lapel hinge seam matching.
            fe = np.asarray(fe_lo, float)
            hem = np.asarray(seams['hem'], float)
            allp = np.vstack([fe, hem, np.asarray(seams['princess'], float)])
            sign = 1.0 if np.mean(fe[:, 0]) < np.mean(allp[:, 0]) else -1.0
            # CONSTANT offset -- no taper. Any taper shrinks the overlap exactly
            # where the closure is stitched (a full-length linear one left 0.85cm
            # instead of 8.5), and ramping it only near the top puts a near
            # horizontal jog through the seam. So the strip keeps its full width
            # and is closed across the top by a short edge from the cut back to
            # the break point B. That also leaves `roll` untouched, so the lapel
            # hinge seam still matches exactly.
            cut = fe + np.array([sign * inset, 0.0])
            i_hem = int(np.argmin(np.linalg.norm(hem - cut[0], axis=1)))
            hem_rest = hem[:i_hem + 1]
            if len(hem_rest) < 2:
                raise ValueError('button-line inset too deep for this front')
            cut[0] = hem_rest[-1]
            # Same fractions as the over front's edge, so the two stitched
            # segments correspond -- and land on the VERTICAL run, not the
            # angled hem sweep.
            parts = _split_frac(cut, f0, f1)
            body_seams = {
                'placket_lo': parts[0], 'placket_mid': parts[1],
                'placket_hi': parts[2],
                'placket_top': np.array([cut[-1], fe[-1]]),
                'roll': roll[::-1], 'lapel': lp_lo,
                'gorge': seams['gorge'], 'shoulder': seams['shoulder'],
                'armhole': seams['armhole'], 'princess': seams['princess'],
                'hem': hem_rest,
            }
        body = DxfPanel(f'blazer_fc_{tag}', body_seams,
                                    single=tuple(k for k in ('roll', 'shoulder', 'armhole',
                                                 'princess', 'placket')
                                     if k in body_seams))
        # The lapel keeps its TRUE shape -- no in-plane reflection. Reflecting it
        # laid it flat in the same plane as the body, which has no crease: the
        # fold is faked and the sim has nothing to hinge. Instead the lapel is
        # placed on the body's own roll line and then rotated OUT of that plane by
        # LAPEL_FOLD degrees, so the roll line is a real crease from frame 0.
        lapel = DxfPanel(f'blazer_lapel_{tag}', {
            'front_edge': fe_hi, 'lapel': lp_hi, 'roll': roll,
        }, single=('roll',),
            # Labelled so its vertices land in vertex_labels.yaml and can be
            # sprung to the front's gorge (see LAPEL_TACK). Safe because this
            # edge is not stitched -- boxmeshgen only objects to labels on the
            # two sides of a stitch disagreeing.
            edge_labels={'front_edge': f'tack_lapel_{tag}'})
        # Place the body panel first, then the lapel clear of it -- giving both
        # the same offset leaves them overlapping at x=0.
        # The lapel sits BETWEEN the two front panels -- innermost, at centre
        # front -- so it is placed first and the body panel clears it.
        # Body and lapel share ONE offset: the lapel hinges on the body's roll
        # line, so their roll edges must coincide before the fold.
        if offsets is not None:              # mirror side: reuse the +x offsets
            x = offsets[1]
        else:
            probe = np.array([list(e.start) for e in body.edges])
            probe[:, 1] += y
            x = _clear_right(probe, list(refs), gap, min_x=self.HALF_GAP)
        for pan in (body, lapel):
            _place_flat(pan, x, y, z, mirror=offsets is not None)
        return [(lapel, x), (body, x)]


class HyperdropTwoPieceSleeve(pyg.Component):
    """Tailored two-piece sleeve: a wide upper panel and a narrow under panel,
    joined along the forearm (seam_b) and hindarm (seam_a) seams.

    The under sleeve's hindarm corner sits higher than its forearm corner, which
    is what identifies the two seams. Its cap therefore runs forearm -> hindarm,
    i.e. across the underarm, and the upper cap runs the other way over the top.
    Both panels are pivoted at their highest point and swung down the arm by the
    body's A-pose angle, as `sleeves.Sleeve` does.
    """

    def __init__(self, name, upper_loop, under_loop, body, seam_a_len,
                 un_cap_front, cap_split, side=+1, upper_x_gap=0.0,
                 under_x_gap=0.0, include_upper=True):
        super().__init__(name)
        keys = HyperdropBlazer.KEYS
        self.under_cap_len = un_cap_front
        self.upper = self._build(f'{name}_upper', upper_loop, keys['us'],
                                 dict(cap=('seam_a', cap_split)),
                                 seam_a_len=seam_a_len) if include_upper else None
        self.under = self._build(f'{name}_under', under_loop, keys['un'],
                                 dict(cap=(None, (un_cap_front,))))

        # Seat the cap apex at the top of the ARMHOLE, not at the neck point.
        # Hanging it from HPS starts the sleeve ~6 cm above where it has to end
        # up, and the cap stitches then drag the whole jacket down off the
        # shoulders as they close.
        shoulder_y = body['height'] - body['head_l'] - 6.0
        placed = ([(self.upper, 10.0, upper_x_gap)] if self.upper is not None
                  else []) + [(self.under, -10.0, under_x_gap)]
        for panel, z, dx in placed:
            panel.translate_to([body['shoulder_w'] / 2 + dx, shoulder_y, z])
            panel.rotate_by(R.from_euler('XYZ', [0, 0, body['arm_pose_angle']],
                                         degrees=True))
        self.subs = [p for p, _, _ in placed]
        if side < 0:
            self.mirror()
        if self.upper is not None:
            face_to(self.upper, [0, 0, 1])
        face_to(self.under, [0, 0, -1])

    @staticmethod
    def _build(name, loop, spec, splits, seam_a_len=None):
        idx, names = spec
        seams = {n: arc(loop, idx[i], idx[(i + 1) % len(idx)])
                 for i, n in enumerate(names)}
        out = {}
        for key, chain in seams.items():
            if key != 'cap':
                out[key] = chain
                continue
            prefix, cuts = splits['cap']
            if prefix:                      # upper: hindarm seam then the cap
                head, chain = split_chain(chain, seam_a_len)
                out[prefix] = head
                labels = ['cap_bs', 'cap_bc', 'cap_fc']
            else:                           # under: forearm half then hindarm
                labels = ['cap_fs', 'cap_bs']
            # `cuts` are segment lengths; each split consumes the head and
            # leaves the tail for the next one, so no cumulative offset.
            for lab, cut in zip(labels, cuts):
                head, chain = split_chain(chain, cut)
                out[lab] = head
            out[labels[-1]] = chain
        panel = DxfPanel(name, out,
                         single=tuple(k for k in out if k != 'hem'),
                         pivot=[0.0, float(np.max(loop[:, 1]))])
        return panel


# --------------------------------------------------------------------------- #
#  Pattern generation
#
#  Pattern building only -- simulation is run_garment.py's job. It calls
#  generate_pattern() for any config with `pattern_source: dxf` and then drives
#  the sim with the sim_props from the config, like every other garment here.
# --------------------------------------------------------------------------- #
class HyperdropDress2(HyperdropDress):
    """HDBAKIRA dress re-cut to match the reference drape rather than the spec.

    The spec sheet and the reference mesh disagree at the neck. Measured on the
    reference (assets/garment_configs/Hyperdrop/dress.obj) against our size-M
    drape on the same avatar:

                          reference   ours   spec sheet
      front neck drop        2.23     5.73      6.50
      centre-back neck y   119.98   115.83        --

    Our pattern is faithful to the spec sheet -- the raw DXF loop measures 6.73
    against the sheet's 6.50, verified before any seam splitting -- so the gap is
    the sheet and the reference being different revisions. This variant follows
    the reference: the front `neck` and back `scoop` are both FREE edges, so
    raising their interiors changes nothing else in the garment.
    """
    # Calibrated against the reference: pow 1.0 + 4.8cm puts the front drop at
    # 2.24 vs the reference's 2.23. A `scoop` lift was tried for the back neck
    # and is inert -- the back opening's low point IS the shared corner with the
    # centre-back band, which this profile holds fixed by design. The user also
    # confirmed the back panel drapes correctly, so the back is left alone.
    # All three knobs are coupled and were calibrated together on the settled
    # drape, not the flat pattern -- translating the back block up pulls the
    # whole garment with it and shallows the FRONT neck too, which is why the
    # neck lift is only 0.6 here (it was 2.7 before the back moved). Final size-M
    # landmarks vs the reference: front drop 2.33 / 2.23, centre-back neck
    # 119.96 / 119.98 -- 0.12cm of total error.
    NECK_LIFT = 0.6
    FREE_EDGE_LIFT_POW = 1.0

    # Left at 0: the mechanism is correct (it shortens the front and back
    # shoulders by the same amount, so the seam still matches -- verified) but it
    # barely reaches the drape. Trimming 2.0cm moved the flat-pattern shoulder
    # corner 1.90cm inboard and the settled garment's widest point in its top 3cm
    # only 0.29cm (15% carry-through), while chamfer to the reference went
    # 2.10 -> 2.19. Our drape is ~1.9cm wider than the reference uniformly over
    # its top 12cm, so the outer edge is set by where cloth lands on the
    # shoulder/deltoid, not by where the pattern cuts it. Closing that gap needs
    # a body-contact or armhole-depth change, not a shoulder trim.
    SHOULDER_TRIM = 0.0

    BACK_SCOOP_RAISE = 4.5     # cm the lower back block rides up
    # The raise alone already takes 6.9% out of the armhole ring (it lifts the
    # underarm); this adds the rest, to 41.94cm from the original 48.02 (-12.7%).
    ARMHOLE_FLATTEN = 0.30     # 0..1; pulls the whole armhole ring toward its chord

    def __init__(self, *a, **kw):
        # built here rather than as class dicts so each knob stays a plain float
        # and the config override in generate_pattern can reach it
        self.FREE_EDGE_LIFT = {'neck': float(self.NECK_LIFT)}
        f = float(self.ARMHOLE_FLATTEN)
        # every chain of the armhole ring: fc's two, plus fs's and bs's 'armhole'
        self.FREE_EDGE_FLATTEN = {'armhole': f, 'armhole_r': f, 'armhole_l': f}
        super().__init__(*a, **kw)


BUILDERS = {
    'tee': HyperdropTee,
    'pants': HyperdropPants,
    'dress': HyperdropDress,
    'dress_2': HyperdropDress2,
    'blazer': HyperdropBlazer,
}


def generate_pattern(size, config, body_yaml_path, output_base,
                     garment_prefix='hyperdrop', labels=False):
    """Build one Hyperdrop garment from its DXF and serialize the pattern.

    `size` is the DXF size label ('M'); `config['hyperdrop']['style']` selects the
    builder. Returns (folder, garment_name) to match the other generate_pattern
    entry points run_garment.py uses.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    from assets.bodies.body_params import BodyParameters

    style = config.get('hyperdrop', {}).get('style')
    if style not in BUILDERS:
        raise ValueError(f'hyperdrop.style must be one of {sorted(BUILDERS)}, '
                         f'got {style!r}')

    body = BodyParameters(body_yaml_path)
    # Optional per-run overrides of a builder's placement constants, so a search
    # can vary them from the config without editing this file. Only keys the
    # builder already defines are accepted.
    for key, val in (config.get('hyperdrop', {}) or {}).items():
        attr = key.upper()
        if key != 'style' and hasattr(BUILDERS[style], attr):
            setattr(BUILDERS[style], attr, float(val))
    garment = BUILDERS[style](body, size=size)
    garment_name = f'{garment_prefix}_size{size}'
    garment.name = garment_name

    pattern = garment.assembly()
    if garment.is_self_intersecting():
        print(f'  WARNING: {garment_name} has self-intersecting panels')
    # labels=True writes panel names and per-edge ids into the pattern png,
    # which is what you need to map a render back to the seam map in code.
    folder = Path(pattern.serialize(
        output_base, tag='_' + datetime.now().strftime('%y%m%d-%H-%M-%S'),
        to_subfolder=True, with_3d=False, with_text=labels,
        view_ids=labels, with_printable=True))

    # Lower-body garments are built with the crotch at Y=0 (body-agnostic), so
    # stamp the same marker the parametric pants use. simulate_pattern sees
    # anchor='crotch_at_zero' and runs _place_pattern_for_body: body crotch lift
    # + pose-X leg alignment against the actual sim body. Upper-body garments are
    # already placed at build time and must NOT carry the marker.
    springs = getattr(garment, 'panel_springs', None)
    if springs:
        spec_path = next(folder.glob('*_specification.json'))
        with open(spec_path) as f:
            spec = json.load(f)
        spec.setdefault('properties', {})['panel_springs'] = springs
        with open(spec_path, 'w') as f:
            json.dump(spec, f, indent=2)
        print(f'  Panel springs declared: {springs}')

    if style == 'pants':
        spec_path = next(folder.glob('*_specification.json'))
        with open(spec_path) as f:
            spec = json.load(f)
        spec.setdefault('properties', {})['placement'] = {
            'anchor': 'crotch_at_zero',
            'back_rise_lift': 0.0,   # the DXF's own back rise; no extra scoop
            'extra_x_sep': 0.0,      # DXF crotch hook is shallow; no extra gap
        }
        with open(spec_path, 'w') as f:
            json.dump(spec, f, indent=2)

    print(f'  Pattern generated: {garment_name} -> {folder}')
    return folder, garment_name
