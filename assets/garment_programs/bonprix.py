"""The five bonprix production styles, read from their AAMA DXF + .rul pair.

Reading is `aama_dxf.py`'s job; this module holds what the files themselves do
not say -- which style each DXF is, whether it goes on the upper or the lower
body, and how a requested EU size maps onto the size run its `.rul` actually
grades.

Two things about these exports differ from the Hyperdrop DXFs and matter to
everything downstream:

  * NO SEW LINE. Every piece is layer 1 (CUT) only; not one of the five files
    has a layer-14 net outline. So the raw boundary is the cut line WITH seam
    allowance, whereas every Hyperdrop garment simulates a net line, and a
    parametric GarmentCode panel is drafted straight onto its finished
    measurements. Left uncorrected every seam sits one allowance outside the
    finished garment.

    `SEAM_ALLOW` is 1.0 cm, read off the files rather than assumed: the shirred
    strips of 8642610003 and the cuff of 8672609700 carry their elastic rows on
    layer 8, and the outermost row sits 1.00-1.07 cm from the cut edge on all
    five of them, with 1.0 cm spacing between rows -- the first row lands on the
    seam line. Two different pattern rooms, the same 1 cm. `Piece.net()` applies
    it, holding layer-6 fold edges fixed since a fold is not a seam.

  * PAIRED SIZE RUNS. Three of the five grade in two-size steps
    ('32/34', '36/38', '40/42', ...), so they have no distinct 36 and 38: both
    requested sizes resolve to the same graded pattern. `size_label()` does that
    mapping and `distinct_sizes()` reports which requests collapse.

Run `python -m assets.garment_programs.bonprix` for the piece inventory, and
with `--preview DIR` to render every graded piece.
"""
from __future__ import annotations

import os

import numpy as np
import pygarment as pyg
from scipy.spatial.transform import Rotation as R

from assets.garment_programs import aama_dxf as ad
from assets.garment_programs.aama_dxf import (
    DxfPanel, arc, corners, face_to, nearest_idx, place_around,
    split_at_points, _auto_rw,
)
from assets.garment_programs.hyperdrop import (
    HyperdropSleeve, Z_BACK, Z_FRONT, _normalised, _rect_seams, _set_ruffle,
    _stitch, _world_pt,
)

DXF_DIR = 'assets/garment_configs/Bonprix'

# Inward offset (cm) applied to the cut line to recover the net sew line.
# 0.0 = simulate the cut line as drawn. See the module docstring.
SEAM_ALLOW = 1.0
# Longest part a mating seam is cut into (cm). This is the real control on how
# many welds a garment has, and therefore on how fragile its stitching is: the
# jogger's 135 welds at a 4.6 cm median came from this being 5.0, against
# hyperdrop's 18 welds at 21.4 cm (it fits one cubic per seam and splits nothing).
# Raising it means fewer, longer welds.
SEAM_MAX_SEG = 25.0
# Fit the DXF's OWN polyline, not a resampled version of it. `read_pieces`
# smooths by default, interpolating through the stored vertices -- measured, it
# leaves every stored vertex exactly on the result (0.00 mm) and adds at most
# 0.67 mm of curve bulge between them, while multiplying the point count by ~5
# (54 -> 289 on a jogger leg). That 5x is what drove the fitter to 282 edges and
# 135 welds at a 4.6 cm median, and short numerous edges are what breaks welding.
# Paying 0.7 mm to get the point count back is the right trade here; the target
# is "better than the original sims", not sub-millimetre.
# TRIED AND REVERTED: fitting the raw polyline instead of the smoothed one saved
# nothing. The jogger's edge count barely moved (282 -> 288) because it is set by
# `_paired_chains`/`_nparts` splitting seams into ~5 cm parts, NOT by the input
# point density -- and fidelity got worse (1.45 -> 4.90 mm at size 38), because
# on a sparse outline the `min_seg` merge starts cutting real corners. The 5x
# resample costs nothing that matters; `max_seg` is the real lever.
SMOOTH_OUTLINES = True
# Shortest fitted edge on the leg WAIST (cm), which is welded to the straight
# casing. Just above the box mesher's 1.0 cm resolution, which is all it needs:
# more headroom (tried 1.6 and 2.0) buys nothing once the weld DIRECTION for the
# seam is decided as a whole -- see `_stitch_matched` -- and costs 1.1 mm of
# fidelity on the waist. Sub-resolution edges still have to go, though: at the
# 4-way side corner a 1.15 cm edge had both its ends pulled into the corner's
# vertex.
WELD_MIN_SEG = 1.1
WAIST_MIN_SEG = 1.1


def _declaw(loop, keep, min_turn=150.0, thr=22.0, window=3.0):
    """Remove needle spikes from a graded outline, keeping the declared darts.

    Grading can fold an outline back on itself. The jogger -- which has no darts
    -- turns 178 deg on its back-leg rise at size 34, where sizes 38 and 40 both
    sit at 109; the polygon stays valid, so nothing else catches it, but no fit
    can follow a needle and the seam measured 100 mm off. Sizes 38 and 40 were
    1.45 mm over the same code.

    The spike is replaced by a straight chord across its base, which is what the
    outline was clearly meant to be. `keep` lists dart apexes, which look
    identical by turn angle and must survive.
    """
    loop = np.asarray(loop, float)
    keys = ad.corners(loop, thr=thr, window=window)
    ang, n = ad.turn_angles(loop)
    protect = {int(k) for k in keep}
    # A needle is a LOCAL pinch: the outline leaves a point, runs out to the tip
    # at k and comes back to essentially the same point, so there is a pair
    # (i, j) straddling k whose vertices nearly coincide. Find the tightest such
    # pair and drop only what lies between them.
    #
    # The previous rule -- drop everything between k's neighbouring CORNERS, on
    # whichever side spans fewer indices -- is unrelated to how far the spike
    # actually reaches, and picks the wrong side outright when the spike straddles
    # index 0. It took 175 of 289 vertices off a jogger leg and 59 off a
    # shirt-dress skirt, both silently.
    BASE_TOL = 0.5          # cm: how close the two legs' feet must come back
    MAX_SPAN = max(4, n // 8)
    drop, notes = set(), []
    for k in keys:
        if k in protect or abs(ang[k]) < min_turn:
            continue
        best = None
        for span in range(2, MAX_SPAN + 1):
            for lead in range(1, span):
                i, j = (k - lead) % n, (k + span - lead) % n
                if float(np.hypot(*(loop[j] - loop[i]))) <= BASE_TOL:
                    best = (i, j, span)
                    break
            if best:
                break
        if best is None:
            print(f'  WARNING: {abs(ang[k]):.0f} deg needle at vertex {k} has no '
                  f'base within {BASE_TOL} cm over {MAX_SPAN} vertices -- left '
                  f'in place rather than guessing which side to cut')
            continue
        i, j, span = best
        drop.update((i + 1 + t) % n for t in range((j - i - 1) % n))
        notes.append((abs(ang[k]), span))
    if not drop:
        return loop
    for turn, span in notes:
        print(f'  Grading artefact: removed a {turn:.0f} deg needle '
              f'({span - 1} vertices) from a graded outline')
    return loop[[i for i in range(n) if i not in drop]]


def _paired_chains(chain_a, chain_b, max_seg=None):
    """Split two chains into the SAME number of equal-arc-length parts.

    Built to keep `StitchingRule.match_interfaces` from running at all. When the
    two sides of a seam do not already share edge fractions, it subdivides them
    to match -- and when it splits an edge, the two halves come back in swapped
    order. In the serialized spec that shows up as the paired edge indices
    zigzagging with period 2 (e18<->e53, e19<->e54, e16<->e55, e17<->e56), i.e.
    every second weld crossing its neighbour. On the jogger legs it accounted for
    38 of 40 out-of-order welds.

    Giving both sides N parts at identical fractions makes `isMatching()` true,
    so nothing is subdivided and the welds pair 1:1 in order. Each part is ONE
    cubic, so the counts cannot drift apart afterwards either; `max_seg` keeps
    each cubic short enough to track the outline.
    """
    def arclen(c):
        c = np.asarray(c, float)
        return float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))

    max_seg = SEAM_MAX_SEG if max_seg is None else max_seg
    n = max(1, int(np.ceil(max(arclen(chain_a), arclen(chain_b)) / max_seg)))
    fr = [k / n for k in range(1, n)]
    return _split_at_fracs(chain_a, fr), _split_at_fracs(chain_b, fr), n


def _align_welds(stitches):
    """Re-decide every seam's weld direction AFTER `match_interfaces` has run.

    `_auto_rw` has to choose before the rule is built, when the two sides may
    carry different edge counts -- it then loops over `min(len(a), len(b))` and
    defaults the remainder, so most of a seam gets an arbitrary answer. Once
    `StitchingRule.__init__` has projected both sides onto the union of their
    fractions, edge k really does face edge k, and the question becomes
    answerable. So it is answered here instead, per seam, summed over all its
    pairs -- the same test `_stitch_matched` applies to the jogger's band, moved
    to the only point where the dresses' curved-to-curved seams satisfy its
    precondition.

    Summed rather than per-edge because a single pair is ambiguous wherever the
    two panels sit far apart in the flat layout, which for an upper-body garment
    is everywhere (the bodice halves are placed 55 cm apart in z).

    Self-seams -- darts, both sides on one panel -- are left alone: their two
    sides are the same edge run and the test is meaningless.
    """
    for rule in getattr(stitches, 'rules', []) or []:
        a, b = rule.int1, rule.int2
        pa = {p.name for p in a.panel}
        pb = {p.name for p in b.panel}
        if pa == pb and len(pa) == 1:
            continue
        n = min(len(list(a.edges)), len(list(b.edges)))
        if n < 1:
            continue

        def p3(interf, i, which):
            e, pan = interf.edges[i], interf.panel[i]
            return np.asarray(pan.point_to_3D(
                list(e.start if which == 's' else e.end)))

        same = flip = 0.0
        for k in range(n):
            aS, aE = p3(a, k, 's'), p3(a, k, 'e')
            bS, bE = p3(b, k, 's'), p3(b, k, 'e')
            same += np.linalg.norm(aS - bS) + np.linalg.norm(aE - bE)
            flip += np.linalg.norm(aE - bS) + np.linalg.norm(aS - bE)
        b.right_wrong = [bool(same < flip)] * len(b.right_wrong)
    return stitches


def _stitch_matched(int_a, int_b, force=None, reverse=None):
    """Stitch two seams whose parts already correspond 1:1, with ONE weld
    direction for the whole seam.

    Use this only where the pairing is matched by construction -- a band or rib
    edge split to its partner's own count and fractions (`_matched_slot`). Such a
    seam is a single continuous run, so it cannot alternate direction, and
    `_auto_rw`'s per-edge decision is a liability there: it compares endpoint
    distances, and the casing sits ~16 cm from the leg waist in the flat layout,
    so its answer is near-arbitrary. One flipped flag in the middle of the seam
    welds that edge's start to its neighbour's end, which drags the edge's own two
    ends onto one vertex -- a 5.8 cm edge on the back waistband collapsed exactly
    that way, and `collapse_stitch_vertices` reported it as two vertices of one
    panel stitched together.

    The direction is decided from ALL the pairs at once, which is what makes this
    safe: summing over the seam swamps the per-edge ambiguity. That is NOT the
    whole-interface endpoint test that collapsed the garment before -- that one
    used only the two ends of the seam and ran while the ORDER was still wrong.

    `_auto_rw`'s per-edge behaviour is still right for the crotch/rise seams,
    which genuinely are a mix after the mirror, so they keep `_stitch_aligned`.

    `force` and `reverse` STATE the direction and the edge order instead of
    measuring them. Both votes compare 3D distances, which needs the two sides to
    be near each other; in the exploded layout several seams are not. The collar
    sat 27 cm from every neckline it took, the yoke bridges front to back, and a
    split cuff's two halves are 30 cm apart in z -- and a wrong ORDER folds a
    seam in half and welds its two ends together, which no direction flag fixes.
    """
    int_a, out = _stitch_aligned(int_a, int_b, reverse=reverse)
    n = min(len(list(int_a.edges)), len(list(out.edges)))
    if n < 1:
        return (int_a, out)

    def p3(interf, i, which):
        e, pan = interf.edges[i], interf.panel[i]
        return np.asarray(pan.point_to_3D(
            list(e.start if which == 's' else e.end)))

    # Summed over ALL the pairs, then applied uniformly. Only valid because the
    # parts correspond 1:1 (`_matched_slot`) and `_stitch_aligned` has aligned
    # their order -- edge k really does face edge k. Summing is what makes it
    # safe: any single pair is ambiguous when the two panels sit far apart in the
    # flat layout, but the seam as a whole is not.
    #
    # Deciding it from the seam's two ENDS instead was tried and is wrong here:
    # the casing sits ~16 cm from the leg waist in the flat layout, so its ends
    # are offset too, and the jogger went from 0 flagged welds to 60.
    same = flip = 0.0
    for k in range(n):
        aS, aE = p3(int_a, k, 's'), p3(int_a, k, 'e')
        bS, bE = p3(out, k, 's'), p3(out, k, 'e')
        same += np.linalg.norm(aS - bS) + np.linalg.norm(aE - bE)
        flip += np.linalg.norm(aE - bS) + np.linalg.norm(aS - bE)
    out.right_wrong = [bool(same < flip) if force is None
                       else bool(force)] * len(out.right_wrong)
    return (int_a, out)


def _stitch_aligned(int_a, int_b, reverse=None):
    """Stitch pair with edge ORDER aligned first, then ONE weld direction.

    `hyperdrop._stitch` -> `aama_dxf._auto_rw` decides the weld direction per
    edge by comparing `edge[i]` of one side with `edge[i]` of the other. That is
    only meaningful if the two interfaces list their edges in the same order
    along the seam, and here they often do not: the jogger's waist casing and
    ankle rib run OPPOSITE to the leg edges they take (measured, the reversed
    pairing is 3x closer -- 37 against 119). Comparing non-corresponding edges
    then gives each edge its own answer, and the flags come out MIXED within one
    seam -- `[True, False, False, False, ...]`. A seam welded partly one way and
    partly the other crosses itself, which is what collapsed the garment.

    So: work out whether the orders correspond by comparing edge midpoints in 3D,
    reverse one side if they do not, and only then let `_auto_rw` choose the
    directions -- on edges that actually face each other.
    """
    out = pyg.Interface.from_multiple(int_b)

    def mid(interf, i):
        e, pan = interf.edges[i], interf.panel[i]
        return 0.5 * (np.array(pan.point_to_3D(list(e.start)))
                      + np.array(pan.point_to_3D(list(e.end))))

    n = min(len(int_a.edges), len(out.edges))
    if n > 1:
        same = sum(np.linalg.norm(mid(int_a, k) - mid(out, k)) for k in range(n))
        rev = sum(np.linalg.norm(mid(int_a, k) - mid(out, n - 1 - k))
                  for k in range(n))
        if reverse if reverse is not None else (rev < same):
            # with_edge_dir_reverse=True: reversing the ORDER without negating
            # the per-edge direction flags leaves every edge traversed backwards,
            # and when `match_interfaces` later splits one, its two halves land
            # in swapped order. That shows up in the spec as the paired edge
            # indices zigzagging in pairs -- e18<->e53, e19<->e54, e16<->e55,
            # e17<->e56 -- i.e. every second weld crossing its neighbour, which
            # is the crossed seam. `pants_clo` reverses its crotch interfaces the
            # same way, with the flags negated.
            out = out.reverse(with_edge_dir_reverse=True)

    # Order fixed, hand the DIRECTION back to `_auto_rw`, which decides it per
    # edge. Forcing one direction for the whole seam was wrong: the per-edge
    # behaviour is deliberate and its docstring says why -- it "handles the
    # mirror seams (crotch) whose left/right segments are a mix after the
    # mirror", where a genuinely mixed set is correct. Overriding that collapsed
    # the garment to a ring at one knee. Only the ORDER was ever broken; with
    # edge[i] now facing edge[i], the per-edge comparison is meaningful again.
    out = _auto_rw(int_a, out)
    # MAJORITY VOTE over the seam. `_auto_rw` decides per edge, and at the two
    # ENDS of a seam the two candidate pairings are nearly equidistant -- the
    # edges meet at a corner -- so its test is ambiguous there and flips one
    # weld. Measured on the jogger, every crossed seam was exactly that: one
    # flipped weld out of 6 or 8, always the first or the last
    # (`T.......` / `.......T`), on the waistband-to-front and rib-to-leg seams.
    # Voting keeps every unambiguous per-edge decision and only overrides the
    # outlier. Note this is NOT the same as deciding the direction from the
    # interface endpoints, which is a different test and picked the wrong
    # direction outright, collapsing the garment.
    # NO majority vote. One was added here when seams came out with mixed flags,
    # but the mixture was an artefact: `_auto_rw` loops over
    # `min(len(a.edges), len(b.edges))` and defaults the remainder to False, so a
    # 1-edge band against an 8-edge leg waist could only ever return
    # [True, False x 7]. Voting then flattened it to one arbitrary direction for
    # the whole seam. With both sides carrying the same edge count and the same
    # fractions (`_band_parts`), every flag is a real per-edge decision and must
    # be left alone -- the crotch seams genuinely are a mix after the mirror.
    return (int_a, out)


def _dart_triples(loop, thr=22.0, window=3.0, min_turn=120.0,
                  min_leg=3.0, leg_tol=0.25):
    """[(base, apex, base)] for every dart spike on an outline.

    Found on the CUT line, where a dart is a 150-170 degree reversal and
    unmistakable. Passed to `inset_loop` so the seam-allowance offset leaves the
    dart alone -- see the note there on why offsetting one destroys it.
    """
    loop = np.asarray(loop, float)
    keys = ad.corners(loop, thr=thr, window=window)
    if len(keys) < 3:
        return []
    ang, _ = ad.turn_angles(loop)

    def arclen(a, b):
        ch = ad.arc(loop, a, b)
        return float(np.sum(np.linalg.norm(np.diff(ch, axis=0), axis=1)))

    out = []
    for i, k in enumerate(keys):
        if abs(ang[k]) < min_turn:
            continue
        a, b = keys[i - 1], keys[(i + 1) % len(keys)]
        la, lb = arclen(a, k), arclen(k, b)
        # A dart's two legs are stitched TO EACH OTHER, so they are drafted the
        # same length -- 14.85 against 14.72 cm on one of these, 13.62 against
        # 12.91 on the other. A near-reversal with grossly unequal legs is not a
        # dart but a GRADING artefact: at size 34 the jogger's graded outline
        # spikes 178 deg with legs of 62 and 12 vertices, and treating that as a
        # dart pulled it out of the offset and cost the leg its real corners.
        if min(la, lb) < min_leg or abs(la - lb) / max(la, lb, 1e-9) > leg_tol:
            continue
        out.append((a, k, b))
    return out


class Style:
    """One bonprix style: its DXF, what it is, and where it goes on the body."""

    def __init__(self, key, dxf, kind, name, has_darts=False):
        self.key = key
        self.dxf = os.path.join(DXF_DIR, dxf)
        self.kind = kind            # 'top' | 'bottom' | 'dress'
        self.name = name
        # Whether any piece carries a DART, declared rather than sniffed. Dart
        # detection keys off a near-reversal of the outline, and a GRADED outline
        # can spike for its own reasons: the jogger -- which has no darts at all
        # -- turns 178 deg at size 34, with legs close enough in length to pass
        # the equal-leg test. Excluding a false dart from the seam-allowance
        # offset cost that leg its real corners.
        self.has_darts = has_darts

    # -- sizing ----------------------------------------------------------- #
    def size_run(self):
        """(size labels, sample size) from the `.rul` beside the DXF."""
        labels, sample, _ = ad.read_grade_rules(ad.grade_path(self.dxf))
        return labels, sample

    def size_label(self, size):
        """The `.rul` label grading `size`, e.g. '38' -> '36/38' where paired.

        Raises rather than guessing when the run holds no label covering the
        requested size -- silently grading a 40 as a 42 would be a wrong
        pattern that still simulates.
        """
        labels, _ = self.size_run()
        size = str(size)
        if size in labels:
            return size
        for lab in labels:
            if size in lab.split('/'):
                return lab
        raise ValueError(
            f'{self.key}: size {size!r} is not in its run {labels}')

    def pieces(self, size, seam_allow=None, **kw):
        """Graded outer-shell pieces at `size` (drops lining/template/notions).

        Every piece comes back with its boundary replaced by the net sew line,
        so callers downstream never have to know these files are cut-line-only.
        """
        sa = SEAM_ALLOW if seam_allow is None else seam_allow
        kw.setdefault('smooth', SMOOTH_OUTLINES)
        out = ad.pieces_for_size(self.dxf, self.size_label(size), **kw)
        for p in out:
            if sa and not p.has_sew_line:
                darts = _dart_triples(p.cut) if self.has_darts else []
                p.cut = _declaw(p.cut, [d[1] for d in darts])
                darts = _dart_triples(p.cut) if self.has_darts else None
                p.sew = ad.inset_loop(p.cut, sa, p.fold, darts=darts)
                # Notches were marked on the CUT edge, so insetting the boundary
                # alone leaves every one of them sitting exactly `sa` outside it
                # -- which is how SEAM_ALLOW got independently confirmed (all of
                # 8642610003's notches measure 1.00 cm off the net line), but it
                # also means any notch-driven seam split would miss. Pull them
                # onto the net line by their closest point on it.
                p.notches = _snap_to(p.notches, p.sew)
        return out


def _snap_to(pts, loop):
    """Move each point onto its nearest vertex of `loop`."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts):
        return pts
    loop = np.asarray(loop, float)
    return np.array([loop[int(np.argmin(np.linalg.norm(loop - q, axis=1)))]
                     for q in pts])


STYLES = {
    # article no.        DXF                        kind      description
    '6812610700': Style('6812610700', 'st3-6812610700-154.dxf', 'top',
                        'V-neck blouse, blouson sleeve + cuff, back yoke',
                        has_darts=True),
    '7492610006': Style('7492610006', 'ST3-7492610006-154-re.dxf', 'bottom',
                        'Jogger, elastic waist + drawcord, ribbed leg cuffs'),
    '8242610411': Style('8242610411', 'ST3-8242610411-154.dxf', 'dress',
                        'Shirt dress, 3/4 sleeve, button placket, tie belt',
                        has_darts=True),
    '8642610003': Style('8642610003', 'ST3-8642610003-154.dxf', 'dress',
                        'Tiered dress, shirred waist + shirred cuffs'),
    '8672609700': Style('8672609700', 'ST3-8672609700-154.dxf', 'dress',
                        'Midi shirt dress, collar, yoke, flared skirt',
                        has_darts=True),
}

# Which body pose each kind is simulated on.
POSE = {'top': 'anim', 'dress': 'anim', 'bottom': 'custom_pose'}


def distinct_sizes(style, sizes):
    """{requested size: graded label}, so collapsed requests are visible."""
    return {s: style.size_label(s) for s in sizes}


def resolve_sizes(style, requested=('36', '38', '40')):
    """Requested sizes, substituted so each one is a DIFFERENT pattern.

    Three of the five styles grade in two-size steps, so 36 and 38 land on the
    same '36/38' rule column and would simulate twice identically. Where that
    happens the 36 is replaced by the next size DOWN the run, i.e. 34, giving
    34/38/40 -- three neighbouring distinct patterns.

    Requests are claimed LARGEST FIRST so the collapsed pair keeps its larger
    label: '36/38' is reported as 38 and the displaced request becomes 34, not
    the other way round. Returns an ascending {size: label}.
    """
    labels, _ = style.size_run()
    out, seen = {}, {}
    for want in sorted(requested, key=int, reverse=True):
        lab = style.size_label(want)
        if lab not in seen:
            out[want] = lab
            seen[lab] = want
            continue
        # Collapsed onto a column already taken -- step down the run instead.
        for j in range(labels.index(lab) - 1, -1, -1):
            if labels[j] not in seen:
                alt = labels[j].split('/')[-1]       # '32/34' -> '34'
                out[alt] = labels[j]
                seen[labels[j]] = alt
                break
        else:
            raise ValueError(f'{style.key}: no distinct size below {lab!r}')
    return {k: out[k] for k in sorted(out, key=int)}


# --------------------------------------------------------------------------- #
#  Inspection entry point
# --------------------------------------------------------------------------- #
def _report(sizes):
    for key, st in STYLES.items():
        labels, sample = st.size_run()
        print('=' * 78)
        print(f'{key}  [{st.kind}]  {st.name}')
        print(f'  DXF   {os.path.basename(st.dxf)}')
        print(f'  run   sample={sample}  {" ".join(labels)}')
        asked = distinct_sizes(st, sizes)
        mapped = resolve_sizes(st, sizes)
        if list(mapped) != list(asked):
            print(f'  sizes {mapped}   <-- {sizes} collapse to '
                  f'{sorted(set(asked.values()))}, substituted')
        else:
            print(f'  sizes {mapped}')
        for size in mapped:
            ps = st.pieces(size)
            tot = sum(p.perimeter() for p in ps)
            print(f'  -- size {size} ({mapped[size]}): {len(ps)} shell pieces, '
                  f'{tot:.0f} cm of seam line')
            for p in ps:
                w, h = p.size_cm()
                print('       %-30s %6.1f x %6.1f cm  n=%-4d %s'
                      % (p.block[:30], w, h, len(p.boundary),
                         'pair' if p.mirrored else 'single'))


def _preview(sizes, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    for key, st in STYLES.items():
        per_size = {s: st.pieces(s) for s in resolve_sizes(st, sizes)}
        n = max(len(v) for v in per_size.values())
        rows, cols = len(per_size), n
        fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 3.2 * rows),
                                 squeeze=False)
        for r, size in enumerate(per_size):
            for c in range(cols):
                ax = axes[r][c]
                ax.set_aspect('equal')
                ax.tick_params(labelsize=5)
                if c >= len(per_size[size]):
                    ax.axis('off')
                    continue
                p = per_size[size][c]
                b = np.vstack([p.boundary, p.boundary[:1]])
                ax.plot(b[:, 0], b[:, 1], '-', lw=0.9, color='k')
                for lay, v in p.internal:
                    ax.plot(v[:, 0], v[:, 1], '-', lw=0.5,
                            color='tab:red' if lay == '8' else 'tab:blue')
                if len(p.notches):
                    ax.plot(p.notches[:, 0], p.notches[:, 1], 'x', ms=3,
                            color='tab:green')
                w, h = p.size_cm()
                ax.set_title('%s\nsize %s  %.1f x %.1f'
                             % (p.block[:22], size, w, h), fontsize=6)
        fig.suptitle(f'{key} -- {st.name}', fontsize=10)
        fig.tight_layout()
        path = os.path.join(out_dir, f'{key}.png')
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print('wrote', path)




# --------------------------------------------------------------------------- #
#  Seam-junction finding
#
#  The Hyperdrop builders each carry a hardcoded list of boundary vertex indices
#  marking where one seam ends and the next begins (`FRONT_KEYS = [25, 26, ...]`).
#  That works because their outlines are fixed, but it is brittle: the index has
#  to be re-read by hand for every piece, and it silently means something else if
#  the outline is ever re-smoothed or re-nested. Here the junctions are DERIVED
#  from the outline instead, which also survives the two DXF mirror orientations
#  these markers use (the jogger nests its front and back legs facing each other,
#  so one is stored with the crotch on +x and the other on -x).
# --------------------------------------------------------------------------- #
def _leg_junctions(loop, notches=None, thr=15.0, window=3.0,
                   band=0.15, edge_band=3.0):
    """(loop, keys) for a trouser leg: hem_out, hem_in, crotch, waist_in, waist_out.

    Junctions are picked by ROLE, not by expecting a particular number of
    corners. Counting corners looked tempting -- a leg has five real ones -- but
    the count is not stable: the offset repair resamples a defective outline
    evenly, and where the hip curve then falls relative to the new vertices
    decides whether it registers as one corner or three. The jogger front leg
    gives 5 corners at size 38 and 7 at size 34 from the same detector.

    So: corners in the bottom `band` of the height are the hem, corners in the
    top `band` are the waist, and of what is left the one furthest sideways from
    the piece's centre is the crotch point. Which side that lands on defines the
    inseam side, and the inner/outer hem and waist corners follow from it. Extra
    corners along the outseam or the rise are simply not selected.

    `notches` travels with the outline, mirrored when the outline is. Without
    that the notch coordinates stay in the original frame, the outseam split
    finds nothing within tolerance and the whole outseam collapses to one cubic
    -- so it silently stops matching the other leg's sub-seams.

    The returned loop is normalised two ways, and both matter:

    * MIRRORED if needed so the crotch sits on the +x side, i.e. the leg extends
      into negative x from its crotch point. This marker nests the front and back
      legs facing each other -- the front is stored with its crotch on -x and the
      back on +x -- so placing them with the same offset points the two panels to
      OPPOSITE sides of the body and no outseam can ever close. (Symptom: legs
      hanging off the body as loose flaps with 14 of 26 stitches warned.)
    * REVERSED if needed so that walking forward from `keys[0]` visits the
      junctions in the returned order, i.e. the seams come out as hem, inseam,
      rise, waist, outseam. Mirroring flips the winding, so this test runs after
      it, not before.
    """
    loop = np.asarray(loop, float)
    notches = (np.zeros((0, 2)) if notches is None
               else np.asarray(notches, float).reshape(-1, 2))
    idx = corners(loop, thr=thr, window=window)
    if len(idx) < 5:
        raise ValueError(f'leg outline: only {len(idx)} corners over {thr} deg')
    y = loop[:, 1]
    lo, hi = y.min(), y.max()
    h = hi - lo
    x_mid = 0.5 * (loop[:, 0].min() + loop[:, 0].max())

    # The hem and waist junctions are the two ENDS OF THOSE EDGES, taken over
    # every vertex within `edge_band` cm of the outline's top and bottom -- not
    # over the corner list. Corner detection is too unstable here: at size 34 the
    # front leg has a corner 12 cm down the outseam that still fell inside a 15%
    # height band, won the 'waist' pick, and turned a 20 cm waist into a 35 cm
    # one while the outseam lost the same 9 cm. The band has to clear the waist's
    # own slope (2.3 cm front to back on the back leg) without reaching so far
    # down the outseam that it matters -- and it does not matter, because the
    # outseam is near-vertical there, so its x at y-3cm is the waist corner's x.
    hem = np.where(y <= lo + edge_band)[0]
    waist = np.where(y >= hi - edge_band)[0]
    mid = [i for i in idx if lo + band * h < y[i] < hi - band * h]
    if not (len(hem) >= 2 and len(waist) >= 2 and mid):
        raise ValueError(f'leg outline: hem={len(hem)} waist={len(waist)} '
                         f'mid={len(mid)}; not a leg shape')

    crotch = max(mid, key=lambda i: abs(loop[i][0] - x_mid))
    if loop[crotch][0] < x_mid:                  # crotch must end up on +x
        loop = loop * [-1.0, 1.0]
        notches = notches * [-1.0, 1.0] if len(notches) else notches
        x_mid = -x_mid
    inner = np.sign(loop[crotch][0] - x_mid)     # the inseam side of the piece

    def pick(group, top):
        """(outer, inner) corner of a group, by how far each sits to each side.

        Then refined ALONG the edge: taking the extreme-x vertex alone lands on
        the far end of the near-vertical run that the rise and the outseam make
        near the waist, not on the corner. The x is right either way -- the run is
        vertical -- but the junction INDEX is what bounds the seam, so the waist
        was swallowing 1.8 cm of rise and 2.9 cm of outseam and measuring 25.3 cm
        where the edge is 20.7. Of the vertices sharing that x, take the one
        highest (waist) or lowest (hem): that is the corner.
        """
        def refine(i):
            same = [j for j in group if abs(loop[j][0] - loop[i][0]) < 0.4]
            return (max(same, key=lambda j: loop[j][1]) if top
                    else min(same, key=lambda j: loop[j][1]))
        far = refine(max(group, key=lambda i: inner * (loop[i][0] - x_mid)))
        near = refine(min(group, key=lambda i: inner * (loop[i][0] - x_mid)))
        return near, far

    hem_out, hem_in = pick(hem, top=False)
    waist_out, waist_in = pick(waist, top=True)
    keys = [hem_out, hem_in, crotch, waist_in, waist_out]
    if len(set(keys)) != 5:
        raise ValueError(f'leg outline: junctions collided at {keys}')

    # Walking forward from the hem's outseam corner, the very next junction has
    # to be the hem's inseam corner. If it is the waist corner instead the loop
    # runs the other way round and every seam would come out named as its
    # opposite, so flip it and re-index.
    others = set(keys[1:])
    n = len(loop)
    step = next(k for k in range(1, n + 1) if (hem_out + k) % n in others)
    if (hem_out + step) % n != hem_in:
        # Reversing the loop reverses the cyclic order of every vertex, so a
        # walk that visited the roles backwards now visits them forwards: the
        # role-ordered list is already the walk order and must NOT be re-sorted.
        # Only the indices move.
        loop = loop[::-1]
        keys = [n - 1 - k for k in keys]
    return loop, notches, keys


def _on_fold(loop, fold, tol=0.05):
    """Indices of the boundary vertices lying on a layer-6 mirror line."""
    a, b = np.asarray(fold, float)[0], np.asarray(fold, float)[-1]
    d = b - a
    d = d / max(np.linalg.norm(d), 1e-12)
    perp = np.array([-d[1], d[0]])
    return np.where(np.abs((np.asarray(loop, float) - a) @ perp) < tol)[0]


def _on_edge(loop, side, band=1.0):
    """Indices within `band` cm of one side of the outline's bounding box.

    Used instead of `extreme_idx` where a whole EDGE has to be found rather than
    a single extreme point: the yoke's outer edge slants, so its lowest vertex
    and its rightmost vertex are different vertices, and asking for
    'right-bottom' returns whichever the lexicographic tie-break happens to pick.
    """
    loop = np.asarray(loop, float)
    col = {'left': 0, 'right': 0, 'top': 1, 'bottom': 1}[side]
    v = loop[:, col]
    lim = v.max() - band if side in ('right', 'top') else v.min() + band
    return np.where(v >= lim if side in ('right', 'top') else v <= lim)[0]


def _pick(loop, idx, axis, want):
    """The index in `idx` that is furthest along an axis ('x'|'y', 'min'|'max')."""
    loop = np.asarray(loop, float)
    col = 0 if axis == 'x' else 1
    key = (lambda i: loop[i][col]) if want == 'min' else (lambda i: -loop[i][col])
    return int(min(idx, key=key))


def _seams_of(loop, keys, names):
    """(seams, source, loop, keys) with the loop simplified ONCE, junctions kept.

    Every panel here goes through this. Simplifying per-seam instead leaves
    collinear triples straddling the junctions, which triangulate into slivers
    the mesher refuses ("Invalid edge lengths [1.069, 0.515, 0.515]"), and it was
    wired into the jogger alone at first -- so the blouse kept failing with those
    exact numbers. `source` is the UNsimplified chain set, which is what the
    fidelity gate must judge against.
    """
    # No pre-simplification: the corner-split cubic fit works off the full
    # chain, and simplifying first is what made junctions collide on a sparse
    # outline. `source` is the same chain set, which is exactly what the fidelity
    # gate should judge the fitted curves against.
    seams = _ordered(loop, keys, names)
    return seams, {k: v.copy() for k, v in seams.items()}, loop, keys


def _ordered(loop, keys, names):
    """{name: chain} for junctions given in walk order, wrapping to close.

    `keys` must already be in the order the outline visits them; the chain for
    name k runs keys[k] -> keys[k+1] and the last wraps to keys[0], so the seams
    tile the whole outline with no gap.
    """
    if len(keys) != len(names):
        raise ValueError(f'{len(keys)} junctions but {len(names)} seam names')
    # The junctions must be a cyclic walk order, i.e. the forward steps between
    # consecutive ones sum to exactly one lap. When they do not, each descending
    # pair silently yields an arc going the LONG way round and the seams overlap
    # instead of tiling -- which is how the shirt-dress front ended up retracing
    # its outline ten times while still reading as a continuous loop. Loud here.
    n = len(loop)
    steps = [(int(keys[(i + 1) % len(keys)]) - int(k)) % n
             for i, k in enumerate(keys)]
    if sum(steps) != n:
        worst = max(range(len(steps)), key=lambda i: steps[i])
        raise ValueError(
            f'junctions are not in walk order: steps {steps} sum to '
            f'{sum(steps)}, not the loop length {n}; worst is '
            f'{names[worst]} spanning {steps[worst]} vertices')
    return {nm: arc(loop, keys[i], keys[(i + 1) % len(keys)])
            for i, nm in enumerate(names)}


def _walk_sorted(loop, keys):
    """Rotate a set of junctions into the order the outline actually visits them.

    Junctions are located by role (topmost vertex, the fold's lower end, ...),
    which says nothing about walk order, and `_ordered` needs walk order. Sorting
    the indices ascending and rotating so the caller's first junction leads gives
    it, because a closed outline visits ascending indices in order by definition.
    """
    ks = sorted(int(k) for k in keys)
    if len(set(ks)) != len(keys):
        raise ValueError(f'junctions collided: {sorted(keys)}')
    i = ks.index(int(keys[0]))
    return ks[i:] + ks[:i]


# Flat placement offsets for a bodice, copied from the parametric one
# (bodice.py: front torso `translate_by([0, 0, 30])`, back `[0, 0, -25]`). The
# asymmetry is deliberate there, and the 55 cm total gap is what gives the side
# seams enough dz to wrap the halves round the body instead of fighting friction.
# Where a bodice half is placed flat, in z. The body's chest surface sits at
# z = +13.6 / -11.6 on this body, so these start a bodice about 15 cm clear of it.
# Tightening them to 18/-16 was TRIED as a fix for the upper-body garments sliding
# off the shoulders and did not fix it (the blouse still ended up on one
# shoulder), so the slide is not initial clearance -- see the note in
# `bonprix_blouse.yaml`.
ZB_FRONT, ZB_BACK = 30.0, -25.0


def _place_flat(panel, loop, centre_x, top_y, z):
    """Place a bodice half FLAT, with its centre-front/back edge on x = 0.

    `place_around` -- laying each panel tangent to an ellipse -- is wrong for
    these: it rotates the panel to face outward, and whether a half's CF edge
    then lands clockwise or anticlockwise of centre front depends on which local
    x the CF happens to be, so one half comes out flipped and the CF-to-CF stitch
    twists the whole garment round the body (the blouse rendered with its front
    over one shoulder and the other side bare).

    The parametric bodice does not wrap either: it builds the half, places it
    flat at z = +30 / -25 and mirrors. Anchoring CF at x = 0 makes the two halves
    exact mirrors whatever the source winding, so the CF seam is a straight
    z-to-z weld.
    """
    panel.translation = np.array([0.0, top_y, z], float)
    return face_to(panel, [0.0, 0.0, float(np.sign(z))])


def _hang(panel, loop, theta, top_y, rx, rz):
    """Place a panel around the body with its TOP EDGE at `top_y`.

    `place_around` translates the panel's ORIGIN, and `_normalised` leaves a
    piece with y running 0..height, so passing the shoulder height straight in
    hangs the panel's hem from the shoulder and its top a whole panel-length
    above the head. (Symptom: the blouse draped as a sheet over the head with the
    body bare.) `hyperdrop.py` subtracts the panel's own top for exactly this
    reason -- `y = hps - loop[:, 1].max()` appears at every one of its upper-body
    placements.
    """
    return place_around(panel, theta, top_y - float(np.asarray(loop)[:, 1].max()),
                        rx, rz)


def _mirror_junctions(loop, keys, notches=None):
    """Mirror an outline in x, leaving indices and junction roles untouched.

    Coordinates only: no reversal. Mirroring does flip the winding to clockwise,
    but `face_to()` is what settles a DxfPanel's facing, and `hyperdrop.py`'s
    legs take the same route -- build from a reflected outline, then face it.
    Reversing as well would keep the winding CCW but run the junction list
    backwards, which relabels every seam as its opposite; that showed up as both
    mirrored leg panels reporting self-intersection.

    This exists so the crotch normalisation inside `_leg_junctions` and a
    caller's deliberate left/right mirror cannot cancel each other. They did:
    `_leg_junctions` mirrors the front leg to put its crotch on +x, and the left
    leg was built by mirroring the piece BEFORE that, so the normalisation undid
    it and both legs came out identical -- one leg assembled, the other hung open.
    """
    notches = (np.zeros((0, 2)) if notches is None
               else np.asarray(notches, float).reshape(-1, 2))
    return (loop * [-1.0, 1.0],
            [int(k) for k in keys],
            notches * [-1.0, 1.0] if len(notches) else notches)


_LEG_SEAMS = ['hem', 'inseam', 'rise', 'waist', 'outseam']


# --------------------------------------------------------------------------- #
#  7492610006 -- jogger
# --------------------------------------------------------------------------- #
class BonprixJoggerLeg(pyg.Component):
    """One leg: front panel + back panel, outseam and inseam closed.

    Follows `HyperdropPantsLeg`: the opposite leg is built from a REFLECTED
    outline rather than by mirroring the assembled leg, because mirroring flips
    each panel's winding and DxfPanel has autonorm disabled, so one leg would
    come out inside out. Every stitched seam is a single cubic for the same
    reason as there -- a multi-edge interface pairs edge i with edge i, and a
    reflection reverses one side's edge order.
    """

    def __init__(self, tag, front_piece, back_piece, crotch_y,
                 z_front=Z_FRONT, z_back=Z_BACK, reflect=False):
        super().__init__(f'pant_{tag}')
        # Panel names are the PARAMETRIC pants' names. run_custom_pants' sim-time
        # placement looks the four leg panels up by exactly these, which is what
        # lets a DXF garment reuse the crotch lift and pose-X leg alignment.
        x_off = 0.5 if reflect else -0.5
        # Both panels are prepared BEFORE either is built, because their shared
        # seams (outseam, inseam) have to be cut into matching parts -- see
        # `_paired_chains`.
        fr = self._prep(front_piece, reflect)
        bk = self._prep(back_piece, reflect)
        self.n_shared = {}
        # The hem meets the rib's TWO top edges (top_r, top_l), so it is cut into
        # two matching parts as well. Left as one multi-edge seam it needed
        # subdividing, and the mirrored side came back with a crossed weld while
        # its twin did not -- the last order break in the garment.
        self.n_hem = {}
        for side, prep in (('f', fr), ('b', bk)):
            ch = prep['seams']['hem']
            L = float(np.sum(np.linalg.norm(np.diff(ch, axis=0), axis=1)))
            # 5 cm parts, each ONE cubic (the counts must not drift or
            # `match_interfaces` subdivides and swaps halves). Going finer made
            # it WORSE, not better -- at 2.5 cm one part came out 34 mm off in
            # length, so the splitter misbehaves at higher counts. Left at the
            # value that measures clean; the residual ~8 mm sits on the hem,
            # which is enclosed by the rib.
            n = max(2, int(np.ceil(L / 5.0)))
            self.n_hem[side] = n
            parts = _split_at_fracs(ch, [k / n for k in range(1, n)])
            self.n_hem[side] = len(parts)
            self._explode(prep['seams'], 'hem',
                          {f'hem_{i}': c for i, c in enumerate(parts)})
        for role in ('outseam', 'inseam'):
            # The two panels traverse the shared seam in OPPOSITE directions, so
            # the back's chain is reversed before splitting; part i then faces
            # part i. The back's parts go back into its dict in panel order
            # (reversed) while keeping the matching names, since DxfPanel builds
            # the loop in dict order but the names are only labels.
            fa, ba, n = _paired_chains(fr['seams'][role], bk['seams'][role][::-1])
            self.n_shared[role] = n
            self._explode(fr['seams'], role, {f'{role}_{i}': c
                                              for i, c in enumerate(fa)})
            self._explode(bk['seams'], role,
                          {f'{role}_{n - 1 - i}': c[::-1]
                           for i, c in enumerate(reversed(ba))})
        (self.front, self.f_waist_y, self.f_hem, self.f_hem_x) = self._build(
            f'pant_f_{tag}', fr, crotch_y, z_front, x_off)
        (self.back, self.b_waist_y, self.b_hem, self.b_hem_x) = self._build(
            f'pant_b_{tag}', bk, crotch_y, z_back, x_off)
        self.subs = [self.front, self.back]

        # ONE stitch per seam, over the multi-part interfaces -- not one stitch
        # per part. Splitting a seam into N stitches makes every junction vertex
        # between adjacent parts belong to TWO stitches, which is the 3+way case
        # `boxmeshgen` warns about ("use of the same interfaces in other stitches
        # ... may fail"); the garment collapsed to a ring at the thigh while both
        # the order and weld-direction checks read clean. The parts still exist
        # and still line up 1:1, so `match_interfaces` has nothing to subdivide.
        self.stitching_rules = pyg.Stitches(
            *[_stitch_aligned(
                self.front.seam(*[f'{role}_{i}'
                                  for i in range(self.n_shared[role])]),
                self.back.seam(*[f'{role}_{i}'
                                 for i in range(self.n_shared[role])]))
              for role in ('outseam', 'inseam')],
        )
        self.interfaces = {}
        for side, pan in (('f', self.front), ('b', self.back)):
            for role in ('waist', 'rise'):
                self.interfaces[f'{role}_{side}'] = pan.interfaces[role]
            n = self.n_hem[side]
            self.interfaces[f'hem_{side}'] = pan.seam(
                *[f'hem_{i}' for i in range(n)])
            for i in range(n):
                self.interfaces[f'hem_{side}_{i}'] = pan.interfaces[f'hem_{i}']

    @staticmethod
    def _explode(seams, role, parts):
        """Replace `seams[role]` with `parts`, IN PLACE in the dict order.

        Order matters: DxfPanel assembles the edge loop by walking the dict, so
        the parts have to sit where the original seam sat.
        """
        out = {}
        for k, v in seams.items():
            if k == role:
                out.update(parts)
            else:
                out[k] = v
        seams.clear()
        seams.update(out)

    def _prep(self, piece, reflect):
        loop, notches = _normalised(piece)
        # Canonicalise FIRST (crotch on +x, seams walking hem->outseam), then
        # apply the left/right mirror. Doing it the other way round lets the two
        # cancel -- see `_mirror_junctions`.
        loop, notches, keys = _leg_junctions(loop, notches)
        if reflect:
            loop, keys, notches = _mirror_junctions(loop, keys, notches)
        # No pre-simplification: the corner-split cubic fit reads the full chain.
        # Simplifying first threw away the very vertices the fit needs and the
        # errors compounded (the leg measured 124 mm out).
        seams = {s: arc(loop, keys[i], keys[(i + 1) % len(keys)])
                 for i, s in enumerate(_LEG_SEAMS)}
        return dict(loop=loop, keys=keys, notches=notches, seams=seams)

    def _build(self, name, prep, crotch_y, z, x_off):
        loop, keys, seams = prep['loop'], prep['keys'], prep['seams']
        source = {k: v.copy() for k, v in seams.items()}
        # One cubic per paired sub-seam, so the two sides of a seam carry equal
        # edge counts and `match_interfaces` has nothing to subdivide. NOT the
        # hem: forcing a 5 cm hem part to a single cubic cost 8.4 mm of fidelity
        # at size 38 and 13.0 mm at size 40 (over the limit), and the hem is
        # welded as ONE rule against the rib, which can subdivide freely.
        shared = tuple(k for k in seams
                       if k.startswith(('outseam_', 'inseam_')))
        panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                         single=shared,
                         source=source,     # judged against the DXF chains
                         # The waist is welded to a straight band and sits at the
                         # 4-way side corner where both bands and both legs meet,
                         # so a merge there cascades: a 1.15 cm edge (4 mesh
                         # vertices) had BOTH its ends pulled into the corner's
                         # global vertex, which reads as two vertices of one panel
                         # stitched together. It needs more headroom over the
                         # mesher's 1 cm than the rest of the outline does.
                         # Waist AND hem: both are welded to a straight band
                         # (casing, ankle rib), so neither may carry an edge
                         # under the mesher's resolution.
                         min_seg={k: WAIST_MIN_SEG for k in seams
                                  if k == 'waist' or k.startswith('hem')},
                         pivot=loop[keys[2]],            # crotch point -> origin
                         translation=[x_off, crotch_y, z])
        face_to(panel, [0, 0, np.sign(z)])
        hem_all = np.vstack([v for k, v in seams.items() if k.startswith('hem')])
        waist_y = float(seams['waist'][:, 1].mean() - loop[keys[2]][1])
        hem_y = float(hem_all[:, 1].mean() - loop[keys[2]][1])
        hem_x = float(hem_all[:, 0].mean() - loop[keys[2]][0]) + x_off
        return panel, waist_y, hem_y, hem_x


class BonprixJogger(pyg.Component):
    """7492610006 jogger, shell only.

    Dropped from the DXF: the pocket bag, the drawcord and the elastic (the
    marker's `Shape` blocks). Kept: both legs, the waist casing, and the ribbed
    ankle cuffs -- the cuffs are not optional trim here, they are what pulls the
    41 cm leg opening in to 20 cm and gives the garment its shape. Leaving them
    off produces a straight-legged trouser, which is not this garment.
    """
    KEY = '7492610006'
    # Ankle rib, from the marker's `Shape 6` block: 20 cm round, 6 cm tall, worn
    # folded double, so 3 cm finished. It is a notion block rather than a named
    # pattern piece, so this is the one dimension here taken on inference rather
    # than from a piece outline -- if the drape shows a cuff that is too tight
    # or too deep, this is the number to change.
    # NO invented rib/band dimensions any more -- both pieces are read from the
    # DXF, see `BAND_BLOCK`/`CUFF_BLOCK` below.
    # The DXF blocks that hold the waistband and the ankle rib, and what they
    # must measure once the fold is expanded. Named explicitly because the block
    # names carry no role ('Shape 5', 'Shape 6'), and dimension-checked so a
    # re-export that moves them fails loudly instead of silently substituting the
    # wrong piece.
    BAND_BLOCK, BAND_SIZE = 'Shape 5', (70.0, 2.5)
    CUFF_BLOCK, CUFF_SIZE = 'Shape 6', (20.0, 6.0)

    # The DXF draws the waist casing as a RECTANGLE, and that is how it is
    # built: the piece's own 82 cm, unaltered. Its bottom is welded to 94.7 cm of
    # leg waist, so the weld carries 1.15x -- that stretch is the gather the
    # simulator cannot represent (nothing here models elastic).
    #
    # This was switched to a length-matched trapezoid once, on the strength of an
    # A/B showing the rectangle spiking at the side seams. That A/B ran BEFORE the
    # two real weld faults at that corner were found -- sub-resolution edges
    # collapsing, and a flipped weld direction mid-seam -- so it was measuring
    # those, not the band's shape. `faithful_bands: 0` restores the trapezoid.
    FAITHFUL_BANDS = True
    # Ankle ribs are rectangles too, but drawn at the LEG OPENING's length rather
    # than at `CUFF_LEN` -- see the note at the `rib = top` line. There is no rib
    # piece in this DXF to be faithful to. `faithful_cuffs: 0` gives the tapered
    # (cinched) rib back.
    FAITHFUL_CUFFS = True
    ELASTIC_EASE = 0.95

    def __init__(self, body, design=None, size='38') -> None:
        super().__init__('bonprix_jogger')
        st = STYLES[self.KEY]
        pieces = st.pieces(size)
        # Front and back leg are the two tall panels; the front is the narrower.
        legs = sorted((p for p in pieces if p.size_cm()[1] > 60.0),
                      key=lambda p: p.size_cm()[0])
        if len(legs) != 2:
            raise ValueError(f'{self.KEY}: expected 2 leg panels, got '
                             f'{[p.block for p in legs]}')
        front, back = legs
        # The waistband and the ankle rib ARE in this DXF -- as blocks named
        # 'Shape 5' and 'Shape 6', which `read_pieces` classifies as notions
        # (anything named `shape <n>`) and `Style.pieces` therefore never
        # returns. Both were confirmed by the exporter. Reading them here instead
        # of inventing dimensions: 'Shape 5' is 35 x 2.5 cut on a short-edge
        # fold, so the band ring is 70 x 2.5; 'Shape 6' is a plain 20 x 6, the
        # rib. `CUFF_LEN`/`BAND_LEN` used to be hand-entered numbers standing in
        # for exactly these two pieces.
        # keyed on PIECE NAME, not block name -- the block carries a size suffix
        # `pieces_for_size`, not `read_pieces`: only 36/38 is in the file, the
        # other sizes are graded from it via the .rul.
        trims = {q.name: q for q in ad.pieces_for_size(
            st.dxf, st.size_label(size), fabric_only=False)
            if q.kind != 'fabric'}
        band = self._trim(trims, self.BAND_BLOCK, self.BAND_SIZE)
        rib = self._trim(trims, self.CUFF_BLOCK, self.CUFF_SIZE)

        # Body-agnostic placement, as the parametric pants: build with the
        # garment crotch at Y=0 and let sim-time placement apply the body's
        # crotch lift and pose-X leg alignment. generate_pattern stamps the
        # properties.placement marker that path keys off.
        crotch_y = 0.0
        self.right = BonprixJoggerLeg('r', front, back, crotch_y)
        self.left = BonprixJoggerLeg('l', front, back, crotch_y, reflect=True)

        # --- waist casing ----------------------------------------------------
        # A trapezoid, not a rectangle: the bottom edge is length-matched to the
        # four leg waists so the weld stretches nothing, and the top edge is the
        # born-stretched ring that grips. See `_taper_seams` -- a gather cannot
        # be expressed to the simulator, so the shape has to carry it.
        wf = self.right.interfaces['waist_f'].edges.length()
        wb = self.right.interfaces['waist_b'].edges.length()
        band_h = float(band[1])
        bot_f, bot_b = 2 * wf, 2 * wb
        band_relaxed = float(band[0])
        top_total = min(band_relaxed, float(body['waist']) * self.ELASTIC_EASE)
        if self.FAITHFUL_BANDS:
            print(f'  Waist casing: RECTANGLE {band_relaxed:.1f}cm '
                  f'(DXF {self.BAND_BLOCK}) '
                  f'against {bot_f + bot_b:.1f}cm of leg waist -> welded '
                  f'{(bot_f + bot_b) / band_relaxed:.2f}x its length; '
                  f'body waist {body["waist"]:.1f}cm')
        else:
            print(f'  Waist casing: bottom {bot_f + bot_b:.1f}cm (matched to the '
                  f'legs) -> top {top_total:.1f}cm on a {body["waist"]:.1f}cm '
                  f'waist ({band_relaxed:.1f}cm relaxed in the DXF)')
        if self.FAITHFUL_BANDS:
            # rectangle, at the DXF's own relaxed length
            bot_f = top_f = band_relaxed * wf / (wf + wb)
            bot_b = top_b = band_relaxed - bot_f
        else:
            top_f = top_total * wf / (wf + wb)
            top_b = top_total - top_f
        # Each bottom half is cut to the proportions of the leg waist it takes,
        # so the weld pairs edge-for-edge with nothing left for
        # `match_interfaces` to subdivide. bottom_l meets the RIGHT leg,
        # bottom_r the left; which end meets which is measured, not assumed.
        self.wb_int = {}
        for band_attr, name, bot, top, y, z, role in (
                ('band_f', 'wb_front', bot_f, top_f,
                 crotch_y + self.right.f_waist_y, Z_FRONT, 'waist_f'),
                ('band_b', 'wb_back', bot_b, top_b,
                 crotch_y + self.right.b_waist_y, Z_BACK, 'waist_b')):
            partners = {'bottom_l': self.right.interfaces[role],
                        'bottom_r': self.left.interfaces[role]}
            make = lambda parts, _n=name, _b=bot, _t=top, _y=y, _z=z: self._band(
                _n, _b, _t, band_h, _y, _z,
                bot_l_parts=parts.get('bottom_l'),
                bot_r_parts=parts.get('bottom_r'))
            plain = {k: [float(e.length()) for e in v.edges]
                     for k, v in partners.items()}
            chosen = {}
            for slot, partner in partners.items():
                others = {k: v for k, v in plain.items() if k != slot}
                chosen[slot] = _matched_slot(make, slot, partner, others)
            band = make({k: w for k, (w, _) in chosen.items()})
            setattr(self, band_attr, band)
            for slot, (_, order) in chosen.items():
                # A seam short enough to need only ONE part is not split at all,
                # so there is no `<slot>_0` to ask for.
                self.wb_int[f'{name}_{slot}'] = (
                    pyg.Interface.from_multiple(
                        *[band.interfaces[f'{slot}_{k}'] for k in order])
                    if len(order) > 1 else band.interfaces[slot])

        # --- ankle ribs ------------------------------------------------------
        # Same trapezoid trick, upside down: the rib's TOP edge takes the leg
        # opening 1:1 and its bottom edge is the narrow rib that closes round
        # the ankle. Each rib is placed under its own leg -- an earlier version
        # left all four at the body centre, ~15 cm from the hem they had to
        # reach, with the left leg's ribs sitting inside the right leg's.
        hf = self.right.interfaces['hem_f'].edges.length()
        hb = self.right.interfaces['hem_b'].edges.length()
        # Rib ring and depth come from the DXF's own 'Shape 6' (20 x 6 cm), split
        # front/back in proportion to the hem each side takes.
        cuff_len, cuff_h = rib
        rib_f = cuff_len * hf / (hf + hb)
        self.cuffs, self.cuff_int = {}, {}
        for tag, leg in (('r', self.right), ('l', self.left)):
            for side, top, ring, z, hem_y, hem_x in (
                    ('f', hf, rib_f, Z_FRONT, leg.f_hem, leg.f_hem_x),
                    ('b', hb, cuff_len - rib_f, Z_BACK, leg.b_hem,
                     leg.b_hem_x)):
                # Named to run_custom_pants' convention: its sim-time pose-X
                # leg rotation moves each leg AND any panel whose name contains
                # `l_cuff`/`r_cuff` as one rigid group, so the rib stays coaxial
                # with the leg. As `cuff_f_l` it matched nothing, so the legs
                # rotated about their top pivot -- displacement largest exactly
                # at the hem -- and the ribs stayed put.
                name = f'pant_{tag}_cuff_{side}'
                if self.FAITHFUL_CUFFS:
                    # A RECTANGLE at the RIB'S OWN length, as 'Shape 6' draws it:
                    # 20 cm round the ring, 6 cm deep, split front/back in
                    # proportion to the hem each side takes. The leg opening is
                    # 47.8 cm, so the weld carries 2.4x -- and for a RIB that is
                    # not an artefact, it is what the piece does: a 20 cm rib is
                    # stretched onto a 47.8 cm opening and its recovery is what
                    # cinches the ankle. (Drawing it at the opening's length
                    # instead removes the stretch but also removes the cinch, so
                    # the ankle hangs straight.)
                    top = ring
                partner = leg.interfaces[f'hem_{side}']
                make = lambda parts, _n=name, _r=ring, _t=top, _y=hem_y, \
                              _z=z, _x=hem_x, _h=cuff_h: self._band(
                    _n, _r, _t, _h,
                    crotch_y + _y - _h, _z, x=_x,
                    top_parts=parts.get('top'))
                widths, order = _matched_slot(make, 'top', partner, {})
                self.cuffs[name] = make({'top': widths})
                self.cuff_int[name] = (
                    pyg.Interface.from_multiple(
                        *[self.cuffs[name].interfaces[f'top_{k}'] for k in order])
                    if len(order) > 1
                    else self.cuffs[name].seam('top_r', 'top_l'))

        self.stitching_rules = pyg.Stitches(
            # centre front and centre back, leg to leg
            _stitch_aligned(self.right.interfaces['rise_f'], self.left.interfaces['rise_f']),
            _stitch_aligned(self.right.interfaces['rise_b'], self.left.interfaces['rise_b']),
            # casing side seams, then casing to the four leg waists
            _stitch_aligned(self.band_f.interfaces['side_r'], self.band_b.interfaces['side_r']),
            _stitch_aligned(self.band_f.interfaces['side_l'], self.band_b.interfaces['side_l']),
            _stitch_matched(self.wb_int['wb_front_bottom_l'],
                            self.right.interfaces['waist_f']),
            _stitch_matched(self.wb_int['wb_front_bottom_r'],
                            self.left.interfaces['waist_f']),
            _stitch_matched(self.wb_int['wb_back_bottom_l'],
                            self.right.interfaces['waist_b']),
            _stitch_matched(self.wb_int['wb_back_bottom_r'],
                            self.left.interfaces['waist_b']),
            # each rib closes into a ring at its own side seams, then its top
            # edge takes the leg opening
            *[_stitch_aligned(self.cuffs[f'pant_{t}_cuff_f'].interfaces[s],
                              self.cuffs[f'pant_{t}_cuff_b'].interfaces[s])
              for t in 'rl' for s in ('side_r', 'side_l')],
            # ONE stitch for the whole rib-to-hem seam, not one per part. Split
            # into separate stitches each part decided its own weld direction and
            # the seam came out mixed again -- the majority vote in
            # `_stitch_aligned` can only act over a seam it can see. The parts
            # still line up 1:1 (matching counts and fractions), so nothing is
            # subdivided; the rib's top is consumed in reverse because it runs
            # +x to -x while the hem runs the other way.
            *[_stitch_matched(
                self.cuff_int[f'pant_{t}_cuff_{sd}'],
                leg.interfaces[f'hem_{sd}'])
              for t, leg in (('r', self.right), ('l', self.left))
              for sd in 'fb'],
        )
        self.subs = [self.right, self.left, self.band_f, self.band_b,
                     *self.cuffs.values()]
        self.interfaces = {
            'top': pyg.Interface.from_multiple(
                self.band_f.interfaces['top_l'], self.band_f.interfaces['top_r'],
                self.band_b.interfaces['top_l'], self.band_b.interfaces['top_r']),
        }
        # 'leg' filters the arms out of body collision for those particles, so
        # the hands cannot push the trousers around; bands stay 'body'.
        self.left.set_panel_label('leg')
        self.right.set_panel_label('leg')
        self.set_panel_label('body', overwrite=False)

    @staticmethod
    def _trim(trims, block, expect, tol=0.18):
        """A named trim block, with its fold expanded and its size sanity-checked.

        `tol` is RELATIVE, because these pieces grade: 'Shape 5' is 64 / 70 / 76
        cm at sizes 34 / 38 / 40, a 6 cm step. `expect` is the size-38 value, so
        the check is only that the block still looks like the piece it is
        supposed to be -- enough to catch a re-export that renumbers the shapes.
        """
        p = trims.get(block)
        if p is None:
            raise ValueError(f'{block!r} not in this DXF; have '
                             f'{sorted(trims)}')
        w, h = p.size_cm()
        if p.fold is not None:
            f = np.asarray(p.fold, float)
            # a fold on a SHORT edge doubles the length, on a long edge the height
            if abs(f[0][0] - f[-1][0]) < abs(f[0][1] - f[-1][1]):
                w *= 2.0
            else:
                h *= 2.0
        if (abs(w - expect[0]) > tol * expect[0]
                or abs(h - expect[1]) > max(tol * expect[1], 0.3)):
            raise ValueError(
                f'{block!r} measures {w:.2f} x {h:.2f} cm, which is not within '
                f'{tol:.0%} of the expected {expect[0]} x {expect[1]} -- has the '
                f'export renumbered its shape blocks?')
        return w, h

    def _band(self, name, bottom, top, height, y, z, x=0.0, n_top=0,
              top_parts=None, bot_l_parts=None, bot_r_parts=None):
        panel = DxfPanel(name, _taper_seams(bottom, top, height, n_top=n_top,
                                            top_parts=top_parts,
                                            bot_l_parts=bot_l_parts,
                                            bot_r_parts=bot_r_parts),
                         verbatim=True, translation=[x, y, z])
        # Only a TAPERED band departs from the pattern; a rectangle is what the
        # DXF draws. Either way there is no DXF chain to check it against, so it
        # is declared rather than quietly validated against itself.
        panel.synthetic = True
        return face_to(panel, [0, 0, np.sign(z)])



# --------------------------------------------------------------------------- #
#  Walk-order helper shared by every builder
# --------------------------------------------------------------------------- #
def _band_parts(interf, from_centre):
    """Relative widths that split a band edge 1:1 against `interf`.

    A band edge is straight, so it can be cut into as many collinear pieces as
    its partner has edges, in the same length PROPORTIONS -- the band stays a
    rectangle and the two sides then share edge fractions exactly.

    This is what was missing. The casing's bottom was ONE edge against the leg
    waist's eight, so `_auto_rw` -- which loops over `min(len(a), len(b))` --
    judged edge 0 only and defaulted the other seven, giving a mixed set that the
    majority vote then flattened to a single arbitrary direction; and because the
    fractions did not match, `match_interfaces` subdivided the band into eight
    welds that all inherited it. Measured on the result: pant_f_r <-> wb_front
    came out fully reversed, mean weld gap 15.0 cm against 5.3 cm for the correct
    pairing, while its mirror pant_f_l was right -- the left/right asymmetry that
    gave it away.

    `from_centre` -- True when the band edge runs centre -> side, False when it
    runs side -> centre. The partner's own direction is read off its endpoints in
    3D, and the widths are reversed when the two disagree, so part i really does
    face edge i.
    """
    lens = [float(e.length()) for e in interf.edges]
    if len(lens) < 2:
        return lens
    p_first, e_first = interf.panel[0], interf.edges[0]
    p_last, e_last = interf.panel[-1], interf.edges[-1]
    x_first = abs(float(np.asarray(p_first.point_to_3D(list(e_first.start)))[0]))
    x_last = abs(float(np.asarray(p_last.point_to_3D(list(e_last.end)))[0]))
    starts_at_centre = x_first < x_last
    if starts_at_centre != bool(from_centre):
        lens = lens[::-1]
    return lens


def _split_straight(a, b, parts):
    """Straight edge a->b cut into collinear pieces of the given relative widths."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.asarray(parts, float)
    f = np.r_[0.0, np.cumsum(w) / w.sum()]
    return [np.array([a + (b - a) * f[i], a + (b - a) * f[i + 1]])
            for i in range(len(w))]


def _sub_chain(chain, f0, f1):
    """The [f0, f1] arc-length slice of a point chain, endpoints interpolated.

    Needed to match a seam that pairs with SEVERAL partners along its length --
    the collar's neck edge takes two fronts and two backs -- so each stretch can
    be given the fractions its own partner needs.
    """
    P = np.asarray(chain, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    tot = float(cum[-1])
    a, b = f0 * tot, f1 * tot

    def at(t):
        i = min(max(int(np.searchsorted(cum, t, 'right') - 1), 0), len(P) - 2)
        u = (t - cum[i]) / max(seg[i], 1e-12)
        return P[i] + u * (P[i + 1] - P[i])

    return np.vstack([at(a), P[(cum > a + 1e-9) & (cum < b - 1e-9)], at(b)])


def _chain_len(chain):
    """Arc length of a raw point chain (a `source_seams` entry)."""
    P = np.asarray(chain, float)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def _mid3(itf, i):
    e, pan = itf.edges[i], itf.panel[i]
    return 0.5 * (np.asarray(pan.point_to_3D(list(e.start)))
                  + np.asarray(pan.point_to_3D(list(e.end))))


def _pair_dist(int_a, int_b):
    """Total 3D distance between correspondingly-indexed edges of two seams."""
    n = min(len(list(int_a.edges)), len(list(int_b.edges)))
    return sum(float(np.linalg.norm(_mid3(int_a, k) - _mid3(int_b, k)))
               for k in range(n))


def _matched_slot(make, slot, partner, others):
    """(widths, consume_order) splitting a band's straight `slot` to face `partner`.

    Both candidates are fraction-matched with `partner` -- widths forward and
    consumed forward, or widths reversed and consumed in reverse -- so either way
    `match_interfaces` has nothing to subdivide. They differ only in which END of
    the band edge meets which end of the partner, and that is settled by building
    each and measuring, rather than by assuming a winding. Hardcoding it was
    wrong for exactly half the cases: the leg hems of a mirrored pair run in
    opposite directions, so one `reversed(range(n))` cannot serve both.
    """
    lens = [float(e.length()) for e in partner.edges]
    n = len(lens)
    if n < 2:
        return lens, list(range(n))
    best = None
    for widths, order in ((lens, list(range(n))),
                          (lens[::-1], list(reversed(range(n))))):
        pan = make({**others, slot: widths})
        itf = pyg.Interface.from_multiple(
            *[pan.interfaces[f'{slot}_{k}'] for k in order])
        d = _pair_dist(itf, partner)
        if best is None or d < best[0]:
            best = (d, widths, order)
    return best[1], best[2]


def _taper_seams(bottom, top, height,
                 names=('bottom_l', 'bottom_r', 'side_r',
                        'top_r', 'top_l', 'side_l'), n_top=0,
                 top_parts=None, bot_l_parts=None, bot_r_parts=None):
    """CCW trapezoid, `bottom` wide at y=0 and `top` wide at y=height.

    This is how an elastic casing or a ribbed cuff is modelled here, and the
    reason is that GATHERS DO NOT REACH THE SIMULATOR. `Interface.ruffle` only
    affects GarmentCode's own 2D projection: the serialized spec stores each
    stitch as a bare (panel, edge) pair with no ratio, so the mesh stitcher welds
    the two edges by arc length and a 1.24 ruffle simply stretches the shorter
    edge by 24%. Measured: a waist casing built 24% short came out of the sim
    stretched 26% on average (against 2.6% for the leg panels) and slid from the
    waist to mid-thigh. `hyperdrop.py`'s tee ran into the same wall and solved it
    the same way -- "the yoke and peplum seams are length-matched, so there is NO
    gather".

    So the shape carries what the gather cannot: the wide edge is length-matched
    to whatever it is stitched to, so the weld stretches nothing, and the narrow
    edge is the born-stretched ring that actually grips. Which is also what the
    garment does -- loose fabric at the seam, cinched at the elastic.

    Long edges are split at x=0 so a band spanning both legs meets each leg with
    its own 1:1 edge pair, as `_rect_seams` does and for the same reason.
    """
    wb, wt, h = bottom / 2, top / 2, height
    out = dict(zip(names, [
        np.array([[-wb, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [wb, 0.0]]),
        np.array([[wb, 0.0], [wt, h]]),
        np.array([[wt, h], [0.0, h]]),
        np.array([[0.0, h], [-wt, h]]),
        np.array([[-wt, h], [-wb, 0.0]]),
    ]))
    parts = None
    if top_parts is not None and len(top_parts) > 1:
        # Cut to the PARTNER's own length proportions, not into equal pieces:
        # equal pieces do not share fractions with a corner-fit hem, so
        # `match_interfaces` subdivided anyway (5 declared edges came out as 7
        # stitches on the ankle ribs).
        parts = _split_straight([wt, h], [-wt, h], top_parts)
    elif n_top and n_top > 1:
        # The top edge is cut into `n_top` EQUAL parts so it can pair 1:1 with a
        # multi-part partner. Splitting a straight edge is exact, so this costs
        # nothing in fidelity and keeps `match_interfaces` -- whose edge splits
        # come back swapped -- from having anything to do.
        full = np.array([[wt, h], [-wt, h]])
        parts = _split_at_fracs(full, [k / n_top for k in range(1, n_top)])
    if parts is not None:
        rebuilt = {}
        for k, v in out.items():
            if k == names[3]:                      # 'top_r' slot
                for i, part in enumerate(parts):
                    rebuilt[f'top_{i}'] = part
            elif k == names[4]:                    # 'top_l' slot, absorbed
                continue
            else:
                rebuilt[k] = v
        out = rebuilt
    # The two bottom halves, each cut to its own leg's proportions. Done in the
    # dict's place so DxfPanel still walks the outline in order.
    for slot, parts_rel, ends in ((names[0], bot_l_parts, ([-wb, 0.0], [0.0, 0.0])),
                                  (names[1], bot_r_parts, ([0.0, 0.0], [wb, 0.0]))):
        if parts_rel is None or len(parts_rel) < 2 or slot not in out:
            continue
        pieces = _split_straight(ends[0], ends[1], parts_rel)
        rebuilt = {}
        for k, v in out.items():
            if k == slot:
                for i, part in enumerate(pieces):
                    rebuilt[f'{slot}_{i}'] = part
            else:
                rebuilt[k] = v
        out = rebuilt
    return out


def _darts(loop, keys, min_turn=120.0):
    """Corners that are dart APEXES, i.e. near-reversals of the outline.

    A dart is drawn as a zero-width spike: the boundary runs out to the point
    and straight back, so it turns by 150-170 deg there against at most ~106 deg
    at any real corner, and it is the only place that turns the opposite way.
    That makes the turn angle the reliable locator. Picking a dart by position
    instead ("the corner nearest centre front") is what selected the blouse's
    neck point, because the neck sits further toward CF than the dart tip does.
    """
    ang, _ = ad.turn_angles(loop)
    return [k for k in keys if abs(ang[k]) >= min_turn]


def _orient(loop, keys):
    """(loop, keys) with the outline walking the junctions in the given order.

    `keys` arrives in INTENDED seam order -- located by role, which says nothing
    about which way round the outline runs. If walking forward from keys[0] does
    not reach keys[1] first, the winding is the mirror of what the seam names
    assume, so the loop is reversed and re-indexed. Same correction, and the same
    reason, as in `_leg_junctions`: a mirrored piece is stored wound the other
    way, and without this every seam comes out labelled as its opposite.
    """
    keys = [int(k) for k in keys]
    if len(set(keys)) != len(keys):
        raise ValueError(f'junctions collided: {keys}')
    n = len(loop)
    others = set(keys[1:])
    step = next(k for k in range(1, n + 1) if (keys[0] + k) % n in others)
    if (keys[0] + step) % n == keys[1]:
        return loop, keys
    return loop[::-1], [n - 1 - k for k in keys]


def _rotate_to(seq, first):
    """Cyclically rotate a list so `first` leads."""
    i = list(seq).index(first)
    return list(seq[i:]) + list(seq[:i])


# --------------------------------------------------------------------------- #
#  6812610700 -- V-neck blouse
# --------------------------------------------------------------------------- #
def _nparts(len_a, len_b, max_seg=None, min_seg=None):
    """How many EQUAL parts to cut both sides of a seam into.

    Equal parts on both sides is the whole trick: the fraction set {k/n} is
    symmetric, so it matches whichever way round either chain runs, and
    `match_interfaces` finds nothing to subdivide. Its subdivision is what
    creates the sub-resolution slivers that collapse when welded.

    Bounded below by `min_seg` on the SHORTER side, so no part can come out under
    the mesher's resolution.
    """
    min_seg = WELD_MIN_SEG if min_seg is None else min_seg
    max_seg = SEAM_MAX_SEG if max_seg is None else max_seg
    lo, hi = min(len_a, len_b), max(len_a, len_b)
    n = int(np.ceil(hi / max_seg)) if max_seg else 1
    return max(1, min(n, int(lo // max(min_seg, 1e-6))))


def _ufracs(n):
    """Interior fractions of n equal parts."""
    return [k / n for k in range(1, n)] if n > 1 else []


def _composite_fracs(f_split, n_a, n_b, a_first):
    """Fractions cutting one chain to face TWO chains meeting at `f_split`.

    The single side is cut at the junction the other side already has, then each
    chunk is cut into equal parts, so both sides end up with the same fraction
    list. `a_first` says whether this chain runs into chunk A first; when it does
    not, the whole list is mirrored -- which is why it has to be measured rather
    than assumed (the sleeve cap runs underarm-to-shoulder while the armhole it
    meets is listed yoke-first, i.e. shoulder-first).
    """
    if not a_first:
        f_split, n_a, n_b = 1.0 - f_split, n_b, n_a
    out = [f_split * k / n_a for k in range(1, n_a)]
    out.append(f_split)
    out += [f_split + (1.0 - f_split) * k / n_b for k in range(1, n_b)]
    return out


def _runs_into(interf, target):
    """True if `interf` starts at the end nearer `target`'s midpoint, in 3D."""
    def mid(i, k):
        e, pan = i.edges[k], i.panel[k]
        return 0.5 * (np.asarray(pan.point_to_3D(list(e.start)))
                      + np.asarray(pan.point_to_3D(list(e.end))))
    n = len(list(interf.edges))
    m = len(list(target.edges))
    t = np.mean([mid(target, k) for k in range(m)], axis=0)
    return (np.linalg.norm(mid(interf, 0) - t)
            < np.linalg.norm(mid(interf, n - 1) - t))


def _concat_seam(first, second):
    """NOT USED -- kept for the finding. Two interfaces joined head-to-tail.

    The observation is real: the blouse's yoke armhole runs upward (y 128.9 ->
    139.4) while the back's runs downward (124.1 -> 120.6), so concatenating them
    gives a seam that doubles back. But reversing a chunk with
    `Interface.reverse` to fix it made things WORSE -- 16 flagged welds became 32
    and 4 edges started collapsing again -- so `reverse` does not mean here what
    this assumed. Left in place because the head-to-head concatenation is still
    the most likely cause of the 16 that remain, and this records what was tried.

    `Interface.from_multiple` keeps each side's own edge order, and two chunks
    from DIFFERENT panels have no reason to be head-to-tail: the blouse's yoke
    armhole runs upward (y 128.9 -> 139.4) while the back's runs downward (124.1
    -> 120.6), so they meet head-to-HEAD. Concatenated as-is the seam doubles back
    on itself, and welding it folded two vertices of the back and two of the
    sleeve onto single points -- every one of the back-to-sleeve welds came out
    invalid. Reversing whichever chunk needs it makes the run continuous.
    """
    def ends(i):
        n = len(list(i.edges))
        p0, e0 = i.panel[0], i.edges[0]
        pN, eN = i.panel[n - 1], i.edges[n - 1]
        return (np.asarray(p0.point_to_3D(list(e0.start))),
                np.asarray(pN.point_to_3D(list(eN.end))))

    best, out = None, None
    for ra in (False, True):
        a = first.reverse(with_edge_dir_reverse=True) if ra \
            else pyg.Interface.from_multiple(first)
        for rb in (False, True):
            b = second.reverse(with_edge_dir_reverse=True) if rb \
                else pyg.Interface.from_multiple(second)
            gap = float(np.linalg.norm(ends(a)[1] - ends(b)[0]))
            if best is None or gap < best:
                best, out = gap, (ra, rb)
    ra, rb = out
    a = first.reverse(with_edge_dir_reverse=True) if ra \
        else pyg.Interface.from_multiple(first)
    b = second.reverse(with_edge_dir_reverse=True) if rb \
        else pyg.Interface.from_multiple(second)
    return pyg.Interface.from_multiple(a, b)


class BonprixBlouse(pyg.Component):
    """6812610700 blouse, shell only.

    Six shell pieces: front half (x2, seamed at CF), back half (x2, seamed at
    CB), back yoke half (x2, seamed at CB), sleeve (x2). Dropped: the front
    facing (`FRONTFACING-2X`, classified as a facing and never returned) and
    `YOKE-OUTSIDE`, a 24 x 38 cm skewed panel whose height cannot belong to this
    garment -- the back is 52 cm and the front 67, so a 24 cm shoulder yoke makes
    the back 76 cm long. The 13 cm yoke does fit (52 + 13 = 65 against the
    front's 67, the CF being longer for the V), so that is the one used.

    KNOWN DEFECT: the finished neckline opens wider than the photographed
    garment -- 87.9 cm against the pattern's 71.4, and 30.5 cm across against
    ~21, measured on the boundary edges of the size-38 sim. The shoulder line
    does not move (141.3 -> 141.1), so the V is spreading, not the garment
    sliding, and a panel-wide stretch average misses it entirely (the front
    panel reads 1.02). A length-matched binding strip stitched along the V and
    the back neck WAS tried: it closed the opening to 18.7 cm but dragged the
    whole garment up and twisted it, because a free strip placed beside the
    neckline has to travel to reach it and pulls the neckline along. Reverted.
    The real garment's front band (the 4 x 46 cm strip, annotated to the front)
    is probably the right piece to face this edge with, placed ON the seam
    rather than beside it.

    The front carries a real SIDE BUST DART: its outline spikes 12.9 cm inward at
    the underarm, from (-13.98, 38.31) out to (-1.08, 38.69) and back to
    (-13.60, 33.34) -- a 5 cm base folded to a point at the bust. It is sewn by
    stitching the two legs of that spike to each other, which is what the
    machinist does, rather than by editing it out of the outline.

    Cut-on-the-fold pieces are built as two halves with a seam rather than
    unfolded into one panel. The extra seam is cosmetic at this mesh resolution
    and it keeps every junction search on the half outline the DXF stores.
    """
    KEY = '6812610700'
    # The DXF has no cuff piece. Its only strip, 4 x 46 cm, is annotated
    # '..._FRO_..._015' -- the FRONT's piece id, not the sleeve's ('..._SLE_') --
    # so it is the band behind the V, not a cuff cut two-up. The photographed
    # garment does have gathered cuffs, so one is built from the body's wrist
    # instead: that at least ties it to the wearer rather than to a number
    # chosen to look right. This is the only dimension in this builder not taken
    # from a piece outline.
    # Half-axes of the ellipse the torso panels start on, as a fraction of the
    # bust circumference's radius: a body section is wider than deep.
    RX_FRAC, RZ_FRAC = 1.30, 0.78
    SIDE_THETA = 40.0         # panel centre angle from CF, degrees

    def __init__(self, body, design=None, size='38') -> None:
        super().__init__('bonprix_blouse')
        st = STYLES[self.KEY]
        pieces = st.pieces(size)

        def by(prefix, tall=None, short=None):
            hits = [p for p in pieces if p.block.startswith(prefix)
                    and (tall is None or p.size_cm()[1] > tall)
                    and (short is None or p.size_cm()[1] < short)]
            if len(hits) != 1:
                raise ValueError(f'{self.KEY}: {prefix!r} matched '
                                 f'{[h.block for h in hits]}')
            return hits[0]

        front = by('14WV', tall=60.0)
        back = by('13WV', tall=40.0)
        yoke = by('13WV', short=20.0)
        sleeve = by('x2WV')

        # Panels hang from the shoulder line, as the parametric bodice does
        # (bodice.py places its panels at `height - head_l`).
        hps = body['height'] - body['head_l']
        r = body['bust'] / (2 * np.pi)
        rx, rz = r * self.RX_FRAC, r * self.RZ_FRAC
        arm_deg = float(body['arm_pose_angle'])

        # Panels are built TWICE. The first pass measures the seams; the second
        # rebuilds with every mating seam cut into matching parts, so the two
        # sides of each stitch share an edge count and a fraction set and
        # `match_interfaces` has nothing left to subdivide. Its subdivision is
        # what produced the sub-resolution slivers that collapsed when welded and
        # pinched whole regions of the underarm onto one vertex.
        def build(parts):
            F, B, Y, S = {}, {}, {}, {}
            for tag, mir in (('r', False), ('l', True)):
                sgn = -1.0 if tag == 'r' else 1.0
                Y[tag] = self._yoke(f'blouse_yoke_{tag}', yoke, mir, hps, sgn,
                                    rx, rz, parts=parts.get('yoke'))
                F[tag] = self._front(f'blouse_front_{tag}', front, mir, hps, sgn,
                                     rx, rz, parts=parts.get('front'))
                # The back hangs from the yoke's lower edge, not the shoulder.
                B[tag] = self._back(f'blouse_back_{tag}', back, mir,
                                    hps - yoke.size_cm()[1], sgn, rx, rz,
                                    parts=parts.get('back'))
                # Sleeve halves: split at the cap apex so each gets the shoulder
                # point at its own local origin and rotates down the arm about
                # it. `tag` 'r' is the half at NEGATIVE x, which is the
                # convention the front, back and yoke pivots produce, so it needs
                # side = -1; +1 put each sleeve on the opposite side of the body
                # from the armhole it is sewn to.
                S[tag] = BonprixSleeve(
                    f'blouse_sleeve_{tag}', sleeve, body,
                    side=-1 if tag == 'r' else +1,
                    front_cap_len=self._cap_len, parts=parts.get('sleeve'))
            return F, B, Y, S

        def L(itf):
            return float(itf.edges.length())

        # -- pass 1: measure ---------------------------------------------------
        self._cap_len = None
        F0, B0, Y0, S0 = build({})
        self._cap_len = L(F0['r'].interfaces['armhole'])

        a_front, cap_f = L(F0['r'].interfaces['armhole']), L(S0['r'].front.interfaces['cap'])
        a_yoke, a_back = L(Y0['r'].interfaces['armhole']), L(B0['r'].interfaces['armhole'])
        cap_b = L(S0['r'].back.interfaces['cap'])
        sh_f, sh_y = L(F0['r'].interfaces['shoulder']), L(Y0['r'].interfaces['shoulder'])
        yk_bot, yk_bk = L(Y0['r'].interfaces['bottom']), L(B0['r'].interfaces['yoke'])
        su, sl = L(F0['r'].interfaces['side_up']), L(F0['r'].interfaces['side_lo'])
        b_side = L(B0['r'].interfaces['side'])
        ua_f, ua_b = L(S0['r'].front.interfaces['underarm']), L(S0['r'].back.interfaces['underarm'])

        n_arm = _nparts(a_front, cap_f)
        n_sh = _nparts(sh_f, sh_y)
        n_yk = _nparts(yk_bot, yk_bk)
        n_ua = _nparts(ua_f, ua_b)
        # cap_b faces yoke.armhole + back.armhole; the front side faces the
        # back's one side seam. Both are a single chain against two, so the
        # single one is cut at the junction the pair already has.
        f_arm = a_yoke / (a_yoke + a_back)
        n_ay = _nparts(a_yoke, cap_b * f_arm)
        n_ab = _nparts(a_back, cap_b * (1.0 - f_arm))
        f_side = su / (su + sl)
        n_su = _nparts(su, b_side * f_side)
        n_sl = _nparts(sl, b_side * (1.0 - f_side))
        # Which end each single chain runs from -- measured, not assumed.
        yoke_first = _runs_into(S0['r'].back.interfaces['cap'],
                                Y0['r'].interfaces['armhole'])
        up_first = _runs_into(B0['r'].interfaces['side'],
                              F0['r'].interfaces['side_up'])

        parts = {
            'front': {'armhole': _ufracs(n_arm), 'shoulder': _ufracs(n_sh),
                      'side_up': _ufracs(n_su), 'side_lo': _ufracs(n_sl)},
            'yoke': {'armhole': _ufracs(n_ay), 'shoulder': _ufracs(n_sh),
                     'bottom': _ufracs(n_yk)},
            'back': {'armhole': _ufracs(n_ab), 'yoke': _ufracs(n_yk),
                     'side': _composite_fracs(f_side, n_su, n_sl, up_first)},
            'sleeve': {
                'f': {'cap': _ufracs(n_arm), 'underarm': _ufracs(n_ua)},
                'b': {'cap': _composite_fracs(f_arm, n_ay, n_ab, yoke_first),
                      'underarm': _ufracs(n_ua)}},
        }
        print(f'  Matched seams: armhole {n_arm}, shoulder {n_sh}, yoke {n_yk}, '
              f'underarm {n_ua}, cap_b {n_ay}+{n_ab}, side {n_su}+{n_sl}')

        # -- pass 2: build matched --------------------------------------------
        self.fronts, self.backs, self.yokes, self.sleeves = build(parts)
        self._parts_n = dict(arm=n_arm, sh=n_sh, yk=n_yk, ua=n_ua,
                             ay=n_ay, ab=n_ab, su=n_su, sl=n_sl,
                             yoke_first=yoke_first, up_first=up_first)

        # The back eases onto the shorter yoke -- that gather IS the blouson.
        # NOTE the ruffle coefficient is recorded for the PATTERN only; it never
        # reaches the simulator (see `_taper_seams`), so what the sim actually
        # does at this seam is stretch the 18.3 cm yoke edge toward the back's
        # 21.6 cm. That is 18% on a 13 cm yoke -- visible but not damaging, and
        # it is what the pattern says, so it is left alone rather than faked.
        for tag in 'rl':
            nb = self._parts_n['yk']
            # `_seamof`, not a `> 1` guard: a seam listed in `parts` is renamed
            # `yoke_0` even when it comes out as ONE part (`_ufracs(1)` is an
            # empty fraction list, which still means "one part"), so the
            # unsuffixed name does not exist. `n_yk` fell to 1 and this raised
            # KeyError 'yoke'.
            b_yoke = _seamof(self.backs[tag], 'yoke', nb)
            y_bot = _seamof(self.yokes[tag], 'bottom', nb)
            _set_ruffle(b_yoke, float(b_yoke.edges.length())
                        / float(y_bot.edges.length()))

        # NO CUFF. This DXF has no cuff piece: its only strip is annotated to
        # the FRONT, not the sleeve, so it is the band behind the V. An earlier
        # version invented one from the body's wrist measurement -- a panel with
        # no counterpart in the pattern, which is exactly what a DXF-derived
        # garment must not contain. The sleeve hem is left free instead.
        self.cuffs = {}

        self.subs = [*self.fronts.values(), *self.backs.values(),
                     *self.yokes.values(), *self.sleeves.values(),
                     *self.cuffs.values()]
        # Every mating seam is now the same number of parts at the same
        # fractions on both sides, so each is welded as ONE rule over the
        # multi-part interfaces and `match_interfaces` has nothing to do.
        # `_stitch_matched` settles the order and then one direction for the
        # whole seam; per-edge direction is meaningless when the two panels sit
        # 55 cm apart in the flat layout.
        n = self._parts_n

        # Every key passed through here is in `parts`, so it is always suffixed
        # -- see the note on `b_yoke` above.
        seam = _seamof

        self.stitching_rules = pyg.Stitches(
            _stitch_aligned(self.fronts['r'].interfaces['cf'],
                            self.fronts['l'].interfaces['cf']),
            _stitch_aligned(self.backs['r'].interfaces['cb'],
                            self.backs['l'].interfaces['cb']),
            _stitch_aligned(self.yokes['r'].interfaces['cb'],
                            self.yokes['l'].interfaces['cb']),
            *[st for tag in 'rl' for st in (
                # the bust dart, folded out
                _stitch_aligned(self.fronts[tag].interfaces['dart_a'],
                                self.fronts[tag].interfaces['dart_b']),
                # shoulder: front to yoke; the yoke's lower edge to the back
                _stitch_matched(seam(self.fronts[tag], 'shoulder', n['sh']),
                                seam(self.yokes[tag], 'shoulder', n['sh'])),
                _stitch_matched(seam(self.yokes[tag], 'bottom', n['yk']),
                                seam(self.backs[tag], 'yoke', n['yk'])),
                # Side seam, ONE rule: the front's side is broken in two by the
                # dart while the back's is a single run, and stitching the halves
                # separately put the back vertex between them into two stitches.
                _stitch_matched(
                    pyg.Interface.from_multiple(
                        seam(self.fronts[tag], 'side_up', n['su']),
                        seam(self.fronts[tag], 'side_lo', n['sl'])),
                    seam(self.backs[tag], 'side', n['su'] + n['sl'])),
                # Sleeve tube: fold to fold, underarm to underarm -- closes the
                # two halves round the arm without hauling on the armhole.
                _stitch_aligned(self.sleeves[tag].front.interfaces['top'],
                                self.sleeves[tag].back.interfaces['top']),
                _stitch_matched(seam(self.sleeves[tag].front, 'underarm', n['ua']),
                                seam(self.sleeves[tag].back, 'underarm', n['ua'])),
                # Front cap to the front armhole; back cap to the yoke's armhole
                # plus the back's, which together are the other half.
                _stitch_matched(seam(self.sleeves[tag].front, 'cap', n['arm']),
                                seam(self.fronts[tag], 'armhole', n['arm'])),
                _stitch_matched(
                    seam(self.sleeves[tag].back, 'cap', n['ay'] + n['ab']),
                    pyg.Interface.from_multiple(
                        seam(self.yokes[tag], 'armhole', n['ay']),
                        seam(self.backs[tag], 'armhole', n['ab']))
                    if n['yoke_first'] else pyg.Interface.from_multiple(
                        seam(self.backs[tag], 'armhole', n['ab']),
                        seam(self.yokes[tag], 'armhole', n['ay']))),
            )],
        )
        _align_welds(self.stitching_rules)
        self.interfaces = {
            'bottom': pyg.Interface.from_multiple(
                *[self.fronts[t].interfaces['hem'] for t in 'rl'],
                *[self.backs[t].interfaces['hem'] for t in 'rl']),
        }
        # Only 'arm' panels keep colliding with the arms; everything else has
        # them filtered out (meshgen/garment.py). Sleeves first, rest after.
        for tag in 'rl':
            self.sleeves[tag].set_panel_label('arm')
        self.set_panel_label('body', overwrite=False)

    # -- piece preparation -------------------------------------------------- #
    @staticmethod
    def _prep(piece, mirror):
        """Outline (and fold line) centred on the bbox, optionally mirrored."""
        loop, notches = _normalised(piece)
        fold = None
        if piece.fold is not None:
            b = piece.boundary
            c = np.array([(b[:, 0].min() + b[:, 0].max()) / 2, b[:, 1].min()])
            fold = piece.fold - c
        if mirror:
            loop = loop * [-1.0, 1.0]
            notches = notches * [-1.0, 1.0] if len(notches) else notches
            fold = fold * [-1.0, 1.0] if fold is not None else None
        return loop, notches, fold

    # -- panel builders ----------------------------------------------------- #
    def _front(self, name, piece, mirror, hps, sgn, rx, rz, parts=None):
        """Front half.

        Nine corners, and the count is the same at 36, 38 and 40, so their cyclic
        ORDER of roles is fixed; only where the walk starts and which way it runs
        have to be pinned. Two role anchors do that: the neck point is the
        highest corner, and the dart point is the one corner that juts sideways
        out of the outline. In seam order the dart point is four junctions past
        the neck, so comparing those two positions settles the direction.
        """
        loop, cut, _, _ = _prep_both(piece, mirror)
        # Junctions are found on the CUT outline and mapped over -- see
        # `_map_idx`. The dart cannot be found on the net line at all.
        keys_c = corners(cut, thr=25.0, window=4.0)
        if len(keys_c) != 9:
            raise ValueError(f'{name}: expected 9 corners, got {len(keys_c)}')
        spikes_c = _darts(cut, keys_c)
        if len(spikes_c) != 1:
            raise ValueError(f'{name}: expected 1 dart spike, found {len(spikes_c)}')
        order_c = _rotate_to(sorted(keys_c), _pick(cut, keys_c, 'y', 'max'))
        if order_c.index(spikes_c[0]) != 4:
            order_c = [order_c[0]] + order_c[1:][::-1]
        fwd = _map_idx(order_c, cut, loop)
        keys = fwd
        neck = fwd[0]
        # The dart POINT is the outline's only near-reversal: the boundary runs
        # 12.9 cm out to the point and straight back, so it turns by 159 deg
        # there against at most 106 deg anywhere else, and it is the only corner
        # that turns the other way. Picking it by x instead -- 'the corner
        # closest to centre front' -- selected the neck point, because the neck
        # sits further toward CF than the dart tip does.
        names = ['shoulder', 'armhole', 'side_up', 'dart_a', 'dart_b',
                 'side_lo', 'hem', 'cf', 'neck']
        weld = {k: WELD_MIN_SEG for k in
                ('shoulder', 'armhole', 'side_up', 'side_lo')}
        loop, keys = _orient(loop, fwd)
        seams, source, loop, keys = _seams_of(loop, keys, names)
        # CF is the straight edge at the extreme x (mirrored for the left half).
        # Taken as a COORDINATE, not via a junction index: `_orient` may have
        # reversed the loop, which leaves coordinates alone but invalidates any
        # index captured before it.
        cf_x = float(loop[:, 0].min() if mirror else loop[:, 0].max())
        panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                         source=source, min_seg=weld, parts=parts,
                         pivot=[cf_x, float(loop[:, 1].max())])
        return _place_flat(panel, loop, cf_x, hps, ZB_FRONT)

    def _back(self, name, piece, mirror, hps, sgn, rx, rz, parts=None):
        """Back half below the yoke.

        Its side seam is left WHOLE. It used to be halved at its own arc midpoint
        so it could meet the front's two side sub-seams either side of the dart,
        but the front's dart sits at 17.6% of that seam, not 50%, so the two
        sides' internal boundary landed in different places and
        `match_interfaces` had to subdivide. `parts` cuts it at the fractions the
        front actually uses instead.
        """
        loop, cut, _, fold = _prep_both(piece, mirror)
        keys_c = corners(cut, thr=25.0, window=4.0)
        if len(keys_c) != 5:
            raise ValueError(f'{name}: expected 5 corners, got {len(keys_c)}')
        onfold = [k for k in keys_c if k in set(_on_fold(cut, fold, tol=0.3))]
        if len(onfold) != 2:
            raise ValueError(f'{name}: {len(onfold)} corners on the CB fold')
        cb_top = _pick(cut, onfold, 'y', 'max')
        cb_bot = _pick(cut, onfold, 'y', 'min')
        fwd_c = _rotate_to(sorted(keys_c), cb_top)
        if fwd_c.index(cb_bot) != 4:        # yoke, armhole, side, hem, THEN cb
            fwd_c = [cb_top] + fwd_c[1:][::-1]
        names = ['yoke', 'armhole', 'side', 'hem', 'cb']
        loop, keys = _orient(loop, _map_idx(fwd_c, cut, loop))
        seams, source, loop, keys = _seams_of(loop, keys, names)
        weld = {k: WELD_MIN_SEG for k in ('yoke', 'armhole', 'side')}
        cb_x = float(np.mean(np.asarray(fold, float)[:, 0]))
        panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                         source=source, min_seg=weld, parts=parts,
                         pivot=[cb_x, float(loop[:, 1].max())])
        return _place_flat(panel, loop, cb_x, hps, ZB_BACK)

    def _yoke(self, name, piece, mirror, hps, sgn, rx, rz, parts=None):
        """Back yoke half.

        Only three corners clear 25 deg here and the count is NOT the same at
        every size, so nothing is assumed about corners at all: the five
        junctions are the fold's two ends, the outer edge's two ends, and the
        neck point (the highest vertex on the piece).
        """
        loop, cut, _, fold = _prep_both(piece, False)
        # The yoke is stored with the OPPOSITE handedness from the back: the
        # back's fold sits at the +x end of its outline and the yoke's at the -x
        # end. Pivoted on their folds they then extend to opposite sides, so the
        # right yoke sat over the left back. Canonicalise to "extends -x" first,
        # then apply the left/right mirror -- the same ordering `_leg_junctions`
        # needs, for the same reason.
        fold_x = float(np.mean(np.asarray(fold, float)[:, 0]))
        if loop[:, 0].max() - fold_x > fold_x - loop[:, 0].min():
            loop, fold = loop * [-1.0, 1.0], fold * [-1.0, 1.0]
        if loop[:, 0].max() - fold_x > fold_x - loop[:, 0].min():
            cut = cut * [-1.0, 1.0]
        if mirror:
            loop, fold, cut = (loop * [-1.0, 1.0], fold * [-1.0, 1.0],
                               cut * [-1.0, 1.0])
        onfold = _on_fold(cut, fold, tol=0.3)
        outer = _on_edge(cut, 'left' if not mirror else 'right', band=1.0)
        keys_c = [_pick(cut, onfold, 'y', 'min'),              # CB, bottom
                  _pick(cut, outer, 'y', 'min'),               # outer, bottom
                  _pick(cut, outer, 'y', 'max'),               # outer, top
                  _pick(cut, range(len(cut)), 'y', 'max'),     # neck point
                  _pick(cut, onfold, 'y', 'max')]              # CB, top
        names = ['bottom', 'armhole', 'shoulder', 'neck', 'cb']
        loop, keys = _orient(loop, _map_idx(keys_c, cut, loop))
        seams, source, loop, keys = _seams_of(loop, keys, names)
        cb_x = float(np.mean(np.asarray(fold, float)[:, 0]))
        panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                         source=source, parts=parts,
                         min_seg={c: WELD_MIN_SEG for c in
                                  ('bottom', 'armhole', 'shoulder')},
                         pivot=[cb_x, float(loop[:, 1].max())])
        return _place_flat(panel, loop, cb_x, hps, ZB_BACK)

# --------------------------------------------------------------------------- #
#  8672609700 -- midi shirt dress
# --------------------------------------------------------------------------- #
class BonprixShirtDress(pyg.Component):
    """8672609700 shirt dress, shell only.

    Bodice front (x2, CF-seamed), bodice back (x2, CB-seamed), shoulder yoke
    (x2), sleeve (x2), cuff (x2), and one flared skirt panel front and back.
    Dropped: the skirt LININGS (classified as lining), the cuff TEMPLATE, the
    1.5 cm finished stand collar and three 0.5 cm finished stay tapes -- at 1 cm
    of seam allowance a 2.5 cm strip nets down to half a centimetre, which is
    below the mesh resolution and is construction rather than drape.

    The front bodice carries TWO darts, a waist dart opening at the waist seam
    and a bust dart opening at the side seam, both drawn as zero-width spikes and
    both sewn by stitching their legs together. Its corner count is not stable
    across the size run (11, 11, 10), so nothing here is indexed off the corner
    list: the darts are found by their reversal angle and every other junction by
    role.

    The skirt panels are single flared pieces whose waist arc is far shorter than
    their hem arc; they take the bodice waist directly, with no gather, because a
    gather cannot be expressed to the simulator (see `_taper_seams`).
    """
    KEY = '8672609700'
    RX_FRAC, RZ_FRAC = 1.30, 0.78
    SIDE_THETA = 40.0
    # Longest part a mating seam may be cut into. As the tiered dress: the
    # global 25 cm forces one cubic over a whole curved seam, which misses by
    # centimetres.
    SEAM_SEG = 5.0
    # Turn angle counted as a corner when cutting a mating seam. This garment's
    # curves are GENTLER than the 20 deg default sees: the back armhole bends 10
    # to 15 deg and reports no corners at all at 20, so a 4.2 cm part spanned two
    # of them and its single cubic missed by 26.5 mm.
    SEAM_CORNER_THR = 10.0

    def __init__(self, body, design=None, size='38') -> None:
        super().__init__('bonprix_shirtdress')
        st = STYLES[self.KEY]
        # Selected by BLOCK NAME, on the exporter's mapping. Annotation suffixes
        # do not identify a piece here: this DXF carries a near-duplicate of
        # almost everything (`x1...-BK1` and `14...-BK` share the annotation
        # `8672609700-BK`), so `by('-BK')` was resolving on whichever copy
        # `dedupe_near` happened to keep.
        def pick(prefix, pool=None):
            hits = [q for q in (pool if pool is not None else st.pieces(size))
                    if (q.name if pool is not None else q.block).startswith(
                        prefix)]
            if len(hits) != 1:
                raise ValueError(
                    f'{self.KEY}: {prefix!r} matched {len(hits)}: '
                    f'{[(q.name if pool is not None else q.block) for q in hits]}')
            return hits[0]

        front, back = pick('x68672609700-FRT-2X'), pick('148672609700-BK')
        yoke, sleeve = pick('x58672609700-YOKE-4X'), pick('x98672609700-SL-2X')
        fskirt = pick('x88672609700-FRTSKIRT')
        bskirt = pick('x78672609700-BKSKIRT')
        # The two cuffs and the collar are NARROW trims. Taken as DRAWN, which is
        # the call already made for 8642610003's collar: a 1 cm allowance off
        # each long edge leaves the back waistband 0.50 cm and the collar 1.47,
        # at or under the mesher's 1.0 cm cell. The cuffs are `kind='template'`
        # besides, so `Style.pieces` drops them entirely -- the same trap as the
        # jogger's `Shape 5`/`Shape 6` and the tiered dress's `Shape 19`, and the
        # reason an earlier version fell back to 'CUFF-2X' (41 x 7.5), which is
        # not either cuff.
        raw = ad.pieces_for_size(st.dxf, st.size_label(size),
                                 fabric_only=False)
        # Both cuffs carry the SAME block name, so they are told apart by size --
        # which is also what distinguishes them: the ring is the short one
        # (19 x 5 at size 38) and the bell the long one (41 x 2.5).
        cuffs = sorted(
            (q for q in raw if q.name.startswith('128672609700-CUFF-TTEM')),
            key=lambda q: float(np.asarray(q.boundary, float)[:, 0].ptp()))
        if len(cuffs) != 2:
            raise ValueError(f'{self.KEY}: expected 2 CUFF-TTEM, got '
                             f'{[q.name for q in cuffs]}')
        cuff_a, cuff_b = cuffs
        collar = pick('118672609700-COLLAR-2X', raw)
        bwaist = pick('108672609700-BKWAIST', raw)

        hps = body['height'] - body['head_l']
        r = body['bust'] / (2 * np.pi)
        rx, rz = r * self.RX_FRAC, r * self.RZ_FRAC
        arm_deg = float(body['arm_pose_angle'])
        self._shoulder_w = float(body['shoulder_w'])
        # Waist height: the bodice hangs from the shoulder, so its waist seam
        # lands one bodice-length down. Taken from the piece, not the body, so
        # the skirt meets the bodice wherever this pattern puts the waist.
        waist_y = hps - float(front.size_cm()[1])

        def dims(q):
            L = np.asarray(q.boundary, float)
            return float(L[:, 0].ptp()), float(L[:, 1].ptp())

        a_len, a_h = dims(cuff_a)
        b_len, b_h = dims(cuff_b)
        # BKWAIST is cut on the fold at x=0, so its 26.97 doubles to the whole
        # back: 53.94 x 2.50 against a 52.17 cm back bodice waist.
        bw_len, bw_h = dims(bwaist)
        bw_len *= 2.0

        # Built TWICE, as the blouse and the tiered dress are. Pass one measures
        # the mating seams; pass two rebuilds with both sides of every seam cut
        # at the SAME fractions, so `match_interfaces` has nothing to subdivide.
        # Its subdivision is what left 32 collapsed edges and 91 of 104 welds
        # invalid here -- the part COUNTS already agreed, which is why this
        # looked like something else for so long.
        _o = os.environ.get('SD_ORD', 'xx')
        _ORD = {'f': None if _o[0] == 'x' else _o[0] == '1',
                'b': None if _o[1] == 'x' else _o[1] == '1'}

        def build(parts):
            fr, bk, yk, sl = {}, {}, {}, {}
            cf, ct, cp, bl, bt, bp = {}, {}, {}, {}, {}, {}
            for tag, mir in (('r', False), ('l', True)):
                sgn = -1.0 if tag == 'r' else 1.0
                # This front piece stores CF at its min-x end, the opposite way
                # round from its own back piece, so the mirror flag is inverted
                # to land both halves of a side seam on the same side.
                fr[tag] = self._front(f'sd_front_{tag}', front, not mir,
                                      hps, sgn, rx, rz,
                                      parts=parts.get('front'))
                bk[tag] = self._back(f'sd_back_{tag}', back, mir, hps, sgn,
                                     rx, rz,
                                     parts=(parts.get('back') or {}).get(tag))
                yk[tag] = self._yoke(f'sd_yoke_{tag}', yoke, mir, hps, sgn,
                                     rx, rz, parts=parts.get('yoke'))
                sl[tag] = BonprixSleeve(
                    f'sd_sleeve_{tag}', sleeve, body,
                    side=-1 if tag == 'r' else +1,
                    front_cap_len=self._cap_len,
                    parts=parts.get('sleeve'))
            for tag in 'rl':
                # Both cuff tiers AS DRAWN: the ring is gathered onto the sleeve
                # opening and the bell flares back off the ring -- the
                # exporter's "the sleeve goes inward and then outward". Neither
                # gather reaches the simulator (see `_taper_seams`), so the
                # ratios are printed rather than absorbed into a shape.
                #
                # Split FRONT/BACK, as the jogger's ankle ribs are. A single
                # panel carrying the whole circumference has to lie in one plane
                # -- the mid-plane between the sleeve's two halves -- so it
                # cannot wrap: it collapsed instead of closing round the wrist.
                # Two halves sit in their own half's plane, each welded to that
                # half's hem at short range, and close into a ring on their two
                # side seams.
                hf = float(_anyseam(sl[tag].front, 'hem').edges.length())
                hb = float(_anyseam(sl[tag].back, 'hem').edges.length())
                side = -1 if tag == 'r' else +1
                for sd, half, share in (('f', sl[tag].front, hf / (hf + hb)),
                                        ('b', sl[tag].back, hb / (hf + hb))):
                    k = f'{tag}{sd}'
                    cf[k], ct[k], cp[k] = _cuff_at_sleeve(
                        f'sd_cuff_{tag}_{sd}', half, a_len * share, a_h, side,
                        order_rev=_ORD[sd])
                    bl[k], bt[k], bp[k] = _bell_below(
                        f'sd_bell_{tag}_{sd}', cf[k], b_len * share, b_h)
            # Skirt panels are placed FLAT at the parametric skirt's own z
            # offsets, not wrapped onto the bodice ellipse. `place_around` is
            # right for a 25 cm bodice panel but not for a 112 cm one: tangent
            # to an ellipse of 10.9 cm depth it lands INSIDE the body, and the
            # sim's first frame then has 16k skirt vertices interpenetrating the
            # torso and legs -- frame 0 blew the 120 s watchdog.
            fsk = self._skirt('sd_skirt_front', fskirt, waist_y, Z_FRONT,
                              parts=parts.get('fskirt'))
            # The back waistband LENGTHENS the back below the bodice, on the
            # exporter's call, so the back skirt drops by the band's own depth
            # while the front skirt still meets the front bodice directly.
            bsk = self._skirt('sd_skirt_back', bskirt, waist_y - bw_h, Z_BACK,
                              parts=parts.get('bskirt'))
            band = self._strip('sd_bkwaist', bw_len, bw_h,
                               waist_y - bw_h, ZB_BACK,
                               top_parts=parts.get('band_top'),
                               bot_parts=parts.get('band_bot'))
            return fr, bk, yk, sl, cf, ct, cp, bl, bt, bp, fsk, bsk, band

        def C(pan, key):
            return np.asarray(pan.source_seams[key], float)

        seg = self.SEAM_SEG
        self._cap_len = None
        p0 = build({})
        f0, b0, y0, s0, *_rest = p0
        fs0, bs0 = _rest[-3], _rest[-2]
        self._cap_len = float(f0['r'].interfaces['armhole'].edges.length())
        p0 = build({})
        f0, b0, y0, s0, *_rest = p0
        fs0, bs0 = _rest[-3], _rest[-2]

        def pair(chain_a, chain_b):
            return _matched_fracs(chain_a, chain_b, seg,
                                  thr=self.SEAM_CORNER_THR)

        # A composite seam -- one chain facing two -- is handled in two steps,
        # because the two sides need DIFFERENT things. The chunk fraction lists
        # are computed ONCE: both bodice halves share one `parts` entry, so they
        # must be identical, and `_matched_fracs`' sets are mirror-symmetric so
        # one list serves both. Only the SPLIT POSITION varies per side, since
        # the mirrored half runs into the other chunk first. Recomputing the
        # chunk lists per side gave the right skirt half 9 parts against the
        # bodice's 5.
        def _split(chunks, lead_first):
            la, lb = _chain_len(chunks[0]), _chain_len(chunks[1])
            return (la if lead_first else lb) / (la + lb)

        def chunk_fracs(single, chunks, lead_first):
            f = _split(chunks, lead_first)
            lead, trail = ((chunks[0], chunks[1]) if lead_first
                           else (chunks[1], chunks[0]))
            fr_lead = pair(_sub_chain(single, 0.0, f), lead)
            fr_trail = pair(_sub_chain(single, f, 1.0), trail)
            return ((fr_lead, fr_trail) if lead_first
                    else (fr_trail, fr_lead))

        def in_order(pan, keys, counts, lead_first):
            """The two chunks as ONE interface, in the order the single chain
            meets them. The fraction mapping already respects that order; the
            interface has to as well, or the two sides are welded chunk-for-
            chunk in opposite senses."""
            pairs = list(zip(keys, counts))
            if not lead_first:
                pairs = pairs[::-1]
            return pyg.Interface.from_multiple(
                *[_seamof(pan, k, c) for k, c in pairs])

        def single_list(chunks, fr0, fr1, lead_first):
            """The single chain's fractions, and how many parts precede the
            split -- cut at the junction the pair already has."""
            f = _split(chunks, lead_first)
            lead_fr, trail_fr = ((fr0, fr1) if lead_first else (fr1, fr0))
            return ([x * f for x in lead_fr] + [f]
                    + [f + x * (1.0 - f) for x in trail_fr]), len(lead_fr) + 1

        # -- waist: each skirt half-waist faces one bodice half. The front's is
        # two chains (the bust dart splits it), the back's one.
        def xmid(itf):
            k = len(list(itf.edges))
            return float(np.mean([_mid3(itf, j) for j in range(k)], axis=0)[0])

        # WHICH bodice half each skirt half-waist takes is measured. The skirt's
        # own r/l labelling is the opposite way round from the bodice's: its
        # `waist_r` sits at x=+11.5 against front_r's waist at x=-5.4, and the
        # back the same, so assuming `waist_r` -> the 'r' panels sent every waist
        # weld straight across the body -- crossed seams at the bottom of both
        # bodices and the top of both skirts.
        wf = {sh: min('rl', key=lambda t: abs(
                  xmid(fs0.interfaces[f'waist_{sh}'])
                  - xmid(f0[t].interfaces['waist_a']))) for sh in 'rl'}
        wbk = {sh: min('rl', key=lambda t: abs(
                   xmid(bs0.interfaces[f'waist_{sh}'])
                   - xmid(b0[t].interfaces['waist']))) for sh in 'rl'}
        if set(wf.values()) != {'r', 'l'} or set(wbk.values()) != {'r', 'l'}:
            raise ValueError(f'{self.KEY}: waist halves not paired 1:1 -- '
                             f'front {wf}, back {wbk}')
        self._wf, self._wbk = wf, wbk
        # Measured PER SIDE. The left panels are mirrored, which reverses the
        # winding, so the chunk a chain runs into first is not the same on both
        # -- and one flag applied to both left the front waist welded chunk-for-
        # chunk in opposite senses on one side.
        fw_lead = {sh: _runs_into(fs0.interfaces[f'waist_{sh}'],
                                  f0[wf[sh]].interfaces['waist_b'])
                   for sh in 'rl'}
        # Computed per side. `_matched_fracs`' own sets are mirror-symmetric (it
        # adds 1-f for every corner), so a chunk's list serves both halves -- but
        # a COMPOSITE list is not symmetric: its split sits at `f`, and the
        # mirrored side needs it at 1-f. Deriving both from the right side left
        # the last 2 collapsed edges on the left front waist.
        waist_one, side_one, w_lead_n, s_lead_n = {}, {}, {}, {}
        w_chunks = [C(f0[wf['r']], 'waist_b'), C(f0[wf['r']], 'waist_a')]
        fr_wb, fr_wa = chunk_fracs(C(fs0, 'waist_r'), w_chunks, fw_lead['r'])
        for sh in 'rl':
            waist_one[sh], w_lead_n[sh] = single_list(
                w_chunks, fr_wb, fr_wa, fw_lead[sh])
        fr_bw = pair(C(bs0, 'waist_r'), C(b0['r'], 'waist'))
        # -- shoulders: front to yoke, yoke to back
        fr_sh_f = pair(C(f0['r'], 'shoulder'), C(y0['r'], 'front'))
        fr_sh_b = pair(C(y0['r'], 'back'), C(b0['r'], 'shoulder'))
        # -- side: the front's is split by the dart, the back's is one run
        sd_lead = {t: _runs_into(b0[t].interfaces['side'],
                                 f0[t].interfaces['side_up']) for t in 'rl'}
        s_chunks = [C(f0['r'], 'side_up'), C(f0['r'], 'side_lo')]
        fr_su, fr_sl = chunk_fracs(C(b0['r'], 'side'), s_chunks, sd_lead['r'])
        for t in 'rl':
            side_one[t], s_lead_n[t] = single_list(
                s_chunks, fr_su, fr_sl, sd_lead[t])
        # -- armholes and the sleeve's own seams
        fr_arm_f = pair(C(s0['r'].front, 'cap'), C(f0['r'], 'armhole'))
        fr_arm_b = pair(C(s0['r'].back, 'cap'), C(b0['r'], 'armhole'))
        fr_ua = pair(C(s0['r'].front, 'underarm'), C(s0['r'].back, 'underarm'))
        # The sleeve's fold-to-fold seam is left unparted: both halves are the
        # same 51.94 cm, it is 1:1 already, and it was never among the flagged
        # welds. `BonprixSleeve` also renames it from `_closing` to `top` after
        # construction, so a `parts` entry would have to use the old name and the
        # rename would then miss.
        # -- skirt side seams
        fr_sk_r = pair(C(fs0, 'side_r'), C(bs0, 'side_r'))
        fr_sk_l = pair(C(fs0, 'side_l'), C(bs0, 'side_l'))

        parts = {
            'front': {'waist_b': fr_wb, 'waist_a': fr_wa,
                      'shoulder': fr_sh_f, 'side_up': fr_su,
                      'side_lo': fr_sl, 'armhole': fr_arm_f},
            'back': {t: {'waist': fr_bw, 'shoulder': fr_sh_b,
                         'side': side_one[t], 'armhole': fr_arm_b}
                     for t in 'rl'},
            'yoke': {'front': fr_sh_f, 'back': fr_sh_b},
            # The hem is cut in HALF so it matches the cuff's own split at x=0.
            # Left as one edge it had FEWER edges than the cuff's top, and
            # `_stitch_aligned` skips its order test entirely when
            # `min(len(a), len(b)) == 1` -- so the reversal it exists to catch
            # went undetected and the front cuffs were welded back-to-front
            # (endpoints: same 24.59 cm against flip 8.46).
            'sleeve': {'f': {'cap': fr_arm_f, 'underarm': fr_ua,
                             'hem': [0.5]},
                       'b': {'cap': fr_arm_b, 'underarm': fr_ua,
                             'hem': [0.5]}},
            'fskirt': {'waist_r': waist_one['r'], 'waist_l': waist_one['l'],
                       'side_r': fr_sk_r, 'side_l': fr_sk_l},
            'bskirt': {'waist_r': fr_bw, 'waist_l': fr_bw,
                       'side_r': fr_sk_r, 'side_l': fr_sk_l},
            # Widths, not fractions: `_taper_seams` splits a straight edge into
            # the given part widths, so both faces of the band pair 1:1 with the
            # halves they take.
            'band_top': [1.0] * (2 * (len(fr_bw) + 1)),
            'band_bot': [1.0] * (len(fr_bw) + 1),
        }
        (self.fronts, self.backs, self.yokes, self.sleeves, self.cuffs,
         self.cuff_top, self.cuff_partner, self.bells, self.bell_top,
         self.bell_partner, self.fskirt, self.bskirt, self.band) = build(parts)

        # Counted off the BUILT panels, not off the fraction lists. `min_seg`
        # merges a part that comes out under the mesher's cell, so a seam can
        # end up with fewer edges than it was given fractions for -- the front
        # waist was asked for 5 and the skirt for 6, and `match_interfaces`
        # split the difference into 4 collapsed edges.
        def ncount(pan, key):
            return max(1, len([x for x in pan.interfaces
                               if x == key
                               or (x.startswith(key + '_')
                                   and x.rsplit('_', 1)[-1].isdigit())]))

        n = {'wb': ncount(self.fronts['r'], 'waist_b'),
             'wa': ncount(self.fronts['r'], 'waist_a'),
             'fw': ncount(self.fskirt, 'waist_r'),
             'bw': ncount(self.bskirt, 'waist_r'),
             'sh_f': ncount(self.fronts['r'], 'shoulder'),
             'sh_b': ncount(self.backs['r'], 'shoulder'),
             'su': ncount(self.fronts['r'], 'side_up'),
             'sl': ncount(self.fronts['r'], 'side_lo'),
             'bs': ncount(self.backs['r'], 'side'),
             'arm_f': ncount(self.fronts['r'], 'armhole'),
             'arm_b': ncount(self.backs['r'], 'armhole'),
             'ua': ncount(self.sleeves['r'].front, 'underarm'),
             'sk_r': ncount(self.fskirt, 'side_r'),
             'sk_l': ncount(self.fskirt, 'side_l')}
        if n['wb'] + n['wa'] != n['fw'] or n['su'] + n['sl'] != n['bs']:
            raise ValueError(
                f"{self.KEY}: composite seam counts disagree -- waist "
                f"{n['wb']}+{n['wa']} vs {n['fw']}, side "
                f"{n['su']}+{n['sl']} vs {n['bs']}")
        print('  Matched seams: waist %d/%d+%d, shoulder %d/%d, side %d+%d, '
              'armhole %d/%d, underarm %d, skirt side %d/%d'
              % (n['bw'], n['wb'], n['wa'], n['sh_f'], n['sh_b'], n['su'],
                 n['sl'], n['arm_f'], n['arm_b'], n['ua'], n['sk_r'],
                 n['sk_l']))
        _open = sum(float(_anyseam(getattr(self.sleeves['r'], h),
                                   'hem').edges.length())
                    for h in ('front', 'back'))
        print(f'  Bell cuff: sleeve opening {_open:.2f} -> ring {a_len:.2f}'
              f'x{a_h:.1f} ({a_len / _open:.2f}x) -> bell {b_len:.2f}'
              f'x{b_h:.1f} ({b_len / a_len:.2f}x)')

        self.subs = [*self.fronts.values(), *self.backs.values(),
                     *self.yokes.values(), *self.sleeves.values(),
                     *self.cuffs.values(), *self.bells.values(),
                     self.fskirt, self.bskirt, self.band]

        def SM(pan, key, count):
            return _seamof(pan, key, count)

        _w = os.environ.get('SD_FW', '00')
        _FW = {'f': None if _w[0] == 'x' else _w[0] == '1',
               'b': None if _w[1] == 'x' else _w[1] == '1'}
        _e = os.environ.get('SD_RV', 'xx')
        _RV = {'f': None if _e[0] == 'x' else _e[0] == '1',
               'b': None if _e[1] == 'x' else _e[1] == '1'}

        # Which half of the band takes which back bodice half, and which half of
        # its bottom takes which skirt half -- measured by x, like the waist
        # pairing, rather than assumed from the r/l labels.
        def _bot(e):
            return (self.band.seam(*[f'bottom_{e}_{i}'
                                     for i in range(n['bw'])])
                    if n['bw'] > 1 else self.band.interfaces[f'bottom_{e}'])

        self._band_top = [
            (k, min('rl', key=lambda t: abs(
                xmid(_bandseam(self.band, k, n['bw']))
                - xmid(SM(self.backs[t], 'waist', n['bw'])))))
            for k in (0, 1)]
        self._band_bot = [
            (e, min('rl', key=lambda sh: abs(
                xmid(_bot(e)) - xmid(SM(self.bskirt, f'waist_{sh}', n['bw'])))))
            for e in ('l', 'r')]
        if ({t for _, t in self._band_top} != {'r', 'l'}
                or {sh for _, sh in self._band_bot} != {'r', 'l'}):
            raise ValueError(f'{self.KEY}: band halves not paired 1:1 -- '
                             f'{self._band_top}, {self._band_bot}')
        print(f'  Back waistband: {bw_len:.2f}x{bw_h:.1f} between a '
              f'{2 * float(SM(self.backs["r"], "waist", n["bw"]).edges.length()):.2f}'
              f'cm bodice waist and a '
              f'{2 * float(SM(self.bskirt, "waist_r", n["bw"]).edges.length()):.2f}'
              f'cm skirt waist')


        self.stitching_rules = pyg.Stitches(
            _stitch(self.fronts['r'].interfaces['cf'],
                    self.fronts['l'].interfaces['cf']),
            _stitch(self.backs['r'].interfaces['cb'],
                    self.backs['l'].interfaces['cb']),
            *[_stitch_matched(SM(self.fskirt, f'side_{s}', n[f'sk_{s}']),
                              SM(self.bskirt, f'side_{s}', n[f'sk_{s}']))
              for s in ('r', 'l')],
            # waist: each skirt half-waist takes one bodice half
            # The front waist in TWO rules, split where the waist dart sits.
            # As one rule it cannot work: the dart's two legs lie between
            # `waist_b` and `waist_a` in the panel's edge order, so the composite
            # interface is DISCONTINUOUS in 3D, and `_stitch_matched` applies one
            # direction across the gap -- which welded the 9.3 cm `waist_a`
            # chunk end-to-end onto itself.
            *[st for sh in 'rl'
              for st in _waist_rules(self, sh, self._wf[sh], n,
                                     fw_lead[sh], w_lead_n[sh])],
            # back: bodice -> waistband -> skirt. The band's top edge runs
            # +x to -x, so its first `n_bw` parts are one half and the last
            # `n_bw` the other; which half is which is measured.
            *[_stitch_matched(_bandseam(self.band, k, n['bw']),
                              SM(self.backs[t], 'waist', n['bw']))
              for k, t in self._band_top],
            *[_stitch_matched(
                self.band.seam(*[f'bottom_{e}_{i}' for i in range(n['bw'])])
                if n['bw'] > 1 else self.band.interfaces[f'bottom_{e}'],
                SM(self.bskirt, f'waist_{sh}', n['bw']))
              for e, sh in self._band_bot],
            *[x for tag in 'rl' for x in (
                _stitch(self.fronts[tag].interfaces['wdart_a'],
                        self.fronts[tag].interfaces['wdart_b']),
                _stitch(self.fronts[tag].interfaces['bdart_a'],
                        self.fronts[tag].interfaces['bdart_b']),
                # front shoulder -> yoke -> back shoulder
                _stitch_matched(SM(self.fronts[tag], 'shoulder', n['sh_f']),
                                SM(self.yokes[tag], 'front', n['sh_f'])),
                _stitch_matched(SM(self.yokes[tag], 'back', n['sh_b']),
                                SM(self.backs[tag], 'shoulder', n['sh_b'])),
                # Side seam in ONE rule: the front's side is broken in two by
                # the bust dart while the back's is a single run, so stitching
                # the halves separately would put the back vertex between them
                # into two stitches.
                _stitch_matched(
                    in_order(self.fronts[tag], ('side_up', 'side_lo'),
                             (n['su'], n['sl']), sd_lead[tag]),
                    SM(self.backs[tag], 'side', n['bs'])),
                # Sleeve closed as two halves round the arm
                _stitch(self.sleeves[tag].front.interfaces['top'],
                        self.sleeves[tag].back.interfaces['top']),
                _stitch_matched(SM(self.sleeves[tag].front, 'underarm', n['ua']),
                                SM(self.sleeves[tag].back, 'underarm', n['ua'])),
                _stitch_matched(SM(self.sleeves[tag].front, 'cap', n['arm_f']),
                                SM(self.fronts[tag], 'armhole', n['arm_f'])),
                _stitch_matched(SM(self.sleeves[tag].back, 'cap', n['arm_b']),
                                SM(self.backs[tag], 'armhole', n['arm_b'])),
                # Each tier's two halves take their own sleeve half's hem,
                # then close into a ring on their two side seams -- the
                # jogger's arrangement.
                # `force=False`: the direction is stated, the order still
                # measured (short range now, so that vote is sound). Left to the
                # vote as well, both tiers' top welds collapsed -- measured
                # across the alternatives, this is the setting that reaches 0.
                *[_stitch_matched(
                    self.cuff_top[f'{tag}{sd}'],
                    self.cuff_partner[f'{tag}{sd}'],
                    force=_weld_sense(self.cuff_top[f'{tag}{sd}'],
                                      self.cuff_partner[f'{tag}{sd}']))
                  for sd in 'fb'],
                *[_stitch_matched(
                    self.bell_top[f'{tag}{sd}'],
                    self.bell_partner[f'{tag}{sd}'],
                    force=_weld_sense(self.bell_top[f'{tag}{sd}'],
                                      self.bell_partner[f'{tag}{sd}']))
                  for sd in 'fb'],
                *[_stitch_aligned(self.cuffs[f'{tag}f'].interfaces[e],
                                  self.cuffs[f'{tag}b'].interfaces[e])
                  for e in ('side_r', 'side_l')],
                *[_stitch_aligned(self.bells[f'{tag}f'].interfaces[e],
                                  self.bells[f'{tag}b'].interfaces[e])
                  for e in ('side_r', 'side_l')],
            )],
        )
        self.interfaces = {
            'bottom': pyg.Interface.from_multiple(
                self.fskirt.interfaces['hem'], self.bskirt.interfaces['hem']),
        }
        for tag in 'rl':
            self.sleeves[tag].set_panel_label('arm')
            for sd in 'fb':
                self.cuffs[f'{tag}{sd}'].set_panel_label('arm')
                self.bells[f'{tag}{sd}'].set_panel_label('arm')
        self.set_panel_label('body', overwrite=False)

    # -- panels ------------------------------------------------------------- #
    @staticmethod
    def _strip(name, length, height, y, z, top_parts=None, bot_parts=None):
        """A rectangle at the given length, as the DXF draws the waistband.

        Its top edge is cut into the part widths of the two bodice waists it
        takes and each half of its bottom into the skirt half it takes, so every
        weld pairs 1:1 and `match_interfaces` has nothing to subdivide.
        """
        panel = DxfPanel(name, _taper_seams(length, length, height,
                                            top_parts=top_parts,
                                            bot_l_parts=bot_parts,
                                            bot_r_parts=bot_parts),
                         verbatim=True, translation=[0.0, y, z])
        panel.synthetic = True
        return face_to(panel, [0.0, 0.0, float(np.sign(z) or 1.0)])

    def _front(self, name, piece, mirror, hps, sgn, rx, rz, parts=None):
        # Everything is located on the CUT outline and mapped across: a 1 cm
        # inward offset collapses a zero-width dart spike, and this piece has
        # two of them (172 and 163 deg on the cut line, a single 108 deg corner
        # on the net line). See `_map_idx`.
        loop, cut, _, _ = _prep_both(piece, mirror)
        keys = corners(cut, thr=22.0, window=3.0)
        apex = _darts(cut, keys)
        if len(apex) != 2:
            raise ValueError(f'{name}: expected 2 dart spikes, found {len(apex)}')
        # CF vs side seam. Both are x extremes of the piece, and picking them
        # off the `mirror` flag had them SWAPPED on this DXF: every role then
        # landed on the wrong edge, `junc` came out non-cyclic, `_orient` read
        # that as mirrored winding and reversed it, and all eleven seams walked
        # the long way round the loop -- 223 fitted edges on a 36-vertex outline,
        # the front bodice retraced ten times. Decide it geometrically instead:
        # only the CF side runs all the way up to the neck, the side seam stops
        # at the underarm, so the taller extreme is CF whichever way the piece
        # is wound.
        xs = cut[:, 0]
        lo, hi = float(xs.min()), float(xs.max())
        reach = lambda x0: float(cut[np.abs(xs - x0) < 4.0][:, 1].max())
        cf_x, side_x = (lo, hi) if reach(lo) > reach(hi) else (hi, lo)
        # The CF line's ends are junctions but its top need not be a CORNER: the
        # neckline meets CF at ~9 deg here, well under the corner threshold. Take
        # both ends off the CF line itself rather than out of `keys`.
        on_cf = np.nonzero(np.abs(xs - cf_x) < 0.4)[0]
        if len(on_cf) < 2:
            raise ValueError(f'{name}: CF line has {len(on_cf)} vertices')
        cf_top = int(on_cf[np.argmax(cut[on_cf, 1])])
        cf_bot = int(on_cf[np.argmin(cut[on_cf, 1])])
        # The waist dart's two bases sit on the waist (the lowest edge); the bust
        # dart's sit on the side seam.
        def bases(a):
            i = keys.index(a)
            return keys[i - 1], keys[(i + 1) % len(keys)]
        pairs = {a: bases(a) for a in apex}
        waist_apex = min(apex, key=lambda a: np.mean([cut[b][1] for b in pairs[a]]))
        bust_apex = [a for a in apex if a != waist_apex][0]
        taken = set(apex) | set(sum(pairs.values(), ())) | {cf_top, cf_bot}
        side = [k for k in keys
                if abs(cut[k][0] - side_x) < 4.0 and k not in taken]
        if len(side) < 2:
            raise ValueError(f'{name}: {len(side)} side-seam corners found')
        underarm = _pick(cut, side, 'y', 'max')
        # Neck point is the highest corner; the shoulder tip is the corner just
        # before it in walk order (the armhole and shoulder carry no corners of
        # their own). Any slip here trips the walk-order guard in `_ordered`.
        neck_pt = _pick(cut, [k for k in keys if k not in apex], 'y', 'max')
        # "Between" has to be CYCLIC -- the walk from the underarm to the neck
        # crosses index 0 on this piece, so a plain `underarm < k < neck_pt` test
        # finds nothing.
        span = (neck_pt - underarm) % len(cut)
        after = sorted((k for k in keys if 0 < (k - underarm) % len(cut) < span),
                       key=lambda k: (k - underarm) % len(cut))
        if not after:
            raise ValueError(f'{name}: no shoulder-tip corner between underarm '
                             f'{underarm} and neck point {neck_pt}')
        shoulder_tip = after[-1]
        junc = [cf_top, cf_bot,
                pairs[waist_apex][0], waist_apex, pairs[waist_apex][1],
                _pick(cut, side, 'y', 'min'),                  # waist/side
                pairs[bust_apex][0], bust_apex, pairs[bust_apex][1],
                underarm, shoulder_tip, neck_pt]
        # 12 junctions, so the neckline is its OWN seam. Folding it into
        # 'shoulder' (11 names) would have stitched the neckline to the yoke.
        names = ['cf', 'waist_a', 'wdart_a', 'wdart_b', 'waist_b', 'side_lo',
                 'bdart_a', 'bdart_b', 'side_up', 'armhole', 'shoulder', 'neck']
        # `junc` is already in seam order (it was built role by role), so it is
        # handed to _orient as-is -- re-sorting it ascending would scramble the
        # correspondence with `names`.
        loop, junc = _orient(loop, _map_idx(junc, cut, loop))
        seams, source, loop, junc = _seams_of(loop, junc, names)
        # Short-edge guard on every WELDED seam, as the blouse and the tiered
        # bodice have. Without it the corner fit emitted edges far under the
        # mesher's 1.0 cm cell -- 0.19 cm on the yoke, 0.21 on the front, 0.38 to
        # 0.49 on the backs and skirt -- and each one collapsed to a point when
        # welded: 38 of this garment's 102 stitches. Its part COUNTS were already
        # matched, which is why it looked like a matching problem and was not.
        panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                         source=source, parts=parts,
                         min_seg={k: WELD_MIN_SEG for k in
                                  ('cf', 'waist_a', 'waist_b', 'side_lo',
                                   'side_up', 'armhole', 'shoulder')},
                         pivot=[cf_x, float(loop[:, 1].max())])
        return _place_flat(panel, loop, cf_x, hps, ZB_FRONT)

    def _back(self, name, piece, mirror, hps, sgn, rx, rz, parts=None):
        loop, _, fold = _prep_piece(piece, mirror)
        keys = corners(loop, thr=22.0, window=3.0)
        if len(keys) != 6:
            raise ValueError(f'{name}: expected 6 corners, got {len(keys)}')
        onfold = [k for k in keys if k in set(_on_fold(loop, fold, tol=0.3))]
        cb_bot = _pick(loop, onfold, 'y', 'min')
        cb_top = _pick(loop, onfold, 'y', 'max')
        fwd = _rotate_to(sorted(keys), cb_top)
        if fwd.index(cb_bot) != 5:      # neck, shoulder, armhole, side, waist, cb
            fwd = [cb_top] + fwd[1:][::-1]
        names = ['neck', 'shoulder', 'armhole', 'side', 'waist', 'cb']
        loop, keys = _orient(loop, fwd)
        cb_x = float(np.mean(np.asarray(fold, float)[:, 0]))
        seams, source, loop, keys = _seams_of(loop, keys, names)
        panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                         source=source, parts=parts,
                         min_seg={k: WELD_MIN_SEG for k in
                                  ('shoulder', 'armhole', 'side', 'waist')},
                         pivot=[cb_x, float(loop[:, 1].max())])
        return _place_flat(panel, loop, cb_x, hps, ZB_BACK)

    def _yoke(self, name, piece, mirror, hps, sgn, rx, rz, parts=None):
        """Shoulder yoke: a small quad bridging the front and back shoulders.
        Its two long edges take the shoulders, the two short ones are the neck
        and armhole ends."""
        loop, _, _ = _prep_piece(piece, mirror)
        keys = corners(loop, thr=22.0, window=3.0)
        if len(keys) != 4:
            raise ValueError(f'{name}: expected 4 corners, got {len(keys)}')
        start = _pick(loop, keys, 'y', 'min')
        fwd = _rotate_to(sorted(keys), start)
        lens = []
        for cand in (fwd, [start] + fwd[1:][::-1]):
            ln, kk = _orient(loop, cand)
            ss = _ordered(ln, kk, ['e0', 'e1', 'e2', 'e3'])
            lens.append((ln, kk, [float(np.sum(np.linalg.norm(
                np.diff(ss[f'e{i}'], axis=0), axis=1))) for i in range(4)]))
        # Name the two LONG edges front/back and the two short ones neck/armhole,
        # starting the walk at whichever orientation puts a long edge first.
        loop, keys, L = lens[0] if lens[0][2][0] > lens[0][2][1] else lens[1]
        names = ['front', 'neck', 'back', 'armhole']
        seams, source, loop, keys = _seams_of(loop, keys, names)
        panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                         source=source, parts=parts,
                         min_seg={k: WELD_MIN_SEG for k in
                                  ('front', 'back', 'armhole')},
                         pivot=[float(loop[:, 0].mean()),
                                float(loop[:, 1].max())])
        # The yoke bridges front to back over the shoulder, so it lies FLAT in
        # the horizontal plane at the shoulder point, not in a vertical plane
        # like the body panels. Rotating -90 about X lays it down; the shoulder
        # offset is the body's own shoulder width.
        panel.translation = np.array(
            [sgn * float(self._shoulder_w) / 2, hps + 1.0, 0.0], float)
        panel.rotate_by(R.from_euler('XYZ', [-90.0, 0, 0], degrees=True))
        return face_to(panel, [0.0, 1.0, 0.0])

    def _skirt(self, name, piece, waist_y, z, parts=None):
        """Flared skirt panel: waist arc on top, two side seams, hem arc below.

        Nothing here is taken from the corner detector. The waist corners of a
        flared panel are GENTLE -- the outline just changes from a straight side
        seam to a shallow arc -- so at any threshold that finds them it also
        finds a dozen points along the hem, and at a threshold that does not it
        returns three corners of which one is spurious. The four junctions are
        instead the panel's extreme points: the two lowest-and-outermost tips
        where the side seams meet the hem, and the two highest points, which are
        the ends of the waist arc either side of its dip.
        """
        loop, _, _ = _prep_piece(piece, False)
        top = _on_edge(loop, 'top', band=3.0)
        w_l = _pick(loop, top, 'x', 'min')
        w_r = _pick(loop, top, 'x', 'max')
        t_l = _pick(loop, range(len(loop)), 'x', 'min')
        t_r = _pick(loop, range(len(loop)), 'x', 'max')
        junc = [t_l, w_l, w_r, t_r]
        names = ['side_l', 'waist', 'side_r', 'hem']
        loop, junc = _orient(loop, junc)
        seams = _ordered(loop, junc, names)
        L = {k: float(np.sum(np.linalg.norm(np.diff(v, axis=0), axis=1)))
             for k, v in seams.items()}
        if L['hem'] < L['waist']:
            raise ValueError(f'{name}: hem {L["hem"]:.0f}cm is shorter than '
                             f'waist {L["waist"]:.0f}cm -- junctions are wrong')
        # Halve the waist arc so each half can take one bodice half; the split
        # point is the arc-length midpoint, which is where the side seam sits.
        w = seams.pop('waist')
        cut = _half_index(w)
        seams = {'side_l': seams['side_l'], 'waist_l': w[:cut + 1],
                 'waist_r': w[cut:], 'side_r': seams['side_r'],
                 'hem': seams['hem']}
        panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                         source={k: v.copy() for k, v in seams.items()},
                         parts=parts,
                         min_seg={k: WELD_MIN_SEG for k in
                                  ('waist_l', 'waist_r', 'side_l', 'side_r')})
        panel.translate_to([0.0, waist_y - float(loop[:, 1].max()), z])
        return face_to(panel, [0.0, 0.0, float(np.sign(z))])


def _matched_split(chains, max_seg=10.0):
    """Split several chains into the SAME number of equal-arc-length parts.

    Fidelity: one cubic cannot follow a seam that is part straight and part
    curved, so it bows -- measured on the jogger, a single-cubic waist came out
    2.66 cm short of its source and 1.15 cm off it at worst, and a drafted
    straight edge reads as a curve. Several short cubics track the source to
    well under a millimetre.

    Correspondence: a multi-edge seam welds edge i to edge i, so the two sides of
    a stitch must be cut into the same NUMBER of parts. Passing both chains here
    guarantees that -- the count comes from the longest of them -- while each
    side still splits at its own equal fractions, so the parts line up
    proportionally even where the partners differ in length (the jogger inseams
    are 75.9 and 71.7).
    """
    lens = [float(np.sum(np.linalg.norm(np.diff(np.asarray(c, float), axis=0),
                                        axis=1))) for c in chains]
    n = max(1, int(np.ceil(max(lens) / float(max_seg))))
    return [_split_equal(c, n) for c in chains], n


def _split_at_fracs(chain, fracs):
    """Split a point chain at exact cumulative arc-length fractions.

    The cut point is INTERPOLATED, not snapped to the nearest existing vertex.
    Snapping leaves the two sides of a paired seam with slightly different
    fractions, `StitchingRule.isMatching` (atol 0.05) then rejects them, and
    `match_interfaces` subdivides after all -- which is the whole thing being
    avoided, since its edge splits come back in swapped order.
    """
    chain = np.asarray(chain, float)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))]
    total = d[-1]
    out, prev_pt, prev_i = [], chain[0], 0
    for f in list(fracs) + [1.0]:
        target = total * float(f)
        j = int(np.searchsorted(d, target))
        j = min(max(j, 1), len(chain) - 1)
        span = d[j] - d[j - 1]
        t = 0.0 if span <= 1e-12 else (target - d[j - 1]) / span
        cut_pt = chain[j - 1] + t * (chain[j] - chain[j - 1])
        seg = np.vstack([prev_pt, chain[prev_i + 1:j], cut_pt])
        seg = seg[np.r_[True, np.linalg.norm(np.diff(seg, axis=0), axis=1) > 1e-9]]
        if len(seg) >= 2:
            out.append(seg)
        prev_pt, prev_i = cut_pt, j - 1
    return out


def _split_equal(chain, n):
    """Split a point chain into `n` sub-chains of equal arc length.

    Used for the leg outseam instead of splitting at the DXF's match notches.
    Notches looked like the right answer -- they mark where the hip curve starts,
    and `hyperdrop.py` uses them -- but they do not survive the mirror: the left
    leg's split came back with 3 chunks against the right leg's 4, the per-index
    stitch pairing then covered only 3, and the fourth stretch of outseam was
    left unsewn (visible as one leg hanging open). Equal quarters correspond by
    construction on every panel and at every size, and they serve the only
    purpose the split has here, which is to keep each cubic short enough to
    follow the curve.
    """
    chain = np.asarray(chain, float)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))]
    cuts = [int(np.argmin(np.abs(d - d[-1] * k / n))) for k in range(n + 1)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        if b > a:
            out.append(chain[a:b + 1])
    return out


def _half_index(chain):
    """Index splitting a point chain into two equal arc lengths."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))]
    return int(np.argmin(np.abs(d - d[-1] / 2.0)))


def _map_idx(idx, src, dst):
    """Map vertex indices from one outline onto the nearest vertices of another.

    Junctions -- corners, and above all DARTS -- have to be found on the CUT
    line. A dart is drawn as a zero-width spike, and a 1 cm inward offset
    collapses it: the shirt-dress front's two spikes of 172 and 163 degrees come
    back off the net line as a single 108 degree corner, which no threshold can
    tell from an ordinary one. On the cut line they are unmistakable. So they are
    located there and carried over here, and the blunted net tip -- a dart whose
    sewing line stops 1 cm short of the cut tip, which is what the garment
    does -- is what the panel gets.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)

    def fracs(loop):
        d = np.r_[0.0, np.cumsum(np.linalg.norm(
            np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1))]
        return d[:-1] / d[-1]

    # Mapped by PROXIMITY. Arc-length fraction was wrong: it assumes the two
    # outlines start at corresponding vertices and wind the same way, and the net
    # line does not -- it comes back from shapely's offset with an arbitrary
    # start vertex and possibly the opposite winding. On the blouse front that
    # sent the neck point at (4.1, 68.3) to a net vertex at (9.8, 0.0), and every
    # junction with it, so every seam took the wrong edge and the garment
    # collapsed to a band across the chest. Proximity does not care where either
    # loop starts or which way it runs; `_orient` settles the winding afterwards
    # and `_ordered` refuses the result if the junctions are not a walk order.
    #
    # Assigned most-certain-first so a junction with one obvious partner claims
    # it before an ambiguous one can, and each net vertex is taken once: two
    # junctions on one vertex is what made seams wrap the whole loop before.
    src_j = src[[int(i) % len(src) for i in idx]]
    dist = np.linalg.norm(dst[None, :, :] - src_j[:, None, :], axis=2)
    out = [None] * len(idx)
    used = set()
    for i in np.argsort(dist.min(axis=1)):
        j = next((int(c) for c in np.argsort(dist[i]) if int(c) not in used), None)
        if j is None:
            raise ValueError(f'cannot map {len(idx)} junctions onto '
                             f'{len(dst)} vertices without collision')
        used.add(j)
        out[int(i)] = j
    # The net line is the cut line inset by the seam allowance, so a junction
    # should land within about that distance (a dart tip lands on its blunted
    # net tip). Anything further means the outlines do not correspond at all.
    MAP_MAX = 4.0                                              # cm
    worst = max((float(dist[i, out[i]]), int(i)) for i in range(len(idx)))
    if worst[0] > MAP_MAX:
        raise ValueError(
            f'junction {idx[worst[1]]} at '
            f'{np.round(src_j[worst[1]], 2).tolist()} is {worst[0]:.2f} cm from '
            f'the nearest free vertex of the net outline (limit {MAP_MAX} cm)')
    return out


def _prep_both(piece, mirror):
    """(net loop, cut loop, notches, fold) under one identical transform."""
    b = piece.boundary
    c = np.array([(b[:, 0].min() + b[:, 0].max()) / 2, b[:, 1].min()])
    net, cut = b - c, np.asarray(piece.cut, float) - c
    notches = piece.notches - c if len(piece.notches) else np.zeros((0, 2))
    fold = piece.fold - c if piece.fold is not None else None
    if mirror:
        net, cut = net * [-1.0, 1.0], cut * [-1.0, 1.0]
        notches = notches * [-1.0, 1.0] if len(notches) else notches
        fold = fold * [-1.0, 1.0] if fold is not None else None
    return net, cut, notches, fold


def _prep_piece(piece, mirror):
    loop, notches = _normalised(piece)
    fold = None
    if piece.fold is not None:
        b = piece.boundary
        c = np.array([(b[:, 0].min() + b[:, 0].max()) / 2, b[:, 1].min()])
        fold = piece.fold - c
    if mirror:
        loop = loop * [-1.0, 1.0]
        notches = notches * [-1.0, 1.0] if len(notches) else notches
        fold = fold * [-1.0, 1.0] if fold is not None else None
    return loop, notches, fold


def _flat_sleeve(name, piece, body, side, hem_lift=2.0):
    """One-piece sleeve, PIVOTED AT THE CAP APEX so placement is symmetric.

    `HyperdropSleeve` is the better model and the blouse uses it, but it splits
    the outline at the cap apex and at the underarm corners, and on a sleeve with
    a nearly flat top -- this one runs level from (-2.5, 40.7) to (-38.5, 40.7)
    before the cap rises -- the underarm corner lands adjacent to the hem corner
    and one arc comes out degenerate ("Start and end of an edge should differ").

    So: keep the flat piece as a single panel, but put the CAP APEX at the local
    origin before placing it. That is the part of hyperdrop's sleeve that
    matters here -- rotating about a bbox-relative origin is not
    mirror-symmetric, which left the two sleeves of a pair at different heights.
    """
    loop, _ = _normalised(piece)
    # Mirror so the sleeve extends AWAY from centre front on its own side. The
    # piece is stored with its bulk on +x of the cap apex, so the side that is
    # to occupy -x is the one that gets flipped.
    if side > 0:
        loop = loop * [-1.0, 1.0]
    keys = corners(loop, thr=22.0, window=3.0)
    if len(keys) != 4:
        raise ValueError(f'{name}: expected 4 corners, got {len(keys)}')
    low = sorted(keys, key=lambda i: loop[i][1])[:2]
    start = _pick(loop, low, 'x', 'min')
    other = [k for k in low if k != start][0]
    fwd = _rotate_to(sorted(keys), start)
    if fwd[1] != other:
        fwd = [start] + fwd[1:][::-1]
    loop, keys = _orient(loop, fwd)
    names = ['hem', 'side_b', 'cap', 'side_f']
    seams, source, loop, keys = _seams_of(loop, keys, names)
    apex = loop[int(np.argmax(loop[:, 1]))]
    panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                     source=source, pivot=apex)
    sgn = 1.0 if side > 0 else -1.0
    panel.translate_to([sgn * float(body['shoulder_w']) / 2,
                        float(body['height']) - float(body['head_l']) - hem_lift,
                        0.0])
    # The sleeve must swing down TOWARD its own side. Local -y is "down the
    # sleeve", and a +Z rotation carries it toward +x, so the half sitting at
    # -x needs a negative angle. Getting this backwards sent each sleeve across
    # the body to the opposite armhole.
    panel.rotate_by(R.from_euler(
        'XYZ', [0, 0, sgn * float(body['arm_pose_angle'])], degrees=True))
    return face_to(panel, [sgn, 0.0, 0.0])


class BonprixSleeve(pyg.Component):
    """Sleeve split at the cap apex into front and back halves.

    This is `HyperdropSleeve`'s layout -- shoulder point at each half's local
    origin, halves placed either side of the arm at z = +/-15 and rotated down it
    -- and it exists because placing the sleeve as ONE flat piece is what pulled
    every upper-body garment off the shoulders. A flat sleeve starts as a 36 cm
    wide sheet standing out past the hand; when its two side seams weld into an
    11 cm tube, that contraction drags the armhole -- and the bodice with it --
    down off the shoulder. Visible in the first six frames of any of the earlier
    runs: the bodice does not fall, the sleeves haul it down. Two halves already
    sit either side of the arm, so closing fold<->fold and underarm<->underarm
    pulls nothing.

    `HyperdropSleeve` was tried here first and abandoned because it left the
    underarm 161 mm from its source -- but that was its `single=('cap',
    'underarm')`, one cubic for a 47 cm seam, the same forcing that cost the
    jogger hem 8.4 mm. Dropped here, so the corner fit runs and long sleeves
    stay faithful.

    `front_cap_len` picks which half is the front: the halves are assigned so the
    front's cap is the better match for the armhole it will be sewn to. Guessing
    by sign of x put each sleeve on the opposite side from its armhole.
    """

    def __init__(self, name, piece, body, side=+1, front_cap_len=None,
                 parts=None):
        super().__init__(name)
        loop, notches = _normalised(piece)
        # Prefer the cap notch the pattern maker marked over the geometric apex.
        i_apex = (nearest_idx(loop, max(notches, key=lambda q: q[1]))
                  if len(notches) else int(np.argmax(loop[:, 1])))
        apex = loop[i_apex].copy()
        hem_y = loop[:, 1].min()
        # The hem corners, taken over a BAND. A 0.05 cm window (what
        # `HyperdropSleeve` uses on its own short sleeve) assumes a dead-level
        # hem: this one drops 0.21 cm from its corners to its middle, so the
        # window caught only the middle and the "corners" came back at x = -4.97
        # and +8.75 instead of -18 and +18. The underarm spans then ran from the
        # underarm THROUGH part of the hem -- 60.32 and 56.53 cm against the
        # DXF's own 47.28/47.29 side seams -- and welding two side seams 3.8 cm
        # apart in length made the vertex matcher collapse 26 vertices of four
        # panels onto one point.
        hem_band = max(1.0, 0.03 * float(loop[:, 1].ptp()))
        hem = np.where(loop[:, 1] < hem_y + hem_band)[0]
        i_hem_p = int(hem[np.argmax(loop[hem, 0])])
        i_hem_m = int(hem[np.argmin(loop[hem, 0])])
        above = loop[:, 1] > hem_y + 0.5
        i_ua_p = int(np.argmax(np.where(above, loop[:, 0], -1e9)))
        i_ua_m = int(np.argmin(np.where(above, loop[:, 0], +1e9)))
        # The hem is halved by INSERTING the split point, not by picking the
        # nearest stored vertex: on these sleeves the hem is one straight edge
        # with only its two end vertices, so "nearest to the middle" returned an
        # end and one half's hem came out zero-length ("seam has fewer than 2
        # distinct points" -- the degenerate edge that made `HyperdropSleeve`
        # look unusable here).
        hem_chain = arc(loop, i_hem_m, i_hem_p)
        hem_parts = split_at_points(hem_chain, [[apex[0], hem_y]],
                                    tol=1e9, min_seg=0.5)
        if len(hem_parts) != 2:
            mid = _half_index(hem_chain)
            hem_parts = [hem_chain[:mid + 1], hem_chain[mid:]]

        halves = {
            'a': (['cap', 'underarm'], [(i_apex, i_ua_m), (i_ua_m, i_hem_m)],
                  ('hem', hem_parts[0])),
            'b': (['underarm', 'cap'], [(i_hem_p, i_ua_p), (i_ua_p, i_apex)],
                  ('hem', hem_parts[1])),
        }
        rot = np.array([[0.0, 1.0], [-1.0, 0.0]])           # -90 deg about z
        built = {}
        for role, (names, spans, (hem_name, hem_part)) in halves.items():
            seams = {n: arc(loop, a, b) for n, (a, b) in zip(names, spans)}
            seams[hem_name] = np.asarray(hem_part, float)
            # A NEEDLE in the hem is a DART the pattern folds out: the outline
            # doubles back on a single vertex (|turn| > 140 deg). 8242610411's
            # sleeve carries one 4 cm deep; the shirt dress's and the tiered
            # dress's have none, so this only fires where a dart exists. Left
            # whole, the hem keeps the spike as an unsewn flap.
            hem_keys = [hem_name]
            hc = np.asarray(seams[hem_name], float)
            if len(hc) >= 4:
                turn = np.abs(ad.turn_angles(hc)[0])
                spike = [i for i in range(1, len(hc) - 1) if turn[i] > 140.0]
                if len(spike) == 1:
                    i = spike[0]
                    del seams[hem_name]
                    seams['hem_0'] = hc[:i]
                    seams['dart_a'] = hc[i - 1:i + 1]
                    seams['dart_b'] = hc[i:i + 2]
                    seams['hem_1'] = hc[i + 1:]
                    hem_keys = ['hem_0', 'dart_a', 'dart_b', 'hem_1']
            # ... and put them back in walk order: half 'a' runs cap, underarm,
            # hem; half 'b' runs hem, underarm, cap. DxfPanel builds the loop in
            # dict order, so the order here is the outline's.
            keys = (['cap', 'underarm'] + hem_keys) if role == 'a' else \
                   (hem_keys + ['underarm', 'cap'])
            seams = {k: (seams[k] - apex) @ rot.T for k in keys}
            if role == 'a':
                # the -x half comes out spanning +y; flip so both halves hang
                # the same way off the shared fold line
                seams = {k: v * [1.0, -1.0] for k, v in seams.items()}
            built[role] = (seams, float(np.sum(np.linalg.norm(
                np.diff(seams['cap'], axis=0), axis=1))))

        # Which half is the front
        roles = ['a', 'b']
        if front_cap_len is not None:
            roles.sort(key=lambda r: abs(built[r][1] - float(front_cap_len)))
        for role, tag in zip(roles, ('f', 'b')):
            seams = built[role][0]
            panel = DxfPanel(f'{name}_{tag}', seams, verbatim=True,
                             presimplified=True,
                             source={k: v.copy() for k, v in seams.items()},
                             min_seg={'cap': WELD_MIN_SEG,
                                      'underarm': WELD_MIN_SEG},
                             parts=(parts or {}).get(tag),
                             translation=[-float(body['shoulder_w']) / 2, 0,
                                          15.0 if tag == 'f' else -15.0])
            panel.interfaces['top'] = panel.interfaces.pop('_closing')
            setattr(self, 'front' if tag == 'f' else 'back', panel)

        self.subs = [self.front, self.back]
        self.translate_by([0, float(body['height']) - float(body['head_l']), 0])
        # Each PANEL rotates about its own origin -- the shoulder point. Rotating
        # the COMPONENT turns it about its bbox pivot and drags the sleeve off the
        # shoulder (see `HyperdropSleeve`).
        arm = R.from_euler('XYZ', [0, 0, float(body['arm_pose_angle'])],
                           degrees=True)
        for panel in self.subs:
            panel.rotate_by(arm)
        if side > 0:
            self.mirror()
        face_to(self.front, [0, 0, 1])
        face_to(self.back, [0, 0, -1])


def _bell_below(name, above, ring, height):
    """A rectangle hung off `above`'s LOWER edge, in its plane and at its angle.

    The shirt dress's cuff is a bell: a 19 cm ring gathered onto the 38.9 cm
    sleeve opening, then a 41 cm tier taken off that ring's lower edge. The
    second tier's partner is the first CUFF, not the sleeve, so it inherits that
    cuff's placement instead of re-deriving one from the hem -- which is also
    what keeps its weld in-plane and short-range.

    `above` spans local y 0..its own height with its origin on that lower edge,
    so the new panel's top lands exactly there whatever the rotation is.
    """
    pan = DxfPanel(name, _taper_seams(ring, ring, height, n_top=2),
                   verbatim=True)
    pan.rotation = above.rotation
    pan.translation = (np.asarray(above.translation, float)
                       - above.rotation.apply([0.0, height, 0.0]))
    pan.synthetic = True
    face_to(pan, [0.0, 0.0, 1.0])
    return (pan, pan.seam('top_0', 'top_1'),
            above.seam('bottom_l', 'bottom_r'))


def _cuff_at_sleeve(name, sleeve, wrist_len, height, side, faithful=True,
                    order_rev=None):
    """Tapered cuff placed at the far end of an already-placed HyperdropSleeve.

    A RECTANGLE at `wrist_len`, as the cuff piece is drawn. It used to be a
    trapezoid -- narrow edge at `wrist_len`, wide edge length-matched to the
    sleeve hem so the weld stretched nothing (see `_taper_seams`) -- but that
    shape is invented, and on the tiered dress the piece is a 19 x 1 cm elastic,
    so the taper ran 39.6 cm down to 19 over 1 cm of height: two nearly
    horizontal 10 cm flanges rather than a cuff. Faithful, the 2x the weld
    carries IS the elastic being stretched onto the opening, and its recovery is
    what grips -- the same call as the jogger's ankle ribs (`FAITHFUL_CUFFS`).
    Pass `faithful=False` for the trapezoid.

    The position is READ OFF the placed sleeve rather than recomputed. A sleeve
    has been translated to the shoulder, rotated down the arm about that point
    and possibly mirrored, so re-deriving where its hem ended up means redoing
    all of that and getting the sign conventions right twice; `_world_pt` just
    asks the panel.
    """
    halves = ([sleeve.front, sleeve.back] if hasattr(sleeve, 'front')
              else [sleeve])
    hem_len = sum(float(_anyseam(h, 'hem').edges.length()) for h in halves)
    pts = []
    for half in halves:
        e = _anyseam(half, 'hem').edges
        for v in (e[0].start, e[-1].end):
            pts.append(_world_pt(half, v))
    centre = np.mean(pts, axis=0)
    partner = pyg.Interface.from_multiple(
        *[_anyseam(h, 'hem') for h in halves])

    # The sleeve has been rotated down the arm, so its hem is a SLANTED edge --
    # 45.6 deg off horizontal at an `arm_pose_angle` of 44.4. A cuff left at the
    # default identity rotation lies horizontal, so the weld has to twist the
    # seam shut through that angle and the cuff never wraps the wrist. Turn the
    # cuff so its top edge is parallel to the hem and its height runs along the
    # arm.
    #
    # The angle is read off the hem itself rather than recomposed from
    # `arm_pose_angle` and the mirror sign: the same +-90 deg turn is the wrong
    # sign on the mirrored side, and the sleeve's own local axes flip with it
    # (local +x is up the arm on the right, -x on the left). Perpendicular-to-
    # the-hem, disambiguated by which side the sleeve body lies on, needs no
    # sign conventions at all.
    rot = R.identity()
    # The chord is taken between the hem's two EXTREME vertices, not between the
    # traversal's first start and last end. Those coincide whenever the seam's
    # parts are listed against the boundary order -- which is what the mirror
    # does: on the right sleeve `hem_1` precedes `hem_0`, so `first.start` IS
    # `last.end`, the chord measured 0.000, and both right cuffs silently fell
    # back to identity and sat horizontal.
    e0 = _anyseam(halves[0], 'hem').edges
    pts2 = np.array([_world_pt(halves[0], v)[:2]
                     for e in e0 for v in (e.start, e.end)], float)
    gap = np.linalg.norm(pts2[:, None, :] - pts2[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(gap)), gap.shape)
    chord = pts2[j] - pts2[i]
    if np.linalg.norm(chord) > 1e-6:
        chord = chord / np.linalg.norm(chord)
        up = np.array([-chord[1], chord[0]])          # hem turned +90 deg
        body = np.mean([_world_pt(h, v) for h in halves
                        for e in h.edges for v in (e.start,)], axis=0)[:2]
        if np.dot(up, body - centre[:2]) < 0:
            up = -up                                  # ...or -90, whichever
        rot = R.from_euler(                           # points up the arm
            'XYZ', [0, 0, np.degrees(np.arctan2(up[1], up[0])) - 90.0],
            degrees=True)

    # The cuff's top is cut to the sleeve hem's own edge count and length
    # proportions, and which end meets which is measured -- the same treatment
    # the jogger's waist casing and ankle ribs needed. Left as one straight edge
    # against a multi-edge hem, `_auto_rw` only ever judges edge 0 (it loops over
    # `min(len(a), len(b))`) and `match_interfaces` then subdivides the cuff into
    # welds that all inherit that one answer.
    def make(parts, _n=name):
        pan = DxfPanel(_n, _taper_seams(wrist_len,
                                        wrist_len if faithful else hem_len,
                                        height, top_parts=parts.get('top')),
                       verbatim=True)
        pan.rotation = rot
        # The top edge's local midpoint is (0, height), so this puts it exactly
        # on the hem centre whatever `rot` is.
        pan.translation = np.asarray(centre, float) - rot.apply([0.0, height,
                                                                0.0])
        pan.synthetic = True
        return face_to(pan, [float(np.sign(side)) or 1.0, 0.0, 0.0])

    widths, order = _matched_slot(make, 'top', partner, {})
    if order_rev is not None:
        # `_matched_slot` measures which end of the cuff meets which end of the
        # hem, but on the sleeve FRONT half its answer comes out swapped: the two
        # candidate panels are geometrically IDENTICAL there (the hem's parts are
        # equal length, so `widths` and `widths[::-1]` are the same list) and the
        # only difference is the consume order, which makes its distance test a
        # near-tie. `order_rev` states it instead.
        n_w = len(widths)
        order = list(reversed(range(n_w))) if order_rev else list(range(n_w))
        widths = widths[::-1] if order_rev else widths
    panel = make({'top': widths})
    if len(widths) > 1:
        top = pyg.Interface.from_multiple(
            *[panel.interfaces[f'top_{k}'] for k in order])
    else:
        top = panel.seam('top_r', 'top_l')
    return panel, top, partner



# --------------------------------------------------------------------------- #
#  8642610003 -- tiered dress with a shirred waist
# --------------------------------------------------------------------------- #
def _corner_fracs(chain, thr=20.0):
    """Arc-length fractions of the corners INSIDE an open chain."""
    c = np.asarray(chain, float)
    if len(c) < 3:
        return []
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(d)]
    if cum[-1] <= 0:
        return []
    out = []
    for i in range(1, len(c) - 1):
        v1, v2 = c[i] - c[i - 1], c[i + 1] - c[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        ang = np.degrees(np.arccos(
            np.clip(float(np.dot(v1, v2)) / (n1 * n2), -1.0, 1.0)))
        if ang > thr:
            out.append(float(cum[i] / cum[-1]))
    return out


def _matched_fracs(chain_a, chain_b, max_seg, thr=20.0, tol=0.03):
    """Interior fractions to cut BOTH sides of a seam at.

    Equal parts alone are not enough. `parts` forces one cubic per part, and a
    cubic cannot cross a corner -- on the tiered bodice a 3.89 cm part with a
    corner inside it came out 7.15 mm off its source. So every corner of EITHER
    chain has to be a part boundary, plus enough uniform cuts to keep parts under
    `max_seg`.

    Each corner fraction is added together with its MIRROR (1 - f), because the
    two chains may run in opposite directions and this is called before the
    stitch settles that. It costs a few extra parts and removes the need to know.
    """
    def arclen(c):
        c = np.asarray(c, float)
        return float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))

    fr = set()
    for ch in (chain_a, chain_b):
        for f in _corner_fracs(ch, thr):
            fr.add(f)
            fr.add(1.0 - f)
    n = max(1, int(np.ceil(max(arclen(chain_a), arclen(chain_b)) / max_seg)))
    fr |= {k / n for k in range(1, n)}
    out = []
    for f in sorted(fr):
        if f < tol or f > 1.0 - tol:
            continue
        if out and f - out[-1] < tol:
            continue
        out.append(f)
    return out


def _weld_sense(int_a, int_b, tol=1.0):
    """Whether a seam must be welded start-to-END, from the two seams' 3D ENDS.

    `_stitch_matched` votes on this by summing over matched edge PAIRS, which
    needs the pairs to correspond; at the wrist they do not. The cuff's top is
    two edges either side of its own x=0 while the sleeve hem is one arc, and the
    front and back sleeve halves are mirrors, so their hems run opposite ways --
    the vote came out the same for both and welded the front cuffs and every bell
    tier back-to-front (ends: 6.2 cm the right way against 12.3 the way it
    chose). Returns None when the two options are within `tol`, i.e. when the
    ends cannot tell them apart, so the caller leaves the vote alone.
    """
    def ends(itf):
        n = len(list(itf.edges))
        return (np.asarray(itf.panel[0].point_to_3D(list(itf.edges[0].start))),
                np.asarray(itf.panel[n - 1].point_to_3D(
                    list(itf.edges[n - 1].end))))

    a_s, a_e = ends(int_a)
    b_s, b_e = ends(int_b)
    same = np.linalg.norm(a_s - b_s) + np.linalg.norm(a_e - b_e)
    flip = np.linalg.norm(a_s - b_e) + np.linalg.norm(a_e - b_s)
    if abs(same - flip) < tol:
        return None
    # `right_wrong=True` means swap=False, i.e. weld start-to-START -- the
    # convention `_auto_rw` documents and `boxmeshgen` reads
    # (`swap = not right_wrong`). Returning `flip < same` inverted every weld.
    return bool(same < flip)


def _anyseam(pan, key):
    """A seam whether or not `parts` renamed it: 'hem' or 'hem_0'..'hem_n'."""
    if key in pan.interfaces:
        return pan.interfaces[key]
    ks = sorted((k for k in pan.interfaces
                 if k.startswith(key + '_') and k.rsplit('_', 1)[-1].isdigit()),
                key=lambda k: int(k.rsplit('_', 1)[-1]))
    if not ks:
        raise KeyError(f'{pan.name}: no seam {key!r}')
    return pan.seam(*ks)


def _seamof(pan, key, count):
    """The parted seam `key_0..key_{count-1}` as one interface."""
    return pan.seam(*[f'{key}_{i}' for i in range(max(count, 1))])


def _waist_rules(g, side, tag, n, lead_first, lead_n):
    """The shirt dress's front waist as two rules, one per bodice chunk.

    The skirt's half-waist is one continuous run; the bodice's is two, with the
    waist dart between them. Split the skirt's run at the same place and stitch
    each stretch to its own chunk, so neither rule spans the dart.
    """
    keys = ('waist_b', 'waist_a') if lead_first else ('waist_a', 'waist_b')
    cnt = ((n['wb'], n['wa']) if lead_first else (n['wa'], n['wb']))
    total = n['fw']
    if lead_n >= total:
        raise ValueError(f'{g.KEY}: waist split {lead_n} of {total}')
    lo = g.fskirt.seam(*[f'waist_{side}_{k}' for k in range(lead_n)])
    hi = g.fskirt.seam(*[f'waist_{side}_{k}' for k in range(lead_n, total)])
    return (_stitch_matched(lo, _seamof(g.fronts[tag], keys[0], cnt[0])),
            _stitch_matched(hi, _seamof(g.fronts[tag], keys[1], cnt[1])))


def _bandseam(pan, half, count):
    """One half of a band's split top edge: half 0 is the first `count` parts."""
    return pan.seam(*[f'top_{half * count + i}' for i in range(count)])


class BonprixTieredDress(pyg.Component):
    """8642610003, outer layer only.

    Bodice front + back (each cut on the fold, built as two halves), sleeves,
    shirred waist band, and the two skirt tiers. Every seam in the skirt chain is
    length-matched and the cinching is carried by the panel shapes, because a
    gather cannot be expressed to the simulator (see `_taper_seams`):

        bodice waist  ->  band top          (matched, 49.7 per side)
        band bottom   ->  upper tier top    (matched, at the SHIRRED girth)
        upper tier    ->  lower tier top    (matched, 132.3)
        lower tier bottom                    (its own 188.1 -- the flare)

    That mirrors what the shirring does: the band is cinched to the body and the
    fabric above and below fans out from it. The file's own numbers confirm the
    tier structure -- the band is 132.1 cm flat and the upper tier 132.3, a 1:1
    join, and the lower tier is 188.1, a 1.42 flare onto it.

    Dropped: the two under-layer panels (80.8 and 65.1 cm wide, 66.7/65.3 tall --
    the same length as the two tiers together, which is what identifies them as a
    second layer) because hanging them off the same band edge as the upper tier
    needs a three-way seam the stitching model has no way to express; and the
    neck binding, after a binding strip measurably made the blouse worse.

    The sleeve cap is 54.9 cm against a 42.6 cm armhole. That 29% is not an
    error, it is a gathered sleeve head -- and it is left alone: the weld pulls
    both edges toward each other, so the cap compresses into gathers, which is
    what the garment does.
    """
    KEY = '8642610003'
    # Longest part a mating seam may be cut into on THIS garment. The global
    # 25 cm is fine for the jogger's near-straight seams but forcing one cubic
    # over a 20 cm curved bodice side seam missed by 58 mm.
    SEAM_SEG = 5.0
    # Band, tiers and cuff are RECTANGLES in the DXF -- all six are 4-point
    # outlines with a fold line -- and that is how they are built. They were
    # tapered so every seam in the chain came out length-matched and the
    # shirring could grip; that shape is invented, and the gathers it stands in
    # for are the garment's own. Same call as the jogger's `FAITHFUL_BANDS`.
    # `faithful_tiers: 0` gives the trapezoids back.
    FAITHFUL_TIERS = True
    # The front opening's two top corners are tied together by a small panel.
    # There is NO such piece in the DXF -- it is the exporter's addition -- so it
    # is built synthetic and declared as such. `STRAP_EASE` is the tie's length
    # as a fraction of the opening it spans: under 1.0 so it goes taut and holds
    # the corners in, 0.7-0.9 on the exporter's call.
    STRAP_EASE = 0.8
    STRAP_GRIP = 2.0         # cm of each front neckline the tie takes

    def __init__(self, body, design=None, size='38') -> None:
        super().__init__('bonprix_tiered')
        st = STYLES[self.KEY]
        by = {p.block.split('_')[0]: p for p in st.pieces(size)}

        def pick(prefix):
            hits = [v for k, v in by.items() if k.startswith(prefix)]
            if len(hits) != 1:
                raise ValueError(f'{self.KEY}: {prefix!r} matched {len(hits)}')
            return hits[0]

        # Role mapping given by the exporter. This DXF's ANNOTATION carries no
        # role words -- only numeric ids -- so there is nothing in the file to
        # derive these from. Corrections against the previous guesses: the upper
        # tiers were swapped front/back, the waistband was taken from x90113 /
        # x70113 (which are not used at all), the cuff from 140113 (which is the
        # front waistband), and the collar was missing entirely.
        f_bod, b_bod = pick('x30113'), pick('x10113')
        sleeve = pick('x20113')
        band_f, band_b = pick('140113'), pick('x40113')
        up_f, up_b = pick('100113'), pick('x50113')
        lo_f, lo_b = pick('120113'), pick('x60113')
        # The sleeve cuff is a 'Shape' block, which `read_pieces` classes as a
        # notion, so it never reaches `Style.pieces` -- same trap as the jogger's
        # waistband and rib.
        trims = {q.name: q for q in ad.pieces_for_size(
            st.dxf, st.size_label(size), fabric_only=False)
            if q.kind != 'fabric'}
        cuff = next((v for k, v in trims.items() if k.startswith('Shape 19')),
                    None)
        if cuff is None:
            raise ValueError(f'{self.KEY}: no Shape 19 (sleeve cuff); '
                             f'have {sorted(trims)}')

        hps = body['height'] - body['head_l']
        waist_y = hps - float(f_bod.size_cm()[1])

        # Built TWICE, as the blouse is: pass one measures the mating seams,
        # pass two rebuilds with each cut into the same number of equal parts, so
        # both sides of every stitch share an edge count and a fraction set and
        # `match_interfaces` has nothing to subdivide. Its subdivision is what
        # left 31 collapsed edges and 64 of 82 welds invalid here. Every seam on
        # this garment is one chain against one chain -- no yoke, no dart -- so
        # uniform splits suffice and no composite fractions are needed.
        def build(parts):
            bod, sl, cf = {}, {}, {}
            ct, cp = {}, {}
            for tag, mir in (('r', False), ('l', True)):
                bod[f'f_{tag}'] = self._bodice(
                    f'td_front_{tag}', f_bod, mir, waist_y, ZB_FRONT,
                    parts=parts.get('front'))
                bod[f'b_{tag}'] = self._bodice(
                    f'td_back_{tag}', b_bod, mir, waist_y, ZB_BACK,
                    parts=parts.get('back'))
            for tag in 'rl':
                side = -1 if tag == 'r' else +1
                sl[tag] = BonprixSleeve(
                    f'td_sleeve_{tag}', sleeve, body, side,
                    front_cap_len=self._cap_len, parts=parts.get('sleeve'))
                # 'Shape 19', the sleeve cuff: 19 cm round x 1 cm, no fold.
                # Used as drawn -- no doubling, no assumed fold.
                cf[tag], ct[tag], cp[tag] = _cuff_at_sleeve(
                    f'td_cuff_{tag}', sl[tag], float(cuff.size_cm()[1]),
                    float(cuff.size_cm()[0]), side)
            return bod, sl, cf, ct, cp

        def L(itf):
            return float(itf.edges.length())

        self._cap_len = None
        b0, s0, _, _, _ = build({})
        self._cap_len = L(b0['f_r'].interfaces['armhole'])
        seg = self.SEAM_SEG
        # Fractions, not counts: every corner of either chain must be a part
        # boundary or the single cubic per part cuts it (see `_matched_fracs`).
        def src(pan, key):
            return pan.source_seams[key]

        f_arm_f = _matched_fracs(src(b0['f_r'], 'armhole'),
                                 src(s0['r'].front, 'cap'), seg)
        f_arm_b = _matched_fracs(src(b0['b_r'], 'armhole'),
                                 src(s0['r'].back, 'cap'), seg)
        f_side = _matched_fracs(src(b0['f_r'], 'side'),
                                src(b0['b_r'], 'side'), seg)
        f_sh = _matched_fracs(src(b0['f_r'], 'shoulder'),
                              src(b0['b_r'], 'shoulder'), seg)
        f_ua = _matched_fracs(src(s0['r'].front, 'underarm'),
                              src(s0['r'].back, 'underarm'), seg)
        f_wf = _matched_fracs(src(b0['f_r'], 'waist'),
                              src(b0['f_r'], 'waist'), seg)
        f_wb = _matched_fracs(src(b0['b_r'], 'waist'),
                              src(b0['b_r'], 'waist'), seg)
        n_arm_f, n_arm_b = len(f_arm_f) + 1, len(f_arm_b) + 1
        n_side, n_sh, n_ua = len(f_side) + 1, len(f_sh) + 1, len(f_ua) + 1
        n_wf, n_wb = len(f_wf) + 1, len(f_wb) + 1
        # The tie grips the top of each front neckline, so that seam is cut
        # once at its reach and the REST -- the neck opening -- is cut only for
        # fidelity: by its own corners as well as by length, the way the waist
        # seams are. Uniform cuts alone left a cubic straddling the corner where
        # the V turns and missed by 11.7 mm at size 40, and that corner can fall
        # inside `_matched_fracs`' 0.03 guard. The guard keeps WELDED parts above
        # the mesher's cell; this stretch welds to nothing, so a short part there
        # costs a small triangle and nothing else.
        f_chain = src(b0['f_r'], 'neck')
        f_neck = _chain_len(f_chain)
        # The neckline is a wide boat neck that TURNS at a corner and runs almost
        # straight down to the centre front -- a 15 cm slit only ~5.8 cm wide at
        # its top. The tie belongs at that corner, holding the slit shut. The
        # corner is the neck edge's one interior corner at every size (fraction
        # 0.388 at 34 up to 0.406 at 48, the other one being cf_top itself), so
        # it is found rather than assumed. NOT the highest point: that is the
        # shoulder neck point, 9.2 cm out, and tying those spans the whole
        # opening instead of the slit.
        inner = [f for f in _corner_fracs(f_chain, 20.0) if 0.03 < f < 0.97]
        if len(inner) != 1:
            raise ValueError(f'{self.KEY}: front neckline has {len(inner)} '
                             f'interior corners ({inner}), expected 1')
        fc = inner[0]
        # The tie grips DOWNWARD from the corner, along the slit: that stretch is
        # near-vertical, so the tie's own vertical end edge lies along it.
        grip = min(self.STRAP_GRIP, 0.5 * (1.0 - fc) * f_neck)
        # Everything either side of the grip welds to nothing, so it is cut only
        # for fidelity -- by the neckline's own corners as well as by length, the
        # way the waist seams are. A cubic straddling the turn missed by 11.7 mm.
        cuts = {fc, fc + grip / f_neck}
        for lo, hi in ((0.0, fc), (fc + grip / f_neck, 1.0)):
            sub = _sub_chain(f_chain, lo, hi)
            cuts |= {lo + f * (hi - lo)
                     for f in _matched_fracs(sub, sub, seg)}
        cuts |= {f for f in _corner_fracs(f_chain, 20.0)}
        keep, last = [], 0.0
        for f in sorted(cuts):
            protected = abs(f - fc) < 1e-9 or abs(f - (fc + grip / f_neck)) < 1e-9
            if protected or ((f - last) * f_neck >= 0.4
                             and (1.0 - f) * f_neck >= 0.4):
                keep.append(f)
                last = f
        f_neck_frac = keep
        # The grip is the part that STARTS at `fc`: parts run [0,c0], [c0,c1] ...
        tie_part = f_neck_frac.index(fc) + 1
        parts = {
            'front': {'armhole': f_arm_f, 'side': f_side,
                      'shoulder': f_sh, 'waist': f_wf, 'neck': f_neck_frac},
            'back': {'armhole': f_arm_b, 'side': f_side,
                     'shoulder': f_sh, 'waist': f_wb},
            'sleeve': {'f': {'cap': f_arm_f, 'underarm': f_ua},
                       'b': {'cap': f_arm_b, 'underarm': f_ua}},
        }
        print(f'  Matched seams: armhole {n_arm_f}/{n_arm_b}, side {n_side}, '
              f'shoulder {n_sh}, underarm {n_ua}, waist {n_wf}/{n_wb}')
        (self.bod, self.sleeves, self.cuffs,
         self.cuff_top, self.cuff_partner) = build(parts)
        self._pn = dict(arm_f=n_arm_f, arm_b=n_arm_b, side=n_side, sh=n_sh,
                        ua=n_ua, wf=n_wf, wb=n_wb)

        # --- band + tiers -------------------------------------------------- #
        w_f = sum(L(_seamof(self.bod[f'f_{t}'], 'waist', n_wf)) for t in 'rl')
        w_b = sum(L(_seamof(self.bod[f'b_{t}'], 'waist', n_wb)) for t in 'rl')
        # The band's OWN length, from the DXF -- no invented ease. These two
        # pieces are folded on a long edge, so each one's length is twice its
        # stored height and its own width is the band depth: 35 + 33 = 68 cm of
        # ring at 6 cm deep. `ELASTIC_EASE` used to set this to 0.95 x the body
        # waist, which was a number I made up.
        cin_f = 2.0 * float(band_f.size_cm()[1])
        cin_b = 2.0 * float(band_b.size_cm()[1])
        cinch = cin_f + cin_b
        up_w_f, up_w_b = 2 * up_f.size_cm()[0], 2 * up_b.size_cm()[0]
        lo_w_f, lo_w_b = 2 * lo_f.size_cm()[0], 2 * lo_b.size_cm()[0]
        bh = float(band_f.size_cm()[0])   # fold is on the long edge
        uh = float(up_f.size_cm()[1])
        lh = float(lo_f.size_cm()[1])
        tiers = up_w_f + up_w_b, lo_w_f + lo_w_b
        if self.FAITHFUL_TIERS:
            print(f'  Skirt, as drawn: bodice waist {w_f + w_b:.1f} -> band '
                  f'{cinch:.1f} -> tier1 {tiers[0]:.1f} -> tier2 {tiers[1]:.1f}cm '
                  f'(body waist {body["waist"]:.1f}cm). Welded ratios: '
                  f'{(w_f + w_b) / cinch:.2f}x at the waist, '
                  f'{tiers[0] / cinch:.2f}x band->tier1, '
                  f'{tiers[1] / tiers[0]:.2f}x tier1->tier2 -- the garment\'s '
                  f'own gathers, which the sim can only take as stretch on the '
                  f'shorter edge.')
        else:
            print(f'  Shirred waist: bodice {w_f + w_b:.1f}cm -> band '
                  f'{cinch:.1f}cm on a {body["waist"]:.1f}cm waist; tiers '
                  f'{tiers[0]:.1f} -> {tiers[1]:.1f}cm')

        # Each panel is a rectangle at its own DXF length; `_strip` still cuts
        # the band's top into the bodice's part count so that weld pairs
        # edge-for-edge.
        def rect(width, matched):
            return width if self.FAITHFUL_TIERS else matched

        self.band = {
            'f': self._strip('td_band_f', cin_f, rect(cin_f, w_f), bh,
                             waist_y - bh, Z_FRONT,
                             top_parts=[1.0] * (2 * n_wf)),  # widths, not fractions
            'b': self._strip('td_band_b', cin_b, rect(cin_b, w_b), bh,
                             waist_y - bh, Z_BACK,
                             top_parts=[1.0] * (2 * n_wb))}
        self.upper = {
            'f': self._strip('td_tier1_f', up_w_f, rect(up_w_f, cin_f), uh,
                             waist_y - bh - uh, Z_FRONT),
            'b': self._strip('td_tier1_b', up_w_b, rect(up_w_b, cin_b), uh,
                             waist_y - bh - uh, Z_BACK)}
        self.lower = {
            'f': self._strip('td_tier2_f', lo_w_f, rect(lo_w_f, up_w_f), lh,
                             waist_y - bh - uh - lh, Z_FRONT),
            'b': self._strip('td_tier2_b', lo_w_b, rect(lo_w_b, up_w_b), lh,
                             waist_y - bh - uh - lh, Z_BACK)}

        # --- front tie ------------------------------------------------------
        # Both welds are IN the front plane. That is the whole point of it: the
        # collar spanned front and back, so it began 27 cm from all four
        # necklines in the exploded layout, and its travel dragged the bodice off
        # the shoulder (cloth self-intersections 143 -> 742 at size 38). Nothing
        # here has to travel.
        self._tie_slot = f'neck_{tie_part}'

        def corner_of(pan):
            # The grip runs from the corner DOWN the slit, so the corner is the
            # upper end of that part.
            e = pan.interfaces[self._tie_slot].edges
            a = np.asarray(pan.point_to_3D(list(e[0].start)))
            c = np.asarray(pan.point_to_3D(list(e[-1].end)))
            return a if a[1] > c[1] else c

        pk = {t: corner_of(self.bod[f'f_{t}']) for t in 'rl'}
        span = float(np.linalg.norm(pk['r'] - pk['l']))
        tie = self.STRAP_EASE * span
        self.tie = self._strip('td_tie', tie, tie, grip,
                               float(min(pk['r'][1], pk['l'][1])) - grip,
                               ZB_FRONT)
        print(f'  Front tie: {tie:.2f}cm across a {span:.2f}cm slit '
              f'({self.STRAP_EASE:.2f}x, so it goes taut), gripping '
              f'{grip:.2f}cm of slit below each front neck corner')

        # Which end takes which front is measured, not assumed.
        def mid(itf):
            return np.mean([_mid3(itf, k)
                            for k in range(len(list(itf.edges)))], axis=0)

        ends = ('side_r', 'side_l')
        d_same = sum(np.linalg.norm(mid(self.tie.interfaces[e])
                                    - mid(self.bod[f'f_{t}']
                                          .interfaces[self._tie_slot]))
                     for e, t in zip(ends, 'rl'))
        d_swap = sum(np.linalg.norm(mid(self.tie.interfaces[e])
                                    - mid(self.bod[f'f_{t}']
                                          .interfaces[self._tie_slot]))
                     for e, t in zip(ends, 'lr'))
        self.tie_pairs = list(zip(ends, 'rl' if d_same <= d_swap else 'lr'))

        self.subs = [*self.bod.values(), *self.sleeves.values(),
                     *self.cuffs.values(), *self.band.values(),
                     *self.upper.values(), *self.lower.values(), self.tie]
        ring = []
        for grp in (self.band, self.upper, self.lower):
            ring += [_stitch(grp['f'].interfaces['side_r'], grp['b'].interfaces['side_r']),
                     _stitch(grp['f'].interfaces['side_l'], grp['b'].interfaces['side_l'])]
        n = self._pn
        self.stitching_rules = pyg.Stitches(
            _stitch(self.bod['f_r'].interfaces['cf'], self.bod['f_l'].interfaces['cf']),
            _stitch(self.bod['b_r'].interfaces['cb'], self.bod['b_l'].interfaces['cb']),
            *ring,
            *[_stitch_matched(self.tie.interfaces[e],
                              self.bod[f'f_{t}'].interfaces[self._tie_slot])
              for e, t in self.tie_pairs],
            # band top -> the four bodice waists; band bottom -> tier 1; tier 1 -> tier 2
            # Band top -> the two bodice waists of its face. The top edge runs
            # +x to -x, so its FIRST n parts are the +x half (the mirrored bodice
            # half) and the last n the -x half.
            *[_stitch_matched(
                _bandseam(self.band['f'], k, n['wf']),
                _seamof(self.bod[f'f_{t}'], 'waist', n['wf']))
              for k, t in ((1, 'r'), (0, 'l'))],
            *[_stitch_matched(
                _bandseam(self.band['b'], k, n['wb']),
                _seamof(self.bod[f'b_{t}'], 'waist', n['wb']))
              for k, t in ((1, 'r'), (0, 'l'))],
            *[_stitch(self.band[k].seam('bottom_r', 'bottom_l'),
                      self.upper[k].seam('top_r', 'top_l')) for k in 'fb'],
            *[_stitch(self.upper[k].seam('bottom_r', 'bottom_l'),
                      self.lower[k].seam('top_r', 'top_l')) for k in 'fb'],
            *[x for tag in 'rl' for x in (
                _stitch_matched(_seamof(self.bod[f'f_{tag}'], 'side', n['side']),
                                _seamof(self.bod[f'b_{tag}'], 'side', n['side'])),
                _stitch_matched(_seamof(self.bod[f'f_{tag}'], 'shoulder', n['sh']),
                                _seamof(self.bod[f'b_{tag}'], 'shoulder', n['sh'])),
                # Sleeve closed as two halves round the arm -- see `BonprixSleeve`
                _stitch_aligned(self.sleeves[tag].front.interfaces['top'],
                                self.sleeves[tag].back.interfaces['top']),
                _stitch_matched(_seamof(self.sleeves[tag].front, 'underarm', n['ua']),
                                _seamof(self.sleeves[tag].back, 'underarm', n['ua'])),
                _stitch_matched(_seamof(self.sleeves[tag].front, 'cap', n['arm_f']),
                                _seamof(self.bod[f'f_{tag}'], 'armhole', n['arm_f'])),
                _stitch_matched(_seamof(self.sleeves[tag].back, 'cap', n['arm_b']),
                                _seamof(self.bod[f'b_{tag}'], 'armhole', n['arm_b'])),
                # Close the cuff into a ring. The tiers and the band get
                # their `side_r`/`side_l` sewn to their partner's; a one-panel
                # cuff has to be sewn to ITSELF, and it was left open -- the two
                # ends only met because both weld to the sleeve's own side-seam
                # corner, so the cuff hung open below that single vertex.
                _stitch_matched(self.cuff_top[tag], self.cuff_partner[tag]),
                _stitch(self.cuffs[tag].interfaces['side_r'],
                        self.cuffs[tag].interfaces['side_l']),
            )],
        )
        self.interfaces = {
            'bottom': pyg.Interface.from_multiple(
                *[self.lower[k].seam('bottom_r', 'bottom_l') for k in 'fb'])}
        for tag in 'rl':
            self.sleeves[tag].set_panel_label('arm')
            self.cuffs[tag].set_panel_label('arm')
        self.set_panel_label('body', overwrite=False)

    def _bodice(self, name, piece, mirror, waist_y, z, parts=None):
        """Bodice half on the fold.

        The outline's highest vertex IS the neck point (the neckline rises from
        the fold to it, and the shoulder then falls away to the tip at the
        armhole). Reading it the other way round -- highest vertex = shoulder
        tip, neck point = the next corner in -- swapped the two junctions, and
        because they sit one vertex apart the resulting order was not a walk
        order at all: 'shoulder' came out spanning 31 of the outline's 32
        vertices instead of 1.
        """
        loop, _, fold = _prep_piece(piece, False)
        fold_x = float(np.mean(np.asarray(fold, float)[:, 0]))
        if loop[:, 0].max() - fold_x > fold_x - loop[:, 0].min():
            loop, fold = loop * [-1.0, 1.0], fold * [-1.0, 1.0]
        if mirror:
            loop, fold = loop * [-1.0, 1.0], fold * [-1.0, 1.0]
        fold_x = float(np.mean(np.asarray(fold, float)[:, 0]))
        # Located by EDGE, not by corner count: this piece gives 5 corners at
        # size 38 and 7 at size 34 (the tight-threshold count is no more stable
        # here than it is on the jogger legs), so nothing depends on how many
        # come back.
        onfold = np.where(np.abs(loop[:, 0] - fold_x) < 0.3)[0]
        if len(onfold) < 2:
            raise ValueError(f'{name}: {len(onfold)} vertices on the fold')
        cf_bot = _pick(loop, onfold, 'y', 'min')
        cf_top = _pick(loop, onfold, 'y', 'max')
        far = 'left' if fold_x > 0 else 'right'
        waist_out = _pick(loop, _on_edge(loop, 'bottom', band=2.0), 'x',
                          'min' if fold_x > 0 else 'max')
        underarm = _pick(loop, _on_edge(loop, far, band=2.0), 'y', 'max')
        peak = int(np.argmax(loop[:, 1]))
        # The shoulder tip is the far end of the shoulder from the neck point:
        # of the corners in the upper half that are off the fold, the one
        # FARTHEST from it. The neck point is the peak itself.
        keys = corners(loop, thr=12.0, window=2.0)
        mid_y = loop[:, 1].min() + 0.5 * loop[:, 1].ptp()
        # The shoulder tip is the HIGHEST corner on the far side of the neck
        # point from the fold. Both halves of that matter. Taking the corner
        # FARTHEST from the fold picked the armhole's widest bulge on the front
        # -- y=20.0, barely above the underarm at 10.6, against the real
        # shoulder tip's 36.6 next to the peak's 39.1 -- and the panel's seams
        # came out shifted by 17 cm: 'shoulder' 23.68 cm against the back's 6.85
        # and 'armhole' 13.25 against 22.10, so the shoulder seam welded 3.5:1
        # and the sleeve cap 2:1 onto the front armhole. And restricting to the
        # far side of the peak is what keeps the NECKLINE's own corners out of
        # the running: on this front the neck turns at (10.33, 32.19), higher
        # than the shoulder tip but on the fold side of it.
        out = -1.0 if fold_x > loop[peak][0] else 1.0
        upper = [k for k in keys if loop[k][1] > mid_y
                 and abs(loop[k][0] - fold_x) > 0.5 and k != peak
                 and (loop[k][0] - loop[peak][0]) * out > 0]
        if not upper:
            raise ValueError(f'{name}: no shoulder-tip corner found')
        shoulder_tip = max(upper, key=lambda k: loop[k][1])
        junc = [cf_bot, waist_out, underarm, shoulder_tip, peak, cf_top]
        names = ['waist', 'side', 'armhole', 'shoulder', 'neck', 'cf']
        loop, junc = _orient(loop, junc)
        seams, source, loop, junc = _seams_of(loop, junc, names)
        key = 'cf' if z > 0 else 'cb'
        seams[key] = seams.pop('cf')
        source[key] = source.pop('cf')
        panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                         source=source, parts=parts,
                         min_seg={c: WELD_MIN_SEG for c in
                                  ('waist','side','armhole','shoulder')},
                         pivot=[fold_x, float(loop[:, 1].min())])
        panel.translation = np.array([0.0, waist_y, z], float)
        return face_to(panel, [0.0, 0.0, float(np.sign(z))])

    @staticmethod
    def _strip(name, bottom, top, height, y, z, top_parts=None):
        panel = DxfPanel(name, _taper_seams(bottom, top, height,
                                            top_parts=top_parts),
                         verbatim=True, translation=[0.0, y, z])
        # Rectangles, as the DXF draws them (see `FAITHFUL_TIERS`). Still
        # declared synthetic rather than silently self-validated: the outline is
        # rebuilt from the piece's dimensions, not read from it.
        # See `BonprixJogger._band`.
        panel.synthetic = True
        return face_to(panel, [0.0, 0.0, float(np.sign(z))])


def _sleeve_by_role(name, piece, body, side, hem_lift=2.0, cap_fracs=None):
    """Sleeve whose junctions are found by ROLE, not by corner count.

    This piece yields only three corners at any usable threshold, so the four
    junctions are located instead: the two ends of the bottom edge, and the two
    extreme-x vertices above it (the underarms) -- the same rule
    `HyperdropSleeve` uses. Pivoted at the cap apex so the pair places
    symmetrically (see `_flat_sleeve`).
    """
    loop, _ = _normalised(piece)
    if side > 0:
        loop = loop * [-1.0, 1.0]
    hem = _on_edge(loop, 'bottom', band=1.5)
    h_a, h_b = _pick(loop, hem, 'x', 'min'), _pick(loop, hem, 'x', 'max')
    above = np.where(loop[:, 1] > loop[:, 1].min() + 2.0)[0]
    ua_a, ua_b = _pick(loop, above, 'x', 'min'), _pick(loop, above, 'x', 'max')
    junc = [h_a, h_b, ua_b, ua_a]
    names = ['hem', 'side_r', 'cap', 'side_l']
    loop, junc = _orient(loop, junc)
    seams, source, loop, junc = _seams_of(loop, junc, names)
    if cap_fracs:
        # Cut the cap into one sub-seam per armhole segment it meets, so each
        # pairs 1:1 instead of welding one long cap against three panels at
        # once. Matching a 45-edge cap to the 28 edges of three armholes made
        # `StitchingRule.match_interfaces` emit stitch segments short enough to
        # build an impossible rest triangle. The cap and the armholes agree to
        # 1% in total length, so the proportional split is what the pattern's
        # own cap notches would mark.
        # Rebuilt IN ORDER: DxfPanel assembles the edge loop in dict order, so
        # popping 'cap' and appending the parts puts them after 'side_l' and the
        # outline stops tiling in sequence.
        def rebuild(d, chain_key, split):
            parts = (_split_at_fracs(d[chain_key], cap_fracs) if split
                     else [d[chain_key]] * (len(cap_fracs) + 1))
            out = {}
            for k, v in d.items():
                if k != chain_key:
                    out[k] = v
                else:
                    for i, part in enumerate(parts):
                        out[f'cap_{i}'] = part
            return out
        seams = rebuild(seams, 'cap', True)
        # Each cap PART is judged against the WHOLE cap chain. Splitting the
        # source at the same fractions looks right but snaps to its own vertices,
        # which sit elsewhere once the panel chain is simplified -- that put
        # cap_2 43.9 mm out while the geometry was fine. Every part lies on the
        # whole chain, and the gate measures distance to the polyline.
        source = rebuild(source, 'cap', False)
        # ... so the LENGTH comparison has to be aggregated instead: each part is
        # a third of the chain it is measured against, which read as a 371 mm
        # length error while the three parts summed to 44.94 cm against a 44.90
        # cm cap. `check_fidelity` sums a group and compares once.
        cap_group = {f'cap_{i}': 'cap' for i in range(len(cap_fracs) + 1)}
    apex = loop[int(np.argmax(loop[:, 1]))]
    panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                     source=source, pivot=apex)
    panel.source_groups = cap_group if cap_fracs else {}
    panel.n_cap = len(cap_fracs) + 1 if cap_fracs else 1
    sgn = 1.0 if side > 0 else -1.0
    panel.translate_to([sgn * float(body['shoulder_w']) / 2,
                        float(body['height']) - float(body['head_l']) - hem_lift,
                        0.0])
    panel.rotate_by(R.from_euler(
        'XYZ', [0, 0, sgn * float(body['arm_pose_angle'])], degrees=True))
    return face_to(panel, [sgn, 0.0, 0.0])

class BonprixMidiDress(pyg.Component):
    """8242610411 -- midi shirt dress: a front, a back and two sleeves.

    The exporter's mapping, and nothing beyond it: x40113 is the front, x30113
    the sleeves, and the back is ONE of x50113 / x60113. x50113 is used, chosen
    on the geometry rather than by preference -- its side seam is 61.82 cm
    against the front's 56.30 + 5.44 either side of the bust dart (61.74, a
    0.08 cm match), while x60113's is 81.85 and would hang the back 20 cm below
    the front. Shoulder (4.45 against 4.20) and armhole (22.04) match both, so
    the side seam is the only discriminator.

    Not used: x70113 (skirt + integrated sash), x20113 (the tapered strip) and
    x10113 (the darted panel). The reference photo also shows a stand collar and
    a button placket that this export has no pieces for.

    Every junction is found by WALKING the outline from the centre seam, not by
    corner index: the sequence round each piece is fixed (hem, side, dart,
    armhole, shoulder, neck) even though the indices move with the grade. The
    walk direction is the one that does not immediately reach the other end of
    the centre seam.
    """
    KEY = '8242610411'
    BACK_PIECE = 'x50113'
    # The centre front is NOT one seam. Measured from the neck: the top 1.5 cm
    # is joined, the next 11 cm is left OPEN, and the rest joins again -- the
    # keyhole the reference photo shows skin through. `cf` is therefore cut into
    # three parts and only parts 0 and 2 are stitched.
    CF_TOP_JOIN = 1.5
    CF_GAP = 11.0
    SEAM_SEG = 5.0
    # As the shirt dress: these curves bend more gently than the 20 deg default
    # sees, and a part spanning two bends misses by centimetres.
    SEAM_CORNER_THR = 10.0

    def __init__(self, body, design=None, size='38') -> None:
        super().__init__('bonprix_midi')
        st = STYLES[self.KEY]
        by = {p.block.split('_')[0]: p for p in st.pieces(size)}

        def pick(prefix):
            hits = [v for k, v in by.items() if k.startswith(prefix)]
            if len(hits) != 1:
                raise ValueError(f'{self.KEY}: {prefix!r} matched {len(hits)}')
            return hits[0]

        front = pick('x40113')
        back = pick(self.BACK_PIECE)
        sleeve = pick('x30113')
        hps = float(body['height']) - float(body['head_l'])
        seg = self.SEAM_SEG

        def build(parts):
            fr, bk, sl = {}, {}, {}
            for tag, mir in (('r', False), ('l', True)):
                fr[tag] = self._half(
                    f'md_front_{tag}', front, mir, hps, ZB_FRONT,
                    ['hem', 'side_lo', 'dart_a', 'dart_b', 'side_up',
                     'armhole', 'shoulder', 'neck'], 7,
                    parts=parts.get('front'))
                bk[tag] = self._half(
                    f'md_back_{tag}', back, not mir, hps, ZB_BACK,
                    ['hem', 'side', 'armhole', 'shoulder', 'neck'], 4,
                    parts=(parts.get('back') or {}).get(tag))
            for tag in 'rl':
                # The sleeve's side is MEASURED off the armhole it takes, not
                # taken from the tag. Which x a bodice half lands on depends on
                # this style's own winding and its mirror flag: here 'r' comes
                # out at +15.1 where the shirt dress's lands at -19.0, so
                # `side=-1 if tag == 'r'` -- correct there -- welded every sleeve
                # straight across the body. Visible from above in the boxmesh as
                # a seam running from one front panel to the opposite sleeve.
                ax = float(np.mean([_mid3(_anyseam(fr[tag], 'armhole'), k)[0]
                                    for k in range(len(list(
                                        _anyseam(fr[tag], 'armhole').edges)))]))
                sl[tag] = BonprixSleeve(
                    f'md_sleeve_{tag}', sleeve, body,
                    side=-1 if ax < 0 else +1,
                    front_cap_len=self._cap_len,
                    parts=parts.get('sleeve'))
            return fr, bk, sl

        def C(pan, key):
            return np.asarray(pan.source_seams[key], float)

        def pair(a, b):
            # `tol` scaled so no part comes out under the mesher's cell. Left at
            # the default 0.03 this produced armhole parts of 0.97-1.34 cm, and
            # the single cubic forced over one of them overshot badly enough
            # that the fitted seam measured 35.53 cm against its own 21.41 cm
            # source -- which then reached `match_interfaces` as a negative
            # split fraction (size 40 only).
            lo = min(_chain_len(a), _chain_len(b))
            return _matched_fracs(a, b, seg, thr=self.SEAM_CORNER_THR,
                                  tol=max(0.03, WELD_MIN_SEG / max(lo, 1e-6)))

        self._cap_len = None
        f0, b0, _ = build({})
        self._cap_len = float(_anyseam(f0['r'], 'armhole').edges.length())
        f0, b0, s0 = build({})

        # The front's side seam is split by the bust dart, the back's is one run
        # -- the shirt dress's composite case, handled the same way.
        sd_lead = {t: _runs_into(b0[t].interfaces['side'],
                                 f0[t].interfaces['side_up']) for t in 'rl'}

        def _split(chunks, lead_first):
            la, lb = _chain_len(chunks[0]), _chain_len(chunks[1])
            return (la if lead_first else lb) / (la + lb)

        s_chunks = [C(f0['r'], 'side_up'), C(f0['r'], 'side_lo')]
        f = _split(s_chunks, sd_lead['r'])
        lead, trail = ((s_chunks[0], s_chunks[1]) if sd_lead['r']
                       else (s_chunks[1], s_chunks[0]))
        single = C(b0['r'], 'side')
        fr_lead = pair(_sub_chain(single, 0.0, f), lead)
        fr_trail = pair(_sub_chain(single, f, 1.0), trail)
        fr_su, fr_sl = ((fr_lead, fr_trail) if sd_lead['r']
                        else (fr_trail, fr_lead))
        side_one = {}
        for t in 'rl':
            ft = _split(s_chunks, sd_lead[t])
            lf, tf = ((fr_su, fr_sl) if sd_lead[t] else (fr_sl, fr_su))
            cuts = ([x * ft for x in lf] + [ft]
                    + [ft + x * (1.0 - ft) for x in tf])
            # Sorted and de-duplicated. These are cumulative POSITIONS along the
            # seam, and a non-monotonic list reaches `_subdivide` as widths that
            # do not sum to 1 (size 40 raised with a sum of 12.29).
            keep, last = [], 0.0
            for x in sorted(cuts):
                if 1e-4 < x < 1.0 - 1e-4 and x - last > 1e-4:
                    keep.append(x)
                    last = x
            side_one[t] = keep

        fr_sh = pair(C(f0['r'], 'shoulder'), C(b0['r'], 'shoulder'))
        fr_arm_f = pair(C(s0['r'].front, 'cap'), C(f0['r'], 'armhole'))
        fr_arm_b = pair(C(s0['r'].back, 'cap'), C(b0['r'], 'armhole'))
        fr_ua = pair(C(s0['r'].front, 'underarm'), C(s0['r'].back, 'underarm'))

        # `cf` runs from the NECK end (`_half` names it c_top -> c_bot), so the
        # keyhole fractions are measured from 0.
        cf_len = _chain_len(C(f0['r'], 'cf'))
        if self.CF_TOP_JOIN + self.CF_GAP >= cf_len:
            raise ValueError(f'{self.KEY}: keyhole {self.CF_TOP_JOIN}+'
                             f'{self.CF_GAP} does not fit a {cf_len:.1f} cm CF')
        fr_cf = [self.CF_TOP_JOIN / cf_len,
                 (self.CF_TOP_JOIN + self.CF_GAP) / cf_len]
        print('  Front keyhole: CF %.2fcm -- joined %.1f / open %.1f / joined '
              '%.2f' % (cf_len, self.CF_TOP_JOIN, self.CF_GAP,
                        cf_len - self.CF_TOP_JOIN - self.CF_GAP))
        parts = {
            'front': {'shoulder': fr_sh, 'armhole': fr_arm_f,
                      'side_up': fr_su, 'side_lo': fr_sl, 'cf': fr_cf},
            'back': {t: {'shoulder': fr_sh, 'armhole': fr_arm_b,
                         'side': side_one[t]} for t in 'rl'},
            'sleeve': {'f': {'cap': fr_arm_f, 'underarm': fr_ua},
                       'b': {'cap': fr_arm_b, 'underarm': fr_ua}},
        }
        self.fronts, self.backs, self.sleeves = build(parts)

        def nc(pan, key):
            return max(1, len([x for x in pan.interfaces
                               if x == key or (x.startswith(key + '_')
                                               and x.rsplit('_', 1)[-1]
                                               .isdigit())]))

        n = {'sh': nc(self.fronts['r'], 'shoulder'),
             'arm_f': nc(self.fronts['r'], 'armhole'),
             'arm_b': nc(self.backs['r'], 'armhole'),
             'su': nc(self.fronts['r'], 'side_up'),
             'sl': nc(self.fronts['r'], 'side_lo'),
             'bs': nc(self.backs['r'], 'side'),
             'ua': nc(self.sleeves['r'].front, 'underarm')}
        if n['su'] + n['sl'] != n['bs']:
            raise ValueError(f"{self.KEY}: side seam {n['su']}+{n['sl']} "
                             f"against the back's {n['bs']}")
        print('  Matched seams: shoulder %d, armhole %d/%d, side %d+%d, '
              'underarm %d' % (n['sh'], n['arm_f'], n['arm_b'], n['su'],
                               n['sl'], n['ua']))

        self.subs = [*self.fronts.values(), *self.backs.values(),
                     *self.sleeves.values()]

        def in_order(pan, keys, counts, lead_first):
            pr = list(zip(keys, counts))
            if not lead_first:
                pr = pr[::-1]
            return pyg.Interface.from_multiple(
                *[_seamof(pan, k, c) for k, c in pr])

        self.stitching_rules = pyg.Stitches(
            # cf_1 -- the 11 cm keyhole -- is deliberately NOT stitched.
            *[_stitch(self.fronts['r'].interfaces[f'cf_{k}'],
                      self.fronts['l'].interfaces[f'cf_{k}'])
              for k in (0, 2)],
            _stitch(self.backs['r'].interfaces['cb'],
                    self.backs['l'].interfaces['cb']),
            *[x for tag in 'rl' for x in (
                # the bust dart, folded out
                _stitch(self.fronts[tag].interfaces['dart_a'],
                        self.fronts[tag].interfaces['dart_b']),
                _stitch_matched(_seamof(self.fronts[tag], 'shoulder', n['sh']),
                                _seamof(self.backs[tag], 'shoulder', n['sh'])),
                # ONE rule for the side seam: the front's is broken in two by
                # the dart while the back's is a single run.
                _stitch_matched(
                    in_order(self.fronts[tag], ('side_up', 'side_lo'),
                             (n['su'], n['sl']), sd_lead[tag]),
                    _seamof(self.backs[tag], 'side', n['bs'])),
                _stitch(self.sleeves[tag].front.interfaces['top'],
                        self.sleeves[tag].back.interfaces['top']),
                # The sleeve's own hem dart, folded out. `BonprixSleeve` splits
                # it off the hem when the outline doubles back; only this style's
                # sleeve has one, and it sits on whichever half the needle falls
                # in, so both halves are asked.
                *[_stitch(pan.interfaces['dart_a'], pan.interfaces['dart_b'])
                  for pan in (self.sleeves[tag].front, self.sleeves[tag].back)
                  if 'dart_a' in pan.interfaces],
                _stitch_matched(
                    _seamof(self.sleeves[tag].front, 'underarm', n['ua']),
                    _seamof(self.sleeves[tag].back, 'underarm', n['ua'])),
                _stitch_matched(
                    _seamof(self.sleeves[tag].front, 'cap', n['arm_f']),
                    _seamof(self.fronts[tag], 'armhole', n['arm_f'])),
                _stitch_matched(
                    _seamof(self.sleeves[tag].back, 'cap', n['arm_b']),
                    _seamof(self.backs[tag], 'armhole', n['arm_b'])),
            )],
        )
        self.interfaces = {
            'bottom': pyg.Interface.from_multiple(
                *[self.fronts[t].interfaces['hem'] for t in 'rl'],
                *[self.backs[t].interfaces['hem'] for t in 'rl']),
        }
        for tag in 'rl':
            self.sleeves[tag].set_panel_label('arm')
        self.set_panel_label('body', overwrite=False)

    def _half(self, name, piece, mirror, hps, z, names, n_mid, parts=None):
        """One body half, junctions found by walking out from the centre seam.

        The centre seam is the long straight run at one extreme x -- identified
        by which extreme carries the greater y span, so it does not matter which
        way the piece was drawn or whether it has been mirrored. Walking from its
        bottom end in the direction that does NOT immediately reach its top end
        passes the remaining junctions in a fixed order, so `names` describes the
        piece once and holds across the grade.
        """
        loop, _, _ = _prep_piece(piece, mirror)
        best = None
        for x in (loop[:, 0].min(), loop[:, 0].max()):
            on = np.where(np.abs(loop[:, 0] - x) < 0.4)[0]
            span = loop[on, 1].ptp() if len(on) > 1 else -1.0
            if best is None or span > best[0]:
                best = (span, x, on)
        _, c_x, on = best
        if len(on) < 2:
            raise ValueError(f'{name}: no centre seam found')
        c_bot = _pick(loop, on, 'y', 'min')
        c_top = _pick(loop, on, 'y', 'max')
        n = len(loop)
        fwd = [(c_bot + i) % n for i in range(1, n)]
        # The corner COUNT is not stable across the grade -- this back reports
        # four at 36/38 and five at 32/34 -- so the THRESHOLD is swept until
        # exactly `n_mid` fall between the centre seam's two ends. Dropping the
        # surplus by turn angle instead was tried and is wrong: at size 40 it
        # discarded the boundary of the 5.4 cm `side_up`, whose corner is weak,
        # and the armhole absorbed it (35.53 cm against a 22.52 cm sleeve cap).
        #
        # Every vertex ON the centre seam is excluded, not just its two ends:
        # that seam is a long straight run carrying intermediate points, so its
        # y-extreme need not be the corner there.
        seg = np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0),
                             axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        ang = np.abs(ad.turn_angles(loop)[0])
        order = fwd if fwd.index(c_top) > n // 2 else fwd[::-1]

        def merged(cand, min_gap=3.0):
            """One junction reported twice is one junction. Corners closer than
            `min_gap` cm along the outline collapse to the sharper of them --
            size 32/34's back reports its five that way, two of them 2.07 cm
            apart."""
            out = []
            for k in cand:
                if out and abs(cum[k] - cum[out[-1]]) < min_gap:
                    if ang[k] > ang[out[-1]]:
                        out[-1] = k
                    continue
                out.append(k)
            return out

        mid, used = None, None
        for thr in (25.0, 20.0, 30.0, 17.0, 35.0, 14.0, 40.0, 11.0, 45.0, 8.0):
            keys = set(corners(loop, thr=thr, window=2.0)) - set(on.tolist())
            cand = [k for k in order if k in keys]
            cand = cand[:cand.index(c_top)] if c_top in cand else cand
            cand = merged(cand)
            if len(cand) == n_mid:
                mid, used = cand, thr
                break
            if mid is None or abs(len(cand) - n_mid) < abs(len(mid) - n_mid):
                mid, used = cand, thr
        if len(mid) != n_mid:
            raise ValueError(f'{name}: {len(mid)} junctions between the centre '
                             f'seam ends at every threshold tried, expected '
                             f'{n_mid} ({mid})')
        junc = [c_bot] + mid + [c_top]
        loop, junc = _orient(loop, junc)
        seams, source, loop, junc = _seams_of(loop, junc, list(names) + [
            'cf' if z > 0 else 'cb'])
        panel = DxfPanel(name, seams, verbatim=True, presimplified=True,
                         source=source, parts=parts,
                         min_seg={k: WELD_MIN_SEG for k in
                                  ('shoulder', 'armhole', 'side', 'side_up',
                                   'side_lo', 'hem')},
                         pivot=[c_x, float(loop[:, 1].max())])
        return _place_flat(panel, loop, c_x, hps, z)


BUILDERS = {'jogger': BonprixJogger, 'blouse': BonprixBlouse,
            'shirtdress': BonprixShirtDress, 'tiered': BonprixTieredDress,
            'midi': BonprixMidiDress}
STYLE_BUILDER = {'7492610006': 'jogger', '6812610700': 'blouse',
                 '8672609700': 'shirtdress',
                 '8642610003': 'tiered',
                 '8242610411': 'midi'}


# --------------------------------------------------------------------------- #
#  Pattern generation
#
#  Pattern building only -- simulation is run_garment.py's job. It calls this for
#  any config with `pattern_source: dxf` and `pattern_module: bonprix`, then
#  drives the sim from the config's sim_props like every other garment here.
# --------------------------------------------------------------------------- #
# How far a panel may sit from its DXF outline before this refuses to emit it.
# 10 mm is a REGRESSION GUARD, not a target: it is set just under the 14.7 mm
# that unsplit cubic fitting produced, so the failure this exists for -- drafted
# straight edges coming out curved -- cannot come back unnoticed. Chasing a
# tighter number here was a mistake: sub-mm settings leave 25 to 49 edges on a
# seam, and the welds then degrade, so the garment gets worse while the outline
# gets tidier. Actual numbers today: jogger 1.45 mm, blouse 7.0 mm. Enforced,
# not assumed: `generate_pattern` refuses to emit a pattern whose panels drift
# further than this from the source outlines. It exists because the drift was
# real and shipped twice unnoticed -- cubic seam fitting turned drafted straight
# edges into curves and missed by up to 14.7 mm, and nothing in the pipeline
# complained. A silent geometry error is the one kind this pipeline cannot
# tolerate, so it is a hard failure rather than a warning.
FIT_TOL_MM = 10.0


def check_fidelity(garment, tol_mm=None, verbose=True):
    """Largest deviation of any panel edge from its DXF source, in mm.

    Raises if it exceeds `tol_mm`. Must be called BEFORE `assembly()`, which
    folds panel translations into the vertices and makes every seam read as
    hundreds of mm out.
    """
    tol = FIT_TOL_MM if tol_mm is None else tol_mm

    def panels(comp):
        for sub in getattr(comp, 'subs', []) or []:
            if hasattr(sub, 'edges'):
                yield sub
            else:
                yield from panels(sub)

    rows, synthetic = [], []
    for panel in panels(garment):
        if getattr(panel, 'synthetic', False):
            # Not from the DXF, so there is nothing to check it against. Listed
            # rather than skipped: a panel with no source silently validating
            # itself is how the jogger's tapered waistband passed a check it was
            # never actually subject to.
            synthetic.append(panel.name)
            continue
        # Seams sharing one source chain (a split sleeve cap) are length-checked
        # as a GROUP: each part covers a fraction of the chain it is measured
        # against, so per-part length errors are meaningless. Deviation stays
        # per-seam -- distance to the shared polyline is exactly right.
        groups = getattr(panel, 'source_groups', {}) or {}
        acc = {}
        for name, src, fit, dev in ad.verify_panel(panel):
            grp = groups.get(name)
            if grp is None:
                rows.append((dev * 10.0, abs(fit - src) * 10.0,
                             f'{panel.name}/{name}'))
                continue
            tot, _ = acc.get(grp, (0.0, src))
            acc[grp] = (tot + fit, src)
            rows.append((dev * 10.0, 0.0, f'{panel.name}/{name}'))
        for grp, (fit, src) in acc.items():
            rows.append((0.0, abs(fit - src) * 10.0,
                         f'{panel.name}/{grp} (sum of parts)'))
    if not rows:
        return 0.0
    rows.sort(reverse=True)
    worst, worst_len, where = rows[0]
    if verbose:
        wl, _, wl_where = max(rows, key=lambda r: r[1])[0], None, max(
            rows, key=lambda r: r[1])[2]
        print(f'  DXF fidelity: worst deviation {worst:.2f} mm ({where}), '
              f'worst length error {max(r[1] for r in rows):.2f} mm ({wl_where})')
        if synthetic:
            print(f'  NOT from the DXF ({len(synthetic)} panels, unchecked): '
                  f'{", ".join(synthetic)}')
    if worst > tol:
        detail = '; '.join(f'{w}: {d:.2f}mm' for d, _, w in rows[:5])
        raise ValueError(
            f'panels differ from the DXF by up to {worst:.2f} mm '
            f'(limit {tol:.2f} mm) -- {detail}')
    return worst


def generate_pattern(size, config, body_yaml_path, output_base,
                     garment_prefix='bonprix', labels=False):
    """Build one bonprix garment from its DXF and serialize the pattern.

    `size` is the requested EU size; the builder resolves it onto the style's
    own size run (see `resolve_sizes`). `config['bonprix']['style']` picks the
    builder. Returns (folder, garment_name), matching the other generate_pattern
    entry points run_garment.py uses.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    from assets.bodies.body_params import BodyParameters

    style = config.get('bonprix', {}).get('style')
    if style not in BUILDERS:
        raise ValueError(f'bonprix.style must be one of {sorted(BUILDERS)}, '
                         f'got {style!r}')
    builder = BUILDERS[style]

    body = BodyParameters(body_yaml_path)
    # Optional per-run overrides of a builder's constants, so a search can vary
    # them from the config without editing this file. Only keys the builder
    # already defines as class attributes are accepted.
    for key, val in (config.get('bonprix', {}) or {}).items():
        attr = key.upper()
        if key != 'style' and hasattr(builder, attr):
            setattr(builder, attr, float(val))

    garment = builder(body, size=str(size))
    garment_name = f'{garment_prefix}_size{size}'
    garment.name = garment_name

    # Gate BEFORE assembly(), which folds translations into the vertices.
    check_fidelity(garment)

    pattern = garment.assembly()
    if garment.is_self_intersecting():
        print(f'  WARNING: {garment_name} has self-intersecting panels')
    folder = Path(pattern.serialize(
        output_base, tag='_' + datetime.now().strftime('%y%m%d-%H-%M-%S'),
        to_subfolder=True, with_3d=False, with_text=labels,
        view_ids=labels, with_printable=True))

    # Lower-body garments are built with the crotch at Y=0 (body-agnostic), so
    # stamp the marker simulate_pattern keys off: it then runs the body crotch
    # lift and pose-X leg alignment against the actual sim body. Upper-body
    # garments are placed at build time and must NOT carry the marker.
    if STYLES[getattr(builder, 'KEY', '')].kind == 'bottom':
        spec_path = next(folder.glob('*_specification.json'))
        with open(spec_path) as f:
            spec = json.load(f)
        spec.setdefault('properties', {})['placement'] = {
            'anchor': 'crotch_at_zero',
            'back_rise_lift': 0.0,   # the DXF's own back rise; no extra scoop
            'extra_x_sep': 0.1,      # small gap so the two crotch tips do not
                                     # start coincident
            # Aim the legs at the ANKLE, not the foot. The default lower band
            # (y=0..20) includes the splayed feet: on this body the left leg
            # centre reads +23.39 there against +17.4 at the ankle, so the leg
            # was aimed 6 cm outboard and the rib finished 4.5 cm beside the
            # ankle instead of wrapping it.
            'bot_band_floor': 8.0,
            'bot_band_height': 12.0,
        }
        with open(spec_path, 'w') as f:
            json.dump(spec, f, indent=2)

    print(f'  Pattern generated: {garment_name} -> {folder}')
    return folder, garment_name


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', nargs='+', default=['36', '38', '40'])
    ap.add_argument('--preview', metavar='DIR',
                    help='render every graded piece to DIR')
    a = ap.parse_args()
    _report(a.sizes)
    if a.preview:
        _preview(a.sizes, a.preview)
