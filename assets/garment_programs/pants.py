from copy import deepcopy
import numpy as np

import pygarment as pyg
from assets.garment_programs.base_classes import BaseBottoms
from assets.garment_programs import bands

# Set by run_garment.py from garment config `symknee` field. Default False keeps
# baseline linear knee taper; True switches to centered-knee placement.
SYMKNEE = False


class PantPanel(pyg.Panel):
    def __init__(
            self, name, body, design,
            length,
            waist,
            hips,
            hips_depth,
            crotch_width,
            dart_position,
            match_top_int_to=None,
            hipline_ext=1,
            double_dart=False,
            thigh_width=None,
            knee_width=None,
            knee_y=None,
            thigh_y=None,
            front_slit=None,
            barrel_bow=None,
            clo_match=False,
            back_rise_lift=0.0) -> None:
        """
            Basic pant panel with option to be fitted (with darts)

            front_slit: if set (cm), cut a thin V-notch this tall into the
                ankle (bottom) edge at its midpoint (center-front). The cut
                edges are left unstitched, opening into a front split hem.

            barrel_bow: if set (fraction of leg length, e.g. 0.06), bow the
                outseam OUTWARD between the pinned measurement points (ankle,
                knee, thigh) without moving those points. Because the bow lives
                strictly between samples, every girth POM is preserved exactly;
                the extra outward excursion is the barrel silhouette. Only
                active in subcurve mode (thigh/knee controls present). Default
                None leaves the outseam untouched — no effect on other configs.
        """
        super().__init__(name)

        flare = body['leg_circ'] * (design['flare']['v']  - 1) / 4
        hips_depth = hips_depth * hipline_ext

        hip_side_incl = np.deg2rad(body['_hip_inclination'])
        dart_depth = hips_depth * 0.8

        # Crotch cotrols
        crotch_depth_diff =  body['crotch_hip_diff']
        crotch_extention = crotch_width

        # eval pants shape
        # TODO Return ruffle opportunity?

        # amount of extra fabric at waist
        w_diff = hips - waist   # Assume its positive since waist is smaller then hips
        # We distribute w_diff among the side angle and a dart
        hw_shift = np.tan(hip_side_incl) * hips_depth
        # Small difference
        if hw_shift > w_diff:
            hw_shift = w_diff

        # Check if we should use thigh/knee subcurves
        use_subcurves = (thigh_width is not None and knee_width is not None
                         and knee_y is not None and thigh_y is not None)

        # --- Edges definition ---
        # Defaults so the non-subcurve / non-barrel paths don't reference these
        # before assignment (both are only set inside the subcurve branch).
        inner_dx = None
        outseam_led_knee_x = None
        if use_subcurves:
            # --- Outseam with 3 subcurves: ankle→knee, knee→thigh, thigh→hip ---
            ankle_pt = [-flare, 0]
            hip_pt = [0, length]

            # Thigh outseam: set so panel width at crotch_y = thigh_width.
            # At crotch_y the inseam is at hips+crotch_ext (crotch endpoint),
            # and outseam ≈ thigh_out_x (only 2cm above on the Bezier).
            thigh_out_x = (hips + crotch_extention) - thigh_width
            # Knee outseam X: baseline (linear taper) by default; the SYMKNEE
            # module flag (set from garment config `symknee` field) switches to
            # centered-knee placement (knee centered between ankle midpoint and
            # thigh midpoint).
            if SYMKNEE:
                _y_ankle = min(0, length - crotch_depth_diff * 1.5)
                _ankle_in_x = (hips + crotch_extention) - 2 + flare
                _ankle_center = (-flare + _ankle_in_x) / 2
                _thigh_center = thigh_out_x + thigh_width / 2
                _t = (knee_y - _y_ankle) / (thigh_y - _y_ankle)
                _knee_center = _ankle_center + _t * (_thigh_center - _ankle_center)
                knee_out_x = _knee_center - knee_width / 2
            else:
                knee_out_x = -flare * (1 - knee_y / length)

            knee_pt_out = [knee_out_x, knee_y]
            thigh_pt_out = [thigh_out_x, thigh_y]

            # Tangent at junctions: catmull-rom style (direction from prev to next point)
            knee_out_tan = np.array([thigh_pt_out[0] - ankle_pt[0],
                                     thigh_pt_out[1] - ankle_pt[1]])
            knee_out_tan = knee_out_tan / np.linalg.norm(knee_out_tan)

            thigh_out_tan = np.array([hip_pt[0] - knee_pt_out[0],
                                      hip_pt[1] - knee_pt_out[1]])
            thigh_out_tan = thigh_out_tan / np.linalg.norm(thigh_out_tan)

            right_ankle_to_knee = pyg.CurveEdgeFactory.curve_from_tangents(
                ankle_pt, knee_pt_out,
                target_tan1=knee_out_tan,
                initial_guess=[0.5, 0]
            )
            right_knee_to_thigh = pyg.CurveEdgeFactory.curve_from_tangents(
                knee_pt_out, thigh_pt_out,
                target_tan0=knee_out_tan,
                target_tan1=thigh_out_tan,
                initial_guess=[0.5, 0]
            )
            right_thigh_to_hip = pyg.CurveEdgeFactory.curve_from_tangents(
                thigh_pt_out, hip_pt,
                target_tan0=thigh_out_tan,
                target_tan1=np.array([0, 1]),  # vertical at hip (original behavior)
                initial_guess=[0.5, 0]
            )

            right_top = pyg.CurveEdgeFactory.curve_from_tangents(
                right_thigh_to_hip.end,
                [hw_shift, length + hips_depth],
                target_tan0=np.array([0, 1]),
                initial_guess=[0.5, 0]
            )

            # --- Barrel bow (arc_follow) ---
            # When barrel_bow is set the outseam becomes a single convex arc
            # (ankle→waist, bulging through a knee-height bow point) and the
            # INNER edge follows it so every girth stays exact (girth = inner −
            # outer width). The crotch / inner-thigh shifts outward as the outer
            # arc bulges → the legs splay: a true barrel inside and out, like the
            # ghost. inner_dx(y) = arc(y) − girth-correct-reference(y) is the
            # outward shift applied to the crotch and inseam below. barrel_bow is
            # the bulge DEPTH (offset = barrel_bow·length); the apex sits at the
            # knee height (knee_y, from the Knee_from_crotch measurement).
            outseam_leg = [right_ankle_to_knee, right_knee_to_thigh, right_thigh_to_hip]
            outseam_led_knee_x = None
            inner_dx = None   # set when barrel_bow: shift the inner edge to follow the arc
            if barrel_bow:
                cx = (ankle_pt[0] + (thigh_pt_out[0] - ankle_pt[0])
                      * (knee_y - ankle_pt[1]) / (thigh_pt_out[1] - ankle_pt[1]))
                knee_bow_x = cx - barrel_bow * length
                outseam_led_knee_x = knee_bow_x
                waist_corner = [hw_shift, length + hips_depth]
                right_top = pyg.CurveEdgeFactory.curve_3_points(
                    list(ankle_pt), waist_corner, [knee_bow_x, knee_y])
                outseam_leg = []
                _ac = right_top.as_curve()
                _AP = np.array([[_ac.point(t).real, _ac.point(t).imag]
                                for t in np.linspace(0, 1, 200)])
                _AP = _AP[np.argsort(_AP[:, 1])]
                _refy = np.array([ankle_pt[1], knee_y, thigh_y, length, waist_corner[1]])
                _refx = np.array([ankle_pt[0], knee_bow_x, thigh_out_x, 0.0, waist_corner[0]])
                def inner_dx(y, _AP=_AP, _refy=_refy, _refx=_refx):
                    return (float(np.interp(y, _AP[:, 1], _AP[:, 0]))
                            - float(np.interp(y, _refy, _refx)))
        else:
            # Original single-curve outseam
            if pyg.utils.close_enough(design['flare']['v'], 1):  # skip optimization
                right_bottom = pyg.Edge(
                    [-flare, 0],
                    [0, length]
                )
            else:
                right_bottom = pyg.CurveEdgeFactory.curve_from_tangents(
                    [-flare, 0],
                    [0, length],
                    target_tan1=np.array([0, 1]),
                    # initial guess places control point closer to the hips
                    initial_guess=[0.75, 0]
                )
            right_top = pyg.CurveEdgeFactory.curve_from_tangents(
                right_bottom.end,
                [hw_shift, length + hips_depth],
                target_tan0=np.array([0, 1]),
                initial_guess=[0.5, 0]
            )

        # back_rise_lift (cm): raise the center-back waist point — a "back-rise
        # scoop". This is the start of crotch_top, so it lengthens the center
        # rise seam and tilts the waist edge up toward center-back. Default 0.0
        # (front always passes 0.0). NOTE: the placement is decoupled from this
        # lift downstream (see PantsHalf), so the scoop changes shape only, not
        # where the legs sit.
        top = pyg.Edge(
            right_top.end,
            [w_diff + waist, length + hips_depth + back_rise_lift]
        )

        # arc_follow shifts the inner (crotch) edge outward to track the bulging
        # outer arc, so the hip/thigh girths stay correct (legs splay).
        _ct_y = length + 0.45 * hips_depth
        _cb_y = length - crotch_depth_diff
        # _ct_y is ABOVE the crotch tip — here the inner edge is the center
        # (rise) seam, not the inseam, so it must NOT take the leg's inner-follow
        # shift (that pulls the hip in by ~4-5 cm). Only the crotch TIP (_cb_y, in
        # the leg region) follows the arc, to hold the thigh girth.
        _ct_dx = 0.0
        _cb_dx = inner_dx(_cb_y) if inner_dx is not None else 0.0
        # CLO-match: bow the upper rise (crotch_top) outward instead of a straight
        # edge, to reproduce CLO's curved front-fly / back-rise line. Gated; the
        # default path keeps the original straight Edge untouched.
        _crz = design.get('clo_match', {}) if isinstance(design.get('clo_match'), dict) else {}
        _bow = float(_crz.get('front_rise_bow', 3.0))
        if clo_match and 'pant_f' in name and _bow > 0.05:
            _s = np.asarray(top.end, float)
            _e = np.array([hips + _ct_dx, _ct_y], float)
            _ch = _e - _s
            _perp = np.array([-_ch[1], _ch[0]], float)
            if _perp[0] < 0:
                _perp = -_perp            # bow toward +x (outward, crotch side)
            _perp /= np.linalg.norm(_perp)
            crotch_top = pyg.CurveEdgeFactory.curve_3_points(
                list(_s), list(_e), list((_s + _e) / 2 + _bow * _perp))
        else:
            crotch_top = pyg.Edge(
                top.end,
                [hips + _ct_dx, _ct_y]  # A bit higher than hip line
                # NOTE: The point should be lower than the minimum rise value (0.5)
            )
        crotch_bottom = pyg.CurveEdgeFactory.curve_from_tangents(
            crotch_top.end,
            [hips + crotch_extention + _cb_dx, _cb_y],
            target_tan0=np.array([0, -1]),
            target_tan1=np.array([1, 0]),
            initial_guess=[0.5, -0.5]
        )
        crotch_bottom.label = 'pants_crotch'

        # arc_follow shifts the crotch point laterally, which shortens the crotch
        # (rise) seam. Restore the rise: re-deepen crotch_bottom (bow it out, same
        # endpoints) until crotch_top+crotch_bottom length matches the UNSHIFTED
        # seam. The crotch point is shared by the thigh cross-section and the rise,
        # so this decouples them — thigh stays put, rise comes back.
        if inner_dx is not None:
            _ref_ct = pyg.Edge(top.end, [hips, _ct_y])
            _ref_cb = pyg.CurveEdgeFactory.curve_from_tangents(
                [hips, _ct_y], [hips + crotch_extention, _cb_y],
                target_tan0=np.array([0, -1]), target_tan1=np.array([1, 0]),
                initial_guess=[0.5, -0.5])
            _target = _ref_ct.length() + _ref_cb.length()
            _cb_s = np.asarray(crotch_top.end, float)
            _cb_e = np.array([hips + crotch_extention + _cb_dx, _cb_y], float)
            if crotch_top.length() + crotch_bottom.length() < _target - 0.05:
                # Deepen by offsetting the mid PERPENDICULAR to the chord (keeps the
                # projection at 0.5 so curve_3_points stays valid). Bulge toward −x
                # (outward, like the original crotch curve).
                _chord = _cb_e - _cb_s
                _perp = np.array([-_chord[1], _chord[0]])
                if _perp[0] > 0:
                    _perp = -_perp
                _perp = _perp / np.linalg.norm(_perp)
                _cmid = (_cb_s + _cb_e) / 2
                _lo, _hi = 0.0, 40.0
                for _ in range(22):
                    _dd = (_lo + _hi) / 2
                    _t = pyg.CurveEdgeFactory.curve_3_points(
                        list(_cb_s), list(_cb_e), list(_cmid + _dd * _perp))
                    if crotch_top.length() + _t.length() < _target:
                        _lo = _dd
                    else:
                        _hi = _dd
                crotch_bottom = pyg.CurveEdgeFactory.curve_3_points(
                    list(_cb_s), list(_cb_e), list(_cmid + ((_lo + _hi) / 2) * _perp))
                crotch_bottom.label = 'pants_crotch'

        # CLO-match: hold the rise (crotch_top + crotch_bottom) at a target length.
        # The split rebalance drifts the rise, so re-bow crotch_bottom (endpoints
        # fixed → thigh width unchanged) until the total rise hits the target. Any
        # deviation from the chord only LENGTHENS, so if the straight chord already
        # overshoots we additionally lower the crotch_top end (_ct_y) to shorten.
        # Gated; default path is untouched.
        if clo_match:
            _crz2 = design.get('clo_match', {}) if isinstance(design.get('clo_match'), dict) else {}
            _rise_tgt = float(_crz2.get('front_rise_len', 24.7) if 'pant_f' in name
                              else _crz2.get('back_rise_len', 34.7))
            _cb_s2 = np.asarray(crotch_top.end, float)
            _cb_e2 = np.asarray(crotch_bottom.end, float)
            _need = _rise_tgt - crotch_top.length()
            _chord2 = _cb_e2 - _cb_s2
            _chordlen = float(np.linalg.norm(_chord2))
            # Bowing crotch_bottom (endpoints fixed → thigh untouched) can only
            # LENGTHEN past the chord. Use it when the target needs more length
            # (the back case). Shortening the front rise is handled upstream by
            # pulling the crotch tip in (front_ext_frac), not here.
            if _need > _chordlen + 0.05:
                _perp2 = np.array([-_chord2[1], _chord2[0]], float)
                if _perp2[0] > 0: _perp2 = -_perp2
                _perp2 /= np.linalg.norm(_perp2)
                _cmid2 = (_cb_s2 + _cb_e2) / 2
                _lo2, _hi2 = 0.0, 45.0
                for _ in range(26):
                    _dd2 = (_lo2 + _hi2) / 2
                    _tc = pyg.CurveEdgeFactory.curve_3_points(
                        list(_cb_s2), list(_cb_e2), list(_cmid2 + _dd2 * _perp2))
                    if _tc.length() < _need: _lo2 = _dd2
                    else: _hi2 = _dd2
                crotch_bottom = pyg.CurveEdgeFactory.curve_3_points(
                    list(_cb_s2), list(_cb_e2), list(_cmid2 + ((_lo2 + _hi2) / 2) * _perp2))
                crotch_bottom.label = 'pants_crotch'

        if use_subcurves:
            # --- Inseam with 3 subcurves: crotch→thigh, thigh→knee, knee→ankle ---
            # Inseam X at intermediate heights = outseam_x + target_width.
            # When barrel_bow (arc_follow) is set, the outseam knee is the
            # smooth-arc x (not the pin), so the inseam follows it via inner_dx
            # to hold the knee/thigh girths.
            thigh_in_x = thigh_out_x + thigh_width
            if inner_dx is not None:        # arc_follow: inseam follows the arc
                thigh_in_x += inner_dx(thigh_y)
            knee_base_x = outseam_led_knee_x if outseam_led_knee_x is not None else knee_out_x
            knee_in_x = knee_base_x + knee_width

            thigh_pt_in = [thigh_in_x, thigh_y]
            knee_pt_in = [knee_in_x, knee_y]

            # Ankle inseam point (same formula as original). Use the UNSHIFTED
            # crotch base (crotch_bottom.end already carries the arc_follow
            # crotch shift _cb_dx) plus the arc's own ≈0 shift at the ankle, so
            # the crotch shift doesn't leak down and shrink the ankle opening.
            y_ankle = min(0, length - crotch_depth_diff * 1.5)
            ankle_in_x = crotch_bottom.end[0] - _cb_dx - 2 + flare
            if inner_dx is not None:
                ankle_in_x += inner_dx(y_ankle)
            ankle_pt_in = [ankle_in_x, y_ankle]

            # Inseam tangent at knee junction (catmull-rom: prev→next)
            knee_in_tan = np.array([ankle_pt_in[0] - crotch_bottom.end[0],
                                    ankle_pt_in[1] - crotch_bottom.end[1]])
            knee_in_tan = knee_in_tan / np.linalg.norm(knee_in_tan)

            # 3 inseam subcurves: crotch→thigh, thigh→knee, knee→ankle
            # Crotch→thigh is always a STRAIGHT edge: a Bezier here bumps
            # above the crotch line because the catmull-rom tangent (pointing
            # steeply downward toward the knee) forces the control point way
            # above the start, causing self-intersection with the crotch curve.
            left_crotch_to_thigh = pyg.Edge(
                crotch_bottom.end, thigh_pt_in)

            # Tangent at thigh: direction of the straight crotch→thigh edge
            thigh_edge_dir = np.array([thigh_pt_in[0] - crotch_bottom.end[0],
                                       thigh_pt_in[1] - crotch_bottom.end[1]])
            thigh_edge_len = np.linalg.norm(thigh_edge_dir)
            thigh_in_tan = (thigh_edge_dir / thigh_edge_len
                            if thigh_edge_len > 0.01 else np.array([0, -1]))

            left_thigh_to_knee = pyg.CurveEdgeFactory.curve_from_tangents(
                thigh_pt_in, knee_pt_in,
                target_tan0=thigh_in_tan,
                target_tan1=knee_in_tan,
                initial_guess=[0.5, 0]
            )
            left_knee_to_ankle = pyg.CurveEdgeFactory.curve_from_tangents(
                knee_pt_in, ankle_pt_in,
                target_tan0=knee_in_tan,
                target_tan1=[flare, y_ankle - crotch_bottom.end[1]],
                initial_guess=[0.3, 0]
            )
            inseam_edges = pyg.EdgeSequence(
                left_crotch_to_thigh, left_thigh_to_knee, left_knee_to_ankle)

            self.edges = pyg.EdgeSequence(
                *outseam_leg,
                right_top, top, crotch_top, crotch_bottom,
                *inseam_edges
            ).close_loop()
            bottom = self.edges[-1]
            bottom = self._maybe_front_slit(bottom, front_slit)

            # Default placement
            self.set_pivot(crotch_bottom.end)
            self.translation = [-0.5, - hips_depth - crotch_depth_diff + 5, 0]

            # Out interfaces
            self.interfaces = {
                'outside': pyg.Interface(
                    self,
                    pyg.EdgeSequence(*outseam_leg, right_top),
                    ruffle=[1] * len(outseam_leg) + [hipline_ext]),
                'crotch': pyg.Interface(self, pyg.EdgeSequence(crotch_top, crotch_bottom)),
                'inside': pyg.Interface(self, inseam_edges),
                'bottom': pyg.Interface(self, bottom)
            }
        else:
            left = pyg.CurveEdgeFactory.curve_from_tangents(
                crotch_bottom.end,
                [
                    # NOTE "Magic value" (-2 cm) which we use to define default width:
                    #   just a little behing the crotch point
                    # NOTE: Ensuring same distance from the crotch point in both
                    #   front and back for matching curves
                    crotch_bottom.end[0] - 2 + flare,
                    # NOTE: The inside edge either matches the length of the outside (0, normal case)
                    # or when the inteded length is smaller than crotch depth,
                    # inside edge covers of the inside leg a bit below the crotch (panties-like shorts)
                    y:=min(0, length - crotch_depth_diff * 1.5)
                ],
                target_tan1=[flare, y - crotch_bottom.end[1]],
                initial_guess=[0.3, 0]
            )

            self.edges = pyg.EdgeSequence(
                right_bottom, right_top, top, crotch_top, crotch_bottom, left
                ).close_loop()
            bottom = self.edges[-1]
            bottom = self._maybe_front_slit(bottom, front_slit)

            # Default placement
            self.set_pivot(crotch_bottom.end)
            self.translation = [-0.5, - hips_depth - crotch_depth_diff + 5, 0]

            # Out interfaces (easier to define before adding a dart)
            self.interfaces = {
                'outside': pyg.Interface(
                    self,
                    pyg.EdgeSequence(right_bottom, right_top),
                    ruffle=[1, hipline_ext]),
                'crotch': pyg.Interface(self, pyg.EdgeSequence(crotch_top, crotch_bottom)),
                'inside': pyg.Interface(self, left),
                'bottom': pyg.Interface(self, bottom)
            }

        # Add top dart
        # NOTE: Ruffle indicator to match to waistline proportion for correct balance line matching
        dart_width = w_diff - hw_shift  
        if w_diff > hw_shift:
            top_edges, int_edges = self.add_darts(
                top, dart_width, dart_depth, dart_position, double_dart=double_dart)
            self.interfaces['top'] = pyg.Interface(
                self, int_edges, 
                ruffle=waist / match_top_int_to if match_top_int_to is not None else 1.
            ) 
            self.edges.substitute(top, top_edges)
        else:
            self.interfaces['top'] = pyg.Interface(
                self, top, 
                ruffle=waist / match_top_int_to if match_top_int_to is not None else 1.
        ) 
        
        

    def _maybe_front_slit(self, bottom, front_slit):
        """Cut a thin V-notch (front split) into the ankle edge midpoint.
        Returns the (possibly replaced) bottom interface edge sequence.
        Same cutout mechanism as panel-skirt slits (pyg.ops.cut_into_edge).
        No-op when front_slit is falsy.
        """
        if not front_slit:
            return bottom
        depth = min(float(front_slit), bottom.length() * 0.45)  # keep inside leg
        new_edges, _, int_edges = pyg.ops.cut_into_edge(
            pyg.EdgeSeqFactory.dart_shape(2, depth=depth),
            bottom,
            offset=bottom.length() / 2,
            right=True,
        )
        self.edges.substitute(bottom, new_edges)
        return int_edges

    def add_darts(self, top, dart_width, dart_depth, dart_position, double_dart=False):
        
        if double_dart:
            # TODOLOW Avoid hardcoding for matching with the top?
            dist = dart_position * 0.5  # Dist between darts -> dist between centers
            offsets_mid = [
                - (dart_position + dist / 2 + dart_width / 2 + dart_width / 4),   
                - (dart_position - dist / 2) - dart_width / 4,
            ]

            darts = [
                pyg.EdgeSeqFactory.dart_shape(dart_width / 2, dart_depth * 0.9), # smaller
                pyg.EdgeSeqFactory.dart_shape(dart_width / 2, dart_depth)
            ]
        else:
            offsets_mid = [
                - dart_position - dart_width / 2,
            ]
            darts = [
                pyg.EdgeSeqFactory.dart_shape(dart_width, dart_depth)
            ]
        top_edges, int_edges = pyg.EdgeSequence(top), pyg.EdgeSequence(top)

        for off, dart in zip(offsets_mid, darts):
            left_edge_len = top_edges[-1].length()
            top_edges, int_edges = self.add_dart(
                dart,
                top_edges[-1],
                offset=left_edge_len + off,
                edge_seq=top_edges, 
                int_edge_seq=int_edges
            )

        return top_edges, int_edges
        

class PantsHalf(BaseBottoms):
    def __init__(self, tag, body, design, rise=None) -> None:
        super().__init__(body, design, tag, rise=rise)
        design = design['pants']
        self.rise = design['rise']['v'] if rise is None else rise
        waist, hips_depth, waist_back = self.eval_rise(self.rise)

        # NOTE: min value = full sum > leg curcumference
        # Max: pant leg falls flat from the back
        # Mostly from the back side
        # => This controls the foundation width of the pant
        # width_v scales the entire panel hip width (not just crotch extension)
        width_v = design['width']['v']
        min_ext = body['leg_circ'] - body['hips'] / 2  + 5  # 2 inch ease: from pattern making book
        front_hip = (body['hips'] - body['hip_back_width']) / 2 * width_v
        crotch_extention = min_ext * width_v
        if 'front_crotch_fraction' in design and design['front_crotch_fraction'].get('v') is not None:
            front_extention = crotch_extention * float(design['front_crotch_fraction']['v'])
        else:
            front_extention = front_hip / 4    # From pattern making book
        back_extention = crotch_extention - front_extention

        length, cuff_len = design['length']['v'], design['cuff']['cuff_len']['v']
        if design['cuff']['type']['v']:
            # Min length from params. Mapper-built designs carry only {'v': ...}
            # (no 'range'), so fall back to a small floor in that case.
            min_length = design['length'].get('range', [0.2])[0]
            if length - cuff_len < min_length:
                # Cannot be longer then a pant
                cuff_len = length - min_length
            # Include the cuff into the overall length,
            # unless the requested length is too short to fit the cuff
            # (to avoid negative length)
            length -= cuff_len
        length *= body['_leg_length']
        cuff_len *= body['_leg_length']

        # For rise > 1.0 (high-rise), offset panel upward so waistband
        # sits above the natural waist line (e.g. belly button level)
        rise_offset = max(0, (self.rise - 1.0)) * body['hips_line']

        # Thigh/knee width controls (if body measurements available)
        back_hip = body['hip_back_width'] / 2 * width_v

        # --- CLO-match path (opt-in via design['clo_match']['v']) ---------------
        # Rebalances the front/back split forward and widens the crotch extension
        # to reproduce CLO's more balanced front/back distribution. Gated: when the
        # flag is absent/false this block is skipped and behaviour is unchanged.
        clo_match = bool(design.get('clo_match', {}).get('v', False)) \
            if isinstance(design.get('clo_match'), dict) else False
        if clo_match:
            clo_cfg = design.get('clo_match', {})
            front_frac = float(clo_cfg.get('front_frac', 0.46))   # hip front share
            ext_frac = float(clo_cfg.get('front_ext_frac', 0.46)) # crotch-ext front share
            _tot_hip = front_hip + back_hip
            front_hip = _tot_hip * front_frac
            back_hip = _tot_hip * (1.0 - front_frac)
            front_extention = crotch_extention * ext_frac
            back_extention = crotch_extention - front_extention
        # ------------------------------------------------------------------------

        leg_shape_kwargs = {}
        if 'thigh_circ' in body and 'knee_circ' in body:
            thigh_v = design.get('thigh', {}).get('v', 1.0)
            knee_v = design.get('knee', {}).get('v', 1.0)

            thigh_circ = body['thigh_circ'] * thigh_v
            knee_circ = body['knee_circ'] * knee_v

            # Ensure crotch extension is wide enough for the thigh.
            # When thigh_circ > per-leg crotch width, the outseam must
            # bulge past the hip line, creating a spike. Widen the crotch
            # extension to match so the outseam stays monotonic.
            per_leg_crotch = front_hip + back_hip + crotch_extention
            if thigh_circ > per_leg_crotch:
                ext_ratio = (front_extention / crotch_extention
                             if crotch_extention > 0 else 0.5)
                crotch_extention = thigh_circ - front_hip - back_hip
                front_extention = crotch_extention * ext_ratio
                back_extention = crotch_extention - front_extention

            # Crotch-to-knee: prefer production override from design, then body YAML
            design_ctk = design.get('crotch_to_knee', {})
            if isinstance(design_ctk, dict) and design_ctk.get('v') is not None:
                crotch_to_knee = design_ctk['v']
            elif 'crotch_to_knee' in body:
                crotch_to_knee = body['crotch_to_knee']
            else:
                crotch_to_knee = 32.0

            # Y positions in panel coords (Y=0 at bottom, Y=length at hip)
            crotch_y = length - body['crotch_hip_diff']
            # Thigh point placed 2cm below crotch: just enough vertical
            # clearance for the straight crotch→thigh edge, while keeping
            # the control point near the thigh measurement cross-section.
            thigh_y = crotch_y - 2
            knee_y = crotch_y - crotch_to_knee

            # Only use subcurves if knee is above the ankle (pants long enough)
            if knee_y > 1:
                # Split thigh circumference using crotch-extension ratio:
                # At crotch_y the inseam X = hips + crotch_ext (fixed by
                # anatomy), so each panel's thigh share must match its
                # crotch-end width for the cross-section to hit the target.
                crotch_front = front_hip + front_extention
                crotch_back = back_hip + back_extention
                crotch_ratio = crotch_front / (crotch_front + crotch_back)

                # Knee uses hip ratio (knee measurement isn't affected by
                # crotch extensions since it's well below the crotch).
                total_hip_per_leg = front_hip + back_hip
                knee_front_ratio = front_hip / total_hip_per_leg

                leg_shape_kwargs = {
                    'knee_y': knee_y,
                    'thigh_y': thigh_y,
                    'front_thigh': thigh_circ * crotch_ratio,
                    'front_knee': knee_circ * knee_front_ratio,
                    'back_thigh': thigh_circ * (1 - crotch_ratio),
                    'back_knee': knee_circ * (1 - knee_front_ratio),
                }

        front_slit = design.get('front_slit', {}).get('v') if isinstance(design.get('front_slit'), dict) else None
        # Barrel bow: outseam-trajectory shaping, opt-in per config. Absent in
        # mapper-built designs, so existing (non-barrel) configs are unaffected.
        barrel_bow = design.get('barrel_bow', {}).get('v') if isinstance(design.get('barrel_bow'), dict) else None
        self.front = PantPanel(
            f'pant_f_{tag}', body, design,
            length=length,
            waist=(waist - waist_back) / 2,
            hips=front_hip,
            hips_depth=hips_depth,
            dart_position = body['bust_points'] / 2,
            crotch_width=front_extention,
            match_top_int_to=(body['waist'] - body['waist_back_width']) / 2,
            thigh_width=leg_shape_kwargs.get('front_thigh'),
            knee_width=leg_shape_kwargs.get('front_knee'),
            knee_y=leg_shape_kwargs.get('knee_y'),
            thigh_y=leg_shape_kwargs.get('thigh_y'),
            front_slit=front_slit,
            barrel_bow=barrel_bow,
            clo_match=clo_match,
            ).translate_by([0, body['_waist_level'] - 5 + rise_offset, 25])
        self.back = PantPanel(
            f'pant_b_{tag}', body, design,
            length=length,
            waist=waist_back / 2,
            hips=back_hip,
            hips_depth=hips_depth,
            hipline_ext=1.1,
            dart_position = body['bum_points'] / 2,
            crotch_width=back_extention,
            match_top_int_to=body['waist_back_width'] / 2,
            double_dart=True,
            thigh_width=leg_shape_kwargs.get('back_thigh'),
            knee_width=leg_shape_kwargs.get('back_knee'),
            knee_y=leg_shape_kwargs.get('knee_y'),
            thigh_y=leg_shape_kwargs.get('thigh_y'),
            barrel_bow=barrel_bow,
            clo_match=clo_match,
            back_rise_lift=(design['back_rise_lift']['v']
                            if isinstance(design.get('back_rise_lift'), dict)
                            and design['back_rise_lift'].get('v') is not None
                            else 0.0),
            ).translate_by([0, body['_waist_level'] - 5 + rise_offset, -20])

        self.stitching_rules = pyg.Stitches(
            (self.front.interfaces['outside'], self.back.interfaces['outside']),
            (self.front.interfaces['inside'], self.back.interfaces['inside'])
        )

        # add a cuff
        # TODOLOW This process is the same for sleeves -- make a function?
        if design['cuff']['type']['v']:
            
            pant_bottom = pyg.Interface.from_multiple(
                self.front.interfaces['bottom'],
                self.back.interfaces['bottom'])

            # Copy to avoid editing original design dict
            cdesign = deepcopy(design)
            cdesign['cuff']['b_width'] = {}
            # Balloon leg: a `target_width` (cm) pins the cuff circumference to
            # a fixed value (e.g. the ankle measurement) and lets the wide leg
            # bottom gather into it. Otherwise the cuff width is derived from
            # top_ruffle (the original gathered-by-ratio behaviour).
            cuff_target = design['cuff'].get('target_width')
            if isinstance(cuff_target, dict) and cuff_target.get('v'):
                cdesign['cuff']['b_width']['v'] = float(cuff_target['v'])
            else:
                cdesign['cuff']['b_width']['v'] = pant_bottom.edges.length() / design['cuff']['top_ruffle']['v']
            cdesign['cuff']['cuff_len']['v'] = cuff_len

            # Init
            cuff_class = getattr(bands, cdesign['cuff']['type']['v'])
            self.cuff = cuff_class(f'pant_{tag}', cdesign)

            # Position. Center the cuff on the leg tube (coaxial). For a
            # balloon leg the cuff is much narrower than the wide leg opening,
            # so edge alignment ('left') would shove the cuff to one side and
            # the gather would wrap unevenly; 'center' keeps it concentric.
            self.cuff.place_by_interface(
                self.cuff.interfaces['top'],
                pant_bottom,
                gap=5,
                alignment='center'
            )

            # Stitch
            self.stitching_rules.append((
                pant_bottom,
                self.cuff.interfaces['top'])
            )

        self.interfaces = {
            'crotch_f': self.front.interfaces['crotch'],
            'crotch_b': self.back.interfaces['crotch'],
            'top_f': self.front.interfaces['top'],
            'top_b': self.back.interfaces['top']
        }



    def length(self):
        if self.design['pants']['cuff']['type']['v']:
            return self.front.length() + self.cuff.length()
        
        return self.front.length()

class Pants(BaseBottoms):
    def __init__(self, body, design, rise=None) -> None:
        super().__init__(body, design)

        self.right = PantsHalf('r', body, design, rise)
        self.left = PantsHalf('l', body, design, rise).mirror()

        self.stitching_rules = pyg.Stitches(
            (self.right.interfaces['crotch_f'], self.left.interfaces['crotch_f']),
            (self.right.interfaces['crotch_b'], self.left.interfaces['crotch_b']),
        )

        self.interfaces = {
            'top_f': pyg.Interface.from_multiple(
                self.right.interfaces['top_f'], self.left.interfaces['top_f']),
            'top_b': pyg.Interface.from_multiple(
                self.right.interfaces['top_b'], self.left.interfaces['top_b']),
            # Some are reversed for correct connection
            'top': pyg.Interface.from_multiple(   # around the body starting from front right
                self.right.interfaces['top_f'].flip_edges(),
                self.left.interfaces['top_f'].reverse(with_edge_dir_reverse=True),
                self.left.interfaces['top_b'].flip_edges(),
                self.right.interfaces['top_b'].reverse(with_edge_dir_reverse=True), # Flips the edges and restores the direction
            )
        }

    def get_rise(self):
        return self.right.get_rise()
    
    def length(self):
        return self.right.length()
