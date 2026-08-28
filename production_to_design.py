"""
Production Measurements → GarmentCode Design Parameters Mapping

Maps absolute garment production measurements (in cm) + person body measurements
to the relative design parameters that GarmentCode expects.

GarmentCode's parametric system uses multipliers/fractions relative to body
dimensions, not absolute cm values. This script performs the inverse mapping.

Usage:
    # Single garment (shirt)
    python production_to_design.py --mode shirt

    # Single garment (pants)
    python production_to_design.py --mode pants

    # Full outfit (shirt + pants, simulated separately, combined)
    python production_to_design.py --mode outfit

    Or import and use programmatically:
        from production_to_design import ProductionToDesign
        mapper = ProductionToDesign(body_yaml_path)
        design = mapper.map_shirt(garment_measurements)
"""

import copy
import yaml
import numpy as np
from pathlib import Path


# =====================================================
#  Supported garment types in GarmentCode
# =====================================================
SUPPORTED_TYPES = """
UPPER GARMENTS (design.meta.upper):
  Shirt          - Straight/loose-fit top (t-shirt, blouse, tunic)
  FittedShirt    - Fitted bodice with bust darts

LOWER GARMENTS (design.meta.bottom):
  Pants          - Trousers (shorts to full-length, slim to wide-leg)
  SkirtCircle    - Circular/flared skirt
  AsymmSkirtCircle - Asymmetric circle skirt (high-low hem)
  SkirtManyPanels  - Multi-panel A-line skirt (4-15 panels)
  PencilSkirt    - Fitted straight skirt (with optional slits)
  GodetSkirt     - Pencil/panel skirt with godet inserts for flare
  SkirtLevels    - Tiered/layered skirt (1-5 levels)

WAISTBANDS (design.meta.wb):
  StraightWB     - Straight waistband
  FittedWB       - Contoured/shaped waistband

SLEEVES (design.sleeve):
  Regular sleeves with ArmholeCurve / ArmholeAngle / ArmholeSquare
  Sleeveless option
  Optional cuffs: CuffBand, CuffSkirt, CuffBandSkirt

COLLARS (design.collar):
  CircleNeckHalf, CurvyNeckHalf, VNeckHalf, SquareNeckHalf,
  TrapezoidNeckHalf, CircleArcNeckHalf, Bezier2NeckHalf
  Optional components: Turtle, SimpleLapel, Hood2Panels

COMBINATIONS:
  Upper only          - Shirt or FittedShirt (with/without sleeves)
  Lower only          - Any skirt or Pants (requires waistband)
  Upper + Lower       - Jumpsuit (connected at waist, single mesh)
  Separate outfit     - Use map_outfit() to simulate each independently
"""


class ProductionToDesign:
    """Maps garment production measurements (cm) to GarmentCode design parameters.

    GarmentCode parameters are relative to body measurements:
        - shirt.width.v = garment_bust_circ / body.bust  (ease multiplier)
        - shirt.length.v = garment_length / body.waist_line
        - sleeve.length.v = sleeve_length / (body.arm_length - armhole_depth)
        - etc.

    This class performs the inverse: given absolute cm values from production
    data + a body, compute the relative design parameters.
    """

    def __init__(self, body_yaml_path):
        """Load body measurements from a GarmentCode body YAML file.

        Args:
            body_yaml_path: Path to body YAML (e.g., assets/bodies/mean_all.yaml)
        """
        with open(body_yaml_path, 'r') as f:
            body_data = yaml.safe_load(f)
        self.body = body_data['body']

        # Compute derived body params (mirrors body_params.py logic)
        self.body['_waist_level'] = (
            self.body['height'] - self.body['head_l'] - self.body['waist_line']
        )
        self.body['_leg_length'] = (
            self.body['_waist_level'] - self.body['hips_line']
        )
        self.body['_base_sleeve_balance'] = self.body['shoulder_w'] - 2
        self.body['_shoulder_incl'] = self.body['shoulder_incl']
        self.body['_armscye_depth'] = self.body['armscye_depth'] + 2.5
        # _bust_line: weighted average of vert_bust_line and bust_line
        if 'vert_bust_line' in self.body:
            self.body['_bust_line'] = (
                (2/3) * self.body['vert_bust_line']
                + (1/3) * self.body['bust_line']
            )
        else:
            self.body['_bust_line'] = self.body['bust_line']

    def _estimate_sleeve_geometry(self, bust_circ, b, collar_width_cm=None):
        """Estimate opening_length, remaining_shoulder, and arm_width from
        body geometry.

        These are needed to correctly convert HPS-to-cuff measurements to the
        sleeve_length_v parameter that SleevePanel uses, and to compute the
        correct cuff end_width ratio.

        Returns:
            (opening_length, remaining_shoulder, arm_width) in cm
        """
        from assets.garment_programs.sleeves import ArmholeCurve
        import pygarment as pyg

        sleeve_balance = b['_base_sleeve_balance'] / 2
        connecting_width = b['_armscye_depth'] * 1.2  # default v=0.2

        # Torso half-widths (matching tee.py)
        front_body_w = (b['bust'] - b['back_width']) / 2
        front_frac = front_body_w / b['bust']
        ftorso_width = front_frac * bust_circ

        # TorsoFrontHalfPanel.get_width(level) for flare=1.0 returns
        # self.width = ftorso_width (the shoulder_w/2 terms cancel)
        front_w = ftorso_width

        rest_angle = max(np.deg2rad(10), np.deg2rad(b['_shoulder_incl']))

        # Run ArmholeCurve to get the opening shape
        _, front_opening = ArmholeCurve(
            front_w - sleeve_balance, connecting_width,
            angle=rest_angle, bottom_angle_mix=0.1)

        # Back opening for even_armhole
        back_body_w = b['back_width'] / 2
        back_frac = back_body_w / b['bust']
        btorso_width = back_frac * bust_circ
        back_w = btorso_width
        _, back_opening = ArmholeCurve(
            back_w - sleeve_balance, connecting_width,
            angle=rest_angle, bottom_angle_mix=0.1)

        if front_w != back_w:
            front_opening, back_opening = pyg.ops.even_armhole_openings(
                front_opening, back_opening,
                tol=0.2 / front_opening.length())

        opening_length = abs(
            front_opening[0].start[0] - front_opening[-1].end[0])
        arm_width = abs(
            front_opening[0].start[1] - front_opening[-1].end[1])

        # Remaining shoulder between collar and armhole cuts
        cw = collar_width_cm if collar_width_cm is not None else b['neck_w']
        remaining_shoulder = (
            (1 / np.cos(np.deg2rad(b['_shoulder_incl'])))
            * (sleeve_balance - cw / 2)
        )

        return opening_length, remaining_shoulder, arm_width

    def map_shirt(self, garment, collar_type='CircleNeckHalf'):
        """Map production measurements for a shirt/t-shirt to design params.

        Args:
            garment: dict with absolute measurements in cm:
                - bust_circumference: Full garment bust circumference (cm)
                - length: Shoulder-to-hem length (cm)
                - sleeve_length: (optional) Underarm-to-cuff length (cm).
                    If not provided, generates sleeveless.
                - hem_circumference: (optional) Circumference at hem (cm).
                    If not provided, assumes same as bust.
                - collar_depth_front: (optional) Front neckline depth (cm)
                - collar_depth_back: (optional) Back neckline depth (cm)
                - collar_width: (optional) Neck opening width (cm)
                - sleeve_end_width_ratio: (optional) Cuff/bicep ratio
            collar_type: Collar shape class name. Options:
                CircleNeckHalf, CurvyNeckHalf, VNeckHalf,
                SquareNeckHalf, TrapezoidNeckHalf, CircleArcNeckHalf,
                Bezier2NeckHalf

        Returns:
            dict: Complete design parameter dict ready for MetaGarment
        """
        b = self.body

        # --- Shirt body (width & length) ---
        # design.shirt.width.v = garment_bust_circ / body.bust
        # This is the ease multiplier: 1.0 = skin-tight, 1.05 = standard ease
        bust_circ = garment['bust_circumference']
        width_v = bust_circ / b['bust']
        # Lower bound 0.5 allows negative ease (knit/jersey garments cut
        # smaller than the body bust — e.g. HDPAIGE off-shoulder, fitted
        # tube tops). Upper bound stays at 1.3 to avoid runaway-loose tops.
        width_v = np.clip(width_v, 0.5, 1.3)

        # --- Length (corrected for shoulder slope and collar cut) ---
        # Back panel center height = length_v * waist_line + sh_tan * btorso_width
        # cut_corner removes (bc_depth + sh_tan * collar_width/2) from center seam
        # where bc_depth = bc_depth_v * _bust_line + sh_tan * (btorso_width - cw/2)
        # The shoulder-slope terms cancel, giving:
        #   nape_to_hem = length_v * waist_line - bc_depth_v * _bust_line
        # So: length_v = (nape_to_hem + bc_depth_v * _bust_line) / waist_line
        #     = (target + bc_depth_target - sh_tan * (btorso_width - cw/2)) / waist_line
        length_cm = garment['length']
        back_frac = (b['back_width'] / 2) / b['bust']
        btorso_width = back_frac * bust_circ
        sh_tan = np.tan(np.deg2rad(b['_shoulder_incl']))

        bc_depth_target = garment.get('collar_depth_back', None)
        if bc_depth_target is not None:
            # Collar width estimate for the slope term
            cw_for_length = garment.get('collar_width', b['neck_w'])
            length_v = (length_cm + bc_depth_target
                        - sh_tan * (btorso_width - cw_for_length / 2)
                        ) / b['waist_line']
        else:
            # Default bc_depth_v=0: nape_to_hem = length_v * waist_line
            length_v = length_cm / b['waist_line']

        length_v = np.clip(length_v, 0.5, 3.5)

        # design.shirt.flare.v = hem_width_ratio / bust_width_ratio
        # Controls A-line shape: 1.0 = straight, >1 = flared
        hem_circ = garment.get('hem_circumference', bust_circ)
        flare_v = (hem_circ / b['bust']) / width_v
        flare_v = np.clip(flare_v, 0.7, 1.6)

        # --- Collar width (needed by sleeve geometry estimation) ---
        collar_width_cm = garment.get('collar_width', None)

        # --- Sleeves ---
        hps_to_cuff = garment.get('hps_to_cuff', None)
        # sleeve_from_shoulder: SLEEVELENGTH measured from the shoulder point to
        # the cuff (the sleeve's own length). This IS sleeve_top_target, so it is
        # used directly — the shoulder seam is built by the torso and must not be
        # added again (that conflation runs the cuff past the wrist).
        sleeve_from_shoulder = garment.get('sleeve_from_shoulder', None)
        sleeve_length_cm = garment.get('sleeve_length', None)
        arm_width_est = None  # Will be set if sleeve geometry is estimated

        has_hps = hps_to_cuff is not None and hps_to_cuff > 0
        has_from_shoulder = sleeve_from_shoulder is not None and sleeve_from_shoulder > 0
        if has_hps or has_from_shoulder:
            sleeveless = False
            # Compute opening_length from ArmholeCurve construction
            # to get the exact denominator for sleeve_length_v
            opening_length, remaining_shoulder, arm_width_est = \
                self._estimate_sleeve_geometry(bust_circ, b, collar_width_cm)
            # Brand-target bicep override: arm_width corresponds to half-bicep
            # (one panel's cap width); full bicep circumference = 2 * arm_width.
            bicep_target = garment.get('bicep_circumference', None)
            if bicep_target is not None and bicep_target > 0:
                arm_width_est = bicep_target / 2
            if has_from_shoulder:
                sleeve_top_target = sleeve_from_shoulder
            else:
                sleeve_top_target = hps_to_cuff - remaining_shoulder
            target_length = sleeve_top_target - opening_length
            available_arm = b['arm_length'] - opening_length
            sleeve_length_v = target_length / available_arm
            sleeve_length_v = np.clip(sleeve_length_v, 0.1, 1.15)
        elif sleeve_length_cm is not None and sleeve_length_cm > 0:
            sleeveless = False
            armhole_depth = b['_armscye_depth']
            available_arm = b['arm_length'] - armhole_depth
            sleeve_length_v = sleeve_length_cm / available_arm
            sleeve_length_v = np.clip(sleeve_length_v, 0.1, 1.15)
        else:
            sleeveless = True
            sleeve_length_v = 0.3  # default, won't be used

        # Sleeve end width relative to armhole width (arm_width in SleevePanel)
        # Use actual arm_width from armhole estimate when available
        sleeve_opening_circ = garment.get('sleeve_opening_circ', None)
        if sleeve_opening_circ is not None and arm_width_est is not None and arm_width_est > 0:
            # cuff_circ_per_arm = 2 * end_width_v * arm_width
            sleeve_end_width = sleeve_opening_circ / (2 * arm_width_est)
        else:
            sleeve_end_width = garment.get('sleeve_end_width_ratio', 1.0)

        # --- Collar ---
        collar_depth_front = garment.get('collar_depth_front', None)
        collar_depth_back = garment.get('collar_depth_back', None)

        # Collar width: interpolation parameter that the bodice will invert to
        # produce a garment collar width equal to the target. bodice.py uses:
        #   if v >= 0: cw = neck_w + v * (max_w - neck_w)
        #   if v <  0: cw = (1 + v) * neck_w
        # Each branch is a different parametrisation, so the inverse here is
        # also branched — but both make the produced garment width exact.
        collar_width_v = 0.2  # default
        if collar_width_cm is not None:
            min_w = b['neck_w']
            if collar_width_cm >= min_w:
                max_w = b['_base_sleeve_balance'] - 2
                # No upper clip: off-shoulder / wide-neck garments need v > 1.
                # The bodice extrapolates linearly past max_w.
                collar_width_v = (collar_width_cm - min_w) / (max_w - min_w)
                collar_width_v = max(collar_width_v, 0.0)
            else:
                # target < neck_w → bodice formula: cw = (1 + v) * neck_w
                collar_width_v = collar_width_cm / min_w - 1.0
                collar_width_v = np.clip(collar_width_v, -1.0, 0.0)

        # Collar depth: bodice.py eval_dep_params does:
        #   strapless_depth = fc_depth_v * _bust_line
        #   fc_depth = strapless_depth + tan(shoulder_incl) * (torso_width - collar_width/2)
        # We need to set fc_depth_v so the final fc_depth matches the target.
        fc_depth_v = 0.4  # default
        bc_depth_v = 0.0  # default

        if collar_depth_front is not None or collar_depth_back is not None:
            # Compute the shoulder adjustment that bodice.py will add
            tg = np.tan(np.deg2rad(b['_shoulder_incl']))
            bust_circ = garment['bust_circumference']

            # Torso half-widths (matching tee.py panel construction)
            front_body_w = (b['bust'] - b['back_width']) / 2
            front_frac = front_body_w / b['bust']
            ftorso_width = front_frac * bust_circ

            back_body_w = b['back_width'] / 2
            back_frac = back_body_w / b['bust']
            btorso_width = back_frac * bust_circ

            # Collar width in cm (as bodice.py will compute it)
            if collar_width_cm is not None:
                cw = collar_width_cm
            else:
                min_w = b['neck_w']
                max_w = b['_base_sleeve_balance'] - 2
                from pygarment.garmentcode.utils import lin_interpolation
                cw = lin_interpolation(min_w, max_w, collar_width_v)

            if collar_depth_front is not None:
                f_depth_adj = tg * (ftorso_width - cw / 2)
                fc_depth_v = (collar_depth_front - f_depth_adj) / b['_bust_line']

            if collar_depth_back is not None:
                b_depth_adj = tg * (btorso_width - cw / 2)
                bc_depth_v = (collar_depth_back - b_depth_adj) / b['_bust_line']

        # --- Build design dict ---
        design = {
            'meta': {
                'upper': {'v': 'Shirt'},
                'wb': {'v': None},
                'bottom': {'v': None},
            },
            'shirt': {
                'strapless': {'v': False},
                'length': {'v': float(length_v)},
                'width': {'v': float(width_v)},
                'flare': {'v': float(flare_v)},
            },
            'collar': {
                'f_collar': {'v': collar_type},
                'b_collar': {'v': 'CircleNeckHalf'},
                'width': {'v': float(collar_width_v)},
                'fc_depth': {'v': float(fc_depth_v)},
                'bc_depth': {'v': float(bc_depth_v)},
                'fc_angle': {'v': 95},
                'bc_angle': {'v': 95},
                'f_bezier_x': {'v': 0.3},
                'f_bezier_y': {'v': 0.55},
                'b_bezier_x': {'v': 0.15},
                'b_bezier_y': {'v': 0.1},
                'f_flip_curve': {'v': False},
                'b_flip_curve': {'v': False},
                'component': {
                    'style': {'v': None},
                    'depth': {'v': 7},
                    'lapel_standing': {'v': False},
                    'hood_depth': {'v': 1},
                    'hood_length': {'v': 1},
                },
            },
            'sleeve': {
                'sleeveless': {'v': sleeveless},
                'armhole_shape': {'v': 'ArmholeCurve'},
                'length': {'v': float(sleeve_length_v)},
                'connecting_width': {'v': 0.2},
                'end_width': {'v': float(sleeve_end_width)},
                'sleeve_angle': {'v': 10},
                'opening_dir_mix': {'v': 0.1},
                'standing_shoulder': {'v': False},
                'standing_shoulder_len': {'v': 5.0},
                'connect_ruffle': {'v': 1},
                'smoothing_coeff': {'v': 0.25},
                'cuff': {
                    'type': {'v': None},
                    'top_ruffle': {'v': 1},
                    'cuff_len': {'v': 0.1},
                    'skirt_fraction': {'v': 0.5},
                    'skirt_flare': {'v': 1.2},
                    'skirt_ruffle': {'v': 1.0},
                },
            },
            'left': {
                'enable_asym': {'v': False},
                'shirt': {
                    'strapless': {'v': False},
                    'width': {'v': float(width_v)},
                    'flare': {'v': float(flare_v)},
                },
                'collar': {
                    'f_collar': {'v': collar_type},
                    'b_collar': {'v': 'CircleNeckHalf'},
                    'width': {'v': float(collar_width_v)},
                    'fc_angle': {'v': 95},
                    'bc_angle': {'v': 95},
                    'f_bezier_x': {'v': 0.5},
                    'f_bezier_y': {'v': 0.3},
                    'b_bezier_x': {'v': 0.5},
                    'b_bezier_y': {'v': 0.3},
                    'f_flip_curve': {'v': False},
                    'b_flip_curve': {'v': False},
                },
                'sleeve': {
                    'sleeveless': {'v': True},
                    'armhole_shape': {'v': 'ArmholeCurve'},
                    'length': {'v': float(sleeve_length_v)},
                    'connecting_width': {'v': 0.2},
                    'end_width': {'v': float(sleeve_end_width)},
                    'sleeve_angle': {'v': 10},
                    'opening_dir_mix': {'v': 0.2},
                    'standing_shoulder': {'v': False},
                    'standing_shoulder_len': {'v': 5.0},
                    'connect_ruffle': {'v': 1},
                    'smoothing_coeff': {'v': 0.25},
                    'cuff': {
                        'type': {'v': None},
                        'top_ruffle': {'v': 1},
                        'cuff_len': {'v': 0.1},
                        'skirt_fraction': {'v': 0.5},
                        'skirt_flare': {'v': 1.2},
                        'skirt_ruffle': {'v': 1.0},
                    },
                },
            },
            # Include stubs for unused sections (MetaGarment may reference them)
            'waistband': {
                'waist': {'v': 1.0},
                'width': {'v': 0.2},
            },
        }

        return design

    def map_pants(self, garment):
        """Map production measurements for pants to design params.

        Args:
            garment: dict with absolute measurements in cm:
                - waist_circumference: Full waist circumference (cm)
                - length: Waist-to-ankle length (cm)
                - hip_circumference: (optional) Full hip circumference (cm)
                - leg_opening: (optional) Circumference at ankle (cm)

        Returns:
            dict: Complete design parameter dict ready for MetaGarment
        """
        b = self.body

        # pants.width.v = hip_circumference / body.hips (ease multiplier)
        hip_circ = garment.get('hip_circumference', garment['waist_circumference'] * 1.1)
        width_v = hip_circ / b['hips']
        width_v = np.clip(width_v, 0.5, 1.5)

        # pants.length.v is relative to leg_length
        # Upper bound of 1.2 allows full-length pants to reach the floor
        # (length_v > 1.0 means the hem extends below the body's hip datum)
        leg_length = b['_leg_length']
        length_v = garment['length'] / leg_length
        length_v = np.clip(length_v, 0.2, 1.2)

        # Thigh/knee: multiplier on body circumference at those levels
        thigh_v = 1.0
        if garment.get('thigh_circumference') and 'thigh_circ' in b:
            thigh_v = float(np.clip(
                garment['thigh_circumference'] / b['thigh_circ'], 0.5, 2.5))

        # Leg opening (ankle) — also used as the knee fallback target below
        leg_opening = garment.get('leg_opening', hip_circ * 0.5)

        # Knee: multiplier on body knee circumference. If the techpack omits
        # Knee but provides Thigh and Ankle, interpolate linearly between them
        # at the knee Y position rather than falling back to the body's knee.
        # The body fallback creates a sharp bottleneck for jeans whose thigh
        # and ankle are both well wider than the body's anatomical knee.
        knee_v = 1.0
        if garment.get('knee_circumference') and 'knee_circ' in b:
            knee_v = float(np.clip(
                garment['knee_circumference'] / b['knee_circ'], 0.5, 2.5))
        elif garment.get('thigh_circumference') and 'knee_circ' in b:
            crotch_to_knee = garment.get('crotch_to_knee') or b.get('crotch_to_knee')
            inseam_len = garment['length'] - b.get('crotch_hip_diff', 0)
            if crotch_to_knee and inseam_len > 0:
                frac = float(np.clip(crotch_to_knee / inseam_len, 0.0, 1.0))
                knee_circ_target = (
                    garment['thigh_circumference']
                    + (leg_opening - garment['thigh_circumference']) * frac
                )
                knee_v = float(np.clip(knee_circ_target / b['knee_circ'], 0.5, 2.5))

        # Flare: solve for flare_v from target leg opening circumference
        # With width_v scaling both hips and crotch_ext:
        #   leg_opening = (hips/2 + min_ext) * width_v - 4 + leg_circ*(flare_v - 1)
        # where min_ext = leg_circ - hips/2 + 5, and the -4 comes from the
        # "magic value" (-2 cm) offset on each panel's inside seam bottom.
        min_ext = b['leg_circ'] - b['hips'] / 2 + 5
        flare_v = (leg_opening - (b['hips'] / 2 + min_ext) * width_v + 4 + b['leg_circ']) / b['leg_circ']

        # Anticipate crotch extension increase from pants.py: when the thigh
        # circumference exceeds the per-leg crotch width, PantsHalf widens the
        # crotch extension, which propagates to the ankle via ankle_in_x.
        # Pre-compensate flare_v so the ankle stays on target.
        if 'thigh_circ' in b:
            thigh_circ = b['thigh_circ'] * thigh_v
            per_leg_crotch = (b['hips'] / 2 + min_ext) * width_v
            if thigh_circ > per_leg_crotch:
                crotch_increase = thigh_circ - per_leg_crotch
                flare_v -= crotch_increase / b['leg_circ']

        flare_v = np.clip(flare_v, 0.3, 2.0)

        # Rise: how high the waistband sits (1.0 = natural waist, >1.0 = high-rise)
        rise_v = garment.get('rise', 1.0)
        rise_v = np.clip(rise_v, 0.2, 1.8)   # low bound 0.2 enables true low-rise (was 0.5, which floored the front rise ~18.9cm); upper 1.8 matches the fitter's RISE_MAX (was 1.5, a silent plateau that stalled the rise fit on short bodies)

        # Crotch-to-knee distance override from production data (cm)
        crotch_to_knee = garment.get('crotch_to_knee')

        # Waistband circumference from production data.
        # waist.v is a multiplier on body.waist; values > 1.0 make the WB larger
        # than the body's natural waist, so the pants settle lower in simulation.
        waist_v = garment['waist_circumference'] / b['waist']

        design = {
            'meta': {
                'upper': {'v': None},
                'wb': {'v': 'StraightWB'},
                'bottom': {'v': 'Pants'},
            },
            'waistband': {
                'waist': {'v': float(waist_v)},
                'width': {'v': 0.2},
            },
            'pants': {
                'length': {'v': float(length_v)},
                'width': {'v': float(width_v)},
                'flare': {'v': float(flare_v)},
                'thigh': {'v': float(thigh_v)},
                'knee': {'v': float(knee_v)},
                'crotch_to_knee': {'v': float(crotch_to_knee) if crotch_to_knee else None},
                'rise': {'v': float(rise_v)},
                'cuff': {
                    'type': {'v': None},
                    'top_ruffle': {'v': 1.0},
                    'cuff_len': {'v': 0.1},
                    'skirt_fraction': {'v': 0.5},
                    'skirt_flare': {'v': 1.2},
                    'skirt_ruffle': {'v': 1.0},
                },
            },
            # Stubs for unused sections
            'shirt': {
                'strapless': {'v': False},
                'length': {'v': 1.2},
                'width': {'v': 1.05},
                'flare': {'v': 1.0},
            },
            'collar': {
                'f_collar': {'v': 'CircleNeckHalf'},
                'b_collar': {'v': 'CircleNeckHalf'},
                'width': {'v': 0.2},
                'fc_depth': {'v': 0.4},
                'bc_depth': {'v': 0},
                'fc_angle': {'v': 95},
                'bc_angle': {'v': 95},
                'f_bezier_x': {'v': 0.3},
                'f_bezier_y': {'v': 0.55},
                'b_bezier_x': {'v': 0.15},
                'b_bezier_y': {'v': 0.1},
                'f_flip_curve': {'v': False},
                'b_flip_curve': {'v': False},
                'component': {
                    'style': {'v': None},
                    'depth': {'v': 7},
                    'lapel_standing': {'v': False},
                    'hood_depth': {'v': 1},
                    'hood_length': {'v': 1},
                },
            },
            'sleeve': {
                'sleeveless': {'v': True},
                'armhole_shape': {'v': 'ArmholeCurve'},
                'length': {'v': 0.3},
                'connecting_width': {'v': 0.2},
                'end_width': {'v': 1.0},
                'sleeve_angle': {'v': 10},
                'opening_dir_mix': {'v': 0.1},
                'standing_shoulder': {'v': False},
                'standing_shoulder_len': {'v': 5.0},
                'connect_ruffle': {'v': 1},
                'smoothing_coeff': {'v': 0.25},
                'cuff': {
                    'type': {'v': None},
                    'top_ruffle': {'v': 1},
                    'cuff_len': {'v': 0.1},
                    'skirt_fraction': {'v': 0.5},
                    'skirt_flare': {'v': 1.2},
                    'skirt_ruffle': {'v': 1.0},
                },
            },
            'left': {
                'enable_asym': {'v': False},
                'shirt': {
                    'strapless': {'v': False},
                    'width': {'v': 1.05},
                    'flare': {'v': 1.0},
                },
                'collar': {
                    'f_collar': {'v': 'CircleNeckHalf'},
                    'b_collar': {'v': 'CircleNeckHalf'},
                    'width': {'v': 0.5},
                    'fc_angle': {'v': 95},
                    'bc_angle': {'v': 95},
                    'f_bezier_x': {'v': 0.5},
                    'f_bezier_y': {'v': 0.3},
                    'b_bezier_x': {'v': 0.5},
                    'b_bezier_y': {'v': 0.3},
                    'f_flip_curve': {'v': False},
                    'b_flip_curve': {'v': False},
                },
                'sleeve': {
                    'sleeveless': {'v': True},
                    'armhole_shape': {'v': 'ArmholeCurve'},
                    'length': {'v': 0.3},
                    'connecting_width': {'v': 0.2},
                    'end_width': {'v': 1.0},
                    'sleeve_angle': {'v': 10},
                    'opening_dir_mix': {'v': 0.2},
                    'standing_shoulder': {'v': False},
                    'standing_shoulder_len': {'v': 5.0},
                    'connect_ruffle': {'v': 1},
                    'smoothing_coeff': {'v': 0.25},
                    'cuff': {
                        'type': {'v': None},
                        'top_ruffle': {'v': 1},
                        'cuff_len': {'v': 0.1},
                        'skirt_fraction': {'v': 0.5},
                        'skirt_flare': {'v': 1.2},
                        'skirt_ruffle': {'v': 1.0},
                    },
                },
            },
        }

        return design

    def map_skirt(self, garment, skirt_type='SkirtCircle'):
        """Map production measurements for a skirt to design params.

        Args:
            garment: dict with absolute measurements in cm:
                - waist_circumference: Full waist circumference (cm)
                - length: Waist-to-hem length (cm)
                - hem_circumference: (optional) Full hem circumference (cm)
            skirt_type: One of: SkirtCircle, AsymmSkirtCircle,
                SkirtManyPanels, PencilSkirt, GodetSkirt, SkirtLevels

        Returns:
            dict: Complete design parameter dict ready for MetaGarment
        """
        b = self.body

        # Skirt length is relative to leg_length
        leg_length = b['_leg_length']
        length_v = garment['length'] / leg_length
        length_v = np.clip(length_v, -0.2, 0.95)

        # Rise
        rise_v = garment.get('rise', 1.0)

        # Fullness (for circle skirts): larger = more fabric
        hem_circ = garment.get('hem_circumference', garment['waist_circumference'] * 2)
        waist_circ = garment['waist_circumference']
        # suns = ratio of full-circle coverage (1.0 = full circle skirt)
        suns_v = hem_circ / (2 * np.pi * garment['length']) if garment['length'] > 0 else 0.75
        suns_v = np.clip(suns_v, 0.1, 1.95)

        design = {
            'meta': {
                'upper': {'v': None},
                'wb': {'v': 'StraightWB'},
                'bottom': {'v': skirt_type},
            },
            'waistband': {
                'waist': {'v': 1.0},
                'width': {'v': 0.2},
            },
            'skirt': {
                'length': {'v': float(length_v)},
                'rise': {'v': float(rise_v)},
                'ruffle': {'v': 1.3},
                'bottom_cut': {'v': 0},
                'flare': {'v': 0},
            },
            'flare-skirt': {
                'length': {'v': float(length_v)},
                'rise': {'v': float(rise_v)},
                'suns': {'v': float(suns_v)},
                'skirt-many-panels': {
                    'n_panels': {'v': 4},
                    'panel_curve': {'v': 0},
                },
                'asymm': {
                    'front_length': {'v': 0.5},
                },
                'cut': {
                    'add': {'v': False},
                    'depth': {'v': 0.5},
                    'width': {'v': 0.1},
                    'place': {'v': -0.5},
                },
            },
            'pencil-skirt': {
                'length': {'v': float(length_v)},
                'rise': {'v': float(rise_v)},
                'flare': {'v': 1.0},
                'low_angle': {'v': 0},
                'front_slit': {'v': 0},
                'back_slit': {'v': 0},
                'left_slit': {'v': 0},
                'right_slit': {'v': 0},
                'style_side_cut': {'v': None},
            },
            # Stubs for unused sections
            'shirt': {
                'strapless': {'v': False},
                'length': {'v': 1.2},
                'width': {'v': 1.05},
                'flare': {'v': 1.0},
            },
            'collar': {
                'f_collar': {'v': 'CircleNeckHalf'},
                'b_collar': {'v': 'CircleNeckHalf'},
                'width': {'v': 0.2},
                'fc_depth': {'v': 0.4},
                'bc_depth': {'v': 0},
                'fc_angle': {'v': 95},
                'bc_angle': {'v': 95},
                'f_bezier_x': {'v': 0.3},
                'f_bezier_y': {'v': 0.55},
                'b_bezier_x': {'v': 0.15},
                'b_bezier_y': {'v': 0.1},
                'f_flip_curve': {'v': False},
                'b_flip_curve': {'v': False},
                'component': {
                    'style': {'v': None},
                    'depth': {'v': 7},
                    'lapel_standing': {'v': False},
                    'hood_depth': {'v': 1},
                    'hood_length': {'v': 1},
                },
            },
            'sleeve': {
                'sleeveless': {'v': True},
                'armhole_shape': {'v': 'ArmholeCurve'},
                'length': {'v': 0.3},
                'connecting_width': {'v': 0.2},
                'end_width': {'v': 1.0},
                'sleeve_angle': {'v': 10},
                'opening_dir_mix': {'v': 0.1},
                'standing_shoulder': {'v': False},
                'standing_shoulder_len': {'v': 5.0},
                'connect_ruffle': {'v': 1},
                'smoothing_coeff': {'v': 0.25},
                'cuff': {
                    'type': {'v': None},
                    'top_ruffle': {'v': 1},
                    'cuff_len': {'v': 0.1},
                    'skirt_fraction': {'v': 0.5},
                    'skirt_flare': {'v': 1.2},
                    'skirt_ruffle': {'v': 1.0},
                },
            },
            'left': {
                'enable_asym': {'v': False},
                'shirt': {
                    'strapless': {'v': False},
                    'width': {'v': 1.05},
                    'flare': {'v': 1.0},
                },
                'collar': {
                    'f_collar': {'v': 'CircleNeckHalf'},
                    'b_collar': {'v': 'CircleNeckHalf'},
                    'width': {'v': 0.5},
                    'fc_angle': {'v': 95},
                    'bc_angle': {'v': 95},
                    'f_bezier_x': {'v': 0.5},
                    'f_bezier_y': {'v': 0.3},
                    'b_bezier_x': {'v': 0.5},
                    'b_bezier_y': {'v': 0.3},
                    'f_flip_curve': {'v': False},
                    'b_flip_curve': {'v': False},
                },
                'sleeve': {
                    'sleeveless': {'v': True},
                    'armhole_shape': {'v': 'ArmholeCurve'},
                    'length': {'v': 0.3},
                    'connecting_width': {'v': 0.2},
                    'end_width': {'v': 1.0},
                    'sleeve_angle': {'v': 10},
                    'opening_dir_mix': {'v': 0.2},
                    'standing_shoulder': {'v': False},
                    'standing_shoulder_len': {'v': 5.0},
                    'connect_ruffle': {'v': 1},
                    'smoothing_coeff': {'v': 0.25},
                    'cuff': {
                        'type': {'v': None},
                        'top_ruffle': {'v': 1},
                        'cuff_len': {'v': 0.1},
                        'skirt_fraction': {'v': 0.5},
                        'skirt_flare': {'v': 1.2},
                        'skirt_ruffle': {'v': 1.0},
                    },
                },
            },
        }

        return design

    def map_outfit(self, shirt_measurements, bottom_measurements,
                   bottom_type='pants', collar_type='CircleNeckHalf',
                   skirt_type='SkirtCircle'):
        """Map production measurements for a full outfit (upper + lower).

        Since GarmentCode simulates one cloth mesh at a time, this returns
        two separate design dicts — one for the shirt and one for the bottom.
        Each is generated and simulated independently, then combined.

        Args:
            shirt_measurements: dict for the shirt (see map_shirt).
            bottom_measurements: dict for the bottom (see map_pants or map_skirt).
            bottom_type: 'pants' or 'skirt'.
            collar_type: Collar shape for the shirt.
            skirt_type: Skirt class name (only used if bottom_type='skirt').

        Returns:
            tuple: (shirt_design, bottom_design) — two separate design dicts.
        """
        shirt_design = self.map_shirt(shirt_measurements, collar_type=collar_type)

        if bottom_type == 'pants':
            bottom_design = self.map_pants(bottom_measurements)
        elif bottom_type == 'skirt':
            bottom_design = self.map_skirt(bottom_measurements, skirt_type=skirt_type)
        else:
            raise ValueError(f"Unknown bottom_type: {bottom_type}. Use 'pants' or 'skirt'.")

        return shirt_design, bottom_design

    def print_mapping_summary(self, garment_type, garment_measurements, design):
        """Print a summary of the measurement-to-parameter mapping for verification."""
        b = self.body
        print(f"\n{'='*60}")
        print(f"  Production → Design Mapping Summary ({garment_type})")
        print(f"{'='*60}")

        if garment_type == 'shirt':
            w = design['shirt']['width']['v']
            l = design['shirt']['length']['v']
            fl = design['shirt']['flare']['v']

            print(f"\n  Body measurements:")
            print(f"    Bust circumference:    {b['bust']:.1f} cm")
            print(f"    Shoulder-to-waist:     {b['waist_line']:.1f} cm")
            print(f"    Arm length:            {b['arm_length']:.1f} cm")
            print(f"    Armscye depth:         {b['_armscye_depth']:.1f} cm")

            print(f"\n  Production measurements → Design parameters:")
            print(f"    Bust circ:   {garment_measurements['bust_circumference']:.1f} cm  →  shirt.width  = {w:.3f}")
            print(f"    Length:      {garment_measurements['length']:.1f} cm  →  shirt.length = {l:.3f}")

            hem = garment_measurements.get('hem_circumference', garment_measurements['bust_circumference'])
            print(f"    Hem circ:    {hem:.1f} cm  →  shirt.flare  = {fl:.3f}")

            if not design['sleeve']['sleeveless']['v']:
                sl = design['sleeve']['length']['v']
                avail = b['arm_length'] - b['_armscye_depth']
                print(f"    Sleeve len:  {garment_measurements.get('sleeve_length', 0):.1f} cm  →  sleeve.length = {sl:.3f}")
                print(f"      (available arm = {b['arm_length']:.1f} - {b['_armscye_depth']:.1f} = {avail:.1f} cm)")

            print(f"\n  Expected garment dimensions (reconstructed):")
            print(f"    Bust circ:   {w * b['bust']:.1f} cm")
            print(f"    Length:      {l * b['waist_line']:.1f} cm")
            print(f"    Hem circ:    {fl * w * b['bust']:.1f} cm")
            if not design['sleeve']['sleeveless']['v']:
                avail = b['arm_length'] - b['_armscye_depth']
                print(f"    Sleeve len:  {sl * avail:.1f} cm")

        elif garment_type == 'pants':
            w = design['pants']['width']['v']
            l = design['pants']['length']['v']
            fl = design['pants']['flare']['v']
            print(f"\n  Body measurements:")
            print(f"    Hip circumference:     {b['hips']:.1f} cm")
            print(f"    Leg length:            {b['_leg_length']:.1f} cm")
            print(f"\n  Production measurements → Design parameters:")
            print(f"    Hip circ:    {garment_measurements.get('hip_circumference', garment_measurements['waist_circumference']*1.1):.1f} cm  →  pants.width = {w:.3f}")
            print(f"    Length:      {garment_measurements['length']:.1f} cm  →  pants.length = {l:.3f}")
            print(f"    Flare:                    →  pants.flare = {fl:.3f}")
            print(f"\n  Expected garment dimensions (reconstructed):")
            print(f"    Hip circ:    {w * b['hips']:.1f} cm")
            print(f"    Length:      {l * b['_leg_length']:.1f} cm")

        print(f"{'='*60}\n")

        if garment_type == 'outfit':
            # Handled externally by printing shirt + pants summaries separately
            pass


def create_body_yaml_from_measurements(measurements, output_path):
    """Create a GarmentCode body YAML from raw person measurements.

    Args:
        measurements: dict with person body measurements in cm.
            Required keys: height, bust, waist, hips, shoulder_w
            Optional keys: arm_length, head_l, back_width, waist_line,
                hips_line, neck_w, wrist, etc.
        output_path: Where to save the YAML file.

    Returns:
        Path to the saved file.
    """
    # Apply defaults for optional measurements using anthropometric ratios
    h = measurements['height']
    defaults = {
        'head_l': h * 0.153,            # ~1/6.5 of height
        'arm_length': h * 0.314,         # typical arm length ratio
        'arm_pose_angle': 45.0,          # standard A-pose angle
        'back_width': measurements['bust'] * 0.477,  # back ~ 47.7% of bust
        'waist_line': h * 0.214,         # shoulder-to-waist
        'hips_line': h * 0.137,          # waist-to-hip
        'neck_w': h * 0.11,
        'wrist': measurements.get('bust', 100) * 0.166,
        'shoulder_w': h * 0.212,
        'shoulder_incl': 21.7,           # degrees
        'armscye_depth': 12.9,           # cm
        'underbust': measurements['bust'] * 0.864,
        'bust_line': 25.7,
        'waist_back_width': measurements.get('waist', 80) * 0.464,
        'hip_back_width': measurements.get('hips', 100) * 0.530,
        'leg_circ': measurements.get('hips', 100) * 0.582,
        'bust_points': 16.9,
        'bum_points': 18.2,
        'crotch_hip_diff': 8.8,
        'waist_over_bust_line': 40.6,
    }

    body = {}
    for key in defaults:
        body[key] = measurements.get(key, defaults[key])

    # Always use provided values for required fields
    for key in ['height', 'bust', 'waist', 'hips']:
        body[key] = measurements[key]
    if 'shoulder_w' in measurements:
        body['shoulder_w'] = measurements['shoulder_w']

    body_yaml = {'body': body}

    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        yaml.dump(body_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Body YAML saved to {output_path}")
    return output_path


# =====================================================
#  Pipeline helpers
# =====================================================

def generate_pattern(name, body_yaml_path, design, output_base):
    """Generate a sewing pattern and save to disk.

    Returns:
        Path to the output folder containing the specification JSON.
    """
    from datetime import datetime
    from assets.garment_programs.meta_garment import MetaGarment
    from assets.bodies.body_params import BodyParameters

    body = BodyParameters(body_yaml_path)
    piece = MetaGarment(name, body, design)
    pattern = piece.assembly()

    if piece.is_self_intersecting():
        print(f'WARNING: {piece.name} has self-intersecting panels')

    folder = pattern.serialize(
        Path(output_base),
        tag='_' + datetime.now().strftime("%y%m%d-%H-%M-%S"),
        to_subfolder=True,
        with_3d=False, with_text=False, view_ids=False,
        with_printable=True
    )
    body.save(folder)

    design_out = {'design': design}
    with open(Path(folder) / 'design_params.yaml', 'w') as f:
        yaml.dump(design_out, f, default_flow_style=False)

    print(f'Pattern saved to {folder}')
    return Path(folder)


def simulate_pattern(pattern_folder, sim_config='./assets/Sim_props/default_sim_props.yaml',
                     body_name='mean_all', smpl_body=False):
    """Run cloth simulation on a generated pattern.

    Returns:
        Path to the output folder containing the sim.obj.
    """
    import pygarment.data_config as data_config
    from pygarment.meshgen.boxmeshgen import BoxMesh
    from pygarment.meshgen.simulation import run_sim
    from pygarment.meshgen.sim_config import PathCofig

    pattern_folder = Path(pattern_folder)
    spec_files = list(pattern_folder.glob('*_specification.json'))
    if not spec_files:
        raise FileNotFoundError(f"No *_specification.json in {pattern_folder}")
    spec_path = spec_files[0]
    garment_name, _, _ = spec_path.stem.rpartition('_')

    props = data_config.Properties(sim_config)
    props.set_section_stats('sim', fails={}, sim_time={}, spf={}, fin_frame={},
                            body_collisions={}, self_collisions={})
    props.set_section_stats('render', render_time={})

    sys_props = data_config.Properties('./system.json')
    paths = PathCofig(
        in_element_path=spec_path.parent,
        out_path=sys_props['output'],
        in_name=garment_name,
        body_name=body_name,
        smpl_body=smpl_body,
        add_timestamp=True
    )

    print(f"\nSimulating {garment_name}...")
    garment_box_mesh = BoxMesh(paths.in_g_spec, props['sim']['config']['resolution_scale'])
    garment_box_mesh.load()
    garment_box_mesh.serialize(
        paths, store_panels=False, uv_config=props['render']['config']['uv_texture'])
    props.serialize(paths.element_sim_props)

    run_sim(
        garment_box_mesh.name, props, paths,
        save_v_norms=False, store_usd=False,
        optimize_storage=False, verbose=False
    )
    props.serialize(paths.element_sim_props)

    print(f"Simulation output: {paths.out_el}")
    return paths.out_el


def combine_outfit_meshes(shirt_sim_folder, bottom_sim_folder, body_obj_path,
                          output_folder):
    """Combine shirt + bottom sim meshes and body into a single output.

    Saves:
        - outfit_combined.obj  (body + shirt + bottom with named groups)
        - outfit_garments.obj  (shirt + bottom only, no body)
        - outfit_render_front.png / outfit_render_back.png
    """
    import os
    import platform
    if platform.system() == 'Linux':
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    import trimesh
    import pyrender
    from PIL import Image

    shirt_sim_folder = Path(shirt_sim_folder)
    bottom_sim_folder = Path(bottom_sim_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load meshes
    shirt_obj = list(shirt_sim_folder.glob('*_sim.obj'))[0]
    bottom_obj = list(bottom_sim_folder.glob('*_sim.obj'))[0]

    shirt_mesh = trimesh.load(str(shirt_obj), process=False)
    bottom_mesh = trimesh.load(str(bottom_obj), process=False)
    body_mesh = trimesh.load(str(body_obj_path), process=False)

    # Scale body to cm if needed (body OBJs are often in meters)
    if body_mesh.vertices.max() < 3.0:
        body_mesh.vertices = body_mesh.vertices * 100.0

    # Apply y-shift to match simulation (simulation shifts so min_y >= 0)
    min_y = body_mesh.vertices[:, 1].min()
    if min_y < 0:
        shift = abs(min_y)
        body_mesh.vertices[:, 1] += shift

    # --- Save garments-only OBJ (shirt + bottom, no body) ---
    shirt_v = np.array(shirt_mesh.vertices)
    shirt_f = np.array(shirt_mesh.faces)
    bottom_v = np.array(bottom_mesh.vertices)
    bottom_f = np.array(bottom_mesh.faces)

    garments_path = output_folder / 'outfit_garments.obj'
    with open(garments_path, 'w') as f:
        f.write("# Outfit: shirt + bottom garment meshes\n\n")
        for v in shirt_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for v in bottom_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        f.write("\ng shirt\n")
        for face in shirt_f:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        f.write("\ng bottom\n")
        bottom_f_offset = bottom_f + len(shirt_v)
        for face in bottom_f_offset:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Garments-only mesh saved to {garments_path}")

    # --- Save full combined OBJ (body + shirt + bottom) ---
    body_v = np.array(body_mesh.vertices)
    body_f = np.array(body_mesh.faces)

    combined_path = output_folder / 'outfit_combined.obj'
    with open(combined_path, 'w') as f:
        f.write("# Outfit: body + shirt + bottom\n\n")
        for v in body_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for v in shirt_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for v in bottom_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        f.write("\ng body\n")
        for face in body_f:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        f.write("\ng shirt\n")
        shirt_f_offset = shirt_f + len(body_v)
        for face in shirt_f_offset:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        f.write("\ng bottom\n")
        bottom_f_offset2 = bottom_f + len(body_v) + len(shirt_v)
        for face in bottom_f_offset2:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Combined mesh (body+outfit) saved to {combined_path}")

    # --- Render front/back images of the outfit ---
    # Scale to meters for pyrender (matching GarmentCode's render convention)
    body_mesh_r = trimesh.Trimesh(body_v / 100, body_f)
    shirt_mesh_r = trimesh.Trimesh(shirt_v / 100, shirt_f)
    bottom_mesh_r = trimesh.Trimesh(bottom_v / 100, bottom_f)

    body_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.0, 0.0, 0.0, 1.0),
        metallicFactor=0.658, roughnessFactor=0.5
    )
    shirt_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.85, 0.85, 0.95, 1.0),
        metallicFactor=0.1, roughnessFactor=0.8
    )
    bottom_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.3, 0.35, 0.5, 1.0),
        metallicFactor=0.1, roughnessFactor=0.8
    )

    pr_body = pyrender.Mesh.from_trimesh(body_mesh_r, material=body_material)
    pr_shirt = pyrender.Mesh.from_trimesh(shirt_mesh_r, material=shirt_material, smooth=True)
    pr_bottom = pyrender.Mesh.from_trimesh(bottom_mesh_r, material=bottom_material, smooth=True)

    for side in ['front', 'back']:
        scene = pyrender.Scene(bg_color=(1., 1., 1., 0.))
        scene.add(pr_body)
        scene.add(pr_shirt)
        scene.add(pr_bottom)

        # Camera setup (borrowed from GarmentCode's render logic)
        bbox_center = body_mesh_r.bounds.mean(axis=0)
        diag = np.linalg.norm(body_mesh_r.bounds[1] - body_mesh_r.bounds[0])
        distance = 1.5 * diag / (2 * np.tan(np.radians(25)))
        cam_pos = bbox_center.copy()
        cam_pos[2] += distance

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 6.)
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = cam_pos

        # Tilt
        rx = np.radians(-15)
        rot_x = np.array([
            [1, 0, 0, 0], [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0], [0, 0, 0, 1]])
        ry = np.radians(20)
        rot_y = np.array([
            [np.cos(ry), 0, np.sin(ry), 0], [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0], [0, 0, 0, 1]])
        cam_pose = rot_x @ cam_pose
        cam_pose = rot_y @ cam_pose
        if side == 'back':
            rot_180 = np.array([
                [np.cos(np.pi), 0, np.sin(np.pi), 0], [0, 1, 0, 0],
                [-np.sin(np.pi), 0, np.cos(np.pi), 0], [0, 0, 0, 1]])
            cam_pose = rot_180 @ cam_pose

        scene.add(camera, pose=cam_pose)

        # Lights
        for lpos in [[1.6, 1.5, 1.2], [1.3, 1.9, -2.5], [-2.8, 1.3, 2.3],
                      [0.2, 1.8, 3.5], [-2.7, 1.4, -1.3]]:
            light = pyrender.PointLight(color=[1., 1., 1.], intensity=80.)
            lp = np.eye(4)
            lp[:3, 3] = lpos
            scene.add(light, pose=lp)

        renderer = pyrender.OffscreenRenderer(1080, 1080)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        img_path = output_folder / f'outfit_render_{side}.png'
        Image.fromarray(color).save(str(img_path), "PNG")
        renderer.delete()
        print(f"Render saved to {img_path}")

    return output_folder


# =====================================================
#  CLI entry point
# =====================================================

if __name__ == '__main__':
    import argparse
    from datetime import datetime

    from pygarment.data_config import Properties

    parser = argparse.ArgumentParser(
        description='Generate garment patterns from production measurements',
        epilog='Supported garment types:\n' + SUPPORTED_TYPES,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--mode', '-m', choices=['shirt', 'pants', 'skirt', 'outfit', 'types'],
        default='shirt',
        help='Generation mode: single garment or full outfit (default: shirt)'
    )
    parser.add_argument(
        '--body', '-b', type=str, default='./assets/bodies/mean_all.yaml',
        help='Path to body measurements YAML'
    )
    parser.add_argument(
        '--body_name', type=str, default='mean_all',
        help='Body model name for simulation (must match an OBJ in assets/bodies/)'
    )
    parser.add_argument(
        '--sim_config', '-s', type=str, default='./assets/Sim_props/default_sim_props.yaml',
        help='Path to simulation config YAML'
    )
    parser.add_argument(
        '--simulate', action='store_true',
        help='Also run the cloth simulation after pattern generation'
    )
    parser.add_argument(
        '--smpl_body', action='store_true',
        help='Use SMPL body segmentation instead of default'
    )
    args = parser.parse_args()

    if args.mode == 'types':
        print(SUPPORTED_TYPES)
        exit(0)

    body_yaml = args.body
    sys_props = Properties('./system.json')
    output_base = sys_props['output']

    mapper = ProductionToDesign(body_yaml)

    if args.mode == 'shirt':
        # ----- Single shirt -----
        garment_measurements = {
            'bust_circumference': 108.0,
            'length': 68.0,
            'sleeve_length': 20.0,
            'hem_circumference': 108.0,
        }
        design = mapper.map_shirt(garment_measurements)
        mapper.print_mapping_summary('shirt', garment_measurements, design)

        folder = generate_pattern('custom_shirt', body_yaml, design, output_base)
        if args.simulate:
            simulate_pattern(folder, args.sim_config, args.body_name, args.smpl_body)

    elif args.mode == 'pants':
        # ----- Single pants -----
        garment_measurements = {
            'waist_circumference': 84.0,
            'length': 75.0,
            'hip_circumference': 104.0,
            'leg_opening': 40.0,
        }
        design = mapper.map_pants(garment_measurements)
        mapper.print_mapping_summary('pants', garment_measurements, design)

        folder = generate_pattern('custom_pants', body_yaml, design, output_base)
        if args.simulate:
            simulate_pattern(folder, args.sim_config, args.body_name, args.smpl_body)

    elif args.mode == 'skirt':
        # ----- Single skirt -----
        garment_measurements = {
            'waist_circumference': 84.0,
            'length': 60.0,
            'hem_circumference': 200.0,
        }
        design = mapper.map_skirt(garment_measurements)
        mapper.print_mapping_summary('skirt', garment_measurements, design)

        folder = generate_pattern('custom_skirt', body_yaml, design, output_base)
        if args.simulate:
            simulate_pattern(folder, args.sim_config, args.body_name, args.smpl_body)

    elif args.mode == 'outfit':
        # ----- Full outfit: shirt + pants -----
        shirt_measurements = {
            'bust_circumference': 108.0,
            'length': 68.0,
            'sleeve_length': 20.0,
            'hem_circumference': 108.0,
        }
        pants_measurements = {
            'waist_circumference': 84.0,
            'length': 75.0,
            'hip_circumference': 104.0,
            'leg_opening': 40.0,
        }

        shirt_design, pants_design = mapper.map_outfit(
            shirt_measurements, pants_measurements, bottom_type='pants')

        mapper.print_mapping_summary('shirt', shirt_measurements, shirt_design)
        mapper.print_mapping_summary('pants', pants_measurements, pants_design)

        # Generate both patterns
        shirt_folder = generate_pattern('outfit_shirt', body_yaml, shirt_design, output_base)
        pants_folder = generate_pattern('outfit_pants', body_yaml, pants_design, output_base)

        if args.simulate:
            # Simulate each garment independently
            shirt_sim_folder = simulate_pattern(
                shirt_folder, args.sim_config, args.body_name, args.smpl_body)
            pants_sim_folder = simulate_pattern(
                pants_folder, args.sim_config, args.body_name, args.smpl_body)

            # Combine the results
            body_obj = Path('./assets/bodies') / f'{args.body_name}.obj'
            outfit_folder = Path(output_base) / f'outfit_{datetime.now().strftime("%y%m%d-%H-%M-%S")}'
            combine_outfit_meshes(
                shirt_sim_folder, pants_sim_folder,
                body_obj, outfit_folder
            )
            print(f"\nOutfit pipeline complete! Results in {outfit_folder}")
        else:
            print(f"\nPatterns generated. Run with --simulate to drape and combine:")
            print(f"  python production_to_design.py --mode outfit --simulate")
