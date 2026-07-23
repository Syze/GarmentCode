import pygarment as pyg


def effective_waist_hips(body, section):
    """Return (waist, hips, waist_back_width, hip_back_width) for a skirt
    section, applying `target_waist` / `target_hips` if set on the section.

    Matches the pants/shirt convention of multiplying body fields inline
    with design knobs (cf. pants.py:width_v) — body is read, never mutated.
    Back-width terms are scaled proportionally so the front/back ratio is
    preserved.
    """
    target_waist = section.get('target_waist', {}).get('v') if section else None
    target_hips = section.get('target_hips', {}).get('v') if section else None

    if target_waist is not None:
        eff_waist = float(target_waist)
        eff_w_back = float(body['waist_back_width'] * (target_waist / body['waist']))
    else:
        eff_waist = body['waist']
        eff_w_back = body['waist_back_width']

    if target_hips is not None:
        eff_hips = float(target_hips)
        eff_h_back = float(body['hip_back_width'] * (target_hips / body['hips']))
    else:
        eff_hips = body['hips']
        eff_h_back = body['hip_back_width']

    return eff_waist, eff_hips, eff_w_back, eff_h_back


class BaseBodicePanel(pyg.Panel):
    """Base class for bodice panels that defines expected interfaces and common functions"""
    def __init__(self, name, body, design) -> None:
        super().__init__(name)
        self.body = body
        self.design = design
        
        self.interfaces = {
            'outside': object(),
            'inside': object(),
            'shoulder': object(),
            'bottom': object(),

            'shoulder_corner': object(),
            'collar_corner': object(),
        }

    def get_width(self, level):
        """Return the panel width at a given level (excluding darts)
           * Level is counted from the top of the panel
        
        NOTE: for fitted bodice, the request is only valid for values between 0 and bust_level
        """
        # NOTE: this evaluation assumes that the top edge width is the same as bodice shoulder width 
        side_edge = self.interfaces['outside'].edges[-1]

        x = side_edge.end[0] - side_edge.start[0]
        y = side_edge.end[1] - side_edge.start[1]

        # If the orientation of the edge is "looking down"
        # instead of "looking up" as calculations above expect, flip the values
        if y < 0:
            x, y = -x, -y

        return (level * x / y) + self.body['shoulder_w'] / 2


class BaseBottoms(pyg.Component):
    """A base class for all the bottom components.
        Defines common elements: 
        * List of interfaces
        * Presence of the rise value
    """
    def __init__(self, body, design, tag='', rise=None) -> None:
        """Base bottoms initialization
        """
        super().__init__(
            self.__class__.__name__ if not tag else f'{self.__class__.__name__}_{tag}')
        
        self.body = body
        self.design = design
        self.rise = rise
        
        # Set of interfaces that need to be implemented
        self.interfaces = {
            'top': object()
        }
        
    def get_rise(self):
        """Return a rise value for a given component"""
        return self.rise
    
    def eval_rise(self, rise):
        """Evaluate updated hip and waist-related measurements,
            corresponding to the provided rise value.
            rise > 1.0 extends above natural waist (high-rise);
            width stays at waist measurement, only depth grows.
        """
        waist, hips = self.body['waist'], self.body['hips']
        hips_level = self.body['hips_line']
        self.adj_hips_depth = rise * hips_level
        # Clamp interpolation factor at 1.0: for rise > 1.0,
        # width stays at waist (narrowest point)
        width_factor = min(rise, 1.0)
        self.adj_waist = pyg.utils.lin_interpolation(hips, waist, width_factor)

        self_adj_back_waist = pyg.utils.lin_interpolation(
            self.body['hip_back_width'], self.body['waist_back_width'], width_factor)

        # Band-match: when the waistband is drafted to a production waist LARGER
        # than the body-interpolated width (oversized/relaxed fits), widen the
        # garment's top edge to match the band, like real jean construction
        # (band sewn 1:1, no gathered seam). Otherwise the band is sewn onto a
        # tube up to ~9cm smaller than itself and the excess circumference
        # stands off the body as a permanent waist gap (verified on the
        # HM_Jeans big-waist group; all POMs stay GT). No-op when the
        # production waist <= the interpolated width (all normal/under-drafted
        # garments) or when no waistband is defined.
        try:
            band = self.design['waistband']['waist']['v'] * self.body['waist']
        except (KeyError, TypeError):
            band = None
        if band and band > self.adj_waist:
            self_adj_back_waist *= band / self.adj_waist
            self.adj_waist = band

        return self.adj_waist, self.adj_hips_depth, self_adj_back_waist

class StackableSkirtComponent(BaseBottoms):
    """
        Abstract definition of a skirt that can be stacked with other stackable skirts
        (connecting bottom to another StackableSkirtComponent())
    """

    def __init__(self, body, design, tag='', length=None, rise=None, slit=True, top_ruffles=True) -> None:
        """Skirt initialization

            Extra parameters (length, sleets, top_ruffles) 
            can be used to overwrite parameters in design dictionary
        """
        super().__init__(body, design, tag, rise=rise)
        
        pass

        # Set of interfaces that need to be implemented
        self.interfaces = {
            'top': object(),
            'bottom_f': object(),
            'bottom_b': object(),
            'bottom': object()
        }


class BaseBand(pyg.Component):
    def __init__(self, body, design, tag='', rise=None) -> None:
        """Base band initialization
        """
        super().__init__(
            self.__class__.__name__ if not tag else f'{self.__class__.__name__}_{tag}')
        self.body = body
        self.design = design
        self.rise = rise
        
        # Set of interfaces that need to be implemented
        self.interfaces = {
            'top': object(),
            'bottom': object()
        }

    def length(self):
        """Base length == Length of a first panel"""
        return self._get_subcomponents()[0].length()