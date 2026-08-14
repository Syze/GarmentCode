import numpy as np
from copy import copy
from argparse import Namespace
from scipy.spatial.transform import Rotation as R

from pygarment.pattern.core import BasicPattern
from pygarment.garmentcode.base import BaseComponent
from pygarment.garmentcode.edge import Edge, EdgeSequence, CircleEdge
from pygarment.garmentcode.utils import close_enough, vector_align_3D
from pygarment.garmentcode.operators import cut_into_edge
from pygarment.garmentcode.interface import Interface


class Panel(BaseComponent):
    """ A Base class for defining a Garment component corresponding to a single
        flat fiece of fabric
    
    Defined as a collection of edges on a 2D grid with specified 3D placement
        (world coordinates)
    
    NOTE: All operations methods return 'self' object to allow sequential
        applications

    """
    def __init__(self, name, label='') -> None:
        """Base class for panel creations
            * Name: panel name. Expected to be a unique identifier of a panel object
            * label: additional panel label (non-unique)
        """
        super().__init__(name)

        self.label = label
        self.translation = np.zeros(3)
        self.rotation = R.from_euler('XYZ', [0, 0, 0])  # zero rotation
        # NOTE: initiating with empty sequence allows .append() to it safely
        self.edges = EdgeSequence()

    # Info
    def pivot_3D(self):
        """Pivot point of a panel in 3D"""
        return self.point_to_3D([0, 0])

    def length(self, longest_dim=False):
        """Length of a panel element in cm
        
            Defaults the to the vertical length of a 2D bounding box
            * longest_dim -- if set, returns the longest dimention out of the bounding box dimentions
        """
        bbox = self.bbox()

        x = abs(bbox[1][0] - bbox[0][0])
        y = abs(bbox[1][1] - bbox[0][1])

        return max(x, y) if longest_dim else y

    def is_self_intersecting(self):
        """Check whether the panel has self-intersection"""
        edge_curves = []
        for e in self.edges:
            if isinstance(e, CircleEdge):  
                # NOTE: Intersections for Arcs (Circle edge) fails in svgpathtools:
                # They are not well implemented in svgpathtools, see
                # https://github.com/mathandy/svgpathtools/issues/121
                # https://github.com/mathandy/svgpathtools/blob/fcb648b9bb9591d925876d3b51649fa175b40524/svgpathtools/path.py#L1960
                # Hence using linear approximation for robustness:
                edge_curves += [eseg.as_curve() for eseg in e.linearize(n_verts_inside=10)]
            else:
                edge_curves.append(e.as_curve())

        # NOTE: simple pairwise checks of edges
        for i1 in range(0, len(edge_curves)):
           for i2 in range(i1 + 1, len(edge_curves)):
                intersect_t = edge_curves[i1].intersect(edge_curves[i2])
                
                # Check exceptions -- intersection at the vertex
                for i in range(len(intersect_t)): 
                    t1, t2 = intersect_t[i]
                    if t2 < t1:
                        t1, t2 = t2, t1
                    if close_enough(t1, 0) and close_enough(t2, 1):
                        intersect_t[i] = None
                intersect_t = [el for el in intersect_t if el is not None]

                if intersect_t:  # Any other case of intersections
                    return True      
        return False

    # ANCHOR - Operations -- update object in-place 
    def set_panel_label(self, label: str, overwrite=True): 
        """If overwrite is not enabled, only updates the label if it's empty."""
        if not self.label or overwrite:
            self.label = label

    def set_pivot(self, point_2d, replicate_placement=False):
        """Specify 2D point w.r.t. panel local space
            to be used as pivot for translation and rotation

        Parameters:
            * point_2d -- desired point 2D point w.r.t current pivot (origin)
                of panel local space
            * replicate_placement -- will replicate the location of the panel
                as it was before pivot change
                default - False (no adjustment, the panel may "jump" in 3D)
        """
        point_2d = copy(point_2d)  # Remove unwanted object reference 
                                   # In case an actual vertex was used as a target point

        if replicate_placement:
            self.translation = self.point_to_3D(point_2d)
            # FIXME Replicate rotation

        # UPD vertex locations relative to new pivot
        # BUG (found 2026-08-06, not fixed here -- behaviour change would move
        # every panel of every existing garment): the vertices are shifted by
        # int() of the pivot while `translation` above is set from its EXACT
        # value, so the panel is displaced in world by the FRACTIONAL part of
        # point_2d (up to ~1cm per axis). Harmless-looking for an unrotated
        # panel -- it is a constant offset -- but assembly() calls this on every
        # panel, so a panel carrying a placement ROTATION has that offset rotated
        # with it and drifts off its shared seams. Measured on a hinged blazer
        # lapel: 0.23cm of seam gap unrotated, 1.22cm once folded 150 degrees.
        # Workaround used by assets/garment_programs/hyperdrop.py: pre-shift the
        # panel so edges[0].start is integral (or exactly (0, 0)), which makes
        # the truncation exact and the displacement vanish.
        for v in self.edges.verts():
            v[0] -= int(point_2d[0])
            v[1] -= int(point_2d[1])

    def top_center_pivot(self):
        """One of the most useful pivots 
            is the one in the middle of the top edge of the panel 
        """
        vertices = np.asarray(self.edges.verts())

        # out of 2D bounding box sides' midpoints choose the one that is
        # highest in 3D
        top_right = vertices.max(axis=0)
        low_left = vertices.min(axis=0)
        mid_x = (top_right[0] + low_left[0]) / 2
        mid_y = (top_right[1] + low_left[1]) / 2
        mid_points_2D = [
            [mid_x, top_right[1]], 
            [mid_x, low_left[1]],
            [top_right[0], mid_y],
            [low_left[0], mid_y]
        ]
        mid_points_3D = np.vstack(tuple(
            [self.point_to_3D(coords) for coords in mid_points_2D]
        ))
        top_mid_point = mid_points_3D[:, 1].argmax()

        self.set_pivot(mid_points_2D[top_mid_point])
        return self

    def translate_by(self, delta_vector):
        """Translate panel by a vector"""
        self.translation = self.translation + np.array(delta_vector)
        # NOTE: One may also want to have autonorm only on the assembly?
        self.autonorm()
        return self
    
    def translate_to(self, new_translation):
        """Set panel translation to be exactly that vector"""
        self.translation = np.asarray(new_translation)
        self.autonorm()
        return self
    
    def rotate_by(self, delta_rotation: R):
        """Rotate panel by a given rotation
            * delta_rotation: scipy rotation object
        """
        self.rotation = delta_rotation * self.rotation
        self.autonorm()
        return self

    def rotate_to(self, new_rot: R):
        """Set panel rotation to be exactly the given rotation
            * new_rot: scipy rotation object
        """   
        if not isinstance(new_rot, R):
            raise ValueError(f'{self.__class__.__name__}::ERROR::Only accepting rotations in scipy format')
        self.rotation = new_rot
        self.autonorm()
        return self

    def rotate_align(self, vector):
        """Set panel rotation s.t. it's norm is aligned with a given 3D
        vector"""

        vector = np.asarray(vector)
        vector = vector / np.linalg.norm(vector)
        n = self.norm()
        self.rotate_by(vector_align_3D(n, vector))
        return self

    def center_x(self):
        """Adjust translation over x s.t. the center of the panel is aligned
        with the Y axis (center of the body)"""

        center_3d = self.point_to_3D(self._center_2D())
        self.translation[0] += -center_3d[0]
        return self

    def autonorm(self):
        """Update right/wrong side orientation, s.t. the normal of the surface
            looks outside he world origin,
            taking into account the shape and the global position.
        
            This should provide correct panel orientation in most cases.

            NOTE: for best results, call autonorm after translation
                specification
        """
        norm_dr = self.norm()

        # Fail loudly rather than silently skipping the flip: a non-finite normal used
        # to make the comparison below False, leaving a mirrored panel inside-out.
        if not np.all(np.isfinite(norm_dr)):
            raise ValueError(
                f'{self.name}::ERROR::panel normal is not finite ({norm_dr}); cannot '
                'determine right/wrong side')

        # NOTE: Nothing happens if self.translation is zero
        if np.dot(norm_dr, self.translation) < 0: 
            # Swap if wrong  
            self.edges.reverse()

    def mirror(self, axis=None):
        """Swap this panel with its mirror image
        
            Axis specifies 2D axis to swap around: Y axis by default
        """
        if axis is None:
            axis = [0, 1]
        # Case Around Y
        if close_enough(axis[0], tol=1e-4):  # reflection around Y

            # Vertices
            self.edges.reflect([0, 0], [0, 1])
            
            # Position
            self.translation[0] *= -1

            # Rotations
            curr_euler = self.rotation.as_euler('XYZ')
            curr_euler[1] *= -1  
            curr_euler[2] *= -1  
            self.rotate_to(R.from_euler('XYZ', curr_euler))  

            # Fix right/wrong side
            self.autonorm()
        else:
            # TODO Any other axis
            raise NotImplementedError(f'{self.name}::ERROR::Mirrowing over arbitrary axis is not implemented')
        return self

    def add_dart(self, dart_shape, edge, offset, right=True, edge_seq=None, int_edge_seq=None, ):
        """ Shortcut for adding a dart to a panel: 
            * Performs insertion of the dart_shape in the given edge (parameters are the same 
                as in pyp.ops.cut_into_edge)
            * Creates stitch to connect the dart sides
            * Modifies edge_sequnces with full set (edge_seq) or only the interface part (int_edge_seq) 
                of the created edges, if those are provided
            
            Returns new edges after insertion, and the interface part (excludes dart edges)
        """
        edges_new, dart_edges, int_new = cut_into_edge(
            dart_shape, 
            edge, 
            offset=offset,
            right=right)
        
        self.stitching_rules.append(
            (Interface(self, dart_edges[0]), Interface(self, dart_edges[1])))
        
        # Update the edges if given
        if edge_seq is not None: 
            edge_seq.substitute(edge, edges_new)
            edges_new = edge_seq
        if int_edge_seq is not None:
            int_edge_seq.substitute(edge, int_new)
            int_new = int_edge_seq

        return edges_new, int_new

    # ANCHOR - Build the panel -- get serializable representation
    def assembly(self):
        """Convert panel into serialazable representation

         NOTE: panel EdgeSequence is assumed to be a single loop of edges
        """
        # FIXME Some panels have weird resulting alignemnt when th
        # is pivot setup is removed -- there is a bug somewhere

        # always start from zero for consistency between panels
        self.set_pivot(self.edges[0].start, replicate_placement=True)

        # Basics
        # BUG (found 2026-08-06, not fixed here): scipy's UPPERCASE 'XYZ' is the
        # INTRINSIC convention, i.e. the matrix Rx.Ry.Rz. The consumer of this
        # field, pygarment/meshgen/boxmeshgen.py, rebuilds it with
        # pattern/rotation.py::euler_xyz_to_R, which is Rz.Ry.Rx -- scipy's
        # LOWERCASE (extrinsic) 'xyz'. The two agree only when the rotation is
        # about a single axis, which is true of every panel in this repo, so the
        # mismatch has never surfaced. Give a panel a genuine 3-axis rotation and
        # it is simulated in the wrong orientation: a blazer lapel folded about a
        # diagonal roll line was written intending world x[-4.25, 11.50] and the
        # boxmesh placed it at x[-27.72, -4.79], mirrored across the hinge.
        # Correct serialization for the mesh path is
        #     self.rotation.as_euler('xyz', degrees=True)
        # See _rot_for_meshgen() in assets/garment_programs/hyperdrop.py, which
        # works around it locally rather than changing this line for all garments.
        panel = Namespace(
            translation=self.translation.tolist(),
            rotation=self.rotation.as_euler('XYZ', degrees=True).tolist(),
            vertices=[self.edges[0].start],
            edges=[])

        for i in range(len(self.edges)):
            vertices, edge = self.edges[i].assembly()

            # add new vertices
            if panel.vertices[-1] == vertices[0]:   # We care if both point to the same vertex location, not necessarily the same vertex object
                vert_shift = len(panel.vertices) - 1  # first edge vertex = last vertex already in the loop
                panel.vertices += vertices[1:] 
            else: 
                vert_shift = len(panel.vertices)
                panel.vertices += vertices

            # upd vertex references in edges according to new vertex ids in 
            # the panel vertex loop
            edge['endpoints'] = [id + vert_shift for id in edge['endpoints']]
                
            edge_shift = len(panel.edges)  # before adding new ones
            self.edges[i].geometric_id = edge_shift   # remember the mapping of logical edge to geometric id in panel loop
            panel.edges.append(edge)

        # Check closing of the loop and upd vertex reference for the last edge
        if panel.vertices[-1] == panel.vertices[0]:
            panel.vertices.pop()
            panel.edges[-1]['endpoints'][-1] = 0

        # Add panel label, if known
        if self.label:
            panel.label = self.label

        spattern = BasicPattern()
        spattern.name = self.name
        spattern.pattern['panels'] = {self.name: vars(panel)}

        # Assembly stitching info (panel might have inner stitches)
        spattern.pattern['stitches'] = self.stitching_rules.assembly()

        return spattern

    # ANCHOR utils
    def _center_2D(self, n_verts_inside = 3):
        """Approximate Location of the panel center. 

            NOTE: uses crude linear approximation for curved edges,
            n_verts_inside = number of vertices (excluding the start
            and end vertices) used to create a linearization of an edge
        """
        # NOTE: assuming that edges are organized in a loop and share vertices
        lin_edges = EdgeSequence([e.linearize(n_verts_inside)
                                  for e in self.edges])
        verts = lin_edges.verts()

        return np.mean(verts, axis=0)

    def point_to_3D(self, point_2d):
        """Calculate 3D location of a point given in the local 2D plane """
        point_2d = np.asarray(point_2d)
        if len(point_2d) == 2:
            point_2d = np.append(point_2d, 0)

        point_3d = self.rotation.apply(point_2d)
        point_3d += self.translation
        return point_3d

    def norm(self):
        """Normal direction for the current panel using bounding box"""

        # To make norm evaluation work for non-convex panels
        # Determine points located on bounding box (b_verts_2d), compute
        # norm of consecutive b_verts_3d and the b_verts_3d mean (b_center_3d),
        # then weight the norms.
        # The dominant norm direction should be the correct one

        _, b_verts_2d = self.edges.bbox()
        b_verts_3d = [self.point_to_3D(bv_2d) for bv_2d in b_verts_2d]
        b_center_3d = np.mean((b_verts_3d), axis=0)

        norms = []
        num_b_verts_3d = len(b_verts_3d)
        for i in range(num_b_verts_3d):
            vert_0 = b_verts_3d[i]
            vert_1 = b_verts_3d[(i + 1) % num_b_verts_3d]
            # Pylance + NP error for unreachanble code -- see https://github.com/numpy/numpy/issues/22146
            # Works ok for numpy 1.23.4+
            norm = np.cross(vert_0 - b_center_3d, vert_1 - b_center_3d)
            n_len = np.linalg.norm(norm)
            if n_len < 1e-9:
                # Degenerate pair: the two bounding-box vertices coincide, or they
                # are collinear with the centre, so the cross product vanishes.
                # Dividing here produced NaN, which then poisoned avg_norm and made
                # autonorm()'s `dot(norm, translation) < 0` test evaluate False --
                # silently skipping the right/wrong-side fix, so a mirrored panel kept
                # its flipped winding and its triangles faced inward. Symptom: pant
                # legs inside-out in the boxmesh, seams stitching a front face to a
                # back face, and the sim thrashing to the frame cap. Seen on barrel
                # legs, where the single-arc outseam can put a curve extremum exactly
                # on another bbox-touching vertex. Skip the pair instead.
                continue
            norms.append(norm / n_len)
        if not norms:
            raise ValueError(
                f'{self.name}::ERROR::panel normal is undefined: every bounding-box '
                'vertex pair is degenerate')

        # Current norm direction
        avg_norm = sum(norms) / len(norms)

        if close_enough(np.linalg.norm(avg_norm), 0):
            # Indecisive averaging, so using just one of the norms
            # NOTE: sometimes happens on thin arcs
            avg_norm = norms[0]   
            if self.verbose:
                print(f'{self.__class__.__name__}::{self.name}::WARNING::Norm evaluation failed, assigning norm based on the first edge')

        final_norm = avg_norm / np.linalg.norm(avg_norm)

        # solve float errors
        for i, ni in enumerate(final_norm):
            if np.isclose([ni], [0.0]):
                final_norm[i] = 0.0

        return final_norm

    def bbox(self):
        """Evaluate 2D bounding box"""
        # Using curve linearization for more accurate approximation of bbox
        lin_edges = EdgeSequence([e.linearize() for e in self.edges])
        verts_2d = np.asarray(lin_edges.verts())

        return verts_2d.min(axis=0), verts_2d.max(axis=0)


    def bbox3D(self):
        """Evaluate 3D bounding box of the current panel"""

        # Using curve linearization for more accurate approximation of bbox
        lin_edges = EdgeSequence([e.linearize() for e in self.edges])
        verts_2d = lin_edges.verts()
        verts_3d = np.asarray([self.point_to_3D(v) for v in verts_2d])

        return verts_3d.min(axis=0), verts_3d.max(axis=0)