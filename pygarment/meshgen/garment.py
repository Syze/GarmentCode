import contextlib
import igl
import json
import os
import pickle
import numpy as np
import yaml

import warp as wp

import warp.sim.render
from warp.sim.utils import implicit_laplacian_smoothing
import warp.collision.panel_assignment as assign
from warp.sim.collide import count_self_intersections, count_body_cloth_intersections
from warp.sim.integrator_xpbd import replace_mesh_points

# Custom
from pygarment.meshgen.sim_config import PathCofig, SimConfig
from pygarment.pattern.core import BasicPattern

class Cloth:
    def __init__(self, 
                 name, config: SimConfig, paths: PathCofig, 
                 caching=False):

        self.caching = caching   # Saves intermediate frames, extra logs, etc.
        self.paths = paths
        self.name = name
        self.config = config

        self.sim_fps = config.sim_fps
        self.sim_substeps = config.sim_substeps
        self.zero_gravity_steps = config.zero_gravity_steps
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.usd_frame_time = 0.0 
        self.sim_use_graph = wp.get_device().is_cuda
        self.device = wp.get_device() if wp.get_device().is_cuda else 'cpu' 
        self.frame = -1

        self.c_scale = 1.0
        self.b_scale = 100.0
        self.body_path = paths.in_body_obj
        
        # collision resolution options
        self.enable_body_smoothing = config.enable_body_smoothing
        self.enable_cloth_reference_drag = config.enable_cloth_reference_drag

        # Build the stage -- model object, colliders, etc.
        self.build_stage(config)

        # -------- Final model settings ----------
        # NOTE: global_viscous_damping: (damping_factor, min_vel_damp, max_vel) 
        # apply damping when vel > min_vel_damp, and clamp vel below max_vel after damping
        # TODO Remove after refactoring Euler integrator
        self.model.global_viscous_damping = wp.vec3(
            (config.global_damping_factor, config.global_damping_effective_velocity, config.global_max_velocity))
        self.model.particle_max_velocity = config.global_max_velocity
        
        self.model.ground = config.ground  

        self.model.global_collision_filter = config.enable_global_collision_filter
        self.model.cloth_reference_drag = self.enable_cloth_reference_drag
        self.model.cloth_reference_margin = config.cloth_reference_margin
        self.model.cloth_reference_k = config.cloth_reference_k
        self.model.cloth_reference_watertight_whole_shape_index = 0
        self.model.enable_particle_particle_collisions = config.enable_particle_particle_collisions
        self.model.enable_triangle_particle_collisions = config.enable_triangle_particle_collisions
        self.model.enable_edge_edge_collisions = config.enable_edge_edge_collisions
        self.model.attachment_constraint = config.enable_attachment_constraint

        self.model.soft_contact_margin = config.soft_contact_margin
        self.model.soft_contact_ke = config.soft_contact_ke
        self.model.soft_contact_kd = config.soft_contact_kd
        self.model.soft_contact_kf = config.soft_contact_kf
        self.model.soft_contact_mu = config.soft_contact_mu

        self.model.particle_ke = config.particle_ke
        self.model.particle_kd = config.particle_kd
        self.model.particle_kf = config.particle_kf
        self.model.particle_mu = config.particle_mu
        self.model.particle_cohesion = config.particle_cohesion
        self.model.particle_adhesion = config.particle_adhesion

        #self.integrator = wp.sim.SemiImplicitIntegrator() #intialize semi-implicit time-integrator
        self.integrator = wp.sim.XPBDIntegrator() #intialize semi-implicit time-integrator
        self.state_0 = self.model.state() #returns state object for model (holds all *time-varying* data for a model)
        self.state_1 = self.model.state() #i.e. body/particle positions and velocities

        # Auto pre-shift: override initial particle positions to the compressed
        # state computed in build_stage. Edge rest lengths in the model are
        # still natural, so springs pull fabric toward natural during sim.
        if getattr(self, '_compressed_initial_q', None) is not None:
            n = self._compressed_initial_q.shape[0]
            arr = wp.array(self._compressed_initial_q.astype(np.float32),
                           dtype=wp.vec3, device=self.device)
            wp.copy(self.state_0.particle_q, arr, count=n)
            wp.copy(self.state_1.particle_q, arr, count=n)

        if self.caching:
            self.renderer = wp.sim.render.SimRenderer(self.model, str(paths.usd), scaling=1.0)

        if self.sim_use_graph:
            self.create_graph()

        self.last_verts = None
        self._verts_np = None
        self._verts_frame = -1
        self._last_verts_frame = None
        self.current_verts = wp.array.numpy(self.state_0.particle_q)

    @property
    def current_verts(self):
        """Particle positions as numpy, fetched from the device on first use
        in a frame. The readback is a full GPU->CPU sync, so with
        config.lazy_vert_fetch it happens only when a consumer asks."""
        if self._verts_np is None:
            self._verts_np = wp.array.numpy(self.state_0.particle_q)
            self._verts_frame = self.frame
        return self._verts_np

    @current_verts.setter
    def current_verts(self, value):
        self._verts_np = value
        self._verts_frame = self.frame

    def build_stage(self, config):

        builder = wp.sim.ModelBuilder(gravity=0.0)
        # --------------- Load body info -----------------
        body_vertices, body_indices, body_faces = self.load_obj(self.paths.in_body_obj)
        body_seg = self.read_json(self.paths.body_seg) 

        body_vertices = body_vertices * self.b_scale
        self.shift_y = self.get_shift_param(body_vertices)

        if self.shift_y:
            body_vertices[:, 1] = body_vertices[:, 1] + self.shift_y

        self.v_body = body_vertices
        self.f_body = body_faces
        self.body_indices = body_indices

        # -------------- Load cloth ------------
        cloth_vertices, cloth_indices, cloth_faces = self.load_obj(self.paths.g_box_mesh)
        cloth_seg_dict = assign.read_segmentation(self.paths.g_mesh_segmentation)
        self.cloth_seg_dict = cloth_seg_dict
        stitching_vertices = cloth_seg_dict["stitch"] if 'stitch' in cloth_seg_dict.keys() else []

        cloth_vertices = cloth_vertices * self.c_scale
        if self.shift_y:
            cloth_vertices[:, 1] = cloth_vertices[:, 1] + self.shift_y

        # Auto pre-shift: when the garment's lower hem would start below the
        # floor (e.g. pants with inseam exceeding the body's crotch height),
        # compress the lower-body portion vertically so the cuff starts ~1 cm
        # above floor for clean stitching. The compression is only applied to
        # the INITIAL particle state (state_0.particle_q below) — natural
        # cloth_vertices are still passed to the builder so edge rest lengths
        # remain at the pattern's true lengths. During sim the springs pull
        # the fabric back toward natural, with excess inseam puddling at the
        # floor under gravity.
        # Compression spreads across the whole garment below body_crotch_Y,
        # so per-edge shrink is small and fabric doesn't get stuck on itself.
        # Disable per-config via sim props options.auto_pre_lift_enabled = false
        # or by setting auto_pre_lift_clearance = 0.
        self._compressed_initial_q = None
        try:
            auto_clearance = float(
                getattr(config, 'auto_pre_lift_clearance', 2.0) or 0)
            auto_enabled = bool(
                getattr(config, 'auto_pre_lift_enabled', True))
        except Exception:
            auto_clearance, auto_enabled = 1.0, True
        if auto_enabled and auto_clearance > 0:
            cloth_min_y = float(cloth_vertices[:, 1].min())
            if cloth_min_y < 0:
                # SMPL landmark needs no measurements; the yaml-based waist
                # formula is only the fallback for non-SMPL bodies.
                body_crotch_Y = self._crotch_y_smpl_or(None)
                if body_crotch_Y is None:
                    with open(self.paths.in_body_mes, 'r') as _bf:
                        bd = yaml.safe_load(_bf)['body']
                    wl = bd.get('_waist_level',
                                bd['height'] - bd['head_l'] - bd['waist_line'])
                    body_crotch_Y = (wl - bd.get('hips_line', 22.0)
                                     - bd.get('crotch_hip_diff', 8.0))
                mask = cloth_vertices[:, 1] < body_crotch_Y
                if mask.any():
                    vy_min = cloth_vertices[mask, 1].min()
                    if vy_min < 0:
                        span_orig = body_crotch_Y - vy_min
                        span_new = body_crotch_Y - auto_clearance
                        if span_orig > 0.1 and span_new > 0.1:
                            ratio = span_new / span_orig
                            compressed = cloth_vertices.copy()
                            compressed[mask, 1] = (
                                auto_clearance
                                + (compressed[mask, 1] - vy_min) * ratio)
                            self._compressed_initial_q = compressed

        self.v_cloth_init = cloth_vertices
        self.f_cloth = cloth_faces

        #Load ground truth stitching lengths
        if not self.paths.g_orig_edge_len.exists():
            orig_lens_dict = None
            print("no original length dict found")
        else:
            with open(self.paths.g_orig_edge_len, 'rb') as file:
                orig_lens_dict = pickle.load(file)

        cloth_pos = (0.0, 0.0, 0.0)
        cloth_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.degrees(0.0)) #no rotation, but orientation of cloth in world space

        builder.add_cloth_mesh_sewing_spring(
            pos=cloth_pos,
            rot=cloth_rot,
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            vertices=cloth_vertices,
            indices=cloth_indices,
            resolution_scale=config.resolution_scale,
            orig_lens=orig_lens_dict,
            stitching_vertices=stitching_vertices,
            density=config.garment_density,
            edge_ke=config.garment_edge_ke,
            edge_kd=config.garment_edge_kd,
            tri_ke=config.garment_tri_ke,
            tri_ka=config.garment_tri_ka,
            tri_kd=config.garment_tri_kd,
            tri_drag=config.garment_tri_drag,
            tri_lift=config.garment_tri_lift,
            radius=config.garment_radius,
            add_springs=True,
            spring_ke=config.spring_ke,
            spring_kd=config.spring_kd,
        )

        # ------------ Add a body -----------      
        if self.enable_body_smoothing:
            # Starts sim from smoothed-out body and slowly restores original details
            smoothing_total_smoothing_factor = config.smoothing_total_smoothing_factor
            smoothing_num_steps = config.smoothing_num_steps
            smoothing_recover_start_frame = config.smoothing_recover_start_frame
            smoothing_frame_gap_between_steps = config.smoothing_frame_gap_between_steps
            smoothing_step_size = smoothing_total_smoothing_factor / smoothing_num_steps
            self.body_smoothing_frames = [smoothing_recover_start_frame + smoothing_frame_gap_between_steps*i for i in range(smoothing_num_steps + 1)]
            self.body_smoothing_vertices_list = []
            self.body_smoothing_vertices_list = implicit_laplacian_smoothing(body_vertices, body_indices.reshape(-1, 3), 
                                                                             step_size=smoothing_step_size, 
                                                                             iters=smoothing_num_steps)
            body_vertices = self.body_smoothing_vertices_list.pop()
            self.body_smoothing_frames.pop()
            self.body_indices = body_indices
            self.body_vertices_device_buffer = wp.array(body_vertices, dtype=wp.vec3, device=self.device)
            self.v_body = body_vertices
        
        # Foot-shrink logic disabled: collapsing foot vertices during zero-gravity
        # caused cuff clipping for short bodies (cuff settled inside the foot Y
        # zone; frame-150 snap-back punched fabric out, producing self-collisions
        # and slow settling). With body collision filters in place the panels
        # converge fine without shrinking the feet.
        # foot_y_threshold = body_vertices[:, 1].max() * 0.06  # ~6% of height ≈ foot top
        # self._foot_mask = body_vertices[:, 1] < foot_y_threshold
        # self._body_verts_original = body_vertices.copy()
        # if self._foot_mask.any():
        #     # Ankle ring: vertices just above the foot
        #     ankle_band = (body_vertices[:, 1] >= foot_y_threshold) & \
        #                  (body_vertices[:, 1] < foot_y_threshold * 1.5)
        #     if ankle_band.any():
        #         ankle_center = body_vertices[ankle_band].mean(axis=0)
        #     else:
        #         ankle_center = body_vertices[~self._foot_mask].mean(axis=0)
        #     # Shrink foot towards ankle center (keep 5% spread for valid tris)
        #     body_vertices = body_vertices.copy()
        #     body_vertices[self._foot_mask] = (
        #         ankle_center + 0.05 * (body_vertices[self._foot_mask] - ankle_center)
        #     )
        self._foot_mask = np.zeros(len(body_vertices), dtype=bool)

        if not hasattr(self, 'body_vertices_device_buffer'):
            self.body_vertices_device_buffer = wp.array(
                body_vertices, dtype=wp.vec3, device=self.device)

        self.body_mesh = wp.sim.Mesh(body_vertices, body_indices)

        # ----- Body pose animation (optional) -----
        # Step the body through a pose sequence during the sim so the cloth
        # tracks it. Stored as DELTAS off the current (base) body, so any global
        # shift the sim applied to the body is preserved. Frames are scaled
        # METRES->cm to match body_vertices (which were scaled by b_scale).
        self._pose_anim = bool(getattr(config, 'pose_animation_npy', None))
        if self._pose_anim:
            seq = np.load(config.pose_animation_npy)  # (N, n_verts, 3) metres
            self._pose_body_base = np.asarray(body_vertices, dtype=np.float64)
            self._pose_delta = (seq - seq[0]).astype(np.float64) * self.b_scale
            gap = int(config.pose_animation_frame_gap)
            start = int(config.pose_animation_start_frame)
            self._pose_anim_frames = [start + gap * i for i in range(len(seq))]
            self._pose_idx = 0
            pose_end = self._pose_anim_frames[-1]
            # End the sim a fixed N frames after the last pose step. Gravity is on
            # during reposing so the garment is settled by pose_end; extra settling
            # only lets it creep down a tilted pose. Also raise the static-check
            # floor so it can't stop mid-animation.
            settle = int(getattr(config, 'pose_animation_settle_frames', 30))
            config.max_sim_steps = pose_end + settle
            config.min_sim_steps = max(getattr(config, 'min_sim_steps', 0) or 0, pose_end + 1)
            print(f'  POSE ANIMATION: {len(seq)} frames, step every {gap} from '
                  f'frame {start} (ends {pose_end}); sim auto-stops at '
                  f'{config.max_sim_steps} (+{settle} settle)')

        body_pos = wp.vec3(0.0, 0, 0.0)
        body_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.degrees(0.0))


        # Cloth-body segemntation
        # (stdout silenced: panel_assignment prints every panel:label pair)
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            cloth_reference_labels, body_parts = assign.panel_assignment(
                            cloth_seg_dict, cloth_vertices, cloth_indices, wp.transform(cloth_pos, cloth_rot),
                            body_seg, body_vertices, body_indices, wp.transform(body_pos, body_rot),
                            device=self.device,
                            panel_init_labels=self._load_panel_labels(),
                            strategy='closest',
                            merge_two_legs=True,
                            smpl_body=self.paths.use_smpl_seg
                            )
        
        face_filters, particle_filter = [], []
        if config.enable_body_collision_filters:
            v_connectivity = self._build_vert_connectivity(cloth_vertices, cloth_indices)
            # Arm filter for the skirts
            face_filters.append(assign.create_face_filter(
                body_vertices, body_indices, body_seg, ['left_arm', 'right_arm', 'arms'], smpl_body=self.paths.use_smpl_seg))
            particle_filter = assign.assign_face_filter_points(
                cloth_reference_labels, 
                ['left_leg', 'right_leg', 'legs'],
                filter_id=0,
                vert_connectivity=v_connectivity
            )

            # Overall filter that ignores internal geometry (and arms for lower-body garments
            # or upper-body garments with sleeves — in both cases the torso/body-labeled
            # particles should not collide with the arms; sleeve-labeled particles stay
            # unfiltered so they still collide with the arms and wrap correctly).
            body_filter_parts = ['face_internal']
            has_leg_panels = any(l in ('left_leg', 'right_leg', 'legs') for l in cloth_reference_labels)
            has_arm_panels = any(l in ('left_arm', 'right_arm', 'arm', 'arms') for l in cloth_reference_labels)
            is_lower_body = has_leg_panels and not has_arm_panels
            if is_lower_body or has_arm_panels:
                body_filter_parts.extend(['left_arm', 'right_arm', 'arms'])
            face_filters.append(assign.create_face_filter(
                body_vertices, body_indices, body_seg, body_filter_parts, smpl_body=self.paths.use_smpl_seg))
            particle_filter = assign.assign_face_filter_points(
                cloth_reference_labels,
                ['body'],
                filter_id=1,
                vert_connectivity=v_connectivity,
                current_vertex_filter=particle_filter
            )

            if is_lower_body:
                # Assign remaining unfiltered particles (stitch verts) to arm filter
                # so no part of pants collides with arms
                for i in range(len(particle_filter)):
                    if particle_filter[i] == -1:
                        particle_filter[i] = 0

        self.body_shape_index = 0   # Body is the first collider object to be added
        builder.add_shape_mesh(
            body=-1,
            mesh=self.body_mesh,
            pos=body_pos,
            rot=body_rot,
            scale=wp.vec3(1.0,1.0,1.0), #performed body scaling above
            thickness=config.body_thickness,  
            mu=config.body_friction,
            face_filters=face_filters if face_filters else [[]],
            model_particle_filter_ids = particle_filter,
        )

        # ----- Extra static collider (layered outfits) -------
        # An already-draped inner garment, frozen as a static obstacle so the
        # current (outer) garment drapes over it. Loaded in cm (body frame), so
        # NO b_scale is applied (unlike the body OBJ, which is in meters).
        static_obj = getattr(config, 'static_collision_obj', None)
        if static_obj:
            sc_v, sc_inds, _ = self.load_obj(static_obj)
            self.static_collider_mesh = wp.sim.Mesh(sc_v, sc_inds)
            sc_mu = getattr(config, 'static_collision_friction', None)
            if sc_mu is None:
                sc_mu = config.body_friction
            builder.add_shape_mesh(
                body=-1,
                mesh=self.static_collider_mesh,
                pos=body_pos,
                rot=body_rot,
                scale=wp.vec3(1.0, 1.0, 1.0),
                thickness=config.body_thickness,
                mu=sc_mu,
            )
            print(f'  Added static collider ({len(sc_v)} verts) from {static_obj}')

        # ----- Attachment constraint -------

        if config.enable_attachment_constraint:
            self._add_attachment_labels(builder, config)

        # ----- Panel-to-panel springs ("tacks") -----
        # Opt-in, driven entirely by the pattern spec, so no existing garment is
        # affected. add_attachment above pins particles to STATIC world points,
        # which cannot hold one panel to another; a spring can, and its rest
        # length is the frame-0 distance, so it preserves the relationship the
        # pattern was built with (e.g. a pre-folded lapel staying folded).
        self._add_panel_springs(builder)

        # ----- Global collision resolution error ---- 
        for part in body_parts:
            part_v, part_inds = assign.extract_submesh(body_vertices, body_indices, body_parts[part])
            builder.add_cloth_reference_shape_mesh(
                mesh = wp.sim.Mesh(part_v, part_inds),
                name = part,
                pos = body_pos,
                rot = body_rot,
                scale = (1.0,1.0,1.0) #performed body scaling above
            )
        # NOTE: has a side-effect of filling up model.particle_reference_label array 
        self.body_parts_names2index = builder.add_cloth_reference_labels(
            cloth_reference_labels, 
            [   # NOTE: Not adding drag between legs and the body as it's useless and contradicts attachment
                ['left_arm', 'body'], 
                ['right_arm', 'body'], 
                ['left_leg', 'right_leg'],
                ['left_arm', 'left_leg'], 
                ['right_arm', 'left_leg'], 
                ['left_arm', 'right_leg'], 
                ['right_arm', 'right_leg'], 
                ['left_arm', 'legs'], 
                ['right_arm', 'legs'], 
            ]
        )  

        # Enlarge the edge-edge contact buffer for heavy-self-contact garments
        # (deep barrels) that overflow it at stitch init and crash with a CUDA
        # illegal-memory-access. builder.edge_contact_max is per-spring and gets
        # multiplied by spring_count in finalize(); scale it by the configured
        # factor (default 1 = warp default). Must be set BEFORE finalize.
        _ecm = getattr(config, 'edge_contact_mult', 1) or 1
        if _ecm != 1:
            builder.edge_contact_max = int(builder.edge_contact_max * _ecm)

        # ------- Finalize --------------
        self.model: wp.sim.Model = builder.finalize(device = self.device) #data is transferred to warp tensors, object used in simulation
        # Read by the XPBD integrator to stride the self-contact solve; a Warp
        # without that edit simply ignores the attribute.
        self.model.self_contact_interval = getattr(config, 'self_contact_interval', 1)

    def _crotch_y_smpl_or(self, formula):
        """Body crotch Y: SMPL crotch landmark vertex 1210 (floor-relative) if
        the body mesh is SMPL (6890 verts), else the provided formula value.
        Matches the placement anchor in run_custom_pants — the measurement
        formula underestimates the true SMPL crotch by a height-dependent
        ~3-4.5cm."""
        try:
            bv = self.v_body
            if bv is not None and len(bv) == 6890:
                return float(bv[1210, 1] - bv[:, 1].min())
        except Exception:
            pass
        return formula

    def _add_panel_springs(self, builder):
        """Spring pairs of labelled vertex sets together (panel-to-panel tacks).

        Reads properties.panel_springs from the pattern spec:
            [[label_a, label_b, ke, kd], ...]
        where label_a/label_b are EDGE labels set by the garment program; each
        vertex of set A is sprung to its nearest vertex in set B. Rest length is
        the frame-0 separation, so the constraint holds the built geometry rather
        than collapsing the two edges together the way a stitch would.
        """
        try:
            with open(self.paths.g_specs) as f:
                pairs = json.load(f).get('properties', {}).get('panel_springs')
        except (OSError, ValueError):
            return
        if not pairs:
            return
        try:
            with open(self.paths.g_vert_labels, 'r') as f:
                vlabels = yaml.load(f, Loader=yaml.SafeLoader) or {}
        except OSError:
            return
        q = np.asarray(builder.particle_q, dtype=float)
        # Record where our springs start: builder.spring_indices already holds
        # the cloth's own edge springs (add_cloth_mesh(add_springs=True)).
        first = len(builder.spring_indices) // 2
        total = 0
        for entry in pairs:
            la, lb = entry[0], entry[1]
            ke = float(entry[2]) if len(entry) > 2 else 1.0e4
            kd = float(entry[3]) if len(entry) > 3 else 1.0e1
            # A label resolves from vertex_labels.yaml (edge labels) OR from the
            # mesh segmentation (whole panels). Using a whole panel as the target
            # set matters: paired against a sparse 9-vertex edge every spring
            # collapsed onto the same few verts and fanned out; against the full
            # panel each source vertex finds the vert directly beneath it, so the
            # springs come out short and mutually parallel.
            A = vlabels.get(la) or self.cloth_seg_dict.get(la) or []
            B = vlabels.get(lb) or self.cloth_seg_dict.get(lb) or []
            if not A or not B:
                print(f'  Panel springs: skipped {la} <-> {lb} '
                      f'(labelled verts {len(A)}/{len(B)})')
                continue
            Bq = q[B]
            lens, targets = [], set()
            for i in A:
                j = B[int(np.argmin(np.linalg.norm(Bq - q[i], axis=1)))]
                builder.add_spring(i, j, ke, kd, 0.0)
                lens.append(float(np.linalg.norm(q[i] - q[j])))
                targets.add(j)
                total += 1
            print(f'  Panel springs: {la} <-> {lb}, {len(A)} springs '
                  f'(ke={ke:g}, kd={kd:g}) len {min(lens):.2f}-{max(lens):.2f}cm, '
                  f'{len(targets)} distinct targets')
        if total:
            print(f'  Panel springs: {total} added')
            self._dump_panel_springs(builder, q, first)

    def _dump_panel_springs(self, builder, q, first=0):
        """Append the springs to the BOXMESH as visible sliver triangles.

        MeshLab (and most viewers) ignore OBJ `l` polyline elements, so a spring
        written as `l i j` is invisible. Each spring is therefore written as a
        degenerate-thin FACE `f i j k`, where k is an existing vertex near the
        segment's midpoint. Both endpoints are cloth particles and particle index
        == boxmesh vertex index, so no `v` lines are added: the vertex count, the
        segmentation and the solver are all untouched. It does add one face per
        spring to this output file (an artefact only -- boxmeshgen rewrites the
        boxmesh from scratch on every run).
        """
        idx = np.asarray(builder.spring_indices, dtype=int).reshape(-1, 2)[first:]
        if not len(idx):
            return
        try:
            with open(self.paths.g_box_mesh, 'a') as f:
                f.write('# panel-to-panel springs, one sliver face per spring\n')
                for a, b in idx:
                    a, b = int(a), int(b)
                    mid = (q[a] + q[b]) / 2.0
                    d = np.linalg.norm(q - mid, axis=1)
                    d[a] = d[b] = np.inf
                    c = int(np.argmin(d))
                    f.write(f'f {a + 1} {b + 1} {c + 1}\n')
            print(f'  Panel springs: {len(idx)} sliver faces appended to '
                  f'{self.paths.g_box_mesh.name}')
        except OSError as e:
            print(f'  Panel springs: could not annotate boxmesh ({e})')

    def _add_attachment_labels(self, builder, config):
        with open(self.paths.in_body_mes, 'r') as file:
            body_dict = yaml.load(file, Loader=yaml.SafeLoader)['body']
        with open(self.paths.g_vert_labels, 'r') as f:
            vertex_labels = yaml.load(f, Loader=yaml.SafeLoader)
        
        lables_present = False
        for i, attach_label in enumerate(config.attachment_labels):     
            if attach_label in vertex_labels.keys() and len(vertex_labels[attach_label]) > 0:
                constaint_verts = vertex_labels[attach_label]
                if attach_label == 'lower_interface':
                    lables_present = True
                    target_y = self._resolve_attachment_target_y(
                        config.attachment_target_y, body_dict)
                    builder.add_attachment(
                        constaint_verts,
                        wp.vec3(0, target_y, 0),
                        wp.vec3(0., 1., 0.),    # Vertical attachment
                        stiffness = config.attachment_stiffness[i],
                        damping = config.attachment_damping[i]
                    )
                elif attach_label == 'right_collar':
                    lables_present = True
                    
                    neck_w = body_dict['neck_w'] - 2
                    builder.add_attachment(
                        constaint_verts, 
                        wp.vec3(-neck_w / 2, 0, 0),   
                        wp.vec3(1., 0., 0.),    # Horizontal attachment
                        stiffness = config.attachment_stiffness[i],
                        damping = config.attachment_damping[i]
                    )
                elif attach_label == 'left_collar':
                    lables_present = True
                    
                    neck_w = body_dict['neck_w'] - 2
                    builder.add_attachment(
                        constaint_verts, 
                        wp.vec3(neck_w / 2, 0, 0),   
                        wp.vec3(-1., 0., 0.),    # Horizontal attachment
                        stiffness = config.attachment_stiffness[i],
                        damping = config.attachment_damping[i]
                    )
                elif attach_label in ('right_lapel', 'left_lapel'):
                    # Lapel labels recognized but no constraint applied —
                    # the pre-folded geometry + collar attachment + body
                    # collision handle lapel positioning
                    lables_present = True
                elif attach_label in ('right_fold_line', 'left_fold_line'):
                    lables_present = True
                    # Pin fold-line vertices at their initial Y position
                    avg_y = float(np.mean([self.v_cloth_init[v][1]
                                           for v in constaint_verts]))
                    builder.add_attachment(
                        constaint_verts,
                        wp.vec3(0, avg_y, 0),
                        wp.vec3(0., 1., 0.),    # Y-direction constraint
                        stiffness=config.attachment_stiffness[i],
                        damping=config.attachment_damping[i]
                    )
                elif attach_label in ('right_armhole', 'left_armhole'):
                    lables_present = True

                    # Pin armhole vertices at their initial Y position so
                    # sleeves stay at the shoulder instead of sliding down.
                    avg_y = float(np.mean([self.v_cloth_init[v][1]
                                           for v in constaint_verts]))
                    builder.add_attachment(
                        constaint_verts,
                        wp.vec3(0, avg_y, 0),
                        wp.vec3(0., 1., 0.),    # Vertical attachment
                        stiffness=config.attachment_stiffness[i],
                        damping=config.attachment_damping[i]
                    )
                elif attach_label == 'strapless_top':
                    lables_present = True

                    # Attach under arm
                    level = body_dict['height'] - body_dict['head_l'] - body_dict['armscye_depth']
                    builder.add_attachment(
                        constaint_verts,
                        wp.vec3(0, level, 0),
                        wp.vec3(0., 1., 0.),    # Vertical attachment
                        stiffness = config.attachment_stiffness[i],
                        damping = config.attachment_damping[i]
                    )
                elif attach_label == 'pants_crotch':
                    lables_present = True

                    # Pin pants crotch vertices to body crotch Y so a lifted
                    # garment (via ankle_clearance) doesn't pull the crotch
                    # into the torso during the zero-gravity phase. Released
                    # after attachment_frames.
                    if '_waist_level' in body_dict:
                        waist_level = body_dict['_waist_level']
                    else:
                        waist_level = (body_dict['height']
                                       - body_dict['head_l']
                                       - body_dict['waist_line'])
                    hips_line = body_dict.get('hips_line', 22.0)
                    crotch_hip_diff = body_dict.get('crotch_hip_diff', 8.0)
                    target_y = self._crotch_y_smpl_or(
                        waist_level - hips_line - crotch_hip_diff)
                    builder.add_attachment(
                        constaint_verts,
                        wp.vec3(0, target_y, 0),
                        wp.vec3(0., 1., 0.),    # Vertical attachment only
                        stiffness=config.attachment_stiffness[i],
                        damping=config.attachment_damping[i]
                    )
                else:
                    print(f'{self.name}::WARNING::Requested attachment label {attach_label} '
                          'is not supported. Skipped')
                    continue

        if not lables_present:
            # Loaded garment is not labeled -- update config
            config.enable_attachment_constraint = False
            config.update_min_steps()
            print(f'{self.name}::WARNING::Requested attachment labels {config.attachment_labels} '
                  'are not present. Attachment is turned off'
                )

    @staticmethod
    def _resolve_attachment_target_y(target, body_dict):
        """Resolve attachment target Y from config value and body measurements.

        Args:
            target: None (use waist), a numeric value, or a named landmark string.
            body_dict: body measurement dict from YAML.

        Supported landmarks:
            'waist'    — natural waist level
            'hip_bone' — iliac crest, ~36% of the way from waist to full hip
            'hip'      — full hip (widest point)
        """
        if '_waist_level' in body_dict:
            waist_level = body_dict['_waist_level']
        else:
            waist_level = body_dict['height'] - body_dict['head_l'] - body_dict['waist_line']

        if target is None or target == 'waist':
            return waist_level

        if isinstance(target, (int, float)):
            return float(target)

        hips_line = body_dict.get('hips_line', 0)

        if target == 'hip_bone':
            return waist_level - hips_line * 0.36
        elif target == 'hip':
            return waist_level - hips_line
        else:
            raise ValueError(
                f'Unknown attachment target "{target}". '
                'Use a number, or one of: waist, hip_bone, hip')

    def _load_panel_labels(self):
        pattern = BasicPattern(self.paths.g_specs)

        labels = {}
        for name, panel in pattern.pattern['panels'].items():
            labels[name] = panel['label'] if 'label' in panel else ''

        return labels     

    def _sim_frame_with_substeps(self):
        """Basic scheme for simulating a frame update"""

        wp.sim.collide(self.model, self.state_0, self.sim_dt * self.sim_substeps)  # Generates contact points for the particles and rigid bodies
        # in the model, to be used in the contact dynamics kernel of the integrator
        # launches kernels

        for s in range(self.sim_substeps):
            self.state_0.clear_forces()  # set particle and body forces to 0s
            self.integrator.simulate(self.model, self.state_0, self.state_1,
                                     self.sim_dt)  # calculate semi-implicit Euler step
            # launches kernels and calculates new particle (and body) positions and velocities
            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)  # swap prev, new state

    def create_graph(self):
        # create update graph
        wp.capture_begin()  # Captures all subsequent kernel launches and memory operations on CUDA devices.
        
        self._sim_frame_with_substeps()

        self.graph = wp.capture_end()  # returns a handle to a CUDA graph object that can be launched with :func:`~warp.capture_launch()`
        # do not capture kernel launches anymore

    def update(self, frame):
        with wp.ScopedTimer("simulate", print=False, active=True):
            if self.model.enable_particle_particle_collisions:
                # FIXME: Produces cuda errors when activated together with "enable_cloth_reference_drag"
                # Reason is unknown. Or not?
                self.model.particle_grid.build(self.state_0.particle_q, self.model.particle_max_radius * 2.0)
            if frame == self.zero_gravity_steps:
                self.model.gravity = np.array((0.0, -9.81, 0.0))
                # Foot-restore disabled (companion to disabled foot-shrink above).
                # if hasattr(self, '_body_verts_original') and self._foot_mask.any():
                #     self._restore_body_feet()
                if self.sim_use_graph:
                    self.create_graph()
            if self.enable_body_smoothing and frame in self.body_smoothing_frames:
                self.update_smooth_body_shape()
                if self.sim_use_graph:
                    self.create_graph()
            if (self._pose_anim and self._pose_idx < len(self._pose_delta)
                    and frame in self._pose_anim_frames):
                self._apply_pose_frame()
                if self.sim_use_graph:
                    self.create_graph()
            if (self.model.attachment_constraint 
                    and frame >= self.config.attachment_frames):  
                self.model.attachment_constraint = False
                if self.sim_use_graph:
                    self.create_graph()
            
            if self.sim_use_graph: #GPU
                wp.capture_launch(self.graph)

            else: #CPU: launch kernels without graph
                self._sim_frame_with_substeps()

            # Update vertices of last frame. Only advance when a snapshot was
            # actually taken -- with a strided static check the previous frames
            # hold no readback, and clobbering last_verts with None would make
            # is_static() unable to ever compare (and so never terminate).
            if self._verts_np is not None:
                self.last_verts = self._verts_np
                self._last_verts_frame = self._verts_frame
            if self.config.lazy_vert_fetch:
                # Deferred: the property fetches on demand (static check,
                # video capture, final save), so quiet frames cost no sync.
                self._verts_np = None
            else:
                # NOTE Makes a copy if particle_q device is not CPU
                self._verts_np = wp.array.numpy(self.state_0.particle_q)
                self._verts_frame = frame
            
    # def _restore_body_feet(self):
    #     """Restore original body vertices (with feet) after zero-gravity."""
    #     body_vertices = self._body_verts_original
    #     self.v_body = body_vertices
    #     wp.copy(self.body_vertices_device_buffer,
    #             wp.array(body_vertices, dtype=wp.vec3, device='cpu', copy=False))
    #     wp.launch(
    #         kernel=replace_mesh_points,
    #         dim=len(body_vertices),
    #         inputs=[self.body_mesh.mesh.id,
    #                 self.body_vertices_device_buffer],
    #         device=self.device
    #     )
    #     self.body_mesh.mesh.refit()

    def update_smooth_body_shape(self):
        body_vertices = self.body_smoothing_vertices_list.pop()
        self.v_body = body_vertices
        wp.copy(self.body_vertices_device_buffer,
                wp.array(body_vertices, dtype=wp.vec3, device='cpu', copy=False))

        # Apply new vertices and refit the sructures
        wp.launch(
            kernel=replace_mesh_points,
            dim = len(body_vertices),
            inputs=[self.body_mesh.mesh.id,
                    self.body_vertices_device_buffer],
            device=self.device
        )
        self.body_mesh.mesh.refit()

    def _apply_pose_frame(self):
        """Advance the body one step along the pose-animation sequence and refit
        the collision mesh, so the cloth tracks the moving body."""
        body_vertices = self._pose_body_base + self._pose_delta[self._pose_idx]
        self.v_body = body_vertices
        self.body_vertices_device_buffer = wp.array(
            body_vertices, dtype=wp.vec3, device=self.device)
        wp.launch(
            kernel=replace_mesh_points,
            dim=len(body_vertices),
            inputs=[self.body_mesh.mesh.id,
                    self.body_vertices_device_buffer],
            device=self.device,
        )
        self.body_mesh.mesh.refit()
        self._pose_idx += 1

        #update render
        if self.caching: 
            self.renderer.render_mesh(
                            f'shape_{self.body_shape_index}',
                            body_vertices,
                            None,
                            is_template=True,
                        )

    def render_usd_frame(self, is_live=False):
        with wp.ScopedTimer("render", print=False, active=True):
            start_time = 0.0 if is_live else self.usd_frame_time

            self.renderer.begin_frame(start_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

        self.usd_frame_time += 1.0 / self.sim_fps
        if not is_live:
            self.renderer.save()

    def run_frame(self):
        self.update(self.frame)

        # NOTE: USD Render
        if self.caching:
            self.render_usd_frame()
    
    def read_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            return data
    
    def load_obj(self, path):
        v, f = igl.read_triangle_mesh(str(path))
        return v, f.flatten(), f

    def get_shift_param(self,body_vertices):
        v_body_arr = np.array(body_vertices)
        min_y = (min(v_body_arr[:, 1]))
        if min_y < 0:
            return abs(min_y)
        return 0.0

    def calc_norm(self, a, b, c):
        """
        This function calculates the norm based on the three points a, b, and c.
        Input:
            * self (BoxMesh object): Instance of BoxMesh class from which the function is called
            * a (ndarray): first point taking part in norm calculation
            * b (ndarray): second point taking part in norm calculation
            * c (ndarray): third point taking part in norm calculation
        Output:
            * n_normalized (bool): norm(a,b,c) with length 1
        """
        # Calculate the vectors AB and AC
        AB = np.array(b - a)
        AC = np.array(c - a)

        # Calculate the cross product of AB and AC
        n = np.cross(AB, AC)
        n_normalized = n / np.linalg.norm(n)

        return n_normalized

    def calc_vertex_norms(self):
        vertex_normals = np.zeros((len(self.v_cloth_init), 4))
        for face in self.f_cloth:
            v0, v1, v2 = np.array(self.current_verts)[face]
            face_norm = list(self.calc_norm(v0, v1, v2))
            temp_update = face_norm + [1]
            vertex_normals[face] += temp_update

        vertex_normals = vertex_normals[:, :3] / (vertex_normals[:, 3][:, np.newaxis])
        return vertex_normals

    def save_frame(self, save_v_norms=False): 
        """Save current garment state as an obj file, 
        re-using all the information from boxmesh 
        except for vertices and vertex normals (e.g. textures and faces)
        """
        
        # NOTE: igl routine is not used here because it cannot write any extra info (e.g. texture coords) into obj

        # stores v, f, vf and vn
        # Save cloth with texture and normals
        if save_v_norms:
            vertex_normals = self.calc_vertex_norms()

        v_cloth_sim = self.current_verts
        # Store simulated cloth mesh
        # Read the boxmesh file
        with open(self.paths.g_box_mesh, 'r') as obj_file:
            lines = obj_file.readlines()

        # Modify the vertex positions and normals, if required
        with open(self.paths.g_sim, 'w') as obj_file:
            v_idx = 0
            vn_idx = 0
            for line in lines:
                if line.startswith('v '):
                    new_vertex = v_cloth_sim[v_idx]
                    obj_file.write(f'v {new_vertex[0]} {new_vertex[1]} {new_vertex[2]}\n')
                    v_idx += 1
                elif line.startswith('vn '):
                    if save_v_norms:
                        new_vertex = vertex_normals[vn_idx]
                        obj_file.write(f'vn {new_vertex[0]} {new_vertex[1]} {new_vertex[2]}\n')
                        vn_idx += 1
                else:
                    obj_file.write(line)

    def is_static(self):
        """
            Checks whether garment is in the static equilibrium
            Compares current state with the last recorded state
        """
        # last_verts may be more than one frame old (strided static checks /
        # lazy readback); the criterion is per-frame displacement, so scale.
        gap = 1
        if self._last_verts_frame is not None and self.frame > self._last_verts_frame:
            gap = self.frame - self._last_verts_frame
        threshold = self.config.static_threshold * gap
        non_static_percent = self.config.non_static_percent

        curr_verts_arr = self.current_verts
        last_verts_arr = self.last_verts

        if self.last_verts is None:  # first iteration
            return False, len(curr_verts_arr)

        # Compare L1 norm per vertex
        # Checking vertices change is the same as checking if velocity is zero
        diff = np.abs(curr_verts_arr - last_verts_arr)
        diff_L1 = np.sum(diff, axis=1)

        non_static_len = len(
            diff_L1[diff_L1 > threshold])  # compare vertex-wise to allow accurate control over outliers

        if non_static_len == 0 or (non_static_len < len(curr_verts_arr) * 0.01 * non_static_percent):
            print('\nStatic with {} non-static vertices out of {}'.format(non_static_len, len(curr_verts_arr)))
            # Store last frame
            return True, non_static_len
        else:
            return False, non_static_len

    def count_self_intersections(self):
        model = self.model

        if model.particle_count and model.spring_count: 
            model.particle_self_intersection_count.zero_()
            wp.launch(
                kernel=count_self_intersections,
                dim=model.spring_count,
                inputs=[
                    model.spring_indices,
                    model.particle_shape.id,
                ],
                outputs=[
                    model.particle_self_intersection_count
                ],
                device=model.device,
            )
            return int(wp.array.numpy(self.model.particle_self_intersection_count)[0])
        else: 
            return 0

    def count_body_intersections(self):
        model = self.model

        if model.particle_count:
            model.body_cloth_intersection_count.zero_()
            wp.launch(
                kernel=count_body_cloth_intersections,
                dim=model.spring_count,
                inputs=[
                    model.spring_indices,
                    model.particle_shape.id,
                    model.shape_geo,
                    self.body_shape_index
                ],
                outputs=[
                    model.body_cloth_intersection_count
                ],
                device=model.device,
            )
            return int(wp.array.numpy(self.model.body_cloth_intersection_count)[0])
        else:
            return 0 
        
    def _build_vert_connectivity(self, vertices, indices):
        vert_connectivity = [[] for _ in range(len(vertices))]

        for face_id in range(int(len(indices) / 3)):
            v1, v2, v3 = indices[face_id*3 + 0], indices[face_id*3 + 1], indices[face_id*3 + 2]
            
            vert_connectivity[v1].append(v2)
            vert_connectivity[v1].append(v3)

            vert_connectivity[v2].append(v1)
            vert_connectivity[v2].append(v3)

            vert_connectivity[v3].append(v1)
            vert_connectivity[v3].append(v2)

        return vert_connectivity
