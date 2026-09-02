# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Cloth
#
# Shows a simulation of an FEM cloth model colliding against a static
# rigid body mesh using the wp.sim.ModelBuilder().
#
###########################################################################

import sys
import os
import time
import threading
import traceback
import platform
import multiprocessing
import signal
import trimesh
import queue
import numpy as np

# Warp
import warp as wp

# Custom code
from pygarment.meshgen.render.pythonrender import (
    render_images, render_frame_to_array, FrameRenderer)
from pygarment.meshgen.garment import Cloth
from pygarment.meshgen.sim_config import SimConfig, PathCofig

# Suppress warp's init banner and per-module load timing prints
# ("Module warp.sim.X load on device 'cuda:0' took N ms")
wp.config.quiet = True
wp.config.verbose = False
wp.init()

class SimulationError(BaseException):
    """To be rised when panel stitching cannot be executed correctly"""
    pass

class FrameTimeOutError(BaseException):
    """To be rised when frame takes too long to simulate"""
    pass

class SimTimeOutError(BaseException):
    """To be rised when simulation takes too long"""
    pass

def optimize_garment_storage(paths: PathCofig):
    """Prepare the data element for compact storage: store the meshes as ply instead of obj, 
        remove texture files 
    """
    # Objs to ply
    try:
        boxmesh = trimesh.load(paths.g_box_mesh)
        boxmesh.export(paths.g_box_mesh_compressed)
        paths.g_box_mesh.unlink()
    except BaseException:
        pass

    try:
        simmesh = trimesh.load(paths.g_sim)
        simmesh.export(paths.g_sim_compressed)
        paths.g_sim.unlink()
    except BaseException:
        pass

    # Remove large texture file and mtl -- not so necessary
    paths.g_texture_fabric.unlink(missing_ok=True)
    paths.g_mtl.unlink(missing_ok=True)


def update_progress(progress, total):
    """Progress bar in console"""
    # https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    amtDone = progress / total
    num_dash = int(amtDone * 50)
    sys.stdout.write('\rProgress: [{0:50s}] {1:.1f}%'.format('#' * num_dash + '-' * (50 - num_dash), amtDone * 100))
    sys.stdout.flush()

class _FrameWatchdog:
    """Daemon-thread wall-clock watchdog that hard-exits the process via
    os._exit() if a frame doesn't complete in time.

    Replaces the previous SIGALRM-based timeout, which silently fails on
    Linux when garment.run_frame() spends its time inside CUDA / Warp
    kernels — Python signal handlers only run between bytecodes, so the
    pending alarm never gets a chance to interrupt a stuck native call.

    Before exiting, a sentinel ``_TIMEOUT`` file is written to the sim
    output dir (garment.paths.out_el) so that the orchestrating batch
    script (or any post-hoc scan) can tag the run as timed out — in-process
    bookkeeping in props['fails'] cannot run because os._exit skips Python
    cleanup.
    """
    def __init__(self, seconds, garment, frame_num):
        self.seconds = float(seconds)
        self.garment = garment
        self.frame_num = frame_num
        self.cancel = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.cancel.set()

    def _run(self):
        if self.cancel.wait(self.seconds):
            return  # frame finished in time, watchdog disarmed
        try:
            out_dir = getattr(self.garment.paths, 'out_el', None)
            if out_dir is not None:
                marker = os.fspath(out_dir) + '/_TIMEOUT'
                with open(marker, 'w') as f:
                    f.write(f'frame={self.frame_num} budget_s={self.seconds}\n')
        except BaseException:
            pass  # never let bookkeeping prevent the kill
        os.write(2, f"\n[WATCHDOG] frame {self.frame_num} exceeded {self.seconds:.0f}s — hard exit\n".encode())
        os._exit(124)  # GNU-timeout convention


def _run_frame_with_timeout(garment, frame_timeout, frame_num):
    """Run frame while keeping a cap on time to run it"""
    if platform.system() == "Windows":
        # Existing Windows path — kept as-is. Note: original code had a
        # subtle bug (target=garment.run_frame() with parens executes in
        # the parent), but that's outside the scope of the watchdog fix.
        try:
            if frame_num == 0:
                p_frame = multiprocessing.Process(target=garment.run_frame(), name="FrameSimulation")
                p_frame.start()
                p_frame.join(frame_timeout)
                if p_frame.is_alive():
                    p_frame.terminate()
                    p_frame.join()
                    raise TimeoutError
            else:
                garment.run_frame()
        except TimeoutError:
            raise FrameTimeOutError
        return

    # Linux / OSX: daemon-thread watchdog with os._exit on overrun.
    with _FrameWatchdog(frame_timeout, garment, frame_num):
        garment.run_frame()

class _VideoRecorder:
    """Renders simulation video frames on a background thread.

    The sim frame loop is GPU-bound while pyrender/EGL is CPU+GL-bound, so
    rendering inline stalls the solver for every captured frame. Here the loop
    only hands over a copy of the cloth vertices (cheap) and a worker thread
    renders it, so video capture overlaps the sim instead of extending it.
    Body vertices ride along, but only on the frames where the body actually
    moved -- otherwise an animated body would sit frozen in the video while the
    garment moved around it.

    The GL context is created inside the worker and never touched from
    outside it. Rendering failures are swallowed: the video is a diagnostic,
    and it must never take the simulation down with it.
    """

    def __init__(self, body_v, body_f, faces, render_props, resolution=None,
                 queue_max=64):
        self.faces = faces
        self.body_v = body_v
        self.body_f = body_f
        self.render_props = render_props
        self.resolution = resolution
        self.frames = []
        self.error = None
        self._last_body = None
        self._q = queue.Queue(maxsize=queue_max)
        self._thread = threading.Thread(target=self._run, name='VideoRecorder',
                                        daemon=True)
        self._thread.start()

    def _run(self):
        renderer = None
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                verts, body = item
                if renderer is None:
                    # Frame the scene on the body this run starts from.
                    renderer = FrameRenderer(
                        self.body_v if body is None else body, self.body_f,
                        self.render_props, self.resolution)
                    body = None  # already in the scene; no need to swap it
                self.frames.append(
                    renderer.render(verts, self.faces, body_verts=body))
        except BaseException as e:
            self.error = f'{type(e).__name__}: {e}'
        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except BaseException:
                    pass

    def submit(self, verts, body_verts=None):
        """Hand a frame to the worker. Copy: the caller reuses its buffer.

        The body is only sent when it changed -- pose animation rebinds v_body
        to a new array per step, so identity is the test -- which keeps a
        static-body run at one body copy for the whole video.
        """
        if self.error is not None:
            return
        send_body = body_verts is not None and body_verts is not self._last_body
        item = (np.array(verts, copy=True),
                np.array(body_verts, copy=True) if send_body else None)
        try:
            self._q.put(item, timeout=30)
        except queue.Full:
            return  # renderer fell too far behind; drop the frame, keep simulating
        # Only after the frame is safely queued: a dropped frame must not make
        # the next one think the worker has already seen this body.
        if send_body:
            self._last_body = body_verts

    def finish(self, hold_frames=0):
        """Drain the queue, stop the worker, and return the rendered frames."""
        self._q.put(None)
        self._thread.join()
        if self.error is not None:
            print(f'Simulation video frames skipped ({self.error})')
            return []
        if self.frames and hold_frames:
            self.frames.extend([self.frames[-1]] * hold_frames)
        return self.frames


def sim_frame_sequence(garment, config, store_usd=False, verbose=False,
                       video_frames=None, render_props=None, frame_interval=10,
                       recorder=None):
    """Run simulation frame loop.

    Args:
        video_frames: if a list is passed, rendered RGBA frames are appended to it
            every `frame_interval` steps for video generation.
        render_props: render config dict (needed when video_frames is not None).
        frame_interval: capture a video frame every N simulation steps.
    """

    # Save initial state
    if store_usd:
        garment.render_usd_frame()

    # Optional per-frame waistband-top trace (debug). Gated behind WB_TRACE
    # env var = path to a CSV to write. Logs garment cloth max-Y per frame so
    # we can see when (zero-gravity settle vs gravity drape) the height is set.
    import os as _os
    _wb_trace_path = _os.environ.get('WB_TRACE')
    _wb_trace = [] if _wb_trace_path else None

    start_time = time.time()
    # A while loop rather than range(): arming the pose animation can raise
    # config.max_sim_steps, and a range built up front would not see it.
    frame = 0
    while frame < config.max_sim_steps:

        if verbose:
            print(f'\n------ Frame {frame + 1} ------')
        else:
            update_progress(frame, config.max_sim_steps)

        garment.frame = frame

        #Run frame and raise FrameTimeOutError if frame takes too long to simulate

        static = False
        if config.max_frame_time is None:
            # No frame time limits
            garment.run_frame()
        else:
            # NOTE: frame timeouts only work in the main thread of the program.
            # disable frame timeout by passing 'null' as a max_frame_time parameter in config
            _run_frame_with_timeout(
                garment,
                frame_timeout=config.max_frame_time if frame > 0 else config.max_frame_time * 2,
                frame_num=frame
            )

        if verbose:
            num_cloth_cloth_contacts = garment.count_self_intersections()
            print(f'\nSelf-Intersection: {num_cloth_cloth_contacts}')

        # Capture video frame. With a recorder the render happens on a worker
        # thread, so this costs only a vertex readback + copy.
        if recorder is not None and frame % frame_interval == 0:
            recorder.submit(garment.current_verts, garment.v_body)
        elif video_frames is not None and frame % frame_interval == 0:
            video_frames.append(render_frame_to_array(
                garment.current_verts, garment.f_cloth,
                garment.v_body, garment.f_body,
                render_props
            ))

        if _wb_trace is not None:
            try:
                import numpy as _np
                cv = _np.asarray(garment.current_verts)
                in_zg = frame < config.zero_gravity_steps
                _wb_trace.append((frame, float(cv[:, 1].max()),
                                  float(cv[:, 1].min()), int(in_zg)))
            except Exception:
                pass

        if (frame >= config.zero_gravity_steps and frame >= config.min_sim_steps
                and (frame - config.zero_gravity_steps) % config.static_check_interval == 0
                and not garment.pose_animation_holding(frame)):
            static, _ = garment.is_static()
        if static:
            # Phase 1 -> 2: the drape has settled, so the body may start moving.
            # Without a pending animation this is the usual stop.
            if garment.arm_pose_animation(frame):
                static = False
            else:
                break
        elif garment.pose_animation_finished(frame):
            # Phases 2 and 3 are a fixed frame budget rather than a settle, so
            # the run is simply over here.
            break

        runtime = time.time() - start_time
        if runtime > config.max_sim_time:
            raise SimTimeOutError

        frame += 1

    if _wb_trace is not None and _wb_trace_path:
        try:
            with open(_wb_trace_path, 'w') as _f:
                _f.write('frame,top_y_cm,bottom_y_cm,in_zero_gravity\n')
                for row in _wb_trace:
                    _f.write(f'{row[0]},{row[1]:.4f},{row[2]:.4f},{row[3]}\n')
        except Exception:
            pass
        

def save_video(frames, video_path, fps=30):
    """Save list of RGBA numpy arrays as an MP4 video.

    Best-effort: the video is a diagnostic nice-to-have, so encoding problems
    (missing ffmpeg backend, PyAV plugin with a different kwargs contract, ...)
    must never fail the simulation run.
    """
    import imageio.v2 as imageio

    if not frames:
        print("No video frames captured.")
        return

    # Convert RGBA to RGB for MP4
    rgb_frames = [f[:, :, :3] for f in frames]

    def _write(**kwargs):
        with imageio.get_writer(str(video_path), fps=fps, codec='libx264',
                                **kwargs) as writer:
            for f in rgb_frames:
                writer.append_data(f)

    try:
        try:
            # ffmpeg backend (imageio-ffmpeg): supports output_params
            _write(output_params=['-pix_fmt', 'yuv420p'])  # broad compatibility
        except TypeError:
            # Other backends (e.g. PyAV) reject ffmpeg-specific kwargs,
            # some only at write time -- retry the whole encode without them.
            _write()
    except Exception as e:
        print(f'Simulation video skipped ({type(e).__name__}: {e})')
        return


def run_sim(
        cloth_name, props, paths: PathCofig,
        save_v_norms=False, store_usd=False,
        optimize_storage=False,
        verbose=False,
        save_sim_video=False, video_frame_interval=10, video_fps=30):
    """Initialize and run the simulation
    !! Important !!
        'store_usd' parameter slows down the simulation to CPU rates because of required CPU-GPU copies and file writes. Use only for debugging

    Args:
        save_sim_video: if True, render frames during simulation and save as MP4.
        video_frame_interval: capture a frame every N simulation steps (default 10).
        video_fps: frames per second in the output video (default 30).
    """
    sim_props = props['sim']
    render_props = props['render']

    start_time = time.time()

    config = SimConfig(sim_props['config'])   # Why separate class at all?
    garment = Cloth(cloth_name, config, paths, caching=store_usd)

    try:
        _body_initial = str(paths.g_sim).replace('_sim.obj', '_body_initial.obj')
        trimesh.Trimesh(vertices=garment.v_body, faces=garment.f_body,
                        process=False).export(_body_initial)
    except Exception as _e:
        print(f'  body_initial export skipped: {_e}')

    # Video capture: interval and resolution are independent of the final
    # still renders -- a 400x400 preview every 15th frame is a perfectly good
    # diagnostic and a fraction of the cost of 800x800 every 10th.
    video_frame_interval = int(sim_props['config'].get(
        'video_frame_interval', video_frame_interval))
    video_resolution = sim_props['config'].get('video_resolution')
    recorder = None
    video_frames = [] if save_sim_video else None
    if save_sim_video:
        try:
            recorder = _VideoRecorder(
                garment.v_body, garment.f_body, garment.f_cloth,
                render_props['config'], resolution=video_resolution)
        except BaseException as e:
            print(f'Video recorder unavailable, rendering inline ({e})')
            recorder = None

    try:
        sim_frame_sequence(
            garment, config, store_usd, verbose=verbose,
            video_frames=video_frames,
            render_props=render_props['config'] if save_sim_video else None,
            frame_interval=video_frame_interval,
            recorder=recorder,
        )

    except FrameTimeOutError:
        print(f"FrameTimeOutError at frame {garment.frame}")
        props.add_fail('sim', 'frame_timeout', cloth_name)
    except SimTimeOutError:
        print("SimTimeOutError")
        props.add_fail('sim', 'simulation_timeout', cloth_name)
    except SimulationError:
        print("Simulation failed")
        props.add_fail('sim', 'gt_edges_creation', cloth_name)
    except BaseException as e:
        print(f'Sim::{cloth_name}::crashed with {e}')

        if isinstance(e, KeyboardInterrupt):
            # Allow to stop simulation loops by keyboard interrupt
            # It's not a real crash, so don't write down the failure
            sec = round(time.time() - start_time, 3)
            min = int(sec / 60)
            print(f"Simulation pipeline took: {min} m {sec - min * 60} s")
            raise e

        traceback.print_exc()
        props.add_fail('sim', 'crashes', cloth_name)
    else:  # Other quality checks
        if garment.frame == config.max_sim_steps - 1:
            _, non_st_count = garment.is_static()
            print('\nFailed to achieve static equilibrium for {} with {} non-static vertices out of {}'.format(
                cloth_name, non_st_count, len(garment.current_verts)))
            props.add_fail('sim', 'static_equilibrium', cloth_name)

        if time.time() - start_time < 0.5:  # 0.5 sec  -- finished suspiciously fast
            props.add_fail('sim', 'fast_finish', cloth_name)

        # 3D penetrations
        num_body_collisions = garment.count_body_intersections()
        print("BODY CLOTH INTERSECTIONS: ", num_body_collisions)
        num_self_collisions = garment.count_self_intersections()

        sim_props['stats']['body_collisions'][cloth_name] = num_body_collisions
        sim_props['stats']['self_collisions'][cloth_name] = num_self_collisions

        if num_body_collisions > config.max_body_collisions:
            props.add_fail('sim', 'cloth_body_intersection', cloth_name)
        if num_self_collisions: 
            print(f'Self-Intersecting with {num_self_collisions}, '
                  f'is fail: {num_self_collisions > config.max_self_collisions}')
            if num_self_collisions > config.max_self_collisions:
                props.add_fail('sim', 'cloth_self_intersection', cloth_name)

    # ---- Postprocessing ----
    # NOTE: Attempt even on failures for accurate picture and post-analysis
    frame = garment.frame
    print(f"Simulation took #frames={frame + 1}")

    sim_props['stats']['sim_time'][cloth_name] = sim_time = time.time() - start_time
    sim_props['stats']['spf'][cloth_name] = sim_time / frame if frame else sim_time
    sim_props['stats']['fin_frame'][cloth_name] = frame

    garment.save_frame(save_v_norms=save_v_norms) #saving after stats

    try:
        _body_final = str(paths.g_sim).replace('_sim.obj', '_body_final.obj')
        trimesh.Trimesh(vertices=garment.v_body, faces=garment.f_body,
                        process=False).export(_body_final)
    except Exception as _e:
        print(f'  body_final export skipped: {_e}')

    # Render images
    s_time = time.time()
    render_images(paths, garment.v_body, garment.f_body, render_props['config'])
    render_image_time = time.time() - s_time
    render_props['stats']['render_time'][cloth_name] = render_image_time  

    # Save simulation video
    if recorder is not None:
        # One last frame of the settled state, then a ~1s hold on it.
        recorder.submit(garment.current_verts, garment.v_body)
        video_frames = recorder.finish(hold_frames=video_fps - 1)
    elif video_frames:
        # Capture final settled state as extra frames for a brief pause at the end
        final_frame = render_frame_to_array(
            garment.current_verts, garment.f_cloth,
            garment.v_body, garment.f_body,
            render_props['config']
        )
        for _ in range(video_fps):  # ~1 second hold on final frame
            video_frames.append(final_frame)

    if video_frames:
        video_path = paths.out_el / f'{paths.sim_tag}_simulation.mp4'
        save_video(video_frames, video_path, fps=video_fps)

    if optimize_storage:
        optimize_garment_storage(paths)

    # Final info output
    # sec = round(time.time() - start_time, 3)
    # min = int(sec / 60)
