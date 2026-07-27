"""Inference server for 3D garment draping (pants).

Exposes the simulation step of the offline pipeline (run_garment.py ->
run_custom_pants.simulate_pattern) over HTTP. Requests reference data that
already exists on disk: a pattern identified by product_id + size (resolved
to {patterns_root}/{product_id}/{size}/, which must contain a
*_specification.json), a body under ./assets/bodies (auto-generated from
SMPL betas/gender when missing), and sim properties (a preset from
./assets/Sim_props or an inline dict).

Jobs are asynchronous: POST /simulate returns a job_id immediately; a single
dispatcher runs one simulation at a time, each in a fresh spawned child
process (the sim's frame watchdog hard-exits its process on timeout, and a
crashed Warp/CUDA context should not poison the server).

Run:
    python service_inference.py --host 0.0.0.0 --port 8600 --gpu 0

Example client:
    curl -X POST localhost:8600/simulate -H 'Content-Type: application/json' -d '{
        "product_id": "1045459",
        "size": "38",
        "body_name": "global_women_size36_apose",
        "sim_props_preset": "default_sim_props"}'
    # -> {"job_id": "...", "status": "pending", "status_url": "/jobs/..."}
    curl localhost:8600/jobs/<job_id>                  # poll until "succeeded"
    curl -o combined.obj localhost:8600/jobs/<job_id>/result
    curl -o combined.glb 'localhost:8600/jobs/<job_id>/result?format=glb'
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import queue
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent
SERVICE_OUTPUT_DIRNAME = 'service'
WORKER_RESULT_FILE = 'worker_result.json'


def _load_system_config():
    with open(REPO_ROOT / 'system.json') as f:
        return json.load(f)


# ============================================================
# Request / response models
# ============================================================

class SimulateRequest(BaseModel):
    # Pattern reference: resolved to {patterns_root}/{product_id}/{size},
    # which must contain a *_specification.json.
    product_id: str
    size: str
    body_name: str
    sim_props_preset: Optional[str] = None
    sim_props: Optional[dict] = None
    garment_name: Optional[str] = None
    normalize_body: bool = True
    # SMPL body auto-generation (used only when body_name does not already
    # exist under assets/bodies): shape coefficients (exactly 10 floats;
    # default = mean body) and gender (selects the SMPL model and the
    # custom pose). Generation produces {body_name}_apose.obj and
    # {body_name}_custompose.obj; the custom pose is the one simulated on.
    betas: Optional[List[float]] = None
    gender: str = 'female'
    # Bypass request deduplication: identical requests normally coalesce onto
    # the in-flight job (or reuse the succeeded one); force=True always re-runs.
    force: bool = False


class JobStatus(str, Enum):
    pending = 'pending'
    running = 'running'
    succeeded = 'succeeded'
    failed = 'failed'


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    request: dict
    sim_folder: Optional[str] = None
    warnings: List[str] = []
    error: Optional[str] = None
    result_url: Optional[str] = None


class SubmitResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
    deduplicated: bool = False


# ============================================================
# Job store
# ============================================================

@dataclass
class Job:
    id: str
    request: dict
    payload: dict
    output_base: Path
    status: JobStatus = JobStatus.pending
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    sim_folder: Optional[Path] = None
    warnings: list = field(default_factory=list)
    error: Optional[str] = None

    def info(self) -> JobInfo:
        return JobInfo(
            job_id=self.id,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            request=self.request,
            sim_folder=str(self.sim_folder) if self.sim_folder else None,
            warnings=list(self.warnings),
            error=self.error,
            result_url=f'/jobs/{self.id}/result' if self.status == JobStatus.succeeded else None,
        )


JOBS: Dict[str, Job] = {}
KEY_TO_JOB: Dict[str, str] = {}  # dedup key -> most recent job id for that work
JOBS_LOCK = threading.Lock()
JOB_QUEUE: 'queue.Queue[Optional[str]]' = queue.Queue()
STOP_EVENT = threading.Event()
ACTIVE_PROCESS: dict = {'proc': None, 'job_id': None}
SERVER_CONFIG = {
    'gpu': '0',
    # Root under which patterns live as {product_id}/{size}/ folders.
    # Relative paths resolve against the repo root.
    'patterns_root': 'assets/Patterns/service',
    # Classic SMPL model pkls (SMPL_FEMALE.pkl / SMPL_MALE.pkl) and per-gender
    # custom pose files ({gender}.txt, 72 axis-angle values), used to
    # auto-generate missing bodies via the repo's numpy implementation in
    # smpl_pose.py.
    'smpl_models_dir': '../swan-comfyui/Muse/models/smpl',
    'smpl_poses_dir': '../swan-comfyui/Muse/poses_smpl',
}


# ============================================================
# Validation
# ============================================================

SMPL_VERT_COUNT = 6890


def _is_smpl_obj(obj_path: Path) -> bool:
    """True if the mesh has exactly the SMPL vertex count (6890)."""
    try:
        n = 0
        with open(obj_path) as f:
            for line in f:
                if line.startswith('v '):
                    n += 1
                    if n > SMPL_VERT_COUNT:
                        return False
        return n == SMPL_VERT_COUNT
    except OSError:
        return False


def _attachment_enabled(sim_props) -> bool:
    """Whether the resolved sim props (preset path or inline dict) enable the
    attachment constraint. Unreadable/unexpected props count as enabled, so
    the conservative path (requiring the body yaml) wins."""
    try:
        props = sim_props
        if isinstance(sim_props, str):
            with open(sim_props) as f:
                props = yaml.safe_load(f)
        return bool(props['sim']['config']['options']['enable_attachment_constraint'])
    except Exception:
        return True


def _validate_request(req: SimulateRequest) -> dict:
    """Resolve and validate a simulate request; returns the worker payload."""
    sys_config = _load_system_config()
    bodies_path = (REPO_ROOT / sys_config['bodies_default_path']).resolve()
    sim_configs_path = (REPO_ROOT / sys_config['sim_configs_path']).resolve()

    patterns_root = Path(SERVER_CONFIG['patterns_root'])
    if not patterns_root.is_absolute():
        patterns_root = (REPO_ROOT / patterns_root).resolve()
    pattern_folder = (patterns_root / req.product_id / req.size).resolve()
    try:
        pattern_folder.relative_to(patterns_root)
    except ValueError:
        raise HTTPException(422, 'product_id/size must not escape the patterns root')
    if not pattern_folder.is_dir():
        raise HTTPException(
            404, f'Pattern folder not found for product {req.product_id} size {req.size}')
    spec_files = list(pattern_folder.glob('*_specification.json'))
    if not spec_files:
        raise HTTPException(
            422, f'No *_specification.json in {pattern_folder}')

    betas = None
    if req.betas is not None:
        if len(req.betas) != 10 or not all(isinstance(b, (int, float)) for b in req.betas):
            raise HTTPException(422, 'betas must be a list of exactly 10 numbers')
        betas = [float(b) for b in req.betas]

    gender = req.gender.lower()
    if gender not in ('female', 'male'):
        raise HTTPException(422, "gender must be 'female' or 'male'")

    def _cfg_path(key):
        p = Path(SERVER_CONFIG[key])
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    body_obj = bodies_path / f'{req.body_name}.obj'
    sim_body_name = req.body_name
    generate_body = False
    smpl_model = _cfg_path('smpl_models_dir') / f'SMPL_{gender.upper()}.pkl'
    pose_file = _cfg_path('smpl_poses_dir') / f'{gender}.txt'
    if not body_obj.is_file():
        # Fall back to a generated SMPL body pair ({name}_apose.obj +
        # {name}_custompose.obj); the custom pose is the one simulated on.
        sim_body_name = f'{req.body_name}_custompose'
        body_obj = bodies_path / f'{sim_body_name}.obj'
        if not body_obj.is_file():
            if smpl_model.is_file() and pose_file.is_file():
                generate_body = True
            else:
                raise HTTPException(
                    404,
                    f'Body mesh not found for {req.body_name} (auto-generation '
                    f'unavailable: needs {smpl_model} and {pose_file})')

    if req.sim_props_preset and req.sim_props:
        raise HTTPException(422, 'Provide either sim_props_preset or sim_props, not both')

    if req.sim_props is not None:
        try:
            req.sim_props['sim']['config']['resolution_scale']
            req.sim_props['render']['config']['uv_texture']
        except (KeyError, TypeError):
            raise HTTPException(
                422,
                'Inline sim_props must contain sim.config.resolution_scale '
                'and render.config.uv_texture')
        sim_props = dict(req.sim_props)
    else:
        preset = req.sim_props_preset or 'default_sim_props'
        preset_path = Path(preset)
        if not preset_path.suffix:
            preset_path = preset_path.with_suffix('.yaml')
        if not preset_path.is_absolute():
            preset_path = sim_configs_path / preset_path
        preset_path = preset_path.resolve()
        try:
            preset_path.relative_to(sim_configs_path)
        except ValueError:
            raise HTTPException(422, f'Sim props preset must live under {sim_configs_path}')
        if not preset_path.is_file():
            raise HTTPException(404, f'Sim props preset not found: {preset_path}')
        sim_props = str(preset_path)

    # The measurements yaml is only consumed when the attachment constraint
    # is enabled or the body mesh is not SMPL topology (pre-lift fallback
    # formula). Otherwise the sim runs without it. A generated body is SMPL
    # by construction.
    body_yaml = bodies_path / f'{sim_body_name}.yaml'
    if not body_yaml.is_file() and (_attachment_enabled(sim_props)
                                    or not (generate_body or _is_smpl_obj(body_obj))):
        raise HTTPException(
            404,
            f'Body measurements not found: {body_yaml} (required because the '
            'attachment constraint is enabled or the body is not SMPL topology)')

    garment_name = req.garment_name or spec_files[0].stem.replace('_specification', '')

    return {
        'repo_root': str(REPO_ROOT),
        'gpu': SERVER_CONFIG['gpu'],
        'pattern_folder': str(pattern_folder),
        'body_name': sim_body_name,
        'garment_name': garment_name,
        'sim_props': sim_props,
        'normalize_body': req.normalize_body,
        'generate_body': generate_body,
        'generate_base': req.body_name,
        'smpl_model': str(smpl_model),
        'pose_file': str(pose_file),
        'betas': betas,
    }


def _dedup_key(payload: dict) -> str:
    """Identity of the work a payload describes.

    Two requests coalesce when their resolved payloads match AND the files
    that feed the sim are unchanged: the spec file and (for presets) the sim
    props file are identified by path+mtime+size, so regenerating a pattern
    or editing a preset automatically yields a new key.
    """
    def file_stamp(path):
        try:
            st = os.stat(path)
            return [str(path), st.st_mtime, st.st_size]
        except OSError:
            return [str(path), None, None]

    spec_files = sorted(Path(payload['pattern_folder']).glob('*_specification.json'))
    key_data = {
        'pattern_folder': payload['pattern_folder'],
        'specs': [file_stamp(p) for p in spec_files],
        'body_name': payload['body_name'],
        'garment_name': payload['garment_name'],
        'normalize_body': payload['normalize_body'],
        'betas': payload.get('betas'),
        'sim_props': (file_stamp(payload['sim_props'])
                      if isinstance(payload['sim_props'], str)
                      else payload['sim_props']),
    }
    return hashlib.sha256(
        json.dumps(key_data, sort_keys=True).encode()).hexdigest()


# ============================================================
# Worker (runs in a spawned child process)
# ============================================================

def _collect_fails(sim_folder: Path) -> list:
    """Fail types from sim.stats.fails in the sim folder's sim_props.yaml.

    The fails ledger maps fail_type -> [garment names] because it is designed
    for batch runs; a service job simulates a single garment, so only the
    fail types are of interest here.
    """
    try:
        with open(sim_folder / 'sim_props.yaml') as f:
            props = yaml.safe_load(f)
        fails = props['sim']['stats'].get('fails', {})
        return [fail_type for fail_type, names in fails.items() if names]
    except Exception:
        return []


def _generate_smpl_body(base_path: Path, model_pkl: str, pose_file: str, betas=None):
    """Generate an SMPL body pair: {base}_apose.obj and {base}_custompose.obj.

    Uses the repo's numpy SMPL implementation (smpl_pose.py), the A-pose
    definition shared with the pose-animation tooling, and the per-gender
    custom pose (72 axis-angle values) from pose_file. Both meshes are in
    metres and share the A-pose Y-shift (feet at Y=0 in A-pose), matching
    the existing assets/bodies meshes and the pose-animation convention.
    """
    import numpy as np
    import trimesh
    from smpl_pose import load_smpl, smpl_forward
    from make_pose_sequence import THETA_A

    betas = np.asarray(betas if betas is not None else np.zeros(10), dtype=np.float64)
    model = load_smpl(model_pkl)
    theta_custom = np.loadtxt(pose_file).reshape(72)

    verts_a = smpl_forward(model, betas, THETA_A)
    yshift = verts_a[:, 1].min()
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for verts, path in (
            (verts_a, base_path.with_name(f'{base_path.name}_apose.obj')),
            (smpl_forward(model, betas, theta_custom),
             base_path.with_name(f'{base_path.name}_custompose.obj'))):
        verts = verts.copy()
        verts[:, 1] -= yshift
        trimesh.Trimesh(verts, model['f'], process=False).export(str(path))
        print(f'  Generated SMPL body: {path}')


def _sim_worker_entry(payload: dict):
    """Child process entry point. Must stay a top-level function (spawn).

    Communicates the outcome via a JSON file in output_base rather than a
    pipe/queue: the sim's frame watchdog may os._exit() this process at any
    moment, and a file survives that where an unflushed pipe may not.
    """
    # Must be set before importing run_custom_pants (it setdefault's GPU 1)
    # and before Warp initializes.
    os.environ['CUDA_VISIBLE_DEVICES'] = payload['gpu']
    os.chdir(payload['repo_root'])
    output_base = Path(payload['output_base'])
    result = {'ok': False, 'sim_folder': None, 'error': None}
    try:
        body_obj = Path(f"./assets/bodies/{payload['body_name']}.obj")
        if payload.get('generate_body') and not body_obj.is_file():
            _generate_smpl_body(Path(f"./assets/bodies/{payload['generate_base']}"),
                                payload['smpl_model'], payload['pose_file'],
                                betas=payload.get('betas'))

        from run_custom_pants import normalize_body_mesh, simulate_pattern

        if payload['normalize_body']:
            normalize_body_mesh(f"./assets/bodies/{payload['body_name']}.obj")

        sim_folder = simulate_pattern(
            Path(payload['pattern_folder']),
            payload['garment_name'],
            str(output_base),
            body_name=payload['body_name'],
            sim_props=payload['sim_props'],
        )
        if sim_folder is None:
            result['error'] = 'No specification file found in pattern folder'
        else:
            result['ok'] = True
            result['sim_folder'] = str(sim_folder)
    except BaseException as e:
        result['error'] = f'{type(e).__name__}: {e}'
        result['traceback'] = traceback.format_exc()
    try:
        with open(output_base / WORKER_RESULT_FILE, 'w') as f:
            json.dump(result, f)
    except Exception:
        pass


# ============================================================
# Dispatcher (single worker thread in the server process)
# ============================================================

def _resolve_outcome(job: Job, exitcode: Optional[int]):
    """Decide the job outcome from artifacts on disk + the child exitcode."""
    combined = sorted(job.output_base.glob('*/combined.obj'))
    sim_folders = sorted(p for p in job.output_base.iterdir() if p.is_dir())
    timeout_markers = sorted(job.output_base.glob('*/_TIMEOUT'))

    worker_result = None
    result_file = job.output_base / WORKER_RESULT_FILE
    if result_file.is_file():
        try:
            with open(result_file) as f:
                worker_result = json.load(f)
        except Exception:
            pass

    if worker_result and worker_result.get('sim_folder'):
        job.sim_folder = Path(worker_result['sim_folder'])
    elif sim_folders:
        job.sim_folder = sim_folders[-1]

    if job.sim_folder:
        job.warnings = _collect_fails(job.sim_folder)

    if combined:
        job.sim_folder = combined[-1].parent
        # A drape that never settled is not a usable result, even though the
        # sim still wrote its meshes. Intersection fails stay as warnings.
        if 'static_equilibrium' in job.warnings:
            job.status = JobStatus.failed
            job.error = ('Simulation did not reach static equilibrium within '
                         'max_sim_steps (mesh was still written to the sim folder)')
        else:
            job.status = JobStatus.succeeded
        return

    job.status = JobStatus.failed
    if exitcode == 124 or timeout_markers:
        detail = ''
        if timeout_markers:
            try:
                detail = f' ({timeout_markers[-1].read_text().strip()})'
            except Exception:
                pass
        job.error = f'Simulation frame watchdog timeout{detail}'
    elif worker_result and worker_result.get('error'):
        job.error = worker_result['error']
        if worker_result.get('traceback'):
            print(f"[job {job.id}] worker traceback:\n{worker_result['traceback']}")
    elif exitcode not in (0, None):
        job.error = f'Simulation worker died (exitcode={exitcode})'
    elif job.warnings:
        job.error = f"Simulation failed: {'; '.join(job.warnings)}"
    else:
        job.error = 'Simulation finished without producing combined.obj'


def _dispatcher_loop():
    mp_ctx = multiprocessing.get_context('spawn')
    while not STOP_EVENT.is_set():
        job_id = JOB_QUEUE.get()
        if job_id is None:
            break
        try:
            _run_job(mp_ctx, job_id)
        except Exception:
            # A dispatcher bug must never kill the thread — later jobs would
            # queue forever. Mark the job failed and keep serving.
            print(f'[job {job_id}] dispatcher error:\n{traceback.format_exc()}')
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None and job.status == JobStatus.running:
                    job.status = JobStatus.failed
                    job.error = job.error or 'Internal dispatcher error (see server log)'
                    job.finished_at = datetime.now().isoformat(timespec='seconds')
            ACTIVE_PROCESS['proc'], ACTIVE_PROCESS['job_id'] = None, None


def _run_job(mp_ctx, job_id):
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.status = JobStatus.running
        job.started_at = datetime.now().isoformat(timespec='seconds')

    job.output_base.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    proc = mp_ctx.Process(
        target=_sim_worker_entry,
        args=({**job.payload, 'output_base': str(job.output_base)},),
        daemon=True,
    )
    ACTIVE_PROCESS['proc'], ACTIVE_PROCESS['job_id'] = proc, job_id
    proc.start()
    proc.join()
    ACTIVE_PROCESS['proc'], ACTIVE_PROCESS['job_id'] = None, None

    with JOBS_LOCK:
        if STOP_EVENT.is_set() and proc.exitcode not in (0, 124):
            job.status = JobStatus.failed
            job.error = 'Server shutdown while job was running'
        else:
            _resolve_outcome(job, proc.exitcode)
        job.finished_at = datetime.now().isoformat(timespec='seconds')
    elapsed = time.monotonic() - t_start
    print(f'[job {job_id}] {job.status.value}'
          f' in {int(elapsed // 60)}m {elapsed % 60:.1f}s'
          + (f' — {job.error}' if job.error else '')
          + '\n\n')


# ============================================================
# FastAPI app
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    dispatcher = threading.Thread(target=_dispatcher_loop, daemon=True, name='sim-dispatcher')
    dispatcher.start()
    yield
    STOP_EVENT.set()
    JOB_QUEUE.put(None)
    proc = ACTIVE_PROCESS['proc']
    if proc is not None and proc.is_alive():
        proc.terminate()
        proc.join(5)
    dispatcher.join(10)


app = FastAPI(title='GarmentCode draping inference server', lifespan=lifespan)


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'queue_depth': JOB_QUEUE.qsize(),
        'running_job': ACTIVE_PROCESS['job_id'],
    }


@app.post('/simulate', response_model=SubmitResponse, status_code=202)
def submit_simulation(req: SimulateRequest):
    payload = _validate_request(req)
    key = _dedup_key(payload)
    sys_config = _load_system_config()

    with JOBS_LOCK:
        # Identical work already known? Coalesce onto the in-flight job, or
        # reuse the succeeded result. Failed jobs are never reused, and
        # force=True always re-runs.
        if not req.force:
            existing = JOBS.get(KEY_TO_JOB.get(key, ''))
            if existing is not None and existing.status in (
                    JobStatus.pending, JobStatus.running, JobStatus.succeeded):
                return SubmitResponse(
                    job_id=existing.id, status=existing.status,
                    status_url=f'/jobs/{existing.id}', deduplicated=True)

        job_id = uuid.uuid4().hex[:12]
        output_base = (REPO_ROOT / sys_config['output']).resolve() / SERVICE_OUTPUT_DIRNAME / job_id
        job = Job(id=job_id, request=req.model_dump(), payload=payload, output_base=output_base)
        JOBS[job_id] = job
        KEY_TO_JOB[key] = job_id
    JOB_QUEUE.put(job_id)
    return SubmitResponse(job_id=job_id, status=job.status, status_url=f'/jobs/{job_id}')


@app.get('/jobs', response_model=List[JobInfo])
def list_jobs():
    with JOBS_LOCK:
        return [job.info() for job in JOBS.values()]


@app.get('/jobs/{job_id}', response_model=JobInfo)
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, f'Unknown job: {job_id}')
        return job.info()


@app.get('/jobs/{job_id}/result')
def get_job_result(job_id: str, format: str = 'obj'):
    if format not in ('obj', 'glb'):
        raise HTTPException(422, "format must be 'obj' or 'glb'")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, f'Unknown job: {job_id}')
        if job.status != JobStatus.succeeded:
            raise HTTPException(
                409, f'Job is {job.status.value}'
                + (f': {job.error}' if job.error else ''))
        result_path = job.sim_folder / f'combined.{format}'
    if not result_path.is_file():
        raise HTTPException(404, f'Result file missing: {result_path.name}')
    return FileResponse(
        result_path,
        media_type='application/octet-stream',
        filename=f'{job_id}_combined.{format}',
    )


if __name__ == '__main__':
    import uvicorn

    parser = argparse.ArgumentParser(description='GarmentCode draping inference server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8600)
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES value for sim worker processes')
    parser.add_argument('--smpl-models-dir', default=SERVER_CONFIG['smpl_models_dir'],
                        help='Directory with SMPL_FEMALE.pkl / SMPL_MALE.pkl')
    parser.add_argument('--smpl-poses-dir', default=SERVER_CONFIG['smpl_poses_dir'],
                        help='Directory with per-gender custom pose files ({gender}.txt)')
    parser.add_argument('--patterns-root', default=SERVER_CONFIG['patterns_root'],
                        help='Root folder holding patterns as {product_id}/{size}/')
    args = parser.parse_args()

    SERVER_CONFIG['gpu'] = args.gpu
    SERVER_CONFIG['smpl_models_dir'] = args.smpl_models_dir
    SERVER_CONFIG['smpl_poses_dir'] = args.smpl_poses_dir
    SERVER_CONFIG['patterns_root'] = args.patterns_root
    os.chdir(REPO_ROOT)
    # No per-request access log: polling clients hit /jobs every second and
    # drown out the useful job-pipeline output.
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
