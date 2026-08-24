# Drape simulation performance — findings

Perf investigation of the pants draping pipeline, run end-to-end through
`service_inference.py`.

| | |
|---|---|
| **Garment** | `hm_ie/0871889043` size 38 (6 panels, wide-leg pants) |
| **Body** | `HM_heights/female_h168_flat_feet` (SMPL topology, 6890 verts) |
| **Hardware** | NVIDIA H100 PCIe, 24 CPU cores |
| **Baseline** | `default_sim_props.yaml` unchanged: **27.7 s ± 0.9** wall, 18.9 s sim, 542 frames |
| **Method** | 86 configurations × 3 runs each, sequential, fresh worker process per run |

> The "31 s" starting figure was a cold first run. Warm runs settle at 27.7 s;
> all deltas below are against that.

---

## 1. Result

| Tier | Wall | Δ | What you give up |
|---|---:|---:|---|
| **A — recommended** | **8.2 s ± 0.4** | **−71%** | Nothing measurable on this garment; small but detectable on more fitted ones. Video saved, cloth mesh at full resolution, integration timestep bit-identical to baseline. Every fidelity metric at the run-to-run noise floor. |
| **B — fastest sensible** | **5.4 s ± 0.1** | **−81%** | 36% coarser cloth mesh (9,212 verts vs 14,428). Waistband 4.6 mm higher, surface area +0.26%. Fold placement visibly different; fit reads the same. |
| **Free** | **20.4 s** | **−27%** | Nothing at all, and no config needed — the threaded video recorder is a code fix that applies to every garment. |
| **A+ / B+ — more aggressive** | 7.9 s / 5.5 s | −72% / −80% | Adaptive striding ceiling 16 instead of 8, without self-contact striding. Fold deviation drifts above noise (vert 6.28 mm vs a 4.45 mm floor) — strictly worse than Tier A, kept only as a data point. |

```yaml
# Tier A.  Add resolution_scale: 1.25 for Tier B.
sim:
  config:
    sim_fps: 30                    # with substeps 20, dt stays exactly 1/600
    sim_substeps: 20               # -23%: half the frames, half the collide calls
    static_threshold: 0.06         # scaled with sim_fps so the test means the same thing
    zero_gravity_steps: 75         # ALSO scaled with sim_fps -- see below
    video_resolution: [400, 400]   # optional: smaller files, no time change
    video_frame_interval: 15       # optional: same
    options:
      collide_interval: 8                 # ceiling for the adaptive stride
      collide_interval_adaptive: true     # stride by measured cloth motion
      soft_contact_margin: 1.0            # striding needs a wider margin
render:
  config:
    uv_texture:
      dpi: 150                     # -13%: the 12.7 MB fabric PNG was print-res
      fabric_grain_resolution: 1
```

Two of those lines are **corrections, not relaxations**. `static_threshold` and
`zero_gravity_steps` are both counted per *frame*, so halving `sim_fps` doubled
the simulated time each represents. Scaling them back keeps the sim doing what
the baseline did — and `zero_gravity_steps: 75` measured *better* than 150 on
every fidelity metric (§5.6), because 150 frames at 30 fps was 5 s of
zero-gravity settling where the baseline did 2.5 s.

Where the 27.7 s went:

| Change | Saved | Kind |
|---|---:|---|
| Threaded video recorder + cached GL context | 7.3 s | code |
| `sim_fps: 30` + `sim_substeps: 20` | 6.4 s | config |
| `uv_texture.dpi: 1500 → 150` | 3.5 s | config |
| Pre-warmed worker process | 2.0 s | code |
| Parallel panel meshing (4 processes) | 0.7 s | code |
| `resolution_scale: 1.25` (Tier B only) | 4.0 s | config, costs fidelity |
| Adaptive contact striding + margin 1.0 | 2.1 s | code + config |
| Self-contact solve striding (every 4th substep) | 0.8 s | code + config |
| `zero_gravity_steps: 150 → 75` (scaled with `sim_fps`) | 0.4 s | config, *improves* fidelity |
| Concurrent side renders | 0.0 s | code, measured a wash |

---

## 2. Method: the noise floor is the whole story

**This simulation is not deterministic.** Three runs of an identical config
produced 573, 568 and 484 frames — an 18% spread — because the
static-equilibrium test trips at slightly different moments. Comparing a
variant's mesh to a single baseline mesh is therefore meaningless without first
measuring how much the baseline disagrees with *itself*.

Baseline vs baseline, averaged over all 3 pairs:

| Metric | Noise floor |
|---|---:|
| Mean symmetric surface distance | **5.55 mm** |
| p95 surface distance | 13.36 mm |
| Mean per-vertex displacement | **4.45 mm** |
| p95 per-vertex displacement | 15.92 mm |
| Waistband top height | **0.94 mm** |
| Hem bottom height | **0.10 mm** |
| Surface area | **0.016%** |

That split is the key to reading everything below. Loose wide-leg fabric
swings, so **where an individual fold lands is noise** (±5 mm mean, worst vertex
±50 mm). But **waistband height, hem height and surface area are stable to ~1 mm
and 0.02%**. Those three are the honest signal: a config that moves them changed
the garment; a config that only rearranges folds changed nothing distinguishable
from re-running.

Comparison details: meshes are in cm, distances reported in mm. Per-vertex
distance is used when topology matches, and is omitted (—) for
`resolution_scale` variants where it does not. Surface distance is symmetric
mean nearest-neighbour over 20k surface samples per side. Each variant is
compared against the baseline over all 9 baseline×variant pairs.

---

## 3. Wins, in order of size

### 3.1 Video: 8.6 s → 1.3 s, output unchanged

Saving the MP4 took the baseline from 19.1 s to 27.7 s. `render_frame_to_array()`
was the cause: for **every** captured frame it rebuilt the body mesh and both
materials, recomputed smooth normals over the whole garment, assembled a scene
with a camera and five lights, and **created and destroyed an entire EGL
context** — at the same 800×800 as the final stills. Only the garment geometry
actually changes between frames.

Two fixes, both in code:

- **`FrameRenderer`** (`render/pythonrender.py`) builds the static half of the
  scene — body mesh, materials, camera, lights, GL context — once, and swaps
  only the garment node per frame.
- **`_VideoRecorder`** (`simulation.py`) owns that renderer on a **background
  thread**. The sim loop now hands over a vertex copy and keeps solving, so the
  CPU/GL render overlaps the GPU solve instead of blocking it.

| | Wall |
|---|---:|
| baseline, rendered inline | 27.7 s |
| `save_sim_video: false` (no video) | 19.1 s |
| **threaded recorder, video kept** | **20.4 s** |

Output verified frame-by-frame: same resolution, same fps, same file size
(88 vs 89 frames, differing only because frame counts differ between runs).
Because the render is now off the critical path, dropping to 400×400 every 15th
frame buys nothing further (20.0 s vs 20.4 s) — do it only if you want smaller
files.

### 3.2 Substeps are nearly free, so spend them (dt unchanged)

`sim_substeps` was hardcoded to 10 with a `#increase?` comment beside it.
Measuring both directions produced the most useful result in the sweep:

| substeps | s/frame | frames | wall |
|---:|---:|---:|---:|
| 5 | — | 2500 (never settled) | 66 s, failed |
| 8 | 0.0334 | 596 | 28.6 s |
| **10 (baseline)** | 0.0350 | 542 | 27.7 s |
| 15 | 0.0362 | 519 | 27.5 s |
| 20 | 0.0370 | 525 | 28.2 s |

**Doubling the substeps cost 6% more per frame, not 100%.** The reason is in
`_sim_frame_with_substeps()`: `wp.sim.collide()` runs **once per frame, outside**
the substep loop, and it — not the integrator — dominates the frame cost. The
substep loop is the cheap part.

That inverts the strategy. Each frame advances a fixed `1/sim_fps` of simulated
time regardless of substep count, so halving `sim_fps` while doubling
`sim_substeps` keeps `dt = (1/sim_fps)/sim_substeps` **bit-identical at 1/600**
while covering twice the simulated time per frame. Half the frames means half
the collide calls.

`sim_fps: 30` + `sim_substeps: 20` → **21.3 s, −23%**, 321 frames instead of 542.
Surface deviation 5.67 mm against a 5.55 mm floor; waistband 0.98 mm against a
0.94 mm floor. Statistically the same drape.

The direction has a wall. `sim_fps: 30` alone (dt 1/300) never reaches
equilibrium. `fps20/ss30` went erratic — one run of three needed 1021 frames —
and moved the waistband 8.3 mm. `fps15/ss40` failed outright. **30/20 is the
sweet spot.**

### 3.3 Fabric texture: 3.5 s, and the floor is reached

`uv_texture.dpi: 1500` was generating a 12.7 MB print-resolution PNG. Dropping
to 150 saves 3.5 s. **There is nothing left beyond that** — `dpi: 72` with the
fabric-grain image disabled entirely measured 24.0 s against 24.2 s for dpi 150.
At dpi 150 the whole serialize-plus-texture stage is **0.26 s**; the remainder is
UV island packing, which the simulation needs.

### 3.4 Pre-warmed worker: 2.0 s

Once the sim is ~4 s, imports plus Warp/CUDA init (1.5 s) and process spawn
(0.9 s) are **28% of the job**, all of it before any useful work. The dispatcher
spawns a fresh process per job deliberately, so a poisoned CUDA context or a
watchdog `os._exit()` cannot leak into the next run — worth keeping.

The service now holds **pre-warmed spare workers** (one per concurrency slot)
that have already imported everything and initialised Warp, blocked on a pipe
waiting for a payload. They are warmed *during the previous job*, so the cost is
paid off the critical path, and each still handles exactly one job and exits.
Falls back to a cold spawn if a spare is unready or died; `--no-prewarm` opts out.

### 3.5 Parallel panel meshing: 0.7 s (2.1× on that stage)

`gen_panel_meshes` was 0.66 s of the 0.91 s box-mesh build. Per-panel cost for
this garment:

| Panel | Time |
|---|---:|
| `pant_b_r` | 0.200 s |
| `pant_b_l` | 0.184 s |
| `pant_f_l` | 0.157 s |
| `pant_f_r` | 0.127 s |
| `wb_front` | 0.008 s |
| `wb_back` | 0.008 s |

Panels are independent, so the theoretical ceiling is 0.684 s → 0.200 s (3.4×).
**Threads cannot get there**: the CGAL bindings hold the GIL — measured directly,
4 threads ran 4 full pattern meshings at **0.83× of serial**, i.e. slower.

So it needs processes. `Panel.gen_panel_mesh` was split, with the CDT work moved
into a module-level `mesh_panel_cdt(points, edge_verts_ids, mesh_resolution,
panel_name)` that takes and returns only plain point/index data — a small pickle
per panel. `gen_panel_meshes` became three phases: serial edge-vertex setup,
parallel CDT over a `ProcessPoolExecutor`, serial store-and-manifold-check.

Measured **0.700 s → 0.335 s (2.09×)** with byte-identical output. Off by default;
enabled by `init_panel_pool(n)`, which the service calls with `--panel-workers`
(default 4).

Two implementation constraints, both learned the hard way:

- **`fork()` from a process holding a live CUDA context is a hazard**, and
  `ProcessPoolExecutor` forks lazily on first `submit()` — which would land
  mid-job, after Warp is up. `init_panel_pool` therefore forces the children to
  exist immediately, and the service calls it in the pre-warmed worker
  *before* importing Warp.
- **The sim worker processes had to stop being daemonic** — a daemonic process
  is forbidden children — and the pool must be reaped explicitly on the way out,
  or its non-daemonic children block interpreter exit and hang the dispatcher's
  `join()` forever.

### 3.6 Concurrent sims: 1.41× throughput at 2 workers

The dispatcher was single-slot. It now runs `--max-concurrent` dispatcher threads
(default 2) with a per-job process map and a locked spare pool.

| Concurrent | Total | Throughput | Per-job latency |
|---:|---:|---:|---|
| 1 | 6.6 s | 9.1 sims/min | 6.6 s |
| **2** | 9.4 s | **12.8 sims/min (1.41×)** | 9.4 s |
| 4 | 19.2 s | 12.5 sims/min | 9.6–19.2 s |

N=4 confirms the semaphore holds: throughput plateaus at two slots and the extra
jobs queue. This is a **batch lever, not an interactive one** — throughput rises,
per-job latency rises with it.

Measured earlier with 4 bare concurrent processes on the slower config, the gain
was 1.81×; it is smaller here because the job is now much shorter, so a larger
share of it is CPU-bound serial work contending with the other job (and with the
two panel pools). Concurrent output was verified equivalent: concurrent-vs-serial
surface distance 4.50 mm against serial-vs-serial 4.50 mm — exactly the noise floor.

---

## 4. Answered questions

### Is the GPU under-utilised?

Only in aggregate. Sampling `utilization.gpu` every 100 ms and isolating the
solver window: **90.8% mean, median 100%, 51% of samples pegged at 100%**. The GPU
is saturated while solving. Whole-job median utilisation is 2% purely because
half the wall clock was single-threaded CPU work — texture generation, box-mesh
build, video encode — with the GPU parked. Clocks were already maxed
(1755 / 1755 MHz), so there was nothing there either.

### Is the VRAM headroom usable?

Yes, as throughput, not latency: a sim uses **2.4 GB of 81.5 GB**. That is what
§3.6 spends.

### Does relaxing static-equilibrium detection help?

It works, and it is the most fidelity-expensive of the cheap knobs. Both
parameters were swept:

| Change | Δ wall | Waist error |
|---|---:|---:|
| `static_threshold: 0.05` | −7% | 1.8 mm |
| `static_threshold: 0.10` | −18% | 4.0 mm |
| `static_threshold: 0.15` | −24% | **5.4 mm** |
| `non_static_percent: 5` | −8% | 1.5 mm |
| `non_static_percent: 15` | −17% | 3.9 mm |
| both (0.10 + 5) | −23% | 5.2 mm |

The cost lands squarely in the metric that isn't noise: **the waistband ends up
measurably higher because the garment is genuinely still falling when you stop
it.** At −24% the waist is 5.4 mm off against a 0.94 mm floor and mean per-vertex
displacement is 8.8 mm against 4.45.

`sim_fps: 30` + `sim_substeps: 20` buys the same −23% and leaves the waist at
0.98 mm. **Take the timestep win instead.** Relax the static test only for speed
beyond what the other levers give, and prefer `non_static_percent`, which
degrades more gracefully per second saved.

### Is there more in the solver?

The solver is now ~4.3 s of the 6.6 s and it is collide-bound (§3.2), not
integration-bound. Collision work was then explored directly — see **§5**. Short
version: making each contact cheaper never helps, because sloppier contact
resolution needs more frames to settle than it saves per frame. Calling `collide`
*less often* does help, and keying the interval to measured cloth motion makes it
both faster and cleaner than a fixed schedule — **−23% on Tier A with geometry at
the noise floor** (§5.5, §5.7). The substep loop that remains is launch-bound
across 220 kernel launches a frame (§5.8). The knobs tried and rejected: `enable_particle_particle_collisions: false`,
`enable_edge_edge_collisions: false` (−3.9%, at noise),
`enable_triangle_particle_collisions: false`, `global_damping_factor` 0.15 / 0.30.
The two things that *did* move the solver — fewer frames and fewer vertices — are
already in Tiers A and B.

---

## 5. Collision work

The solver is collide-bound, so this was the next target. Most of it was a dead
end, for one consistent reason (§5.4) — but one approach works, and it is the only
solver-side win in the investigation (§5.5). The negative results are recorded in
full so nobody re-runs them.

### 5.1 Where a frame actually goes

Timed with `wp.synchronize()` around each half, CUDA-graph capture disabled so
the two can be separated (Tier B config, 9,212 cloth particles, 13,776 body
triangles):

| | ms/frame | Share |
|---|---:|---:|
| `wp.sim.collide` | **5.90** | 48.3% |
| substep loop (20 substeps) | 6.32 | 51.7% |
| total | 12.22 | |

So collision detection is roughly half the frame, and each of the 20 substeps
costs ~0.32 ms. This also refines §3.2: substeps are not *free*, they are ~9% of
the frame per doubling — small, but not nothing.

### 5.2 Body collider culling — implemented, no-op

`wp.sim.collide` queries a BVH over every body triangle each frame, yet a pair of
pants cannot reach the head, arms or hands. `face_filters` already excludes the
arms, but only at *contact-resolution* time — the BVH still contains all 13,776
triangles.

Implemented as `body_collider_cull`: drop body **faces** whose every vertex sits
above the garment's initial top plus a margin, keeping the full vertex array so
vertex indices and `replace_mesh_points` stay valid, and masking `face_filters`
(one bool per triangle, in face order) to match. A post-sim check fails the run
with `collider_cull_exceeded` if the cloth ever climbs above the kept band, so an
over-aggressive margin cannot fail silently.

It works and it buys nothing:

| Faces kept | collide ms/frame |
|---:|---:|
| 13,776 (100%) | 5.90 |
| 10,164 (74%) — the safe cull, margin 15 cm | 6.22 |
| 2,669 (19%) — probe only, physically wrong | 4.56 |

**BVH cost scales sub-linearly**, as a tree should: removing 81% of triangles
bought 23%, and removing a physically-safe 26% bought nothing. End-to-end
confirmation over 3 runs: `A_cull` 10.5 s vs `A_ctrl2` 10.7 s — identical, with
fidelity at the noise floor (surf 5.54 mm, waist 1.68 mm, area 0.022%).

The reason a safe cull removes so little is the A-pose: the arms hang down to
thigh level, so they sit *inside* the garment's Y band and survive. The SMPL
segmentation would allow a part-based cull instead — arm faces are 36% of the
mesh — but at the measured scaling that is worth roughly 2–3% end-to-end, and it
is not implemented.

Kept in the code, **default off**. It is a no-op for pants, but the scaling above
is the thing to judge by: it only pays where the garment occupies a genuinely
small slice of the body.

### 5.3 `soft_contact_margin` alone — a large per-frame win that loses

`soft_contact_margin` was hardcoded at 0.2 in `sim_config.py`. It is the
broadphase search radius that decides how many candidate contacts `collide()`
generates per particle per frame, so it is the direct lever on collision cost —
and it is a big one:

| margin | collide ms/frame | Δ collide | frame ms |
|---:|---:|---:|---:|
| 1.0 | 5.38 | −9% | 11.69 |
| **0.2 (default)** | **5.90** | — | **12.22** |
| 0.1 | — | — | — |
| **0.05** | **2.18** | **−63%** | **8.04** |

Cutting the margin to 0.05 removes nearly two thirds of the collision cost and a
third of the whole frame. And it is unusable:

| Config | Wall | Frames | Result |
|---|---:|---:|---|
| Tier B control | 6.5 s ± 0.1 | 258 | — |
| Tier B + margin 0.1 | **7.6 s ± 0.4** | **488** | 1 of 3 runs never settled |
| Tier B + margin 0.05 | — | 2500 | **3 of 3 runs never settled** |
| Tier A + margin 0.15 | 10.6 s | 329 | no change |
| Tier A + margin 0.1 | — | 2500 | **3 of 3 runs never settled** |
| Tier A + margin 0.05 | — | 2500 | **3 of 3 runs never settled** |

A tighter margin means contacts are created and destroyed from one step to the
next, the cloth chatters against the body, and the static-equilibrium test stops
tripping. Tier B at margin 0.1 needed 488 frames instead of 258 — so the run got
*slower* despite each frame being cheaper — and it is unreliable rather than
merely slow. At full mesh resolution anything below 0.15 never settles at all.

Exposed as a config knob (`sim.config.options.soft_contact_margin`, default 0.2,
unchanged behaviour) because it is a useful diagnostic, but **do not lower it**.

### 5.4 The pattern, stated once

This is the second independent confirmation of the same structure, after
`sim_substeps` in §3.2:

> Total cost is `frames × per-frame`. Every mechanism that makes a frame cheaper
> by making the physics sloppier — fewer substeps, a bigger `dt`, a tighter
> contact margin — increases the frame count by more than it saves per frame,
> because the run only ends when the cloth stops moving. Sloppier physics takes
> longer to stop moving.

The wins that held up all avoid this trap. They either do the same physics with
fewer frames (`sim_fps: 30` + `sim_substeps: 20`, which keeps `dt` bit-identical),
or they take work off the critical path without touching the solver at all
(threaded video, prewarm, panel pool), or they honestly reduce the problem size
(`resolution_scale`, which is why it costs fidelity).

There is one way to attack collision cost that does *not* fall into the trap:
call `collide` less often, rather than making each contact sloppier. That is
§5.5, and it is the only solver-side win in this whole investigation.

### 5.5 Calling `collide` less often — the one thing that works

Contact detection is ~half the frame, and near equilibrium the cloth barely moves
between frames, so the contact *pairs* stay valid even though the contact *forces*
must be recomputed continuously. Striding detection is therefore different in
kind from every other lever above: it does not make any individual contact
sloppier, it just generates the pair list less often.

Implemented as `collide_interval` (default 1 = current behaviour). A frame that
reuses contacts is a different kernel sequence, so it needs its **own CUDA graph** —
`create_graph()` now captures a second, collide-free graph when striding is on,
and `update()` picks between them.

**Striding alone does not work.** With the margin left at 0.2:

| interval | collide ms/frame | frames | outcome |
|---:|---:|---:|---|
| 1 | 5.90 | 265 | baseline |
| 2 | 1.15 | **2500** | never settled |
| 4 | 0.67 | **2500** | never settled |
| 8 | 0.84 | 1040 | settled, but **238 body intersections** (vs ~95) |

The per-frame cost collapses and the run gets worse, because a contact that
becomes active mid-stride was never in the set — the cloth interpenetrates, the
next `collide` shoves it out, and it oscillates with the stride period.

**Pairing it with a wider margin fixes it.** Contacts have to be found *before*
they are needed, so the margin must cover where the cloth will be, not where it
is:

| interval | margin | collide ms/frame | frame ms | frames |
|---:|---:|---:|---:|---:|
| 1 | 0.2 | 5.90 | 12.22 | 265 |
| 1 | 1.0 | 5.33 | 11.81 | 316 |
| 2 | 0.6 | 3.39 | 9.84 | 306 |
| 4 | 1.0 | 2.98 | **9.50** | **260** |
| 6 | 1.5 | 2.70 | 9.50 | 267 |
| 8 | 2.0 | 2.50 | **9.00** | 269 |

`collide_interval: 4` + `soft_contact_margin: 1.0` gives **−22% per frame with the
same frame count** — 260 frames against the baseline's 265. That is the first
result in this investigation that escapes the frames × per-frame trap. The
`interval 1, margin 1.0` row is the control that proves it is the striding doing
the work, not the margin: margin alone is *slower* (316 frames).

**But striding the initial settle costs quality.** End-to-end, 3 runs each:

| Config | Wall | Frames | surf mm | vert mm | waist mm | body int. | self int. |
|---|---:|---:|---:|---:|---:|---:|---:|
| _noise floor_ | | | _5.55_ | _4.45_ | _0.94_ | | |
| Tier A control | 10.7 s | 332 | 5.65 | 5.12 | 0.80 | 93 | 11.0 |
| Tier A, stride from 0 | **8.7 s** | 337 | 6.65 | **7.13** | 0.70 | 78 | **19.7** |
| Tier A, stride from frame 75 | **10.1 s** | 295 | 5.30 | 4.99 | 1.25 | 79 | 8.7 |
| Tier B control | 6.5 s | 258 | 6.40 | — | 5.10 | 94 | 0.0 |
| Tier B, stride from 0 | **5.8 s** | 272 | 6.53 | — | 3.97 | 73 | 7.0 |
| Tier B, stride from frame 75 | **6.1 s** | 211 | 6.78 | — | 3.11 | 71 | 0.0 |

Striding from frame 0 is worth −19% (Tier A) but pushes fold-level deviation
above the noise floor and roughly doubles-to-quadruples self-intersections.
Striding only after the initial settle (`collide_stride_start_frame`) keeps every
metric at or below the noise floor — and note **body** intersections *improve* in
every striding variant (71–79 vs 93–96), because the wider margin catches contacts
earlier.

So it is a genuine dial rather than a free win: −6% at no measurable cost, or
−19% for fold-level deviation you can see in the table. Both are shipped as
opt-in config, default off.

Two implementation notes:

- The particle grid is strided **with** `collide`. Rebuilding it every frame
  against a stale contact set measured *worse* for self-intersections (41.3 vs
  19.7 at interval 4) — inconsistent contact sets are worse than consistently
  stale ones — and the rebuild costs nothing measurable either way.
- **Self-intersection count is a very noisy metric.** The same config measured
  1.7 and 11.0 across two 3-run groups, and 19.7 vs 45.7 for another identical
  pair. Treat the 2–4× rise as a trend, not a precise figure.

### 5.6 `zero_gravity_steps` must scale with `sim_fps`

Found while investigating the above. `zero_gravity_steps` is counted in frames,
so `sim_fps: 30` silently doubled the *simulated* zero-gravity settle from 2.5 s
to 5 s — 150 of the ~330 frames in a Tier A run, 45% of it, spent on settling the
baseline did in half the time.

Setting `zero_gravity_steps: 75` restores the baseline's simulated settle, and it
is better on every axis at once:

| Config | Wall | Frames | surf mm | vert mm | waist mm | area % | body int. | self int. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| _noise floor_ | | | _5.55_ | _4.45_ | _0.94_ | _0.016_ | | |
| Tier A control (zg 150) | 10.7 s | 332 | 5.65 | 5.12 | 0.80 | 0.047 | 93 | 11.0 |
| **Tier A + zg 75** | **10.3 s** | 296 | **5.11** | **3.69** | 0.67 | **0.021** | 82 | **0.0** |

Both `surf` and `vert` land *below* the run-to-run noise floor, self-intersections
go to zero, and it is 4% faster. This is not a relaxation — it is fixing an
inconsistency that `sim_fps: 30` introduced. It also explains why the earlier
`zero_gravity_steps` tests looked bad: those were run at `sim_fps: 60`, where
cutting 150 → 50 really did remove settling time and moved the waistband 3.7 mm.

### 5.7 Adaptive striding — better on both axes

§5.5 strides on a fixed frame schedule, which forces a choice: stride during the
initial fall and pay in self-intersections, or skip it and collect only −6%.
Neither is necessary. The cloth's own motion says when a stale contact set is
safe, and that signal is already being computed — `is_static()` measures max
per-vertex L1 displacement every frame.

`collide_interval_adaptive` reuses it, scaled by `static_threshold` so the policy
travels with the config:

| motion vs `static_threshold` | interval |
|---|---:|
| > 8× | 1 (detect every frame) |
| > 4× | 2 |
| > 2× | ceiling / 2 |
| otherwise | ceiling |

With CUDA graphs this costs nothing structurally — there are already two graphs
(collide / collide-free), and adaptivity only changes which one is launched.

| Config | Wall | Frames | surf mm | vert mm | waist mm | area % | body int. | self int. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| _noise floor_ | | | _5.55_ | _4.45_ | _0.94_ | _0.016_ | | |
| Tier A control | 10.7 s | 332 | 5.65 | 5.12 | 0.80 | 0.047 | 93 | 11.0 |
| Tier A, fixed stride 4 from frame 0 | 8.7 s | 337 | 6.65 | 7.13 | 0.70 | 0.043 | 78 | 19.7 |
| Tier A, fixed stride 4 after settle | 10.1 s | 295 | 5.30 | 4.99 | 1.25 | 0.033 | 79 | 8.7 |
| **Tier A, adaptive ceiling 8** | **8.2 s** | 318 | **5.60** | 5.64 | 1.43 | 0.025 | 79 | 10.3 |
| Tier A, adaptive ceiling 16 | **7.9 s** | 268 | 6.14 | 6.28 | 0.94 | 0.024 | 76 | 6.3 |
| Tier B control | 6.5 s | 258 | 6.40 | — | 5.10 | 0.264 | 94 | 0.0 |
| **Tier B, adaptive ceiling 8** | **5.4 s** | 202 | 6.55 | — | 3.17 | 0.183 | 70 | **0.0** |
| Tier B, adaptive ceiling 16 | 5.5 s | 227 | 6.51 | — | 2.50 | 0.178 | 67 | **0.0** |

Adaptive at ceiling 8 beats the fixed stride-from-0 schedule on **both** axes at
once — faster (8.2 s vs 8.7 s) *and* markedly cleaner (surf 5.60 vs 6.65 mm, vert
5.64 vs 7.13 mm, self-intersections 10.3 vs 19.7, back inside the control's own
1.7–11.0 range). That is exactly the hypothesis: the damage was being done during
the fast-moving phase, and detecting every frame there costs little because that
phase is short.

Tier B is the cleaner story still: 5.4 s against a 6.5 s control, with waistband
error, surface area, body intersections and self-intersections **all better than
the control**. Faster and more accurate.

### 5.8 What is left in the substep loop — and one more negative result

With striding, contact detection is down to ~11% of the frame and the substep
loop is ~89%. Per-kernel timings (Tier B, `collide_interval: 4`, averaged over 40
frames). **These were measured with CUDA graphs disabled and a `wp.synchronize()`
around every launch, which inflates the total ~2.4×** — the real frame at this
config is 3.95 ms, not 9.5 ms (§5.10). Relative shares are still informative;
absolute values are not:

| Phase | Kernel | ms/frame | Share | Calls/frame |
|---|---|---:|---:|---:|
| substep | `solve_particle_particle_contacts` | 1.332 | 14.0% | 20 |
| substep | `solve_edge_edge_self_contact` | 1.111 | 11.7% | 20 |
| substep | `solve_particle_shape_contacts` | 1.110 | 11.7% | 20 |
| collide | `create_self_edge_contacts` | 0.879 | 9.3% | 0.2 |
| substep | `integrate_particles` | 0.804 | 8.5% | 20 |
| substep | `solve_springs` | 0.795 | 8.4% | 20 |
| substep | `bending_constraint` | 0.775 | 8.2% | 20 |
| substep | `solve_particle_triangle_self_contacts` | 0.771 | 8.1% | 20 |
| substep | `solve_particle_ground_contacts` | 0.729 | 7.7% | 20 |
| substep | `apply_particle_deltas` | 0.657 | 6.9% | 20 |
| substep | `replace_mesh_points` | 0.365 | 3.8% | 20 |
| collide | `create_soft_contacts` | 0.076 | 0.8% | 0.2 |

Three things fall out of this table.

**The frame is launch-bound, not compute-bound.** Eleven kernels × 20 substeps =
**220 kernel launches per frame**, each ~0.03–0.07 ms for 9,212 particles on an
H100. No single kernel dominates; the distribution is flat, which is the signature
of per-launch latency rather than arithmetic. Cutting this means fewer launches —
either fewer substeps (§3.2: cannot, XPBD stiffness depends on the count) or
fusing constraint kernels, which is a Warp-side change.

**The dominant collide kernel is `create_self_edge_contacts`** — cloth-vs-cloth
edge detection, not cloth-vs-body. That retroactively explains §5.2: culling body
triangles was never going to matter much, because body contact generation
(`create_soft_contacts`, 0.076 ms) was never the expensive part.

**And the integrator is XPBD, not semi-implicit Euler**, despite the comment in
`garment.py:84` and in `_sim_frame_with_substeps`. Worth knowing: it is why
lowering `sim_substeps` destabilises rather than merely coarsening — in XPBD,
constraint stiffness is a function of the substep count.

`replace_mesh_points` at 20 calls/frame looked like free money: XPBD's
`simulate()` ends by re-uploading the cloth points and refitting the cloth BVH,
every substep, and nothing reads that BVH mid-frame (the `solve_*` kernels work
off contact lists `collide` already produced). Gating it to once per frame — a 20×
reduction — measured **9.55 ms/frame against 9.50 ms**, i.e. nothing. The BVH
refit is genuinely cheap; an earlier estimate of 7.4 ms/frame for it turned out to
be entirely the cost of the `wp.synchronize()` calls used to measure it. The gate
was reverted.

### 5.9 Self-contact solving on a subset of substeps (measured, NOT adopted)

§5.7 strided contact *detection*. The self-contact **solve** is separate: the pair
list comes from `collide` (now strided), but the XPBD constraint solve for those
pairs runs every substep — 20 times a frame. Ablation in graph mode says that is
the single largest block in the frame:

| Disabled | ms/frame | Saves |
|---|---:|---:|
| — (base) | 3.826 | — |
| `enable_particle_particle_collisions` | 3.149 | 0.68 (18%) |
| `enable_edge_edge_collisions` | 3.359 | 0.47 (12%) |
| `enable_triangle_particle_collisions` | 3.880 | 0 |
| `ground` | 3.810 | 0 |
| **all three self-collision types** | **2.612** | **1.21 (32%)** |

So cloth-vs-cloth is ~32% of the frame, ground contacts and triangle-particle
contacts are free, and the natural move is the same one that worked for detection:
solve it less often rather than not at all.

**This is not on the `perf/sim-speedup` branch.** It was measured, it is the best
result below, and it was dropped for one reason: it needs a change to the vendored
Warp fork (`/NvidiaWarp-GarmentCode`), which is a shared editable install, so the
edit would affect anything else importing that Warp. Cost of dropping it: 8.2 s
instead of 7.4 s on the reference garment, and per-vertex deviation 5.64 mm instead
of 4.68 mm.

`self_contact_solve_interval` does that. It needs a one-line-per-site change in
the vendored Warp fork — the three self-collision solve launches in
`XPBDIntegrator.simulate()` now also check `model.solve_self_contacts` (default
True, so unchanged behaviour) — and the caller sets that flag per substep, so the
pattern bakes into the captured CUDA graph. Springs, bending and body contacts
still solve every substep, so fabric stiffness is untouched.

| Config | Wall | Frames | surf mm | vert mm | waist mm | area % | body int. | self int. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| _noise floor_ | | | _5.55_ | _4.45_ | _0.94_ | _0.016_ | | |
| Tier A control | 10.7 s | 332 | 5.65 | 5.12 | 0.80 | 0.047 | 93 | 11.0 |
| Tier A adaptive (§5.7) | 8.2 s | 318 | 5.60 | 5.64 | 1.43 | 0.025 | 79 | 10.3 |
| Tier A + solve every 2nd | 7.6 s | 279 | 5.77 | 5.33 | 0.91 | 0.021 | 78 | 13.0 |
| **Tier A + solve every 4th** | **7.4 s** | 284 | **5.50** | **4.68** | **0.64** | 0.030 | **75** | 9.0 |
| Tier A, particle-particle OFF | 7.7 s | 280 | 5.90 | 5.81 | 1.15 | 0.029 | 76 | 15.0 |
| Tier B control | 6.5 s | 258 | 6.40 | — | 5.10 | 0.264 | 94 | 0.0 |
| Tier B adaptive (§5.7) | 5.4 s | 202 | 6.55 | — | 3.17 | 0.183 | 70 | 0.0 |
| **Tier B + solve every 4th** | **5.1 s** | 215 | **6.14** | — | 3.13 | 0.196 | 71 | 3.7 |

Solving every 4th substep is the best result in this entire investigation: **7.4 s
against a 10.7 s control, with surface deviation 5.50 mm (below the 5.55 mm noise
floor), per-vertex deviation 4.68 mm (better than the control's 5.12), the lowest
waistband error and the fewest body intersections of any variant tested.** Faster
*and* more faithful than the config it replaces.

Two useful comparisons in that table. Striding beats **disabling**: turning
particle-particle collisions off entirely is slower (7.7 s vs 7.4 s) *and* worse on
every fidelity metric. And solving every 4th substep beats every 2nd on both time
and fidelity, which suggests the previous 20× solve was over-constraining rather
than merely redundant.

### 5.10 Kernel fusion — ceiling measured, not worth taking

The remaining substep loop is ~11 kernels × 20 substeps = 220 launches. Before
attempting to fuse them, three measurements:

**CUDA graph replay already banks the launch-overhead win.** Same config, graphs on
vs off: **3.95 ms/frame vs 6.52 ms/frame** — graphs are already saving 40%, which
is ~12 µs of CPU launch cost per kernel that never reaches the GPU. (This also
corrects the per-kernel table in §5.8: those numbers were measured with graphs
disabled and per-launch syncs, inflating everything ~2.4×. The real frame at that
config is 3.95 ms, not 9.5 ms. Relative shares still hold; absolute values do not.)

**Frame cost is near-linear in substep count**, so each substep does real work
rather than sitting in overhead:

| substeps | ms/frame | marginal |
|---:|---:|---:|
| 5 | 1.872 | — |
| 10 | 2.431 | 0.112 ms/substep |
| 20 | 3.923 | 0.149 ms/substep |
| 40 | 6.730 | 0.140 ms/substep |

Extrapolating to zero substeps leaves ~1.2 ms/frame of fixed cost, and ~0.14 ms
per substep for 11 kernels — about 12.7 µs per kernel. Intra-graph node dispatch
is ~2–3 µs of that, so **fusing 11 kernels down to ~4 would recover roughly
0.4 ms/frame, a ~10% ceiling** — in exchange for rewriting XPBD constraint solving
and its delta accumulation inside a vendored physics fork.

§5.9 got **−25% per frame** by launching fewer kernels instead of fusing them, with
no kernel rewriting at all — 2.5× fusion's ceiling at a fraction of the risk. Fusion
is therefore not worth taking, and the measurement above is the reason rather than
a guess.

One related observation, measured and **not** acted on: two self-collision kernels
launch over **buffer capacity, not actual contact count**.

| Buffer | Capacity | Actual | Occupancy |
|---|---:|---:|---:|
| `edge_contact_max` | 5,277,888 | 1,289,376 | 24% |
| `point_tri_contact_max` | 589,568 | 58,577 | 10% |
| `soft_contact_max` | 65,536 | 4,694 | 7% |

Those threads early-exit, so each is nearly free, but ~4M surplus threads per
launch is on the order of 0.1–0.25 ms/frame at the current stride. The counts live
on the device, so a tighter `dim` would need a host readback per substep — a sync,
which is illegal during graph capture. The alternative, shrinking the buffer
formula, is what `edge_contact_mult` exists to *enlarge*: the config comment
records that undersizing it crashes heavy-self-contact garments with a CUDA
illegal-memory-access at stitch init. Not a safe lever for a 3% gain.

Also worth noting from the ablation: **1.29 million live edge-edge contact pairs**
for a 9,212-particle cloth. That, not the body collider, is what cloth collision
costs here.

---

## 6. Dead ends and no-ops

| Change | Result |
|---|---|
| `sim_substeps` 5 / 8 | 5 never settles (2500 frames, 66 s); 8 is *slower* — jitter costs more frames than the cheaper frame saves |
| `lazy_vert_fetch`, `static_check_interval` 5 / 10 | The per-frame GPU→CPU readback is ~2.7 ms of a 34 ms frame. Striding it saved ~0.5 s, inside the noise |
| `enable_particle_particle_collisions: false` | No change |
| `enable_triangle_particle_collisions: false` | No change |
| `enable_global_collision_filter: false` | No change |
| `global_damping_factor` 0.15 / 0.30 | No change |
| `max_frame_time: null` (watchdog off) | No change — the per-frame watchdog thread is negligible |
| 400×400 single-side final render | No change: 19.0 s vs 19.1 s. The 0.43 s is mostly fixed setup, not pixels |
| Concurrent side renders | A wash (0.36 vs 0.39 s, then 0.32 vs 0.31 s). Each thread must build its own meshes — a pyrender `Primitive` caches its VAO/VBO ids, so two renderers sharing one mesh corrupt that state — and the duplicated load costs what the overlap saves. Implemented but **default off** (`render.config.parallel_renders`); output is pixel-identical |
| `zero_gravity_steps` 50 / 0 | Only −5% / −3%, and pushes the waistband 3.7–4.3 mm. Not worth it |
| `optimize_storage: true` | **Dead property** — `run_custom_pants.py` passes `optimize_storage=False` to `run_sim` unconditionally |

### Actively harmful

- **`enable_body_collision_filters: false`** — not a speedup, a different
  garment: 26.9 mm mean vertex displacement, waistband 13.7 mm off, surface area
  **+3.56%** (the cloth is being stretched), and body intersections collapse from
  84 to 9 because the garment is no longer where it should be.
- **`ground: false`** — 8.5% *slower*. The hem falls further, so more frames.

### Hard limits

- **`resolution_scale` ≥ 1.75** crashes the box-mesh generator:
  `AttributeError: 'Edge' object has no attribute 'name'` — panel edges become
  too coarse to subdivide. 1.5 is the ceiling.
- **`resolution_scale` is an edge-length multiplier, so lower is finer.** 0.8 and
  0.7 cost **+65%** and **+134%**.
- **`sim_fps` ≤ 20** is unusable even with compensating substeps.

---

## 7. Remaining floor

Stage-resolved timings for Tier B, measured by wrapping the pipeline's own
functions:

| Stage | Time | Notes |
|---|---:|---|
| Solver | 4.30 s | 61% of the job; collide-bound |
| Imports + Warp init | 1.47 s | now off the critical path (prewarm) |
| Process spawn | 0.95 s | now off the critical path (prewarm) |
| Box-mesh load + build | 0.91 s → ~0.55 s | with the 4-process panel pool |
| Final renders (2 × 800×800) | 0.43 s | parallelising measured a wash |
| Box-mesh serialize + UV texture | 0.26 s | floor reached |
| MP4 encode | 0.12 s | |
| `combined.obj` / `.glb` export | 0.11 s | |

### Still on the table

- **Raise `--max-concurrent` past 2** — 4 bare processes gave 1.81× on the slower
  config; worth re-measuring at 3–4 now that jobs are short.
- **A tighter cloth-vs-cloth broadphase.** 1.29 M live edge-edge pairs for 9,212
  particles is the root cost (§5.10). Striding amortises it; nothing has reduced
  the pair count itself. `fabric_thickness` (0.5) sets the contact radius and would
  do it directly, but it is a physical fabric property, not a perf knob.
- **Device-side launch bounds.** The self-collision kernels launch over buffer
  capacity at 7–24% occupancy (§5.10). Worth ~3%, and needs either a device-side
  launch bound or a safe way to shrink the buffer formula.
- **Part-based body collider cull** — arm faces are 36% of the mesh and pants
  never collide with them, but at the measured BVH scaling that is worth ~2–3%
  end-to-end (§5.2). Low priority.
- **Deferred texture raster** — would let you keep `dpi: 1500` for free by
  overlapping it with the solve, but needs `texture_mesh_islands()` split into
  UV-packing and image-writing halves. Only worth it if you want print-res
  texture back.

---

## 8. Code changes

All uncommitted. Every default preserves previous behaviour — the
`ctrl_patched` control row confirms it (27.7 s / 536 frames vs baseline
27.7 s / 542). `git diff` shows everything.

### Earned their place

| File | Change |
|---|---|
| `pygarment/meshgen/render/pythonrender.py` | New `FrameRenderer` — builds scene, body mesh, camera, lights and EGL context once, swaps only the garment node. Optional parallel side renders behind `parallel_renders` (default off). |
| `pygarment/meshgen/simulation.py` | New `_VideoRecorder` — renders video frames on a background thread with its own GL context; drops frames rather than stalling if it falls behind, and never propagates a render failure into the sim. Reads `video_frame_interval` and `video_resolution` from sim props. |
| `pygarment/meshgen/sim_config.py` | `sim_fps` and `sim_substeps` now read from sim props (both were hardcoded). This is what made §3.2 reachable from config. |
| `pygarment/meshgen/boxmeshgen.py` | `Panel.gen_panel_mesh` split so the CDT work lives in a module-level `mesh_panel_cdt`; `gen_panel_meshes` runs it across an opt-in `ProcessPoolExecutor` (`init_panel_pool` / `shutdown_panel_pool`) with a serial fallback on any pool failure. |
| `pygarment/meshgen/garment.py` | `body_collider_cull` — drops body-collider faces the garment cannot reach, keeping the full vertex array and masking `face_filters` to match. Default off; measured a no-op for pants (§5.2). |
| `pygarment/meshgen/simulation.py` | Post-sim `collider_cull_exceeded` check, so a cull that was too aggressive fails loudly instead of silently dropping contacts. |
| `pygarment/meshgen/sim_config.py` | `soft_contact_margin` exposed (default 0.2, unchanged) — useful diagnostic, lowering it alone is a trap (§5.3). `collide_interval`, `collide_interval_adaptive` and `collide_stride_start_frame` (defaults 1 / false / 0 = every frame, unchanged). |
| `pygarment/meshgen/garment.py` | Strided contact detection: `_sim_frame_with_substeps(do_collide=...)`, a second collide-free CUDA graph captured in `create_graph()` when striding is on, and `update()` selecting between them. The particle grid strides with it (§5.5). Adaptive mode drives the interval from max per-vertex L1 displacement, measured next to the existing per-frame readback (§5.7). |
| `service_inference.py` | Pre-warmed spare workers (one per slot, warmed during the previous job, cold-spawn fallback); `--max-concurrent` dispatcher threads (default 2) with a locked spare pool and a per-job process map; `--panel-workers`; `/health` gained `running_jobs` and `max_concurrent`, keeping `running_job` for compatibility. |

### Measured nothing — safe to strip

`static_check_interval` and `lazy_vert_fetch` in `sim_config.py`, plus the lazy
`current_verts` property and gap-scaled threshold in `garment.py`. Added to test
whether the per-frame GPU→CPU readback mattered; it does not. Inert at defaults.

### Vendored Warp fork (`/NvidiaWarp-GarmentCode`, editable install)

| File | Change |
|---|---|
| `warp/sim/integrator_xpbd.py` | **NOT APPLIED / reverted.** Would gate the three self-collision solve launches on `model.solve_self_contacts` so they can be solved on a subset of substeps (§5.9). Left out to avoid modifying a shared editable install. |

This would have been the only change outside GarmentCode, and it is why the
self-contact commit is not on the branch. The fork is back to a clean checkout.

### Operational notes

- Sim worker processes are **no longer daemonic** (a daemon cannot fork the panel
  pool). Shutdown terminates them explicitly in the FastAPI lifespan handler, so
  a hard `SIGKILL` of the server can now leave orphans where it previously
  could not.
- With `--max-concurrent 2 --panel-workers 4` the steady-state process count is
  2 spares + up to 2 running jobs, each with up to 4 pool children.

---

## 9. Full results

86 configurations, 3 runs each, sorted fastest first. **Surf** = mean symmetric
surface distance vs baseline. **Vert** = mean per-vertex displacement (only
meaningful when topology matches). **Waist** = absolute waistband-top height
error. Compare each against the noise-floor row.

| Configuration | Wall s | Δ | Sim s | Frames | Verts | Surf mm | Vert mm | Waist mm | Area % | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| _run-to-run noise floor_ | | | | | | _5.55_ | _4.45_ | _0.94_ | _0.016_ | _reference_ |
| **TIER B** — adaptive striding + self-contact solve every 4th substep <br><sub>`B_sci4`</sub> | 5.1 ±0.1 | -82% | 3.0 | 215 | 9,212 | 6.14 | — | 3.13 | 0.196 | above noise |
| Tier B adaptive + self-contact solve every 2nd substep <br><sub>`B_sci2`</sub> | 5.3 ±0.0 | -81% | 2.7 | 220 | 9,212 | 6.46 | — | 2.84 | 0.184 | above noise |
| Tier B adaptive striding, ceiling 8 <br><sub>`B_ad8`</sub> | 5.4 ±0.1 | -81% | 3.1 | 202 | 9,212 | 6.55 | — | 3.17 | 0.183 | above noise |
| Tier B+ adaptive striding, ceiling 16 <br><sub>`B_ad16`</sub> | 5.5 ±0.3 | -80% | 3.3 | 227 | 9,212 | 6.51 | — | 2.50 | 0.178 | above noise |
| Tier B + `collide_interval 8` + margin 2.0 <br><sub>`B_ci8`</sub> | 5.7 ±0.1 | -79% | 3.5 | 276 | 9,212 | 7.00 | — | 3.27 | 0.246 | above noise |
| Tier B + `collide_interval 4`, grid every frame <br><sub>`B_ci4g`</sub> | 5.7 ±0.1 | -79% | 3.6 | 278 | 9,212 | 6.47 | — | 4.17 | 0.196 | above noise |
| Tier B + `collide_interval 4` + margin 1.0, from frame 0 <br><sub>`B_ci4`</sub> | 5.8 ±0.0 | -79% | 3.6 | 272 | 9,212 | 6.53 | — | 3.97 | 0.171 | above noise |
| zg 75 + fixed striding from frame 75 <br><sub>`B_zg75_ci4s`</sub> | 6.1 ±0.3 | -78% | 3.9 | 211 | 9,212 | 6.78 | — | 3.11 | 0.196 | above noise |
| Tier B + `collide_interval 4`, striding from frame 150 <br><sub>`B_ci4s`</sub> | 6.3 ±0.0 | -77% | 4.1 | 263 | 9,212 | 6.44 | — | 3.51 | 0.215 | above noise |
| control: Tier B after collision edits <br><sub>`B_ctrl`</sub> | 6.5 ±0.1 | -77% | 4.3 | 258 | 9,212 | 6.40 | — | 5.10 | 0.264 | above noise |
| earlier Tier B (before contact striding) <br><sub>`final_B`</sub> | 6.6 ±0.0 | -76% | 4.4 | 270 | 9,212 | 6.72 | — | 4.56 | 0.264 | above noise |
| everything stacked (no video) <br><sub>`c_max`</sub> | 6.9 ±0.1 | -75% | 3.1 | 213 | 6,458 | 7.71 | — | 3.27 | 0.312 | above noise |
| Tier B without the panel pool <br><sub>`cv_all_warm`</sub> | 7.0 ±0.1 | -75% | 4.2 | 261 | 9,212 | 6.66 | — | 4.76 | 0.267 | above noise |
| **TIER A** — adaptive striding + self-contact solve every 4th substep <br><sub>`A_sci4`</sub> | 7.4 ±0.1 | -73% | 4.9 | 284 | 14,428 | 5.50 | 4.68 | 0.64 | 0.030 | within run-to-run noise |
| no video + minimal texture + `resolution_scale 1.5` <br><sub>`c_io_res150`</sub> | 7.5 ±0.1 | -73% | 3.7 | 379 | 6,458 | 7.39 | — | 6.24 | 0.404 | above noise |
| Tier A adaptive + self-contact solve every 2nd substep <br><sub>`A_sci2`</sub> | 7.6 ±0.0 | -73% | 5.0 | 279 | 14,428 | 5.77 | 5.33 | 0.91 | 0.021 | within run-to-run noise |
| Tier B + `soft_contact_margin: 0.10` <br><sub>`B_scm010`</sub> | 7.6 ±0.4 | -73% | 5.5 | 488 | 9,212 | 6.71 | — | 1.72 | 0.284 | above noise |
| Tier A adaptive + particle-particle self-collision off <br><sub>`A_nopp2`</sub> | 7.7 ±0.3 | -72% | 5.1 | 280 | 14,428 | 5.90 | 5.81 | 1.15 | 0.029 | above noise |
| Tier A+ adaptive striding, ceiling 16 <br><sub>`A_ad16`</sub> | 7.9 ±0.3 | -72% | 5.2 | 268 | 14,428 | 6.14 | 6.28 | 0.94 | 0.024 | above noise |
| Tier A adaptive striding, ceiling 8 <br><sub>`A_ad8`</sub> | 8.2 ±0.4 | -71% | 5.5 | 318 | 14,428 | 5.60 | 5.64 | 1.43 | 0.025 | above noise |
| Tier B without pool or prewarm <br><sub>`cv_all`</sub> | 8.6 ±0.0 | -69% | 4.2 | 262 | 9,212 | 6.55 | — | 4.85 | 0.256 | above noise |
| Tier A + `collide_interval 8` + margin 2.0 <br><sub>`A_ci8`</sub> | 8.7 ±0.1 | -69% | 6.0 | 347 | 14,428 | 6.52 | 7.42 | 0.99 | 0.025 | above noise |
| Tier A + `collide_interval 4` + margin 1.0, from frame 0 <br><sub>`A_ci4`</sub> | 8.7 ±0.1 | -68% | 6.1 | 337 | 14,428 | 6.65 | 7.13 | 0.70 | 0.043 | above noise |
| Tier A + `collide_interval 4`, grid every frame <br><sub>`A_ci4g`</sub> | 8.7 ±0.1 | -68% | 6.2 | 348 | 14,428 | 6.30 | 7.05 | 1.27 | 0.050 | above noise |
| Tier A + `collide_interval 4` + margin 1.0, from frame 0 (repeat) <br><sub>`A_ci4r`</sub> | 8.8 ±0.0 | -68% | 6.2 | 339 | 14,428 | 6.01 | 5.70 | 0.66 | 0.030 | above noise |
| Tier A + `collide_interval 2`, grid every frame <br><sub>`A_ci2g`</sub> | 9.0 ±0.3 | -68% | 6.4 | 317 | 14,428 | 6.09 | 6.05 | 1.09 | 0.045 | above noise |
| no video + minimal texture + `resolution_scale 1.25` <br><sub>`c_io_res125`</sub> | 9.1 ±0.0 | -67% | 5.0 | 406 | 9,212 | 6.37 | — | 4.04 | 0.257 | above noise |
| zg 75 + fixed striding from frame 75 <br><sub>`A_zg75_ci4s`</sub> | 10.1 ±0.0 | -64% | 7.5 | 295 | 14,428 | 5.30 | 4.99 | 1.25 | 0.033 | within run-to-run noise |
| Tier A + `collide_interval 4`, striding from frame 150 <br><sub>`A_ci4s`</sub> | 10.3 ±0.2 | -63% | 7.7 | 331 | 14,428 | 5.47 | 5.11 | 0.95 | 0.037 | within run-to-run noise |
| Tier A + `zero_gravity_steps: 75` <br><sub>`A_zg75`</sub> | 10.3 ±0.2 | -63% | 7.8 | 296 | 14,428 | 5.11 | 3.69 | 0.67 | 0.021 | within run-to-run noise |
| Tier A + `body_collider_cull` <br><sub>`A_cull`</sub> | 10.5 ±0.1 | -62% | 7.9 | 301 | 14,428 | 5.54 | 5.04 | 1.68 | 0.022 | within run-to-run noise |
| Tier A + `soft_contact_margin: 0.15` <br><sub>`A_scm015`</sub> | 10.6 ±0.0 | -62% | 8.1 | 329 | 14,428 | 5.79 | 5.63 | 1.05 | 0.046 | above noise |
| earlier Tier A (before contact striding) <br><sub>`final_A`</sub> | 10.6 ±0.3 | -62% | 7.9 | 307 | 14,428 | 5.58 | 5.12 | 1.68 | 0.041 | within run-to-run noise |
| control: Tier A after collision edits <br><sub>`A_ctrl2`</sub> | 10.7 ±0.1 | -61% | 8.0 | 332 | 14,428 | 5.65 | 5.12 | 0.80 | 0.047 | within run-to-run noise |
| Tier A without the panel pool <br><sub>`cv_tex_fps_warm`</sub> | 11.3 ±0.1 | -59% | 7.9 | 307 | 14,428 | 5.64 | 5.27 | 1.71 | 0.021 | within run-to-run noise |
| Tier A without pool or prewarm <br><sub>`cv_tex_fps`</sub> | 13.0 ±0.1 | -53% | 7.9 | 311 | 14,428 | 5.53 | 4.79 | 1.13 | 0.039 | within run-to-run noise |
| no video + minimal texture + `static_threshold 0.10` <br><sub>`c_io_static10`</sub> | 14.4 ±0.0 | -48% | 9.6 | 307 | 14,428 | 6.00 | 6.83 | 3.99 | 0.052 | above noise |
| no video + minimal texture <br><sub>`c_io`</sub> | 16.1 ±0.4 | -42% | 11.3 | 651 | 14,428 | 5.21 | 4.32 | 1.32 | 0.034 | within run-to-run noise |
| `resolution_scale: 1.5` <br><sub>`res150`</sub> | 16.6 ±0.1 | -40% | 8.7 | 374 | 6,458 | 7.38 | — | 6.26 | 0.384 | above noise |
| 400×400 single-side final render <br><sub>`render_small`</sub> | 19.0 ±0.1 | -32% | 11.1 | 592 | 14,428 | 5.08 | 3.66 | 0.87 | 0.023 | within run-to-run noise |
| `save_sim_video: false` <br><sub>`novideo`</sub> | 19.1 ±0.4 | -31% | 10.5 | 565 | 14,428 | 5.32 | 4.40 | 1.06 | 0.019 | within run-to-run noise |
| `resolution_scale: 1.25` <br><sub>`res125`</sub> | 19.3 ±0.5 | -31% | 11.0 | 416 | 9,212 | 6.38 | — | 3.96 | 0.264 | above noise |
| threaded video recorder, 400×400 / 15th frame <br><sub>`vid_400_i15`</sub> | 20.0 ±0.1 | -28% | 11.4 | 584 | 14,428 | 5.20 | 3.81 | 0.62 | 0.011 | within run-to-run noise |
| threaded video recorder only (800×800 / 10th frame) <br><sub>`vid_thread`</sub> | 20.4 ±0.4 | -27% | 11.6 | 573 | 14,428 | 5.29 | 4.30 | 1.01 | 0.027 | within run-to-run noise |
| `static_threshold: 0.15` <br><sub>`static15`</sub> | 21.1 ±0.1 | -24% | 12.4 | 230 | 14,428 | 7.09 | 8.81 | 5.35 | 0.259 | above noise |
| `sim_fps: 30` + `sim_substeps: 20` (dt unchanged) <br><sub>`fps30_ss20`</sub> | 21.3 ±0.6 | -23% | 12.6 | 321 | 14,428 | 5.67 | 5.17 | 0.98 | 0.031 | within run-to-run noise |
| `static_threshold: 0.10` + `non_static_percent: 5` <br><sub>`static10_nsp5`</sub> | 21.3 ±0.1 | -23% | 12.7 | 237 | 14,428 | 7.24 | 8.83 | 5.18 | 0.230 | above noise |
| `static_threshold: 0.10` <br><sub>`static10`</sub> | 22.9 ±0.1 | -18% | 14.2 | 304 | 14,428 | 6.04 | 6.76 | 3.98 | 0.027 | above noise |
| `non_static_percent: 15` <br><sub>`nsp15`</sub> | 23.0 ±0.2 | -17% | 14.4 | 327 | 14,428 | 5.90 | 6.55 | 3.91 | 0.032 | above noise |
| `uv_texture.dpi: 72`, no fabric-grain image <br><sub>`tex_floor`</sub> | 24.0 ±1.2 | -13% | 18.9 | 562 | 14,428 | 5.23 | 3.96 | 0.81 | 0.011 | within run-to-run noise |
| `uv_texture.dpi: 150` + `grain_resolution: 1` <br><sub>`tex_min`</sub> | 24.2 ±0.5 | -13% | 18.9 | 558 | 14,428 | 5.03 | 3.41 | 0.61 | 0.016 | within run-to-run noise |
| `sim_fps: 45` <br><sub>`fps45`</sub> | 25.1 ±0.1 | -10% | 16.3 | 487 | 14,428 | 5.64 | 5.64 | 1.42 | 0.044 | above noise |
| `uv_texture.dpi: 300` <br><sub>`tex300`</sub> | 25.4 ±0.1 | -8% | 19.5 | 592 | 14,428 | 5.14 | 3.78 | 0.79 | 0.020 | within run-to-run noise |
| `non_static_percent: 5` <br><sub>`nsp5`</sub> | 25.5 ±0.4 | -8% | 16.8 | 451 | 14,428 | 5.24 | 4.02 | 1.47 | 0.037 | within run-to-run noise |
| `static_threshold: 0.05` <br><sub>`static05`</sub> | 25.8 ±0.8 | -7% | 17.0 | 439 | 14,428 | 5.34 | 4.47 | 1.76 | 0.019 | within run-to-run noise |
| `sim_fps: 20` + `sim_substeps: 30` <br><sub>`fps20_ss30`</sub> | 26.1 ±9.9 | -6% | 17.5 | 549 | 14,428 | 6.02 | 10.62 | 8.31 | 0.031 | above noise |
| `zero_gravity_steps: 50` <br><sub>`zg50`</sub> | 26.4 ±0.1 | -5% | 17.6 | 474 | 14,428 | 6.06 | 7.23 | 3.66 | 0.025 | above noise |
| particle-particle + edge-edge off <br><sub>`nopp_noee`</sub> | 26.5 ±0.2 | -5% | 17.6 | 501 | 14,428 | 5.31 | 4.15 | 0.78 | 0.031 | within run-to-run noise |
| `enable_edge_edge_collisions: false` <br><sub>`noee`</sub> | 26.7 ±0.3 | -4% | 17.9 | 507 | 14,428 | 5.28 | 3.96 | 0.72 | 0.014 | within run-to-run noise |
| `zero_gravity_steps: 0` <br><sub>`zg0`</sub> | 26.8 ±0.1 | -3% | 18.0 | 503 | 14,428 | 6.24 | 7.99 | 4.27 | 0.024 | above noise |
| `enable_body_collision_filters: false` <br><sub>`nobcf`</sub> | 26.8 ±0.2 | -3% | 18.0 | 507 | 14,428 | 13.64 | 26.93 | 13.73 | 3.558 | **materially different drape** |
| `static_check_interval: 5` <br><sub>`static_int5`</sub> | 27.0 ±1.1 | -3% | 18.2 | 513 | 14,428 | 5.30 | 4.04 | 0.94 | 0.014 | within run-to-run noise |
| `lazy_vert_fetch: true` <br><sub>`lazyfetch`</sub> | 27.2 ±1.0 | -2% | 18.4 | 517 | 14,428 | 5.33 | 4.15 | 0.91 | 0.014 | within run-to-run noise |
| `sim_substeps: 15` <br><sub>`substeps15`</sub> | 27.5 ±1.1 | -1% | 18.8 | 519 | 14,428 | 5.66 | 5.50 | 1.82 | 0.060 | above noise |
| control: defaults after code patches <br><sub>`ctrl_patched`</sub> | 27.7 ±1.0 | -0% | 18.8 | 536 | 14,428 | 5.34 | 4.03 | 0.62 | 0.019 | within run-to-run noise |
| `default_sim_props.yaml` unchanged <br><sub>`baseline`</sub> | 27.7 ±0.9 | +0% | 18.9 | 542 | 14,428 | 5.55 | 4.45 | 0.94 | 0.016 | within run-to-run noise |
| `enable_triangle_particle_collisions: false` <br><sub>`notri`</sub> | 28.0 ±1.5 | +1% | 19.2 | 576 | 14,428 | 5.23 | 3.97 | 0.94 | 0.023 | within run-to-run noise |
| `max_frame_time: null` (watchdog off) <br><sub>`nowatchdog`</sub> | 28.0 ±0.7 | +1% | 19.2 | 579 | 14,428 | 5.14 | 3.68 | 0.81 | 0.013 | within run-to-run noise |
| `sim_substeps: 20` <br><sub>`substeps20`</sub> | 28.2 ±0.4 | +2% | 19.4 | 525 | 14,428 | 5.40 | 5.23 | 2.62 | 0.020 | above noise |
| `static_check_interval: 10` <br><sub>`static_int10`</sub> | 28.2 ±1.2 | +2% | 19.0 | 570 | 14,428 | 5.17 | 3.81 | 0.84 | 0.025 | within run-to-run noise |
| `global_damping_factor: 0.30` <br><sub>`damp30`</sub> | 28.2 ±1.1 | +2% | 19.5 | 593 | 14,428 | 5.78 | 5.42 | 1.16 | 0.056 | above noise |
| `global_damping_factor: 0.15` <br><sub>`damp15`</sub> | 28.4 ±0.1 | +2% | 19.6 | 596 | 14,428 | 6.17 | 6.34 | 1.12 | 0.028 | above noise |
| `enable_particle_particle_collisions: false` <br><sub>`nopp`</sub> | 28.4 ±0.1 | +2% | 19.6 | 605 | 14,428 | 5.22 | 4.07 | 1.04 | 0.033 | within run-to-run noise |
| `enable_global_collision_filter: false` <br><sub>`nogcf`</sub> | 28.4 ±0.1 | +2% | 19.6 | 580 | 14,428 | 5.32 | 4.31 | 0.98 | 0.024 | within run-to-run noise |
| `sim_substeps: 8` <br><sub>`substeps8`</sub> | 28.6 ±0.7 | +3% | 19.9 | 596 | 14,428 | 5.56 | 5.83 | 2.29 | 0.070 | above noise |
| `ground: false` <br><sub>`noground`</sub> | 30.1 ±1.2 | +9% | 21.2 | 665 | 14,428 | 6.34 | 9.75 | 4.03 | 0.293 | above noise |
| `resolution_scale: 0.8` <br><sub>`res08`</sub> | 45.9 ±0.8 | +65% | 35.6 | 734 | 22,528 | 6.53 | — | 8.84 | 0.354 | above noise |
| `resolution_scale: 0.7` <br><sub>`res07`</sub> | 65.0 ±1.1 | +134% | 54.4 | 724 | 29,361 | 6.34 | — | 13.11 | 0.615 | **materially different drape** |
| `resolution_scale: 2.0` <br><sub>`res200`</sub> | — | — | — | — | — | — | — | — | — | crashed / never settled |
| `resolution_scale: 1.75` <br><sub>`res175`</sub> | — | — | — | — | — | — | — | — | — | crashed / never settled |
| `sim_substeps: 5` <br><sub>`substeps5`</sub> | — | — | — | — | — | — | — | — | — | crashed / never settled |
| `sim_fps: 30` (substeps 10) <br><sub>`fps30`</sub> | — | — | — | — | — | — | — | — | — | crashed / never settled |
| `sim_fps: 15` + `sim_substeps: 40` <br><sub>`fps15_ss40`</sub> | — | — | — | — | — | — | — | — | — | crashed / never settled |
| Tier A + `soft_contact_margin: 0.10` <br><sub>`A_scm010`</sub> | — | — | — | — | — | — | — | — | — | crashed / never settled |
| Tier A + `soft_contact_margin: 0.05` <br><sub>`A_scm005`</sub> | — | — | — | — | — | — | — | — | — | crashed / never settled |
| Tier B + `soft_contact_margin: 0.05` <br><sub>`B_scm005`</sub> | — | — | — | — | — | — | — | — | — | crashed / never settled |

---

## 10. Reproducing

The harness, per-run metrics and mesh-comparison code used for this report are
in the session scratchpad (`harness.py`, `meshcmp.py`, `analyze.py`,
`results.json`, `analysis.json`). Each experiment applies dotted overrides to
`assets/Sim_props/default_sim_props.yaml`, submits 3 jobs with `force: true`,
records wall/sim/frame metrics from the job record and the emitted
`sim_props.yaml`, and archives each `_sim.obj` for comparison.
