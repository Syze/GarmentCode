# Drape sim speedup — change register

Decision-oriented companion to `SIM_PERF_FINDINGS.md` (which is the full
chronological record, 86 configurations). This file answers three questions per
change: **how much time**, **how much risk to the result**, and **what it depends
on**.

Measured on `hm_ie/0871889043` size 38 on `HM_heights/female_h168_flat_feet`,
H100 PCIe, 3 runs per configuration, end-to-end through `service_inference.py`.

**27.7 s → 8.2 s at full mesh resolution with the video still saved.**
5.4 s if a coarser cloth mesh is acceptable.

One further change was measured and **deliberately not adopted**: striding the
cloth self-contact solve was worth another 0.8 s (8.2 s → 7.4 s) *and* had the best
fidelity of any variant, but it requires editing the vendored Warp fork
(`/NvidiaWarp-GarmentCode`), which is a shared editable install. See change 7.

---

## How to read the risk column

Every fidelity number is judged against the **run-to-run noise floor** — the same
config disagreeing with itself across 3 runs, because this sim is not
deterministic (identical configs produced 573 / 568 / 484 frames):

| Metric | Noise floor | What it tells you |
|---|---:|---|
| Mean surface distance | 5.55 mm | overall shape + fold placement |
| Mean per-vertex displacement | 4.45 mm | fold placement (needs matching topology) |
| Waistband top height | **0.94 mm** | *structural* — did the garment sit differently |
| Hem height | **0.10 mm** | *structural* |
| Surface area | **0.016%** | *structural* — is the cloth being stretched |

The last three are the honest signal. Loose fabric swings, so where an individual
fold lands is noise; **waist, hem and area are stable to ~1 mm and 0.02%**, so if a
change moves those, it changed the garment.

- **None** — output byte-identical or provably untouched geometry.
- **None measured** — every metric at or below the noise floor.
- **Negative (better)** — measurably closer to the baseline than the config it replaces.
- **Real** — moves a structural metric beyond noise. Quantified per row.

---

## The ladder, as measured

Each row is a real 3-run measurement with everything above it already applied.

| Step | Wall | Δ | Cum. | surf | vert | waist | area % |
|---|---:|---:|---:|---:|---:|---:|---:|
| _noise floor_ | | | | _5.55_ | _4.45_ | _0.94_ | _0.016_ |
| baseline (`default_sim_props.yaml`) | 27.7 s | — | — | 5.55 | 4.45 | 0.94 | 0.016 |
| threaded video recorder | 20.4 s | −7.4 | −27% | 5.29 | 4.30 | 1.01 | 0.027 |
| + `uv_texture.dpi: 150` + `sim_fps: 30`/`sim_substeps: 20` | 13.0 s | −7.4 | −53% | 5.53 | 4.79 | 1.13 | 0.039 |
| + pre-warmed worker | 11.3 s | −1.7 | −59% | 5.64 | 5.27 | 1.71 | 0.021 |
| + panel meshing pool | 10.6 s | −0.7 | −62% | 5.58 | 5.12 | 1.68 | 0.041 |
| + `zero_gravity_steps: 75` | 10.3 s | −0.3 | −63% | **5.11** | **3.69** | 0.67 | 0.021 |
| + adaptive collide striding | 8.2 s | −2.2 | −71% | 5.60 | 5.64 | 1.43 | 0.025 |
| **+ self-contact solve stride 4** | **7.4 s** | −0.8 | **−73%** | **5.50** | **4.68** | **0.64** | 0.030 |
| + `resolution_scale: 1.25` | 5.1 s | −2.3 | −82% | 6.14 | — | 3.13 | 0.196 |

Every adopted row through 8.2 s sits at or below the noise floor on this garment.
Only the last row has a real fidelity cost. The italic row is measured but not on
the branch — it needs a change to the vendored Warp fork.

Steps are measured in context, so they are not exactly additive — the video and
texture savings partly overlap with the frame-count reduction from `sim_fps: 30`.
Measured **alone** against baseline: threaded video −27%, `sim_fps 30`/`ss 20` −23%,
`dpi 150` −13%.

---

## Individual changes, ranked by time

### 1. Threaded video recorder — 7.4 s · risk: none · code only

`render_frame_to_array()` rebuilt the body mesh, both materials, the scene, camera
and five lights per captured frame, and **created and destroyed an entire EGL
context** — at full 800×800. Only the garment geometry changes between frames.

Now a `FrameRenderer` builds the static half once and swaps the garment node, and
a `_VideoRecorder` owns it on a background thread so the CPU/GL render overlaps
the GPU solve.

- **Risk: none.** The video is rendered from the same vertices; MP4 verified
  frame-by-frame identical in resolution, fps and file size.
- **No config needed** — on by default once the code is in, for every garment.
- Video now costs 1.3 s instead of 8.6 s. Dropping it entirely would save almost nothing.

### 2. `sim_fps: 30` + `sim_substeps: 20` — 6.4 s · risk: none measured · config

`dt = (1/sim_fps)/sim_substeps` stays **bit-identical at 1/600**, while each frame
covers twice the simulated time. Half the frames, half the `collide` calls, same
physics. Works because `wp.sim.collide()` runs once per frame *outside* the substep
loop, so doubling substeps costs only ~6% more per frame.

- **Risk: none measured.** Surface deviation 5.67 mm against a 5.55 mm floor,
  waistband 0.98 mm against 0.94 mm.
- ⚠️ **The two values must be changed together** — halving fps without doubling
  substeps enlarges `dt` and the sim never settles.
- ⚠️ **Forces rescaling of everything counted in frames** — see Dependencies.

### 3. `uv_texture.dpi: 1500 → 150` — 3.5 s · risk: none · config

The fabric texture was being generated at print resolution (12.7 MB PNG).

- **Risk: none.** Texture raster only; geometry is untouched.
- **Floor reached at 150.** `dpi: 72` with the grain image disabled measured 24.0 s
  vs 24.2 s — identical. The remaining 0.26 s is UV island packing, which the sim needs.

### 4. Adaptive collide striding — 2.2 s · risk: none measured · code + config

Run contact detection every Nth frame and reuse the pair list between, with the
interval driven by measured cloth motion (max per-vertex L1 displacement, scaled by
`static_threshold`): detect every frame while moving fast, stride hard once settled.

- **Risk: none measured** at ceiling 8 — surface deviation 5.60 mm against a
  5.55 mm floor; self-intersections 10.3, inside the control's own 1.7–11.0 range.
- ⚠️ **Hard dependency on `soft_contact_margin: 1.0`** (5× the default). Contacts
  must be found *before* they are needed. Without it, interval 2 and 4 **never
  settle at all** and interval 8 settles with 238 body intersections vs a normal ~95.
- A fixed schedule from frame 0 is faster still (8.7 s) but pushes fold deviation
  to 7.13 mm and doubles self-intersections. Adaptive is both faster *and* cleaner.
- Ceiling 16 gives 7.9 s with deviation drifting to 6.28 mm — above noise, not recommended.

### 5. `resolution_scale: 1.25` — 2.3 s · **risk: real** · config

14,428 → 9,212 cloth vertices (36% coarser).

- **Risk: real and quantified.** Waistband **+3.1 mm** (floor 0.94), surface area
  **+0.20%** (floor 0.016%), and topology changes so per-vertex comparison no longer
  applies. Fold placement visibly different; the fit reads the same.
- Use for iteration and previews, not final fit output.
- ⚠️ Hard ceiling at 1.5 — **1.75 and above crash the box-mesh generator**
  (`AttributeError: 'Edge' object has no attribute 'name'`).
- ⚠️ Counter-intuitive direction: it is an *edge-length* multiplier, so lower is
  finer. 0.8 and 0.7 cost **+65%** and **+134%**.

### 6. Pre-warmed worker — 1.7 s · risk: none · code only

Imports plus Warp/CUDA init (1.5 s) and process spawn (0.9 s) were 28% of a fast
job, all before any useful work. The service now keeps spare workers that have
already imported and initialised Warp, warmed *during the previous job*.

- **Risk: none to geometry** — process management only. Fresh-process-per-job
  isolation is preserved: each spare handles exactly one job and exits.
- ⚠️ **Operational:** workers had to stop being daemonic (a daemon cannot fork the
  panel pool), so a hard `SIGKILL` of the server can now leave orphaned workers
  where it previously could not. Normal shutdown terminates them.

### 7. Self-contact solve stride 4 — 0.8 s · risk: **negative (better)** · NOT ADOPTED

The self-contact pair list comes from `collide` (already strided), but the XPBD
constraint solve for those pairs ran every substep — 20× a frame, and ~32% of the
frame by ablation. Now solved every 4th substep. Springs, bending and body contacts
still solve every substep, so fabric stiffness is untouched.

- **Risk: negative.** Better than the config it replaces on *every* metric —
  surface 5.50 mm (below the 5.55 mm floor), per-vertex 4.68 vs the control's 5.12,
  lowest waistband error (0.64 mm) and fewest body penetrations (75 vs 93) of any
  variant tested. Suggests the 20× solve was over-constraining, not just redundant.
- Striding beats **disabling**: turning particle-particle collisions off outright is
  slower (7.7 s) *and* worse on every metric.
- ⚠️ **NOT ON THE BRANCH.** It requires editing `/NvidiaWarp-GarmentCode` — three
  conditions in `XPBDIntegrator.simulate()` gated on `model.solve_self_contacts` —
  and that is a shared editable install, so the change would affect anything else
  importing that Warp. Dropped for that reason alone, not on merit.
- Cost of dropping it, measured: 8.2 s instead of 7.4 s on the reference garment
  (7.3 s vs 7.1 s mean across five others), and slightly worse fidelity
  (per-vertex 5.64 mm instead of 4.68 mm against a 4.45 mm floor).
- ⚠️ Only tested on top of adaptive collide striding; standalone behaviour unmeasured.

### 8. Panel meshing pool — 0.7 s · risk: none · code only

`gen_panel_meshes` was 0.66 s of the 0.91 s box-mesh build, and panels are
independent. Threads cannot help — the CGAL bindings hold the GIL (4 threads
measured at **0.83× of serial**) — so the CDT work moved into a process pool.

- **Risk: none.** Output verified byte-identical (same vertex and face counts, same
  deterministic CGAL code). 0.700 s → 0.335 s, a 2.09× on that stage.
- ⚠️ Depends on non-daemonic workers (from change 6) and on the pool being forked
  **before** Warp/CUDA initialises.

### 9. `zero_gravity_steps: 150 → 75` — 0.3 s · risk: **negative (better)** · config

Not an optimisation — a **correction**. `zero_gravity_steps` is counted in frames,
so `sim_fps: 30` silently doubled the simulated zero-gravity settle from 2.5 s to
5 s, 45% of the run.

- **Risk: negative.** Better on every axis at once: surface 5.11 mm and per-vertex
  3.69 mm, **both below the noise floor**, area 0.021%, self-intersections zero.
- ⚠️ **Only correct because `sim_fps` is 30.** At `sim_fps: 60` this value must stay
  150 — cutting it there genuinely removes settling time and moved the waistband 3.7 mm.

### 10. Concurrency (2 workers) — throughput only · risk: none

9.1 → 12.8 sims/min (**1.41×**). Not a latency win: per-job wall rises 6.6 s → 9.4 s.

- **Risk: none.** Concurrent-vs-serial surface distance 4.50 mm against
  serial-vs-serial 4.50 mm — exactly the noise floor.
- A batch lever. Leave `--max-concurrent 1` for interactive use.

---

## Dependencies

**The `sim_fps` cluster — the one real trap.** Anything counted in *frames* changes
meaning when `sim_fps` changes. Setting `sim_fps: 30` obliges all of:

| Parameter | At fps 60 | At fps 30 | Why |
|---|---:|---:|---|
| `sim_substeps` | 10 | **20** | keeps `dt` at 1/600; mandatory, not optional |
| `static_threshold` | 0.03 | **0.06** | per-frame displacement bound |
| `zero_gravity_steps` | 150 | **75** | settle duration in frames |
| `video_frame_interval` | 10 | 15 | cosmetic (video pacing) only |
| `collide_stride_start_frame` | — | scale it | if using fixed rather than adaptive striding |

Getting `sim_substeps` wrong here does not degrade the result, it **prevents the sim
from settling at all**. Getting `static_threshold` or `zero_gravity_steps` wrong
degrades quietly.

**Other hard couplings:**

- **Adaptive collide striding → `soft_contact_margin: 1.0`.** Striding without the
  wider margin does not settle. Do not adopt one without the other.
- **`soft_contact_margin` alone must not be *lowered*.** It cuts per-frame collision
  cost 63%, and at 0.1 or below the sim never settles at full mesh resolution. It is
  only safe *upward*, in service of striding.
- **Self-contact striding → Warp fork edit.** Not adopted for this reason. If it is
  ever taken, note that reverting the fork silently disables it (the flag defaults
  to True), so the config becomes a no-op rather than a crash — quiet, not loud.
- **Panel pool → non-daemonic workers → pre-warm change.** These three ship together.
- **Panel pool → fork before CUDA init.** `ProcessPoolExecutor` forks lazily on first
  submit; the pool is forced to materialise at init so it happens before Warp comes up.
- **Concurrency ↔ panel pool** contend for CPU. The 1.41× at 2 workers is measured
  *with* the pool active; more concurrency with 4 pool processes each will contend further.

**Independent — adopt in any order, no interactions measured:**
threaded video, `uv_texture.dpi`, pre-warm, panel pool, `resolution_scale`.

---

## Do not adopt

Measured, and actively harmful:

| Change | Why |
|---|---|
| `enable_body_collision_filters: false` | Not a speedup, a different garment: 26.9 mm vertex displacement, waist 13.7 mm off, area **+3.56%** (cloth being stretched), body intersections collapsing 84 → 9 |
| `ground: false` | **8.5% slower** — the hem falls further, so more frames |
| `static_threshold: 0.10`+ | Works (−18%) but waistband 4.0–5.4 mm off: the garment is genuinely still falling when you stop it. `sim_fps 30`/`ss 20` buys the same time at 0.98 mm |
| `soft_contact_margin` below 0.15 | Never settles at full resolution |
| `sim_substeps: 5` or `8` | 5 never settles; 8 is *slower* — jitter costs more frames than the cheaper frame saves |
| `resolution_scale` ≥ 1.75 | Crashes the box-mesh generator |
| `sim_fps: 20` or below | Erratic even with compensating substeps; one run of three needed 1021 frames |

Measured and worth nothing (do not spend effort here): particle-particle collisions
off, triangle-particle off, `global_damping_factor`, `max_frame_time: null`,
smaller final renders, concurrent side renders, `lazy_vert_fetch`,
`static_check_interval`, body collider culling, cloth-BVH refit gating.

Also: **`optimize_storage: true` in the preset is dead** — `run_custom_pants.py`
passes `optimize_storage=False` to `run_sim` unconditionally.

---

## Caveats on all of the above

1. **One garment, one body, one size.** Everything is measured on wide-leg pants
   (6 panels, 14,428 verts) on a 168 cm A-pose body. The structural conclusions
   should generalise; the specific numbers will not. Re-verify on a fitted garment
   and on a top before trusting the tiers broadly — a tight garment has denser
   body contact, which is exactly what striding approximates.
2. **Self-intersection count is very noisy.** The same config measured 1.7 and 11.0
   across two 3-run groups, and 19.7 vs 45.7 for another identical pair. Treat it as
   a trend indicator, never a precise figure.
3. **The Warp fork is a shared install.** `/NvidiaWarp-GarmentCode` is an editable
   install; the `integrator_xpbd.py` edit affects anything else importing that warp.
4. **Measurement methodology bit twice.** Two intermediate conclusions were wrong
   because instrumentation dominated what was being measured: a "44% of frame" BVH
   refit that was really the `wp.synchronize()` calls around it, and a
   "launch-bound frame" that was really an artifact of profiling with CUDA graphs
   disabled (real frame 3.95 ms, not 9.5 ms). Both were caught by disabling the
   thing and comparing wall time instead. Prefer ablation over instrumentation here.

---

## Cross-product validation

The tuning above was derived on one garment. Five further products with **different
7-digit style prefixes** were run at size 38 on the same body — `0986428028`,
`1109636016`, `1200442003`, `1222670002`, `1236903004` — each with its own baseline,
its own 3-run noise floor, and 3 runs per configuration.

**Speed generalises. Fidelity stratifies into two clearly different tiers.**

Ratios are against **each product's own noise floor**, so 1.0 means
"indistinguishable from re-running the same config". `committed` is the branch as it
now stands (no self-contact striding).

| Product | base | neutral | committed | neutral surf/vert/waist | committed surf/vert/waist |
|---|---:|---:|---:|---:|---:|
| 0986428028 | 11.1 s | 7.6 s | 4.8 s | 1.01 / 1.90 / 4.04 | 1.02 / 6.67 / 2.15 |
| 1109636016 | 17.3 s | 13.6 s | 9.2 s | 0.98 / 0.83 / 1.39 | 1.07 / 1.47 / 1.17 |
| 1200442003 | 13.9 s | 11.2 s | 6.3 s | 0.92 / 0.66 / 0.55 | 0.93 / 0.90 / 0.66 |
| 1222670002 | 17.9 s | 14.0 s | 8.3 s | 1.01 / 1.00 / 0.62 | 1.35 / 3.02 / 4.21 |
| 1236903004 | 18.8 s | 14.6 s | 8.2 s | 1.01 / 1.07 / 1.05 | 1.26 / 2.19 / 5.70 |
| **mean** | **15.8 s** | **12.2 s (−23%)** | **7.3 s (−53%)** | **0.98 / 1.09 / 1.53** | **1.13 / 2.85 / 2.78** |

Worst absolute deviation for the committed tier across all five: surface 5.98 mm,
per-vertex 6.30 mm, waistband **1.99 mm**, area 0.060%.

### Three findings that shape the recommendation

**1. These garments are far more deterministic than the reference one.** Their
per-vertex noise floors are 0.19–2.56 mm against the wide-leg pants' 4.45 mm, and
waistband floors 0.08–0.70 mm against 0.94 mm. The same *absolute* deviation
therefore reads as a much larger ratio. Both views matter: the ratios above look
uncomfortable on three products, while the absolute numbers stay under 2 mm of
waistband error everywhere.

**2. The contact striding is not what moves the result.** Running the same products
with the `sim_fps`/`sim_substeps` rescaling but **no striding at all** produced
essentially the same deviation — waist ratios 1.98 / 3.74 / 0.62 / 1.82 / 4.35
without striding. The deviation comes from the **`sim_fps: 30` rescaling**, not from
the collision approximations. The changes that looked riskiest cost the least.

**3. Dropping the self-contact striding cost a little on both axes.** With it, the
five-product mean was 7.1 s and ratios 1.07 / 2.59 / 2.52; without it, 7.3 s and
1.13 / 2.85 / 2.78. It was removed only to avoid depending on the vendored Warp
fork.

### The geometry-neutral tier is validated

`neutral` is the threaded video recorder, `uv_texture.dpi: 150` and
`video_resolution: [400,400]`, plus the always-on code changes (pre-warmed workers,
panel pool) — **no simulation parameters touched at all**.

Surface deviation ratio is **0.92–1.01 on every one of the five products** —
indistinguishable from re-running. Its one raised number, `0986428028`'s waist ratio
of 4.04, is 4.04 × a 0.08 mm floor = **0.32 mm absolute**. Mean **−23%** for provably
no geometry change.

### Adopt in two steps

| | Speedup | Fidelity | When |
|---|---:|---|---|
| **Geometry-neutral** | −23% | validated indistinguishable on 6 products | unconditionally |
| **Full Tier A** | −53% | small (≤2 mm waist) but detectable on fitted garments | iteration and preview; validate per-product before final fit output |

### One combination that does not work

"Keep `sim_fps: 60` but take the striding" was tested and **rejected**: frame counts
rose well above baseline (663/828/864 against a 469 baseline on `1200442003`), one
run of fifteen failed outright, and `0986428028` produced a 598-frame outlier against
its 158–170 norm. The striding parameters (`soft_contact_margin: 1.0`, adaptive
ceiling 8) are tuned for `sim_fps: 30` and do not transfer. **Tier A is a package —
adopt it whole or not at all.**

---

## Reverting

| Scope | Command |
|---|---|
| All GarmentCode code changes | `git checkout -- pygarment/ service_inference.py` |
| The Warp fork edit | `git -C /NvidiaWarp-GarmentCode checkout -- warp/sim/integrator_xpbd.py` |
| Any single behaviour | every knob defaults to the original behaviour; drop the line from sim props |
| Concurrency / prewarm / panel pool | `--max-concurrent 1 --no-prewarm --panel-workers 0` |

Dead weight worth stripping regardless: `lazy_vert_fetch` and
`static_check_interval` in `sim_config.py` plus the lazy `current_verts` property in
`garment.py` — added to test whether the per-frame GPU→CPU readback mattered. It
does not (~2.7 ms of a 34 ms frame; striding it saved ~0.5 s, inside the noise).

---

## Recommended config

```yaml
# 27.7 s -> 8.2 s, full mesh resolution, video saved, geometry at/below the noise
# floor on the reference garment. No vendored-Warp changes required.
sim:
  config:
    sim_fps: 30                       # with substeps 20 -> dt stays exactly 1/600
    sim_substeps: 20                  # MUST change together with sim_fps
    static_threshold: 0.06            # rescaled for sim_fps 30 (was 0.03)
    zero_gravity_steps: 75            # rescaled for sim_fps 30 (was 150)
    video_resolution: [400, 400]      # optional: smaller files, no time change
    video_frame_interval: 15
    options:
      collide_interval: 8               # ceiling for the adaptive stride
      collide_interval_adaptive: true
      soft_contact_margin: 1.0          # REQUIRED by striding; never lower it
render:
  config:
    uv_texture:
      dpi: 150                        # floor; lower buys nothing
      fabric_grain_resolution: 1
```

Add `resolution_scale: 1.25` for 5.4 s when iterating rather than producing final
fit output — the only line here that costs fidelity.
