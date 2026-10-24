# Humanoid Queue Experiment Checkpoint

## Current: Physical CUDA Task Completed (2026-10-24)

The top-loading washer and deformable washcloth task now works in the calibrated
simulation. Actual G1 joint actuators open the passive lid through an idealized
grasp, lift and insert the washcloth, release it and retract. No pelvis weld,
lid actuator, scripted object motion or low-level joint-delay injection is used.

Final CUDA delayed and timely-observation reference episodes each completed five
skills at 23.56 s, followed by 3 s retention. Each trace contains 665 observations
at 25 Hz; physics integrates at 500 Hz on the RTX 5070 Ti Laptop GPU. Stress:
270 duplicate attempts blocked, 30 repeated/stale observation updates rejected,
five unique physical skill starts. This is not a policy-superiority comparison.

The first CUDA pilot failed the conservative cloth acceptance region; its trace
and source are preserved at commit 615aa4d. The insertion x target moved 0.43 to
0.39 m, with unchanged acceptance bounds. Do not erase that calibration history.

All 32 tests pass (10 historical, 16 protocol, 6 new physical/replay tests).
The 66 s narrated, captioned MP4 is 1920x1080 at 25 fps; all 1,650 frames decode,
audio timing passes, and every stored stress observation is represented. Encoded
opening, insertion and final retention frames were inspected. The video shows
CUDA states; rendering reports Intel Graphics and encoding is CPU libx264.

Output: `laundry_task/output/humanoid_laundry_task_narrated.mp4`.
Release: `laundry-task-2026-10-24` in rbajaj5/humanoid-laundry-queues.
Front-page README emphasizes this one movie. The old README is preserved as
STANDING_REACHING.md; v1 source, results and media remain unchanged.

Exact local resume commands, from this repository root:

```powershell
..\.venv-laundry-gpu\Scripts\python.exe -m unittest test_model laundry_task.test_supervisor laundry_task.test_physical -v
..\.venv-laundry-gpu\Scripts\python.exe -m laundry_task.validate_task
# Recompute only intentionally; do not overwrite retained results casually:
..\.venv-laundry-gpu\Scripts\python.exe -m laundry_task.run_physical --backend cuda --scenario delayed --name laundry_cuda_delayed
..\.venv-laundry-gpu\Scripts\python.exe -m laundry_task.run_physical --backend cuda --scenario immediate --name laundry_cuda_immediate
..\.venv-laundry-gpu\Scripts\python.exe -m laundry_task.render_task
.\laundry_task\narrate_task.ps1 -Python ..\.venv-laundry-gpu\Scripts\python.exe
```

Limitations: hand-authored IK/ankle feedback, stiff model gains, idealized point
grasps, small 16-vertex cloth without self-contact, oracle physical observations,
one calibrated setup, no durable ledger, no hardware or trained-policy validation.
Remaining work: learned skill integration, realistic grasping/cloth, physical
uncertainty, retry backoff and frozen multi-seed policy comparisons.

Gardening is a **separate repository**, not a new laundry subfolder:
https://github.com/rbajaj5/humanoid-gardening-queues . Do not mix their results.

## Historical Pre-Physical Checkpoint (2026-10-24)

Latest user direction supersedes the original emphasis below: assume a capable
humanoid within a stated envelope; use the queue-aware research to coordinate
useful laundry skills. Do not keep producing internal-joint-delay videos. Keep
this work in the standalone repository, with one meaningful physical task movie
when the task actually works.

Added `laundry_task/supervisor.py`: pending skill identity, same-ID retransmission,
receipt/completion separation, fresh physical postconditions, execution-time
precondition checks, session isolation and in-memory receiver deduplication.
Sixteen protocol tests pass. The ten existing physics/provenance tests also
still pass, including the retained v1 replay. Existing v1 source and evidence
have not been changed.

`laundry_task/README.md` maps the earlier final horizon study, coordination
experiments and RAG to this work. It distinguishes transferred design ideas from
nontransferable numerical results. NVIDIA GR00T/SONIC, OpenHLM and Unitree's G1
washing-machine demonstration dataset are verified public candidates, not
installed/trained policies in this repository.

An isolated sibling `.venv-laundry-gpu` now contains MuJoCo/MuJoCo Warp 3.17.0,
Warp 1.17.0, NumPy 2.4.6 and SciPy 1.17.1. Warp 1.18's default CUDA 13 build did
not support driver 577.13; the CUDA 12.9 build in Warp 1.17 detects the RTX 5070
Ti Laptop GPU. No system driver was changed. CUDA forward-dynamics verification
passes via `laundry_task/gpu_probe.py`: four identical G1 standing worlds, 500
physics steps per world, 2 ms timestep, one simulated second. The generated
`laundry_task/output/gpu_probe.json` records numerical/posture checks, a CPU
reference comparison, device, versions and source hashes. This verifies actual
CUDA stepping, not general body reliability or laundry-task success.

Resume from this repository root:

```powershell
..\.venv-humanoid\Scripts\python.exe -m laundry_task.validate_protocol
..\.venv-humanoid\Scripts\python.exe -m unittest test_model -v
..\.venv-laundry-gpu\Scripts\python.exe -m laundry_task.gpu_probe
```

First GPU run compiles collision/dynamics kernels into ignored `.warp-cache/`.
Do not start duplicate runs while compilation is active. No new videos are
generated by these checks. The remaining physical work is an operable hollow
washer, compliant grasping/cloth, a validated body/skill adapter, and actual
release-and-retention checks. The Python protocol's synthetic observations do
not demonstrate those physical capabilities. Real checkpoint integration and
hardware validation remain pending.

## Historical v1 Checkpoint (Preserved)

## Scope

Requested: a humanoid with mass distribution, internal motion delays, a simulation
movie and public GitHub source. This is the first standing/reaching prerequisite,
not a trained robot operating a washing machine.

User explicitly requested a **separate repository**. All new work was moved out
of the queue-assurance repository before publication. Its README changes were
removed; no new commit was made there. The RAG repository is unchanged.

## Completed

- Retrieved pinned licensed G1 model; recorded hashes and per-link inertias.
- Tested original and lower-gain MJX descriptions. Original stands; lower-gain
  static posture falls. Negative probe is retained, not hidden.
- Built free-base MuJoCo model with gravity, contact and torque-limited actuators.
- Generated mass-aware nominal reach poses using constrained inverse kinematics.
- Ran 64 matched episodes (4 delay profiles x 8 seeds x 2 queue policies).
- All episodes reached without falling. Asymmetric delay increased peak tilt;
  coherent bundles reduced it. Zero/uniform controls match across policies.
- Stored per-episode results and exactly reproducible full poses for movie seed
  24100 under skew_600ms (2 of 64 episodes).
- Ten tests pass: queue semantics, no future messages, model integrity, pairing,
  aggregation and exact selected replay regeneration.
- Narrated/captioned 48-second MP4 rendered at 1920x1080, 25 fps; all 1,200
  frames decode successfully. Five encoded-frame samples inspected visually.
- Narration fitted into nonoverlapping windows; movie indices link every frame
  back to its retained physical pose. See output/validation.json and
  output/audio_validation.json.

## Compute

During development, the isolated Python environment is a sibling directory:
`../.venv-humanoid/Scripts/python.exe`. In a fresh clone, create `.venv` inside
this standalone repository using the README commands.

Physics is MuJoCo CPU. The renderer reports Intel Graphics. This experiment does
not use CUDA physics, train a policy or claim NVIDIA compute utilization.

## Resume / Reproduce

In this directory, with the Python above (or a local environment from README):

```powershell
python fetch_model.py
python simulate.py
python -m unittest test_model -v
python render.py
.\narrate.ps1 -Python python
python validate.py
```

The experiment's configuration and upstream model are pinned. Do not replace the
model or interpret the high-gain exploratory result as a hardware safety case.
See README for complete definitions, replay coverage and limitations. The full
class RAG/PDF corpus remains private and is not part of this publication.

## Not Yet Implemented

Adaptive/learned balance control, realistic low-gain recovery, sensor and torque
delays, measured latency distributions, payload/friction sweeps, operable washer
door, grasping, clothing, washing cycle, or hardware validation.

Publication target: `rbajaj5/humanoid-laundry-queues`, release
`humanoid-queues-2026-10-24`. Preserve all earlier experiments and unrelated local
files in the separate research repositories.
