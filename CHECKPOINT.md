# Humanoid Queue Experiment Checkpoint

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
