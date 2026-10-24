# Queue-Aware Laundry Task Coordination

Current development direction: **a capable humanoid doing a useful task, with
queue-aware supervision above its body controller**. The published joint-delay
experiment remains historical evidence; additional internal-delay videos are not
the objective. This directory now implements the tested protocol, a CUDA standing
probe, and a physical top-loading washer/cloth demonstration. The current movie
and result table are on the [front page](../README.md).

## Where the Earlier Research Goes

The research is linked, not copied wholesale or pooled into a new result.

| Earlier work | Role in this task | Transfer boundary |
| --- | --- | --- |
| [Final horizon study](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/tree/main/course_study) | Pending actions are part of the supervisor state. Test the actual intervention/scoring mechanism, including harmful cases. | The quadrotor's N=4 observation is not a laundry horizon or safety guarantee. |
| [Queue and coordination experiments](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/tree/main/analysis) | Separate issued, received, executing and physically completed actions. Preserve identity on retry; recheck state when consuming queued commands. | Their performance counts do not become humanoid results. |
| [Fair-queue experiments](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/blob/main/analysis/run_fair_queue_multiagent_simulation.py) | Future concurrent perception/planning: measure stale work and starvation, not just throughput. | Not yet implemented here; priority alone is not an assurance mechanism. |
| [RAG](https://github.com/rbajaj5/assured-autonomy-rag) | Retrieve mechanism-level literature and retain source provenance for design decisions. | Retrieval is neither a controller nor validation evidence. The class PDF corpus stays private. |
| [Historical humanoid probe](../STANDING_REACHING.md#question-and-mechanism) | Retain the mass/inertia model and reproducibility lessons. | No new emphasis on arbitrary joint delays; no claim the earlier reach opens a washer. |

The final course evidence distinguishes a quarantined v2 conversion bug,
registered N=3/N=4 comparison, exploratory deeper-horizon results, and a separate
CBF follow-up. Those categories must stay separate here too. More queue knowledge
does not by itself imply a better action or safe fallback.

## Conditional Body Reliability

For task-level studies, assume a skill executor can stand, reach and follow a
bounded manipulation command in a specified operating envelope. This is an
explicit assumption, not inferred from a sports demonstration and not produced
by the protocol tests. A physical adapter must supply `body_ready` from a
validated balance/contact/joint-limit monitor, rather than hardcoding it true.

The task supervisor does not replace balance control. If body readiness is lost,
it stops issuing task commands and leaves the low-level recovery controller in
charge. It does not assume that freezing every joint is a safe recovery action.

Useful public developments after the August 2025 Beijing games, reviewed
2026-10-24:

- [NVIDIA GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl):
  initial public release November 2025; SONIC releases in 2026 include G1
  checkpoints and simulation/deployment tooling. Code and model weights have
  different licensing terms. A candidate low-level baseline, **not installed or
  evaluated in this repository**.
- [OpenHLM](https://github.com/OpenHLM-project/OpenHLM): a 2026 whole-body
  manipulation pipeline with demonstration data, checkpoints and training code.
  Relevant to perception-to-skill learning, but not a drop-in verified laundry
  controller. Not integrated here.
- [Unitree's G1 washing-machine dataset](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine):
  the public card lists 300 episodes at 30 fps, whole-body/hand state and action
  channels, four camera streams, and Apache-2.0 licensing. These are demonstration
  data, not a measured autonomous success rate. No corpus or video has been copied
  into this repository, and no model has been trained on it here.

The existing pinned G1 model can therefore remain the physical embodiment while
low-level control is improved independently of task coordination.

## Implemented Protocol

`supervisor.py` separates a planner-side coordinator from the receiving skill
executor. The ordered task is open door, grasp cloth, insert cloth, release cloth,
retract hand. Each skill has preconditions and observable postconditions.

- A missing receipt preserves the pending action. A retry uses the **same ID**.
- A receipt marked `done` does not advance the task without a fresh observation.
- Old sessions, expired proposals, invalid timestamps, reordered observations,
  changed payloads under the same ID, and conflicting active skills are rejected.
- The receiver revalidates preconditions against current observations at dequeue.
- Release requires an open door and cloth already inside. Completion then requires
  the cloth inside without a grasp; retraction also requires the hand to be clear.
- Command expiry prevents a late start; it does **not** cancel an action that
  already started. Uncertainty can block progress indefinitely.

The clock is a single simulation clock. Default observation age is 0.25 s,
retransmission timeout 0.5 s, command TTL 2 s, and open-door threshold 0.8 rad.
These are explicit synthetic protocol settings, not calibrated hardware limits.
The ledger is in memory: receiver crashes, distributed ownership, durable
recovery, authentication, perception errors and adversarial messages are outside
this implementation. Do not call it exactly-once physical execution or a formal
safety certificate. The physical adapter below computes cloth predicates from
all simulated vertices and requires three seconds of final retention.

## Reproduce

From the repository root, protocol tests use the standard Python library only:

```powershell
python -m laundry_task.validate_protocol
```

CUDA compatibility probe in an isolated environment:

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r laundry_task/requirements-gpu.txt
.\.venv-gpu\Scripts\python.exe fetch_model.py
.\.venv-gpu\Scripts\python.exe -m laundry_task.gpu_probe
```

The probe uses actual MuJoCo Warp forward dynamics and refuses CPU fallback.
It checks a small batch of identical standing worlds; it is **not** evidence of
laundry-task success or statistical robot reliability. This is separate from the
published v1 CPU-only experiment. Its default Warp pin uses CUDA 12.9, avoiding
the incompatible CUDA 13 default build on the development machine's driver.

[Recorded CUDA probe](output/gpu_probe.json) identifies the actual device,
numerical checks and source hashes. [Protocol test report](output/protocol_validation.json)
records the 16 passing protocol tests. The 10 historical tests remain separate.
GPU floating-point reductions need not reproduce a bit-identical final-state
hash; the recorded CPU comparison uses an explicit numerical tolerance. The four
worlds share initial conditions and controls and are not four independent trials.

## Physical Task Implementation

`physical.py` extends the pinned G1 description with a hollow top-loading washer,
a passive 0.27 kg lid, a side table and a 4-by-4 deformable washcloth. No lid or
cloth coordinate is scripted during integration. The only actuators are the 29
original humanoid position servos. Whole-body IK supplies nominal joint targets;
small pitch/roll/COM feedback adjusts ankle targets through those actuators.
The free base receives no artificial stabilizing force or weld.

Two compliant point constraints idealize gripping. They are enabled only when
the designated hand is within 25 mm of the handle or cloth corner. Cloth grasp
verification also requires the measured point gap below 35 mm. Hand-object
collisions are excluded for the idealized grasp; lid-housing collisions are
excluded because hinge limits and friction represent that closure mechanism.
The cloth has edge constraints, contact and gravity, but no self-collision or
calibrated bending law. This is not a dexterous-grasp or fabric benchmark.

`run_physical.py` integrates at 2 ms on CUDA, with 40 ms control and observation
ticks. CPU kinematic readback computes actual lid angle, all cloth vertex bounds,
hand clearance, base height and tilt. The body-ready envelope is 0.65-0.90 m base
height and tilt below 15 degrees; violation terminates the run as a failure. This
is an empirical envelope check, not a verified recovery controller.

Physical task settings override protocol defaults: command TTL 10 s and lid-open
threshold 1.10 rad. The freshness limit remains 0.25 s. Both final runs have 0.12 s
command delivery; the reference has immediate observations and 0.04 s receipts.
The stress run has 0.12 s observation delay, no new observations during [4,5.2) s,
2.5 s receipts and a completely withheld opening receipt. Observations are oracle
simulation state with delayed delivery, not noisy camera measurements.

The coordinator may verify a skill's physical postcondition before the executor
finishes its return movement. A following command can then wait in transport,
but the receiver's active-skill guard prevents overlapping execution. The movie
therefore distinguishes verified postconditions from the currently executing
movement. A `TASK_COMPLETE` protocol event alone is insufficient: the experiment
waits for executor completion and three further seconds of retention.

The acceptance region is x in (0.28,0.62), y in (-0.19,0.19), z in (0.488,0.835)
metres. Every cloth vertex must be inside at each retained observation. Hands
must be outside the opening or above it, and the cloth constraint must be off.
One second of initial settling is outside the reported episode clock.

## Reproduce the Movie

After the environment/model commands above, from the repository root:

```powershell
.\.venv-gpu\Scripts\python.exe -m laundry_task.run_physical --backend cuda --scenario delayed --name laundry_cuda_delayed
.\.venv-gpu\Scripts\python.exe -m laundry_task.run_physical --backend cuda --scenario immediate --name laundry_cuda_immediate
.\.venv-gpu\Scripts\python.exe -m unittest test_model laundry_task.test_supervisor laundry_task.test_physical -v
.\.venv-gpu\Scripts\python.exe -m laundry_task.render_task
.\laundry_task\narrate_task.ps1 -Python .\.venv-gpu\Scripts\python.exe
.\.venv-gpu\Scripts\python.exe -m laundry_task.validate_task
```

The final runs share the same frozen model/controller after disclosed pilot
calibration. They are not a matched policy comparison or a randomized sample.
Both finish at 23.56 s, and all 665 observations per run are retained. The movie
shows only the stress trace, plus an explicitly labeled reference summary. Each
rendered frame selects a stored observation with no interpolated dynamics.
Model/source/trace/video hashes and narration timing are retained alongside it.

The first GPU pilot is preserved in `output/calibration_cuda_retention.*` and
commit `615aa4d`. Its x-boundary failure was corrected by changing the insertion
target from 0.43 m to 0.39 m, not by weakening the retention criterion. CPU
development pilots guided grasp geometry, collision masks and ankle feedback;
they are not independent confirmation or statistical evidence.

## Remaining Work

Integrate and validate a learned body/skill policy, realistic fingers and cloth,
vision-based observations, tool/task variations and durable recovery. Compare
queue-aware and non-idempotent/receipt-only baselines under a frozen multi-seed
protocol before claiming a measured advantage. Add retry backoff: the current
tick-level retries deliberately exercise deduplication, but waste messages.
Do not infer hardware safety or general reliability from two successful runs.
