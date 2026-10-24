# Mass-Bearing Humanoid Joint Queues

Initial standalone exploratory release. See the front-page MP4 and README for
the exact model, experiment and limitations.

- Unitree G1: 29 actuated joints, 33.341142 kg, free-base MuJoCo physics.
- 64 matched episodes across four internal command-delay profiles.
- Independent group arrivals versus coherent whole-body command bundles.
- All episodes remained upright and reached; the measured improvement is less
  peak tilt under asymmetric delay, with slower first reaching.
- Retained full-pose replay coverage: 2/64 episodes, fixed seed 24100.
- 48-second, 1920x1080 narrated and captioned MP4 with synchronized telemetry.
- Ten focused tests and exact selected replay checks. Encoded-media validation
  and SHA-256 are supplied as a release asset and in `output/validation.json`.

This is not a trained laundry robot or a hardware safety result. No cloth,
grasping, operable door, or washing cycle is modeled. CPU rigid-body physics;
graphics acceleration is identified separately. Model licensing and attribution
are in THIRD_PARTY_NOTICES.md. Private class papers and RAG corpus are excluded.
