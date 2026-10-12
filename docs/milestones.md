# Milestones

- [x] **M1 Simulator + scripted expert** - MuJoCo Panda pick-and-place scene, DLS IK, randomised cube, success detector,
      scripted privileged-state expert (200/200 on fresh seeds), 7 tests, filmstrip figure.
- [x] **M2 Demo format + replay baseline + eval harness** - `Demo`/`DemoSet` HDF5 schema, expert dataset generation,
      open-loop replay baseline, seeded evaluation with Wilson 95% CI, run directories, `d2p` CLI. Result: open-loop replay of one expert demo on randomised cubes succeeds 4% (2/50, CI 1.1-13.5%); the expert itself ~100%. That gap is what learning has to close.
- [x] **M3 Behavior cloning** - BC-MLP on expert demos (state obs); compare vs replay on randomised cube positions;
      ablation over number of demos. Result (3 training seeds x 50 seeded episodes, `docs/results/bc_sweep.md`): plain BC 31% (10 demos) -> 45% (100 demos); adding 0.05 train-time observation noise -> **92.7%** at 100 demos; replay baseline 4%; scripted expert ~100%. Noise 0.1 is too much (78%). Not yet explained why +time helps only on some seeds (high variance 44-92%).
- [x] **M4 Geometry + hand perception** - camera model, monocular lifting with a documented depth assumption, tested on
      synthetic projections with ground truth; MediaPipe Hands adapter behind a `HandTracker` interface. Lifting = closed-form linear PnP with the rotation known from the tracker's world landmarks (palm-only Procrustes for orientation). Error table `docs/results/lifting_error.md`: at 0.6 m with 2 px / 3 mm noise, median lateral error 3.8 mm, depth 6.4 mm, rotation 4.2 deg; **a 5-10% error in the tracker's metric scale gives 27-57 mm depth error** (depth is the weak axis and rests on that assumption). MediaPipe's world-landmark axis convention is *assumed*, to be verified on real footage in M6.
- [ ] **M5 Cleaning, segmentation, retargeting** - One-Euro filtering, gap handling, phase segmentation, hand->gripper
      retargeting with IK-feasibility checks, quality gate with rejection reasons; synthetic "fake human video" round-trip.
- [ ] **M6 Real video end to end** - run the chain on real recorded videos, replay in sim, train on video-derived demos,
      report video-vs-expert success with honest failure analysis.
- [ ] **M7 Polish** - README with mermaid architecture + results + decisions/tradeoffs, CI, resume bullets.
