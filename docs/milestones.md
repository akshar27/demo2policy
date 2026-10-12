# Milestones

- [x] **M1 Simulator + scripted expert** - MuJoCo Panda pick-and-place scene, DLS IK, randomised cube, success detector,
      scripted privileged-state expert (200/200 on fresh seeds), 7 tests, filmstrip figure.
- [x] **M2 Demo format + replay baseline + eval harness** - `Demo`/`DemoSet` HDF5 schema, expert dataset generation,
      open-loop replay baseline, seeded evaluation with Wilson 95% CI, run directories, `d2p` CLI. Result: open-loop replay of one expert demo on randomised cubes succeeds 4% (2/50, CI 1.1-13.5%); the expert itself ~100%. That gap is what learning has to close.
- [ ] **M3 Behavior cloning** - BC-MLP on expert demos (state obs); compare vs replay on randomised cube positions;
      ablation over number of demos.
- [ ] **M4 Geometry + hand perception** - camera model, monocular lifting with a documented depth assumption, tested on
      synthetic projections with ground truth; MediaPipe Hands adapter behind a `HandTracker` interface.
- [ ] **M5 Cleaning, segmentation, retargeting** - One-Euro filtering, gap handling, phase segmentation, hand->gripper
      retargeting with IK-feasibility checks, quality gate with rejection reasons; synthetic "fake human video" round-trip.
- [ ] **M6 Real video end to end** - run the chain on real recorded videos, replay in sim, train on video-derived demos,
      report video-vs-expert success with honest failure analysis.
- [ ] **M7 Polish** - README with mermaid architecture + results + decisions/tradeoffs, CI, resume bullets.
