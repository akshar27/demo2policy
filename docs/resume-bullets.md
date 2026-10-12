# Resume bullets - demo2policy (every claim traces to code/results in this repo)

- Built a pipeline that turns monocular human video into robot demonstrations: MediaPipe hand tracking, a closed-form
  3D lifting step, track cleaning with explicit rejection reasons, and retargeting onto a simulated Franka Panda
  (MuJoCo), with every demo replayed in simulation before it is kept.
- Measured the pipeline against ground truth instead of eyeballing it: a synthetic-human generator injects tracker noise,
  dropouts and metric-scale error; lifting error tables and a round-trip study (30/30 clean, 27/30 typical, 8 px noise
  rejected by a reprojection gate) drove fixes such as anchor averaging and wider smoothing.
- Trained behavior-cloning policies on video-derived demos: retargeting each video to many cube positions took 10
  synthetic videos to 70% simulated pick-and-place success (vs 22% for 10 expert demos, 94% for 100); train-time
  observation noise alone lifted BC from 45% to 93%.
- Wrote the evaluation harness (seeded episodes, Wilson 95% CIs, paired initial states) and a 34-test suite with CI.
