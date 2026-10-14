| frame | Spearman(pred disparity, true 1/z) | abs-rel depth error, calibrated on 5 random pixels | on 20 pixels | oracle affine fit |
|---|---|---|---|---|
| 0 | 0.981 | 1.4% | 1.3% | 1.3% |
| 1 | 0.981 | 1.8% | 1.6% | 1.5% |
| 2 | 0.988 | 1.8% | 1.5% | 1.5% |

Depth Anything V2 Small on 640x480 renders of the simulator scene (arm + cube + floor); true depth from MuJoCo. Median absolute relative depth error over all valid pixels. A robot-arm render is NOT a human hand on real video; see the script docstring.
