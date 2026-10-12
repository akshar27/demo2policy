| training data | demos used | BC success (3 seeds x 100 eval episodes) |
|---|---|---|
| scripted expert, 100 demos (reference) | 100 | 94.3% (96% / 95% / 92%) |
| scripted expert, 10 demos | 10 | 22.0% (26% / 15% / 25%) |
| 1 synthetic human video(s) x 100 scenes | 100 | 40.3% (43% / 47% / 31%) |
| 10 synthetic human video(s) x 10 scenes | 100 | 70.3% (76% / 67% / 68%) |
| 20 synthetic human video(s) x 5 scenes | 94 | 70.3% (72% / 66% / 73%) |

All rows: BC with 0.05 observation-noise augmentation, same seeded evaluation episodes. 'Human videos' are synthetic (expert motion filmed by a virtual camera, 1.5 px / 3 mm noise, dropouts, one glitch frame); real footage is milestone 6b.
