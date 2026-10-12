| condition | replay success | rejected (reasons) |
|---|---|---|
| clean tracker | 30/30 (100%) | 0 (-) |
| typical (1.5px, 3mm, 2 dropouts, 1 glitch) | 27/30 (90%) | 2 (gap_too_long:2) |
| noisy (4px, 6mm) | 23/30 (77%) | 2 (gap_too_long:2) |
| very noisy (8px, 10mm) | 0/30 (0%) | 30 (low_valid_ratio:30) |
| many dropouts (6) | 24/30 (80%) | 6 (gap_too_long:6) |
| metric scale +5% | 26/30 (87%) | 2 (gap_too_long:2) |
| metric scale +15% | 26/30 (87%) | 2 (gap_too_long:2) |
| metric scale -15% | 28/30 (93%) | 2 (gap_too_long:2) |

30 scripted-expert demos per condition act as the 'human'; each is filmed by a virtual camera (unknown human-world rotation 0.35 rad / scale 1.15 / shift), tracked, lifted, cleaned, retargeted with anchor alignment, and replayed in the simulator. Success = the replayed video-derived demo places the cube in the target zone.
