| task | readout | plain median | 1% line | best attack: change (gap closed) | Opus-with-scores arms, mean change (gap closed) | valid frozen-test replies under the line (every writer's frozen arms, incl. last-round arms) | plain under |
|---|---|---|---|---|---|---|---|
| maths 27B (m22) | EMA peak (deployed) | +8.9 | -5.8 | -5.9 (40%) | -5.2 (35%) | 0 / 7829 | 0 / 500 |
| maths 27B (m22) | per-token max | +11.3 | -3.7 | -5.0 (34%) | -4.3 (29%) | 0 / 7829 | 0 / 500 |
| maths 27B (m22) | mean over reply | -1.0 | -11.3 | -6.1 (59%) | -4.8 (47%) | 0 / 7829 | 0 / 500 |
| maths 27B (m22) | EMA peak from first answer token | +8.3 | -3.2 | -4.6 (40%) | -4.0 (34%) | 0 / 7829 | 0 / 500 |
| maths 27B (m22) | max over answer span | +11.2 | -4.6 | -4.9 (31%) | -4.3 (27%) | 0 / 7829 | 0 / 500 |
| maths 9B (m23) | EMA peak (deployed) | +8.6 | -4.7 | -8.9 (67%) | -6.3 (47%) | 36 / 3584 | 0 / 448 |
| maths 9B (m23) | per-token max | +12.1 | +0.8 | -7.3 (65%) | -5.6 (50%) | 177 / 3584 | 0 / 448 |
| maths 9B (m23) | mean over reply | -2.6 | -10.6 | -7.7 (96%) | -6.1 (76%) | 300 / 3584 | 0 / 448 |
| maths 9B (m23) | EMA peak from first answer token | +6.7 | -1.7 | -4.7 (56%) | -3.4 (40%) | 100 / 3584 | 0 / 448 |
| maths 9B (m23) | max over answer span | +10.4 | -2.6 | -5.6 (43%) | -4.5 (35%) | 70 / 3584 | 0 / 448 |
| maths 122B (m23) | EMA peak (deployed) | +4.1 | -2.1 | -3.5 (58%) | -2.7 (44%) | 0 / 3271 | 0 / 498 |
| maths 122B (m23) | per-token max | +5.8 | -1.3 | -4.0 (56%) | -2.4 (34%) | 0 / 3271 | 0 / 498 |
| maths 122B (m23) | mean over reply | -0.7 | -4.7 | -2.8 (70%) | -1.6 (40%) | 0 / 3271 | 0 / 498 |
| maths 122B (m23) | EMA peak from first answer token | +4.1 | -1.1 | -3.5 (68%) | -2.6 (50%) | 4 / 3271 | 0 / 498 |
| maths 122B (m23) | max over answer span | +5.8 | -1.3 | -4.1 (58%) | -2.5 (36%) | 10 / 3271 | 0 / 498 |
| maths Gemma (m23) | EMA peak (deployed) | +10.2 | -0.5 | -8.6 (80%) | -2.1 (19%) | 12 / 3071 | 0 / 435 |
| maths Gemma (m23) | per-token max | +13.6 | +2.8 | -7.6 (71%) | -1.4 (13%) | 10 / 3071 | 0 / 435 |
| maths Gemma (m23) | mean over reply | -2.2 | -5.0 | -2.3 (83%) | +0.7 (-25%) | 53 / 3071 | 7 / 435 |
| maths Gemma (m23) | EMA peak from first answer token | +10.8 | -0.5 | -8.5 (75%) | -1.8 (16%) | 0 / 3071 | 0 / 435 |
| maths Gemma (m23) | max over answer span | +13.6 | -3.7 | -7.6 (44%) | -1.4 (8%) | 0 / 3071 | 0 / 435 |
| code (m25, m22 description) | EMA peak (deployed) | +9.9 | -5.4 | -4.4 (29%) | -4.0 (26%) | 0 / 2664 | 0 / 478 |
| code (m25, m22 description) | per-token max | +15.2 | +1.3 | -5.2 (37%) | -4.4 (31%) | 3 / 2664 | 0 / 478 |
| code (m25, m22 description) | mean over reply | +0.0 | -11.9 | -7.6 (63%) | -5.6 (47%) | 0 / 2664 | 0 / 478 |
| code (m25, m22 description) | EMA peak from first answer token | +8.9 | -3.6 | -2.1 (17%) | -1.6 (13%) | 0 / 2664 | 0 / 478 |
| code (m25, m22 description) | max over answer span | +13.8 | +1.3 | -3.7 (30%) | -2.9 (23%) | 3 / 2664 | 0 / 478 |
| code (m25, told label kinds) | EMA peak (deployed) | +9.9 | -5.4 | -5.3 (35%) | -4.5 (30%) | 0 / 4209 | 0 / 478 |
| code (m25, told label kinds) | per-token max | +15.2 | +1.3 | -6.3 (45%) | -5.4 (39%) | 0 / 4209 | 0 / 478 |
| code (m25, told label kinds) | mean over reply | +0.0 | -11.9 | -7.9 (66%) | -6.0 (50%) | 0 / 4209 | 0 / 478 |
| code (m25, told label kinds) | EMA peak from first answer token | +8.9 | -3.6 | -3.1 (25%) | -2.4 (19%) | 0 / 4209 | 0 / 478 |
| code (m25, told label kinds) | max over answer span | +13.8 | +1.3 | -4.8 (39%) | -4.0 (32%) | 25 / 4209 | 0 / 478 |
| ordering (m26) | EMA peak (deployed) | +10.6 | -7.3 | -11.2 (63%) | -9.5 (53%) | 0 / 5044 | 0 / 478 |
| ordering (m26) | per-token max | +18.1 | -5.7 | -15.3 (64%) | -10.0 (42%) | 0 / 5044 | 0 / 478 |
| ordering (m26) | mean over reply | -1.9 | -10.6 | -5.0 (58%) | -3.7 (42%) | 0 / 5044 | 0 / 478 |
| ordering (m26) | EMA peak from first answer token | +9.4 | -4.4 | -8.1 (59%) | -5.4 (39%) | 0 / 5044 | 0 / 478 |
| ordering (m26) | max over answer span | +18.1 | -6.0 | -15.5 (64%) | -10.0 (42%) | 0 / 5044 | 0 / 478 |
| maths 27B, white-box template (m19 Stage 2) | EMA peak (deployed) | +8.9 | -5.8 | -15.5 (105%) | - | 385 / 478 | 0 |
| maths 27B, white-box template (m19 Stage 2) | per-token max | +11.3 | -3.7 | -16.1 (108%) | - | 338 / 478 | 0 |
| maths 27B, white-box template (m19 Stage 2) | mean over reply | -1.2 | -11.3 | -8.8 (87%) | - | 0 / 478 | 0 |
| maths 27B, white-box template (m19 Stage 2) | EMA peak from first answer token | +8.4 | -3.2 | -13.0 (112%) | - | 428 / 478 | 0 |
| maths 27B, white-box template (m19 Stage 2) | max over answer span | +11.2 | -4.6 | -17.3 (109%) | - | 332 / 478 | 0 |
