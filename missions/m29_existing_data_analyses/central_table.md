Share of the gap closed (median over instructions; each instruction's value in brackets, in %). Frozen-test numbers except zero-shot (eval set: 40 problems x 2 seeds).

| setup | arm type | n instr. | to the 1% line | to the in-context non-answer level | to the styled-twin level | valid replies under the 1% line |
|---|---|---|---|---|---|---|
| maths 27B (Claude-text probe) | zero-shot (eval set) | 8 | 10% (11, 7, 4, 11, 9, 7, 11, 11) | 9% (10, 6, 4, 10, 9, 7, 10, 11) | - | 0 / 609 |
| maths 27B (Claude-text probe) | model writes for itself, best | 3 | 36% (28, 39, 36) | 34% (26, 37, 34) | - | 0 / 1472 |
| maths 27B (Claude-text probe) | Qwen-122B writes, best | 3 | 46% (43, 46, 52) | 43% (41, 43, 48) | - | 0 / 1163 |
| maths 27B (Claude-text probe) | with scores, best | 3 | 44% (54, 39, 44) | 41% (51, 37, 41) | - | 0 / 982 |
| maths 27B (Claude-text probe) | white-box template (m19) | 1 | 114% | 106% | - | 407 / 484 |
| maths 27B (own-reply probe) | zero-shot (eval set) | 8 | 4% (0, 2, 15, -1, 5, 10, 14, 3) | 4% (0, 2, 16, -2, 5, 11, 15, 3) | - | 0 / 630 |
| maths 27B (own-reply probe) | model writes for itself, best | 3 | 25% (27, 24, 25) | 27% (29, 26, 27) | - | 0 / 1138 |
| maths 27B (own-reply probe) | Qwen-122B writes, best | 3 | 21% (6, 21, 32) | 23% (6, 23, 34) | - | 0 / 1409 |
| maths 27B (own-reply probe) | no scores, best | 3 | 27% (16, 27, 28) | 29% (17, 29, 29) | 27% (15, 28, 27) | 0 / 1373 |
| maths 27B (own-reply probe) | no scores, last | 3 | 24% (15, 26, 24) | 25% (16, 27, 25) | - | 0 / 1407 |
| maths 27B (own-reply probe) | with scores, best | 3 | 33% (33, 40, 32) | 35% (35, 43, 34) | 40% (40, 44, 35) | 0 / 1207 |
| maths 27B (own-reply probe) | with scores, last | 3 | 29% (29, 21, 31) | 31% (31, 22, 33) | - | 0 / 1295 |
| maths 27B (own-reply probe) | white-box template (m19) | 1 | 105% | 111% | 91% | 384 / 485 |
| maths 9B | zero-shot (eval set) | 8 | 12% (6, 11, 23, 23, 27, 3, 13, 9) | 11% (5, 10, 20, 20, 25, 3, 12, 8) | - | 0 / 577 |
| maths 9B | model writes for itself, best | 3 | 33% (36, 33, 23) | 30% (32, 30, 21) | 37% (37, 39, 21) | 0 / 1395 |
| maths 9B | with scores, best | 3 | 48% (67, 48, 26) | 43% (60, 43, 24) | 37% (50, 37, 30) | 31 / 1072 |
| maths 9B | with scores, last | 3 | 22% (52, 16, 22) | 20% (47, 15, 20) | - | 5 / 1117 |
| maths 9B | m19 template, written for 27B | 2 | 90% (105, 76) | 81% (95, 68) | - | 274 / 681 |
| maths 122B | zero-shot (eval set) | 8 | -0% (17, -2, -0, 0, -2, 5, -1, 0) | - | - | 0 / 601 |
| maths 122B | model writes for itself, best | 3 | 25% (24, 30, 25) | - | - | 0 / 1022 |
| maths 122B | with scores, best | 3 | 41% (57, 41, 32) | - | - | 0 / 1127 |
| maths 122B | with scores, last | 3 | 37% (58, 37, 35) | - | - | 0 / 1122 |
| maths 122B | m19 template, written for 27B | 2 | 71% (76, 65) | - | - | 10 / 951 |
| maths Gemma | zero-shot (eval set) | 8 | 2% (-2, -0, 9, 4, -0, 4, -2, 39) | - | - | 3 / 400 |
| maths Gemma | model writes for itself, best | 3 | 3% (3, 22, 3) | - | - | 0 / 1198 |
| maths Gemma | with scores, best | 3 | 8% (-5, 55, 8) | - | - | 0 / 1048 |
| maths Gemma | with scores, last | 3 | 58% (80, 58, 2) | - | - | 12 / 825 |
| maths Gemma | m19 template, written for 27B | 2 | 50% (45, 54) | - | - | 0 / 320 |
| code | zero-shot (eval set) | 8 | 11% (16, 17, 26, 5, 11, 8, 9, 11) | 11% (15, 16, 25, 5, 11, 7, 9, 10) | - | 0 / 637 |
| code | with scores, best | 3 | 27% (28, 27, 25) | 26% (27, 26, 24) | 27% (28, 23, 27) | 0 / 1316 |
| code | with scores, last | 3 | 24% (29, 4, 24) | 23% (28, 4, 23) | - | 0 / 1348 |
| ordering | zero-shot (eval set) | 7 | 20% (33, 13, 22, 3, 20, 27, 17) | 23% (40, 16, 26, 3, 23, 32, 21) | - | 0 / 515 |
| ordering | model writes for itself, best | 3 | 40% (40, 40, 39) | 47% (47, 48, 46) | - | 0 / 1363 |
| ordering | Qwen-122B writes, best | 3 | 53% (53, 28, 53) | 62% (62, 33, 63) | - | 0 / 1279 |
| ordering | no scores, best | 3 | 50% (50, 45, 63) | 59% (59, 53, 74) | 65% (65, 52, 77) | 0 / 1132 |
| ordering | with scores, best | 3 | 61% (36, 61, 62) | 72% (43, 72, 74) | 70% (43, 73, 70) | 0 / 1270 |
