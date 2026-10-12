## Per arm

| task | arm family | arm | source | valid attack / clean twins | attack change (log-odds) | styled-twin shift | gap closed within style [95%] | AUROC attack vs styled twins |
|---|---|---|---|---|---|---|---|---|
| code | searched, Opus with scores (best round) | opus/own/chain_0 | m27 | 464 / 200 | -4.2 | +2.3 | 37% [35, 40] | 1.000 |
| code | searched, Opus with scores (best round) | opus/own/chain_1 | m27 | 359 / 182 | -4.1 | -0.0 | 23% [20, 28] | 1.000 |
| code | searched, Opus with scores (best round) | opus/own/chain_2 | m27 | 493 / 200 | -3.8 | +3.8 | 43% [41, 46] | 1.000 |
| code | searched, Opus with scores (last round) | opus/own/chain_0@last | m34 | 459 / 200 | -4.4 | +2.3 | 38% [35, 41] | 1.000 |
| code | searched, Opus with scores (last round) | opus/own/chain_1@last | m34 | 391 / 200 | -0.7 | +0.0 | 4% [2, 5] | 1.000 |
| code | searched, Opus with scores (last round) | opus/own/chain_2@last | m34 | 498 / 200 | -3.7 | +1.8 | 31% [27, 35] | 1.000 |
| maths 27B | Qwen-122B writes | q122/own/chain_0 | m34 | 497 / 184 | -0.9 | +1.8 | 15% [12, 18] | 1.000 |
| maths 27B | Qwen-122B writes | q122/own/chain_1 | m34 | 490 / 179 | -3.2 | -0.2 | 16% [14, 20] | 1.000 |
| maths 27B | Qwen-122B writes | q122/own/chain_2 | m34 | 422 / 182 | -4.7 | +4.2 | 50% [48, 52] | 1.000 |
| maths 27B | Qwen-27B writes for itself | q27/own/chain_0 | m34 | 327 / 175 | -4.0 | +3.8 | 43% [38, 48] | 1.000 |
| maths 27B | Qwen-27B writes for itself | q27/own/chain_1 | m34 | 413 / 184 | -3.5 | +5.1 | 48% [45, 50] | 1.000 |
| maths 27B | Qwen-27B writes for itself | q27/own/chain_2 | m34 | 398 / 184 | -3.7 | +0.0 | 21% [19, 24] | 1.000 |
| maths 27B | no readout spec (Opus fallback) | opus_nospec/chain_0@best | m28 | 440 / 195 | -2.8 | +4.5 | 41% [39, 44] | 1.000 |
| maths 27B | no readout spec (Opus fallback) | opus_nospec/chain_0@last | m28 | 495 / 191 | -0.1 | +3.2 | 18% [16, 20] | 1.000 |
| maths 27B | no readout spec (Opus fallback) | opus_nospec/chain_1@best | m28 | 456 / 195 | -3.5 | +3.8 | 40% [35, 42] | 1.000 |
| maths 27B | no readout spec (Opus fallback) | opus_nospec/chain_1@last | m28 | 500 / 199 | -0.9 | +4.6 | 31% [29, 33] | 1.000 |
| maths 27B | no readout spec (Opus fallback) | opus_nospec/chain_2@best | m28 | 455 / 184 | -3.6 | +4.4 | 44% [42, 47] | 1.000 |
| maths 27B | no readout spec (Opus fallback) | opus_nospec/chain_2@last | m28 | 500 / 184 | -1.6 | +5.8 | 41% [39, 43] | 1.000 |
| maths 27B | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_0@best | m28 | 496 / 184 | -2.4 | +3.1 | 31% [28, 33] | 1.000 |
| maths 27B | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_0@last | m28 | 500 / 189 | -0.2 | +0.1 | 2% [0, 5] | 1.000 |
| maths 27B | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_1@best | m28 | 424 / 196 | -2.7 | +3.8 | 36% [33, 41] | 1.000 |
| maths 27B | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_1@last | m28 | 500 / 184 | -0.4 | +0.1 | 2% [1, 5] | 1.000 |
| maths 27B | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_2@best | m28 | 417 / 184 | -3.0 | +3.8 | 38% [34, 42] | 1.000 |
| maths 27B | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_2@last | m28 | 500 / 185 | -2.4 | +3.1 | 30% [28, 33] | 1.000 |
| maths 27B | pinned Opus 4.8 no scores (best round) | opus48_pinned_noscores/chain_0@best | m34 | 500 / 184 | -1.3 | +2.9 | 23% [21, 25] | 1.000 |
| maths 27B | pinned Opus 4.8 no scores (best round) | opus48_pinned_noscores/chain_1@best | m34 | 458 / 184 | -3.8 | +4.4 | 45% [42, 48] | 1.000 |
| maths 27B | pinned Opus 4.8 no scores (best round) | opus48_pinned_noscores/chain_2@best | m34 | 455 / 181 | -0.8 | +1.9 | 15% [14, 17] | 1.000 |
| maths 27B | pinned Opus 4.8 no scores (last round) | opus48_pinned_noscores/chain_0@last | m34 | 500 / 184 | -1.1 | +4.4 | 31% [29, 33] | 1.000 |
| maths 27B | pinned Opus 4.8 no scores (last round) | opus48_pinned_noscores/chain_1@last | m34 | 500 / 182 | -0.6 | +5.0 | 31% [30, 33] | 1.000 |
| maths 27B | pinned Opus 4.8 no scores (last round) | opus48_pinned_noscores/chain_2@last | m34 | 500 / 184 | -1.1 | +2.6 | 21% [19, 22] | 1.000 |
| maths 27B | pinned Opus 4.8 with scores (best round) | opus48_pinned/chain_0@best | m34 | 328 / 185 | -6.0 | +4.4 | 58% [52, 60] | 1.000 |
| maths 27B | pinned Opus 4.8 with scores (best round) | opus48_pinned/chain_1@best | m34 | 348 / 184 | -3.9 | +6.0 | 55% [50, 60] | 1.000 |
| maths 27B | pinned Opus 4.8 with scores (best round) | opus48_pinned/chain_2@best | m34 | 482 / 183 | -5.8 | +3.3 | 50% [49, 53] | 1.000 |
| maths 27B | pinned Opus 4.8 with scores (last round) | opus48_pinned/chain_0@last | m34 | 245 / 170 | -5.8 | +4.4 | 56% [50, 60] | 1.000 |
| maths 27B | pinned Opus 4.8 with scores (last round) | opus48_pinned/chain_1@last | m34 | 338 / 183 | -4.0 | +6.0 | 55% [52, 60] | 1.000 |
| maths 27B | pinned Opus 4.8 with scores (last round) | opus48_pinned/chain_2@last | m34 | 494 / 184 | -5.8 | +3.3 | 50% [49, 53] | 1.000 |
| maths 27B | random styles (best of 16) | random/block_0@best16 | m28 | 498 / 182 | -1.7 | +3.3 | 28% [25, 30] | 1.000 |
| maths 27B | random styles (best of 16) | random/block_1@best16 | m28 | 492 / 179 | -2.0 | +1.9 | 21% [18, 25] | 1.000 |
| maths 27B | random styles (best of 16) | random/block_2@best16 | m28 | 267 / 142 | -2.3 | +5.4 | 42% [40, 46] | 1.000 |
| maths 27B | random styles (unselected pick) | random/block_0@pick | m28 | 500 / 183 | -0.5 | +1.4 | 10% [9, 12] | 1.000 |
| maths 27B | random styles (unselected pick) | random/block_1@pick | m28 | 485 / 183 | -1.8 | +5.4 | 40% [38, 41] | 1.000 |
| maths 27B | random styles (unselected pick) | random/block_2@pick | m28 | 484 / 184 | -1.3 | +5.0 | 35% [33, 37] | 1.000 |
| maths 27B | random styles in words (m22-informed) | words/block_0@best16 | m28 | 326 / 185 | -3.8 | +6.6 | 58% [54, 60] | 1.000 |
| maths 27B | searched, Opus no scores (best round) | opus_noscores/own/chain_0 | m27 | 498 / 184 | -2.3 | +3.1 | 30% [28, 33] | 1.000 |
| maths 27B | searched, Opus no scores (best round) | opus_noscores/own/chain_1 | m27 | 439 / 184 | -3.9 | +4.2 | 45% [42, 47] | 1.000 |
| maths 27B | searched, Opus no scores (best round) | opus_noscores/own/chain_2 | m27 | 436 / 184 | -4.1 | +3.1 | 40% [37, 42] | 1.000 |
| maths 27B | searched, Opus no scores (last round) | opus_noscores/own/chain_0@last | m34 | 495 / 184 | -2.2 | +2.8 | 28% [24, 30] | 1.000 |
| maths 27B | searched, Opus no scores (last round) | opus_noscores/own/chain_1@last | m34 | 443 / 186 | -3.8 | +3.2 | 38% [35, 41] | 1.000 |
| maths 27B | searched, Opus no scores (last round) | opus_noscores/own/chain_2@last | m34 | 469 / 184 | -3.5 | +5.0 | 47% [44, 49] | 1.000 |
| maths 27B | searched, Opus with scores (best round) | opus/own/chain_0 | m27 | 283 / 184 | -4.9 | +5.9 | 60% [56, 62] | 1.000 |
| maths 27B | searched, Opus with scores (best round) | opus/own/chain_1 | m27 | 437 / 197 | -5.9 | +4.7 | 58% [55, 61] | 1.000 |
| maths 27B | searched, Opus with scores (best round) | opus/own/chain_2 | m27 | 487 / 198 | -4.7 | +4.5 | 51% [47, 53] | 1.000 |
| maths 27B | searched, Opus with scores (last round) | opus/own/chain_0@last | m34 | 424 / 185 | -4.2 | +4.5 | 48% [45, 50] | 1.000 |
| maths 27B | searched, Opus with scores (last round) | opus/own/chain_1@last | m34 | 399 / 196 | -3.1 | +7.2 | 57% [55, 60] | 1.000 |
| maths 27B | searched, Opus with scores (last round) | opus/own/chain_2@last | m34 | 472 / 189 | -4.6 | +2.1 | 37% [35, 40] | 1.000 |
| maths 27B | white-box template (m19) | m19/s2_own | m19 | 485 / 184 | -15.3 | +1.2 | 92% [91, 93] | 0.951 |
| maths 9B | Qwen-9B writes for itself | self/own/chain_0 | m27 | 472 / 147 | -4.8 | +4.4 | 53% [49, 58] | 0.992 |
| maths 9B | Qwen-9B writes for itself | self/own/chain_1 | m27 | 482 / 170 | -4.4 | +6.1 | 60% [55, 68] | 0.989 |
| maths 9B | Qwen-9B writes for itself | self/own/chain_2 | m27 | 441 / 175 | -3.1 | +2.3 | 31% [25, 37] | 1.000 |
| maths 9B | searched, Opus with scores (best round) | opus/own/chain_0 | m27 | 476 / 180 | -8.9 | -0.6 | 48% [45, 55] | 0.999 |
| maths 9B | searched, Opus with scores (best round) | opus/own/chain_1 | m27 | 305 / 141 | -6.4 | +0.3 | 38% [35, 46] | 0.995 |
| maths 9B | searched, Opus with scores (best round) | opus/own/chain_2 | m27 | 291 / 173 | -3.5 | +5.7 | 53% [48, 58] | 0.995 |
| ordering | Qwen-122B writes | q122/own/chain_0 | m34 | 363 / 187 | -9.4 | +5.8 | 80% [78, 82] | 0.989 |
| ordering | Qwen-122B writes | q122/own/chain_1 | m34 | 469 / 198 | -5.1 | +6.0 | 58% [57, 60] | 1.000 |
| ordering | Qwen-122B writes | q122/own/chain_2 | m34 | 447 / 190 | -9.5 | +1.9 | 61% [56, 63] | 1.000 |
| ordering | Qwen-27B writes for itself | q27/own/chain_0 | m34 (audit ignored) | 458 / 193 | -7.1 | +1.9 | 47% [46, 49] | 1.000 |
| ordering | Qwen-27B writes for itself | q27/own/chain_1 | m34 (audit ignored) | 472 / 192 | -7.2 | +3.1 | 54% [53, 56] | 1.000 |
| ordering | Qwen-27B writes for itself | q27/own/chain_2 | m34 (audit ignored) | 433 / 196 | -6.9 | +1.9 | 47% [44, 49] | 1.000 |
| ordering | no readout spec (Opus fallback) | opus_nospec/chain_0@best | m28 | 438 / 200 | -8.3 | +4.3 | 66% [64, 68] | 1.000 |
| ordering | no readout spec (Opus fallback) | opus_nospec/chain_0@last | m28 | 468 / 162 | -4.6 | +4.8 | 50% [48, 51] | 1.000 |
| ordering | no readout spec (Opus fallback) | opus_nospec/chain_1@best | m28 | 466 / 198 | -4.7 | -0.2 | 24% [22, 25] | 1.000 |
| ordering | no readout spec (Opus fallback) | opus_nospec/chain_1@last | m28 | 455 / 198 | -1.2 | +3.4 | 25% [23, 26] | 1.000 |
| ordering | no readout spec (Opus fallback) | opus_nospec/chain_2@best | m28 | 455 / 192 | -5.9 | +4.4 | 55% [53, 56] | 1.000 |
| ordering | no readout spec (Opus fallback) | opus_nospec/chain_2@last | m28 | 444 / 195 | -4.0 | +4.4 | 44% [43, 45] | 1.000 |
| ordering | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_0@best | m28 | 470 / 89 | -1.4 | +2.8 | 22% [20, 24] | 1.000 |
| ordering | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_0@last | m28 | 465 / 176 | -0.6 | +3.7 | 23% [21, 27] | 1.000 |
| ordering | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_1@best | m28 | 474 / 198 | -0.6 | +3.4 | 21% [19, 24] | 1.000 |
| ordering | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_1@last | m28 | 474 / 198 | -0.6 | +3.4 | 21% [19, 24] | 1.000 |
| ordering | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_2@best | m28 | 476 / 184 | -1.4 | +4.5 | 31% [30, 33] | 1.000 |
| ordering | no readout spec (pinned 4.8) | opus48_pinned_nospec/chain_2@last | m28 | 483 / 177 | -0.6 | +3.8 | 23% [22, 26] | 1.000 |
| ordering | pinned Opus 4.8 no scores (best round) | opus48_pinned_noscores/chain_0@best | m34 | 407 / 184 | -8.9 | +5.2 | 75% [72, 76] | 1.000 |
| ordering | pinned Opus 4.8 no scores (best round) | opus48_pinned_noscores/chain_1@best | m34 | 454 / 195 | -8.5 | +4.3 | 68% [67, 70] | 1.000 |
| ordering | pinned Opus 4.8 no scores (best round) | opus48_pinned_noscores/chain_2@best | m34 | 457 / 195 | -7.5 | +4.7 | 65% [63, 66] | 1.000 |
| ordering | pinned Opus 4.8 no scores (last round) | opus48_pinned_noscores/chain_0@last | m34 | 443 / 189 | -5.5 | +4.5 | 53% [50, 55] | 1.000 |
| ordering | pinned Opus 4.8 no scores (last round) | opus48_pinned_noscores/chain_1@last | m34 | 460 / 194 | -6.4 | +5.2 | 61% [59, 62] | 1.000 |
| ordering | pinned Opus 4.8 no scores (last round) | opus48_pinned_noscores/chain_2@last | m34 | 450 / 195 | -6.4 | +4.7 | 59% [56, 60] | 1.000 |
| ordering | pinned Opus 4.8 with scores (best round) | opus48_pinned/chain_0@best | m34 | 404 / 192 | -9.7 | +4.5 | 75% [73, 77] | 1.000 |
| ordering | pinned Opus 4.8 with scores (best round) | opus48_pinned/chain_1@best | m34 | 444 / 188 | -9.0 | +4.9 | 73% [71, 75] | 1.000 |
| ordering | pinned Opus 4.8 with scores (best round) | opus48_pinned/chain_2@best | m34 | 330 / 198 | -10.5 | +4.8 | 81% [79, 85] | 1.000 |
| ordering | pinned Opus 4.8 with scores (last round) | opus48_pinned/chain_0@last | m34 | 432 / 185 | -9.6 | +4.5 | 75% [72, 77] | 1.000 |
| ordering | pinned Opus 4.8 with scores (last round) | opus48_pinned/chain_1@last | m34 | 466 / 193 | -9.3 | +4.9 | 75% [74, 77] | 1.000 |
| ordering | pinned Opus 4.8 with scores (last round) | opus48_pinned/chain_2@last | m34 | 446 / 192 | -10.7 | +3.7 | 76% [73, 79] | 0.999 |
| ordering | random styles (best of 16) | random/block_0@best16 | m28 | 484 / 188 | -5.7 | +1.1 | 36% [34, 39] | 1.000 |
| ordering | random styles (best of 16) | random/block_1@best16 | m28 | 441 / 196 | -4.9 | +3.6 | 45% [43, 46] | 1.000 |
| ordering | random styles (best of 16) | random/block_2@best16 | m28 | 469 / 195 | -3.3 | +2.6 | 32% [28, 35] | 1.000 |
| ordering | random styles (unselected pick) | random/block_0@pick | m28 | 252 / 89 | -3.8 | +4.3 | 43% [41, 46] | 1.000 |
| ordering | random styles (unselected pick) | random/block_1@pick | m28 | 484 / 200 | -0.4 | +1.0 | 7% [6, 9] | 1.000 |
| ordering | random styles (unselected pick) | random/block_2@pick | m28 | 454 / 193 | -5.1 | +2.9 | 42% [40, 45] | 1.000 |
| ordering | searched, Opus no scores (best round) | opus_noscores/own/chain_0 | m27 | 415 / 198 | -8.9 | +5.2 | 75% [74, 77] | 0.995 |
| ordering | searched, Opus no scores (best round) | opus_noscores/own/chain_1 | m27 | 442 / 197 | -8.1 | +3.5 | 61% [59, 64] | 1.000 |
| ordering | searched, Opus no scores (best round) | opus_noscores/own/chain_2 | m27 | 275 / 195 | -11.2 | +4.3 | 82% [80, 83] | 1.000 |
| ordering | searched, Opus with scores (best round) | opus/own/chain_0 | m27 | 462 / 198 | -6.5 | +3.7 | 54% [52, 57] | 1.000 |
| ordering | searched, Opus with scores (best round) | opus/own/chain_1 | m27 | 435 / 191 | -10.9 | +4.0 | 79% [77, 81] | 1.000 |
| ordering | searched, Opus with scores (best round) | opus/own/chain_2 | m27 | 373 / 194 | -11.2 | +3.0 | 75% [73, 77] | 1.000 |

## Per arm family (median over arms, range in brackets)

| task | arm family | arms | gap closed within style | of which: task replies falling | of which: styled twins rising | attack change | styled-twin shift |
|---|---|---|---|---|---|---|---|
| code | searched, Opus with scores (best round) | 3 | 37% [23, 43] | 23% [22, 24] | 13% [-0, 21] | -4.1 [-4.2, -3.8] | +2.3 [-0.0, +3.8] |
| code | searched, Opus with scores (last round) | 3 | 31% [4, 38] | 21% [4, 25] | 10% [0, 13] | -3.7 [-4.4, -0.7] | +1.8 [+0.0, +2.3] |
| maths 27B | Qwen-122B writes | 3 | 16% [15, 50] | 17% [5, 26] | 10% [-1, 23] | -3.2 [-4.7, -0.9] | +1.8 [-0.2, +4.2] |
| maths 27B | Qwen-27B writes for itself | 3 | 43% [21, 48] | 21% [20, 22] | 21% [0, 28] | -3.7 [-4.0, -3.5] | +3.8 [+0.0, +5.1] |
| maths 27B | no readout spec (Opus fallback) | 6 | 41% [18, 44] | 12% [0, 20] | 25% [18, 32] | -2.2 [-3.6, -0.1] | +4.5 [+3.2, +5.8] |
| maths 27B | no readout spec (pinned 4.8) | 6 | 30% [2, 38] | 13% [1, 17] | 17% [1, 21] | -2.4 [-3.0, -0.2] | +3.1 [+0.1, +3.8] |
| maths 27B | pinned Opus 4.8 no scores (best round) | 3 | 23% [15, 45] | 7% [5, 21] | 16% [10, 24] | -1.3 [-3.8, -0.8] | +2.9 [+1.9, +4.4] |
| maths 27B | pinned Opus 4.8 no scores (last round) | 3 | 31% [21, 31] | 6% [3, 6] | 25% [14, 28] | -1.1 [-1.1, -0.6] | +4.4 [+2.6, +5.0] |
| maths 27B | pinned Opus 4.8 with scores (best round) | 3 | 55% [50, 58] | 32% [22, 33] | 24% [18, 33] | -5.8 [-6.0, -3.9] | +4.4 [+3.3, +6.0] |
| maths 27B | pinned Opus 4.8 with scores (last round) | 3 | 55% [50, 56] | 32% [22, 32] | 24% [18, 33] | -5.8 [-5.8, -4.0] | +4.4 [+3.3, +6.0] |
| maths 27B | random styles (best of 16) | 3 | 28% [21, 42] | 11% [9, 13] | 18% [10, 30] | -2.0 [-2.3, -1.7] | +3.3 [+1.9, +5.4] |
| maths 27B | random styles (unselected pick) | 3 | 35% [10, 40] | 7% [3, 10] | 27% [8, 30] | -1.3 [-1.8, -0.5] | +5.0 [+1.4, +5.4] |
| maths 27B | random styles in words (m22-informed) | 1 | 58% [58, 58] | 21% [21, 21] | 37% [37, 37] | -3.8 [-3.8, -3.8] | +6.6 [+6.6, +6.6] |
| maths 27B | searched, Opus no scores (best round) | 3 | 40% [30, 45] | 22% [13, 23] | 17% [17, 23] | -3.9 [-4.1, -2.3] | +3.1 [+3.1, +4.2] |
| maths 27B | searched, Opus no scores (last round) | 3 | 38% [28, 47] | 19% [12, 21] | 18% [15, 28] | -3.5 [-3.8, -2.2] | +3.2 [+2.8, +5.0] |
| maths 27B | searched, Opus with scores (best round) | 3 | 58% [51, 60] | 27% [26, 33] | 26% [25, 33] | -4.9 [-5.9, -4.7] | +4.7 [+4.5, +5.9] |
| maths 27B | searched, Opus with scores (last round) | 3 | 48% [37, 57] | 23% [17, 25] | 25% [12, 40] | -4.2 [-4.6, -3.1] | +4.5 [+2.1, +7.2] |
| maths 27B | white-box template (m19) | 1 | 92% [92, 92] | 85% [85, 85] | 7% [7, 7] | -15.3 [-15.3, -15.3] | +1.2 [+1.2, +1.2] |
| maths 9B | Qwen-9B writes for itself | 3 | 53% [31, 60] | 25% [18, 27] | 25% [13, 35] | -4.4 [-4.8, -3.1] | +4.4 [+2.3, +6.1] |
| maths 9B | searched, Opus with scores (best round) | 3 | 48% [38, 53] | 37% [20, 51] | 2% [-3, 33] | -6.4 [-8.9, -3.5] | +0.3 [-0.6, +5.7] |
| ordering | Qwen-122B writes | 3 | 61% [58, 80] | 50% [27, 50] | 31% [10, 32] | -9.4 [-9.5, -5.1] | +5.8 [+1.9, +6.0] |
| ordering | Qwen-27B writes for itself | 3 | 47% [47, 54] | 38% [37, 38] | 10% [10, 16] | -7.1 [-7.2, -6.9] | +1.9 [+1.9, +3.1] |
| ordering | no readout spec (Opus fallback) | 6 | 47% [24, 66] | 25% [7, 44] | 23% [-1, 25] | -4.6 [-8.3, -1.2] | +4.3 [-0.2, +4.8] |
| ordering | no readout spec (pinned 4.8) | 6 | 23% [21, 31] | 3% [3, 8] | 19% [15, 24] | -0.6 [-1.4, -0.6] | +3.6 [+2.8, +4.5] |
| ordering | pinned Opus 4.8 no scores (best round) | 3 | 68% [65, 75] | 45% [40, 47] | 25% [23, 27] | -8.5 [-8.9, -7.5] | +4.7 [+4.3, +5.2] |
| ordering | pinned Opus 4.8 no scores (last round) | 3 | 59% [53, 61] | 34% [29, 34] | 25% [24, 28] | -6.4 [-6.4, -5.5] | +4.7 [+4.5, +5.2] |
| ordering | pinned Opus 4.8 with scores (best round) | 3 | 75% [73, 81] | 51% [47, 56] | 26% [24, 26] | -9.7 [-10.5, -9.0] | +4.8 [+4.5, +4.9] |
| ordering | pinned Opus 4.8 with scores (last round) | 3 | 75% [75, 76] | 51% [49, 57] | 24% [19, 26] | -9.6 [-10.7, -9.3] | +4.5 [+3.7, +4.9] |
| ordering | random styles (best of 16) | 3 | 36% [32, 45] | 26% [18, 30] | 14% [6, 19] | -4.9 [-5.7, -3.3] | +2.6 [+1.1, +3.6] |
| ordering | random styles (unselected pick) | 3 | 42% [7, 43] | 20% [2, 27] | 15% [5, 23] | -3.8 [-5.1, -0.4] | +2.9 [+1.0, +4.3] |
| ordering | searched, Opus no scores (best round) | 3 | 75% [61, 82] | 47% [43, 59] | 23% [18, 28] | -8.9 [-11.2, -8.1] | +4.3 [+3.5, +5.2] |
| ordering | searched, Opus with scores (best round) | 3 | 75% [54, 79] | 58% [34, 59] | 19% [16, 21] | -10.9 [-11.2, -6.5] | +3.7 [+3.0, +4.0] |
