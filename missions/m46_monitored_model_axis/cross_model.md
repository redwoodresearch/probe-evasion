# Cross-model results (B = 16; S_offref, mean over the 14 writers)

| task | monitored model (design) | mean S_offref | slope per 10 ECI [95%] | D vs Qwen3.5-27B | 95% CI writers + chains | 95% CI + seeds and problems | 95% t | writers higher |
|---|---|---|---|---|---|---|---|---|
| maths | Qwen3.5-27B (switch) | 0.176 | +0.066 [+0.032, +0.108] | (baseline) | | | | |
| maths | Qwen3.8-27B (switch) | 0.139 | +0.053 [+0.024, +0.089] | -0.037 | [-0.072, -0.002] | [-0.078, -0.001] | [-0.069, -0.005] | 3 of 14 |
| maths | DeepSeek-V4-Flash (slot swap) | 0.337 | +0.082 [+0.054, +0.125] | +0.160 | [+0.112, +0.207] | [+0.103, +0.213] | [+0.112, +0.208] | 14 of 14 |
| ordering | Qwen3.5-27B (switch) | 0.302 | +0.125 [+0.078, +0.178] | (baseline) | | | | |
| ordering | Qwen3.8-27B (switch) | 0.227 | +0.104 [+0.069, +0.150] | -0.075 | [-0.142, -0.008] | [-0.149, -0.003] | [-0.139, -0.010] | 4 of 14 |
| ordering | DeepSeek-V4-Flash (slot swap) | 0.496 | +0.169 [+0.122, +0.230] | +0.194 | [+0.115, +0.273] | [+0.113, +0.272] | [+0.118, +0.270] | 13 of 14 |
| ordering | Qwen3.8-27B (slot swap; ordering primary) | 0.357 | - | +0.055 | [-0.008, +0.119] | [-0.007, +0.124] | [-0.005, +0.115] | 9 of 14 |

## Design gap on Qwen3.8 (slot swap minus switch) and DeepSeek adjusted by it (not the headline)

| task | gap source | gap D | 95% CI | writers higher | DeepSeek measured mean | DeepSeek adjusted mean | DeepSeek D vs 27B measured | adjusted |
|---|---|---|---|---|---|---|---|---|
| maths | design check: the same switch-ON replies, slot-swap OFF vs switch OFF | +0.070 | [+0.048, +0.093] | 14 of 14 | +0.337 | +0.266 | +0.160 | +0.090 |
| ordering | Qwen3.8 ordering: slot-swap arm minus switch arm (separate searches) | +0.130 | [+0.079, +0.183] | 13 of 14 | +0.496 | +0.367 | +0.194 | +0.065 |

Same design (ordering, both slot swap): DeepSeek minus Qwen3.8 D = +0.139, 95% CI writers + chains [+0.069, +0.211], + seeds and problems [+0.065, +0.207]; 12 of 14 writers higher.

## The 2x2 on DeepSeek's cross-context subsample (B = 16, seed 10; mean over writers)

Reply part = the two replies read after the same instruction; context part = one reply read after both instructions. All on the S_offref scale.

| task | model (design) | S_offref (subsample) | reply part, ON context | reply part, OFF context | context part, ON reply | context part, OFF reply |
|---|---|---|---|---|---|---|
| maths | Qwen3.5-27B (switch) | +0.180 | +0.129 | +0.163 | +0.021 | +0.059 |
| maths | Qwen3.8-27B (switch) | +0.140 | +0.092 | +0.120 | +0.021 | +0.054 |
| maths | DeepSeek-V4-Flash (slot swap) | +0.339 | +0.182 | +0.269 | +0.087 | +0.182 |
| ordering | Qwen3.5-27B (switch) | +0.303 | +0.223 | +0.296 | -0.003 | +0.111 |
| ordering | Qwen3.8-27B (switch) | +0.229 | +0.161 | +0.236 | -0.017 | +0.087 |
| ordering | Qwen3.8-27B (slot swap) | +0.357 | +0.093 | +0.272 | +0.095 | +0.295 |
| ordering | DeepSeek-V4-Flash (slot swap) | +0.495 | +0.089 | +0.263 | +0.328 | +0.448 |

## OFF leakage check: S_offref on compliant pairs only (ON follows the style, OFF does not; switch arms)

| batch | task | writer | S_offref all valid pairs | S_offref compliant pairs | OFF follows share | compliant pairs / valid |
|---|---|---|---|---|---|---|
| q27_c | maths | gemini38flash | +0.099 | +0.053 | 0.33 | 327 / 474 |
| q27_c | maths | gpt54mini | +0.028 | +0.034 | 0.05 | 473 / 496 |
| q27_c | maths | gpt5nano | +0.077 | +0.178 | 0.96 | 17 / 467 |
| q27_c | maths | gpt6sol | +0.178 | +0.229 | 0.50 | 192 / 378 |
| q27_c | maths | haiku45 | +0.099 | +0.126 | 0.32 | 331 / 489 |
| q27_c | maths | llama4maverick | +0.168 | +0.181 | 0.07 | 228 / 317 |
| q27_c | maths | opus48 | +0.317 | +0.316 | 0.00 | 399 / 399 |
| q27_c | maths | opus55 | +0.290 | +0.295 | 0.01 | 393 / 399 |
| q27_c | maths | qwen35_397b | +0.105 | +0.151 | 0.65 | 163 / 470 |
| q27_c | maths | qwen35_9b | +0.007 | +0.042 | 0.52 | 77 / 436 |
| q27_c | maths | sonnet55 | +0.243 | +0.259 | 0.12 | 333 / 379 |
| q27_d | maths | fable51 | +0.326 | +0.326 | 0.00 | 381 / 382 |
| q27_d | maths | gpt61sol | +0.248 | +0.248 | 0.00 | 337 / 339 |
| q27_d | maths | gpt6astra | +0.284 | +0.284 | 0.00 | 368 / 368 |
| q38 | maths | fable51 | +0.284 | +0.269 | 0.50 | 179 / 356 |
| q38 | maths | gemini38flash | +0.089 | +0.095 | 0.16 | 391 / 463 |
| q38 | maths | gpt54mini | +0.052 | +0.232 | 0.48 | 179 / 420 |
| q38 | maths | gpt5nano | +0.018 | - | 1.00 | 0 / 479 |
| q38 | maths | gpt61sol | +0.169 | +0.179 | 0.09 | 306 / 337 |
| q38 | maths | gpt6astra | +0.247 | +0.244 | 0.19 | 331 / 412 |
| q38 | maths | gpt6sol | +0.182 | +0.232 | 0.69 | 123 / 433 |
| q38 | maths | haiku45 | +0.069 | +0.148 | 0.94 | 10 / 466 |
| q38 | maths | llama4maverick | +0.116 | +0.176 | 0.28 | 147 / 451 |
| q38 | maths | opus48 | +0.154 | +0.168 | 0.32 | 232 / 345 |
| q38 | maths | opus55 | +0.210 | +0.211 | 0.26 | 285 / 380 |
| q38 | maths | qwen35_397b | +0.053 | +0.060 | 0.56 | 196 / 469 |
| q38 | maths | qwen35_9b | +0.085 | +0.163 | 0.93 | 32 / 473 |
| q38 | maths | sonnet55 | +0.217 | +0.224 | 0.01 | 334 / 353 |
| q27_c | ordering | gemini38flash | +0.246 | +0.245 | 0.10 | 409 / 452 |
| q27_c | ordering | gpt54mini | +0.085 | +0.056 | 0.21 | 328 / 421 |
| q27_c | ordering | gpt5nano | +0.107 | - | 1.00 | 0 / 346 |
| q27_c | ordering | gpt6sol | +0.345 | +0.346 | 0.01 | 439 / 441 |
| q27_c | ordering | haiku45 | +0.246 | +0.260 | 0.09 | 349 / 388 |
| q27_c | ordering | llama4maverick | +0.086 | +0.150 | 0.62 | 162 / 435 |
| q27_c | ordering | opus48 | +0.401 | +0.401 | 0.00 | 395 / 400 |
| q27_c | ordering | opus55 | +0.583 | +0.583 | 0.00 | 399 / 399 |
| q27_c | ordering | qwen35_397b | +0.185 | +0.217 | 0.33 | 269 / 402 |
| q27_c | ordering | qwen35_9b | +0.121 | +0.115 | 0.51 | 159 / 432 |
| q27_c | ordering | sonnet55 | +0.452 | +0.452 | 0.00 | 435 / 435 |
| q27_d | ordering | fable51 | +0.637 | +0.637 | 0.00 | 395 / 395 |
| q27_d | ordering | gpt61sol | +0.296 | +0.383 | 0.00 | 295 / 414 |
| q27_d | ordering | gpt6astra | +0.440 | +0.444 | 0.00 | 411 / 430 |
| q38sw | ordering | fable51 | +0.376 | +0.402 | 0.13 | 343 / 408 |
| q38sw | ordering | gemini38flash | +0.373 | +0.373 | 0.02 | 423 / 432 |
| q38sw | ordering | gpt54mini | +0.022 | +0.160 | 0.67 | 122 / 436 |
| q38sw | ordering | gpt5nano | +0.016 | - | 1.00 | 0 / 458 |
| q38sw | ordering | gpt61sol | +0.268 | +0.270 | 0.03 | 407 / 418 |
| q38sw | ordering | gpt6astra | +0.360 | +0.364 | 0.02 | 401 / 410 |
| q38sw | ordering | gpt6sol | +0.208 | +0.219 | 0.06 | 418 / 446 |
| q38sw | ordering | haiku45 | +0.059 | +0.057 | 0.17 | 361 / 437 |
| q38sw | ordering | llama4maverick | +0.090 | +0.264 | 0.67 | 122 / 434 |
| q38sw | ordering | opus48 | +0.404 | +0.404 | 0.00 | 409 / 410 |
| q38sw | ordering | opus55 | +0.299 | +0.311 | 0.04 | 371 / 402 |
| q38sw | ordering | qwen35_397b | +0.170 | +0.178 | 0.23 | 316 / 421 |
| q38sw | ordering | qwen35_9b | +0.074 | -0.009 | 0.77 | 85 / 404 |
| q38sw | ordering | sonnet55 | +0.465 | +0.475 | 0.03 | 394 / 405 |

Qwen3.8 (switch) minus Qwen3.5-27B on compliant pairs only (writers with compliant pairs on both models):

- maths: D = -0.011, 95% CI writers + chains [-0.061, +0.049], t [-0.067, +0.045]; 5 of 13 writers higher.
- ordering: D = -0.063, 95% CI writers + chains [-0.141, +0.016], t [-0.143, +0.017]; 5 of 13 writers higher.
