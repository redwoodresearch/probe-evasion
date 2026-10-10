### Probe quality

| probe | held-out AUROC as asked / word removed / swapped | plain answers caught 0.1/1/5% (of 68) | plain test valid missed at 1% (of 477) | matched-strength line FA | request-word shift (logit) |
|---|---|---|---|---|---|
| claude@L48s0 | 0.983 / 0.981 / 0.978 | 55/66/67 | 31 | 1.58% | +0.91 |
| claude@L48s1 | 0.984 / 0.982 / 0.977 | 53/64/68 | 38 | 1.85% | +0.83 |
| claude@L48s2 | 0.985 / 0.985 / 0.981 | 55/64/68 | 28 | 1.30% | +1.04 |
| claude@L16s0 | 0.982 / 0.978 / 0.967 | 58/62/67 | 50 | 2.81% | +1.12 |
| claude@L16s1 | 0.979 / 0.973 / 0.961 | 55/64/67 | 55 | 2.60% | +1.15 |
| claude@L16s2 | 0.981 / 0.974 / 0.962 | 57/65/67 | 43 | 2.40% | +1.25 |
| qwen@L48s0 | 0.983 / 0.984 / 0.981 | 59/66/68 | 15 | 0.75% | +0.91 |
| qwen@L48s1 | 0.983 / 0.982 / 0.979 | 62/66/68 | 13 | 0.55% | +0.86 |
| qwen@L48s2 | 0.981 / 0.980 / 0.973 | 62/66/68 | 11 | 0.75% | +1.16 |
| qwen@L16s0 | 0.975 / 0.973 / 0.963 | 60/66/67 | 27 | 1.30% | +0.73 |
| qwen@L16s1 | 0.975 / 0.972 / 0.963 | 57/64/67 | 31 | 1.30% | +0.77 |
| qwen@L16s2 | 0.973 / 0.970 / 0.959 | 58/66/67 | 30 | 1.30% | +0.87 |
| r0_peak@L48 | 0.981 / 0.982 / 0.981 | 53/66/68 | 29 | 1.64% | +1.03 |
| r0_peak@L16 | 0.979 / 0.975 / 0.967 | 57/64/67 | 56 | 3.29% | +1.04 |
| m3a3_on_peak@L48 | 0.994 / 0.990 / 0.985 | 55/67/68 | 6 | 0.55% | +3.31 |
| claude_strict@L48s0 | 0.989 / 0.987 / 0.975 | 60/64/68 | 39 | 1.37% | +1.64 |
| qwen_strict@L48s0 | 0.989 / 0.985 / 0.978 | 62/67/68 | 8 | 0.41% | +1.15 |

### Every attack, layer 48: valid evasions per attempt, instruction removed, 1% line (seed 0 / 1 / 2) and FA90 (seed 0)

| attack | attempts | valid | Claude-text probe | own-reply probe | Claude-text FA90 | own-reply FA90 | paired off minus on, seed 0 [95%] |
|---|---|---|---|---|---|---|---|
| plain | 500 | 477 | 31 / 38 / 28 | 15 / 13 / 11 | 0.6% | 0.5% | +0.032 [+0.008, +0.058] |
| m3a_searched | 500 | 348 | 195 / 163 / 165 | 214 / 210 / 215 | 12.7% | 17.4% | -0.038 [-0.074, -0.004] |
| r0_search | 500 | 454 | 220 / 237 / 228 | 121 / 82 / 113 | 9.5% | 3.8% | +0.198 [+0.140, +0.260] |
| r1_search | 500 | 406 | 392 / 398 / 388 | 361 / 363 / 356 | 40.2% | 30.2% | +0.062 [+0.036, +0.092] |
| m3a_deadpan | 300 | 250 | 85 / 85 / 67 | 75 / 86 / 79 | 4.2% | 5.7% | +0.033 [-0.010, +0.080] |
| m3a_understatement | 300 | 254 | 53 / 69 / 48 | 39 / 40 / 40 | 2.7% | 1.6% | +0.047 [+0.010, +0.083] |
| m3a_neutral_mock_praise | 300 | 240 | 28 / 26 / 20 | 38 / 31 / 37 | 1.0% | 1.2% | -0.033 [-0.063, +0.000] |
| m3a_absurd_suggestion | 300 | 238 | 47 / 43 / 39 | 18 / 19 / 16 | 2.7% | 0.8% | +0.097 [+0.060, +0.133] |
| r2b_search | 500 | 350 | 290 / 298 / 300 | 339 / 345 / 344 | 27.1% | 60.8% | -0.098 [-0.138, -0.062] |
| r2c_search | 500 | 244 | 198 / 211 / 203 | 197 / 185 / 171 | 29.7% | 48.5% | +0.002 [-0.016, +0.020] |
| r2cpk_search | 500 | 448 | 300 / 317 / 303 | 122 / 175 / 130 | 5.3% | 1.5% | +0.356 [+0.284, +0.430] |
| m3a3_on_search | 500 | 402 | 300 / 284 / 248 | 103 / 122 / 118 | 15.2% | 3.4% | +0.394 [+0.330, +0.458] |
| m3a4_s4_search | 500 | 442 | 253 / 261 / 244 | 188 / 166 / 172 | 11.7% | 10.6% | +0.130 [+0.082, +0.178] |
| m13_blind | 500 | 389 | 147 / 148 / 130 | 143 / 123 / 125 | 5.3% | 6.5% | +0.008 [-0.026, +0.042] |
| m13_blind_allval | 500 | 340 | 213 / 208 / 196 | 218 / 202 / 208 | 12.7% | 15.4% | -0.010 [-0.038, +0.020] |
| m13_score_seeing_hidden | 500 | 337 | 301 / 314 / 302 | 188 / 176 / 177 | 25.2% | 6.8% | +0.226 [+0.170, +0.284] |
| m13_seed_hint | 500 | 367 | 73 / 62 / 55 | 60 / 59 / 52 | 1.9% | 2.2% | +0.026 [-0.004, +0.056] |
| m14_claude_r2 | 500 | 385 | 271 / 262 / 258 | 90 / 91 / 81 | 9.9% | 2.2% | +0.362 [+0.294, +0.430] |
| m14_claude_r3 | 500 | 381 | 369 / 360 / 353 | 381 / 379 / 376 | 16.9% | 42.9% | -0.024 [-0.054, -0.004] |
| m14_q122_nothink | 500 | 462 | 166 / 158 / 124 | 152 / 141 / 124 | 4.0% | 4.9% | +0.028 [-0.018, +0.072] |
| m14_q122_think | 500 | 435 | 206 / 190 / 184 | 218 / 228 / 216 | 7.6% | 10.7% | -0.024 [-0.060, +0.012] |
| m14_q122_think_r2 | 500 | 361 | 254 / 242 / 231 | 236 / 231 / 226 | 16.6% | 18.8% | +0.036 [+0.010, +0.062] |
| m14_q122_think_r3 | 500 | 401 | 160 / 154 / 135 | 158 / 147 / 139 | 4.8% | 6.7% | +0.004 [-0.026, +0.036] |
| m14_q122_think_rec | 500 | 354 | 266 / 264 / 256 | 271 / 280 / 271 | 18.3% | 30.2% | -0.010 [-0.042, +0.024] |
| m14_q27_think | 500 | 443 | 243 / 247 / 215 | 245 / 262 / 232 | 8.4% | 11.8% | -0.004 [-0.046, +0.036] |
| m20_search_claude | 500 | 401 | 247 / 237 / 237 | 311 / 325 / 313 | 18.0% | 33.7% | -0.128 [-0.166, -0.088] |
| m20_search_qwen | 500 | 383 | 224 / 221 / 210 | 298 / 305 / 307 | 18.6% | 36.2% | -0.148 [-0.190, -0.108] |
| m20_q122_claude | 500 | 457 | 49 / 51 / 35 | 57 / 44 / 43 | 1.1% | 1.2% | -0.016 [-0.042, +0.010] |
| m20_q122_qwen | 500 | 422 | 231 / 217 / 197 | 258 / 264 / 250 | 9.1% | 19.4% | -0.054 [-0.090, -0.016] |
| m20_search_claude_r2 | 500 | 360 | 271 / 268 / 272 | 304 / 322 / 320 | 47.2% | 49.6% | -0.066 [-0.100, -0.034] |
| m20_search_qwen_r2 | 500 | 411 | 145 / 158 / 166 | 283 / 335 / 311 | 9.5% | 43.1% | -0.276 [-0.328, -0.226] |
| m20_q122_claude_r2 | 500 | 431 | 172 / 181 / 154 | 179 / 175 / 170 | 4.1% | 6.5% | -0.014 [-0.060, +0.028] |
| m20_q122_qwen_r2 | 500 | 356 | 197 / 187 / 179 | 223 / 221 / 217 | 9.1% | 16.7% | -0.052 [-0.090, -0.014] |
| m20_fd_search_claude | 500 | 477 | 307 / 322 / 300 | 342 / 420 / 357 | 9.0% | 13.8% | -0.070 [-0.126, -0.016] |
| m20_fd_search_qwen | 500 | 449 | 324 / 328 / 221 | 387 / 345 / 366 | 5.3% | 8.4% | -0.126 [-0.180, -0.076] |
| m20_fd_search_claude_r2 | 500 | 436 | 267 / 267 / 274 | 345 / 345 / 347 | 25.2% | 45.4% | -0.156 [-0.200, -0.110] |
| m20_fd_search_qwen_r2 | 500 | 477 | 207 / 184 / 127 | 356 / 352 / 233 | 5.9% | 7.5% | -0.298 [-0.374, -0.224] |
| m20_fd_q122_claude | 500 | 419 | 152 / 132 / 130 | 182 / 182 / 170 | 3.9% | 7.7% | -0.060 [-0.100, -0.020] |
| m20_fd_q122_qwen | 500 | 448 | 137 / 134 / 107 | 140 / 141 / 132 | 3.6% | 4.9% | -0.006 [-0.042, +0.028] |

### Strata (layer 48, instruction removed, 1% line)

| seed | stratum | off | on | attempts | off minus on [95%] | arms off higher / on higher |
|---|---|---|---|---|---|---|
| L48s0 | not aimed | 286 | 230 | 1700 | +0.033 [+0.012, +0.054] | 4 / 1 |
| L48s0 | aimed at Claude-data probes | 4444 | 3842 | 9000 | +0.067 [+0.052, +0.082] | 11 / 7 |
| L48s0 | aimed at an earlier on-policy probe | 300 | 103 | 500 | +0.394 [+0.330, +0.462] | 1 / 0 |
| L48s1 | not aimed | 285 | 235 | 1700 | +0.029 [+0.011, +0.048] | 3 / 2 |
| L48s1 | aimed at Claude-data probes | 4432 | 3786 | 9000 | +0.072 [+0.057, +0.087] | 12 / 6 |
| L48s1 | aimed at an earlier on-policy probe | 284 | 122 | 500 | +0.324 [+0.258, +0.392] | 1 / 0 |
| L48s2 | not aimed | 229 | 224 | 1700 | +0.003 [-0.015, +0.021] | 3 / 2 |
| L48s2 | aimed at Claude-data probes | 4215 | 3676 | 9000 | +0.060 [+0.046, +0.074] | 9 / 8 |
| L48s2 | aimed at an earlier on-policy probe | 248 | 118 | 500 | +0.260 [+0.204, +0.318] | 1 / 0 |

### Matched searches

| writer, line, context, counting | off-aimed on off probe | on-aimed on on probe | plain off / on | raw diff [95%] | gain diff [95%] |
|---|---|---|---|---|---|
| Claude Opus 5.5 | 1% line | removed | valid | 247 | 298 | 31 / 15 | -0.102 [-0.166, -0.036] | -0.134 [-0.204, -0.062] |
| Claude Opus 5.5 | 1% line | removed | full_strength | 94 | 135 | 10 / 3 | -0.082 [-0.136, -0.024] | -0.096 [-0.152, -0.040] |
| Claude Opus 5.5 | 1% line | visible | valid | 254 | 308 | 31 / 15 | -0.108 [-0.174, -0.040] | -0.140 [-0.214, -0.068] |
| Claude Opus 5.5 | 1% line | visible | full_strength | 104 | 146 | 10 / 3 | -0.084 [-0.138, -0.026] | -0.098 [-0.154, -0.040] |
| Claude Opus 5.5 | matched-strength line | removed | valid | 220 | 311 | 23 / 23 | -0.182 [-0.246, -0.114] | -0.182 [-0.252, -0.112] |
| Claude Opus 5.5 | matched-strength line | removed | full_strength | 81 | 140 | 7 / 5 | -0.118 [-0.174, -0.060] | -0.122 [-0.178, -0.064] |
| Claude Opus 5.5 | matched-strength line | visible | valid | 211 | 329 | 23 / 23 | -0.236 [-0.310, -0.160] | -0.236 [-0.316, -0.160] |
| Claude Opus 5.5 | matched-strength line | visible | full_strength | 82 | 158 | 7 / 5 | -0.152 [-0.218, -0.090] | -0.156 [-0.222, -0.092] |
| Qwen3.5-122B-A10B | 1% line | removed | valid | 49 | 258 | 31 / 15 | -0.418 [-0.476, -0.358] | -0.450 [-0.516, -0.388] |
| Qwen3.5-122B-A10B | 1% line | removed | full_strength | 18 | 132 | 10 / 3 | -0.228 [-0.278, -0.176] | -0.242 [-0.294, -0.190] |
| Qwen3.5-122B-A10B | 1% line | visible | valid | 76 | 222 | 31 / 15 | -0.292 [-0.356, -0.228] | -0.324 [-0.398, -0.252] |
| Qwen3.5-122B-A10B | 1% line | visible | full_strength | 35 | 113 | 10 / 3 | -0.156 [-0.206, -0.106] | -0.170 [-0.224, -0.118] |
| Qwen3.5-122B-A10B | matched-strength line | removed | valid | 42 | 283 | 23 / 23 | -0.482 [-0.540, -0.422] | -0.482 [-0.544, -0.420] |
| Qwen3.5-122B-A10B | matched-strength line | removed | full_strength | 16 | 150 | 7 / 5 | -0.268 [-0.328, -0.212] | -0.272 [-0.334, -0.214] |
| Qwen3.5-122B-A10B | matched-strength line | visible | valid | 50 | 252 | 23 / 23 | -0.404 [-0.468, -0.344] | -0.404 [-0.472, -0.340] |
| Qwen3.5-122B-A10B | matched-strength line | visible | full_strength | 23 | 133 | 7 / 5 | -0.220 [-0.278, -0.166] | -0.224 [-0.284, -0.168] |
| Claude Opus 5.5, replicate | 1% line | removed | valid | 271 | 283 | 31 / 15 | -0.024 [-0.100, +0.050] | -0.056 [-0.136, +0.026] |
| Claude Opus 5.5, replicate | 1% line | removed | full_strength | 97 | 148 | 10 / 3 | -0.102 [-0.164, -0.042] | -0.116 [-0.180, -0.054] |
| Claude Opus 5.5, replicate | 1% line | visible | valid | 307 | 381 | 31 / 15 | -0.148 [-0.222, -0.076] | -0.180 [-0.262, -0.100] |
| Claude Opus 5.5, replicate | 1% line | visible | full_strength | 119 | 213 | 10 / 3 | -0.188 [-0.254, -0.120] | -0.202 [-0.270, -0.134] |
| Claude Opus 5.5, replicate | matched-strength line | removed | valid | 254 | 299 | 23 / 23 | -0.090 [-0.166, -0.018] | -0.090 [-0.168, -0.014] |
| Claude Opus 5.5, replicate | matched-strength line | removed | full_strength | 87 | 159 | 7 / 5 | -0.144 [-0.206, -0.086] | -0.148 [-0.212, -0.088] |
| Claude Opus 5.5, replicate | matched-strength line | visible | valid | 283 | 394 | 23 / 23 | -0.222 [-0.290, -0.156] | -0.222 [-0.290, -0.150] |
| Claude Opus 5.5, replicate | matched-strength line | visible | full_strength | 103 | 222 | 7 / 5 | -0.238 [-0.306, -0.168] | -0.242 [-0.308, -0.174] |
| Qwen3.5-122B-A10B, replicate | 1% line | removed | valid | 172 | 223 | 31 / 15 | -0.102 [-0.156, -0.044] | -0.134 [-0.192, -0.074] |
| Qwen3.5-122B-A10B, replicate | 1% line | removed | full_strength | 71 | 71 | 10 / 3 | +0.000 [-0.044, +0.042] | -0.014 [-0.060, +0.028] |
| Qwen3.5-122B-A10B, replicate | 1% line | visible | valid | 216 | 225 | 31 / 15 | -0.018 [-0.076, +0.040] | -0.050 [-0.110, +0.012] |
| Qwen3.5-122B-A10B, replicate | 1% line | visible | full_strength | 93 | 77 | 10 / 3 | +0.032 [-0.012, +0.076] | +0.018 [-0.028, +0.064] |
| Qwen3.5-122B-A10B, replicate | matched-strength line | removed | valid | 149 | 244 | 23 / 23 | -0.190 [-0.250, -0.128] | -0.190 [-0.254, -0.126] |
| Qwen3.5-122B-A10B, replicate | matched-strength line | removed | full_strength | 60 | 81 | 7 / 5 | -0.042 [-0.086, +0.000] | -0.046 [-0.090, -0.002] |
| Qwen3.5-122B-A10B, replicate | matched-strength line | visible | valid | 171 | 255 | 23 / 23 | -0.168 [-0.220, -0.114] | -0.168 [-0.224, -0.110] |
| Qwen3.5-122B-A10B, replicate | matched-strength line | visible | full_strength | 70 | 89 | 7 / 5 | -0.038 [-0.080, +0.002] | -0.042 [-0.088, +0.000] |
| Claude Opus 5.5, log-odds display | 1% line | removed | valid | 307 | 387 | 31 / 15 | -0.160 [-0.242, -0.070] | -0.192 [-0.278, -0.104] |
| Claude Opus 5.5, log-odds display | 1% line | removed | full_strength | 204 | 276 | 10 / 3 | -0.144 [-0.260, -0.026] | -0.158 [-0.270, -0.046] |
| Claude Opus 5.5, log-odds display | 1% line | visible | valid | 442 | 447 | 31 / 15 | -0.010 [-0.066, +0.052] | -0.042 [-0.100, +0.022] |
| Claude Opus 5.5, log-odds display | 1% line | visible | full_strength | 300 | 312 | 10 / 3 | -0.024 [-0.124, +0.080] | -0.038 [-0.138, +0.060] |
| Claude Opus 5.5, log-odds display | matched-strength line | removed | valid | 268 | 419 | 23 / 23 | -0.302 [-0.384, -0.220] | -0.302 [-0.384, -0.220] |
| Claude Opus 5.5, log-odds display | matched-strength line | removed | full_strength | 173 | 298 | 7 / 5 | -0.250 [-0.360, -0.144] | -0.254 [-0.364, -0.150] |
| Claude Opus 5.5, log-odds display | matched-strength line | visible | valid | 421 | 448 | 23 / 23 | -0.054 [-0.118, +0.012] | -0.054 [-0.120, +0.010] |
| Claude Opus 5.5, log-odds display | matched-strength line | visible | full_strength | 286 | 312 | 7 / 5 | -0.052 [-0.162, +0.060] | -0.056 [-0.162, +0.052] |
| Claude Opus 5.5, log-odds display, replicate | 1% line | removed | valid | 267 | 356 | 31 / 15 | -0.178 [-0.272, -0.088] | -0.210 [-0.308, -0.118] |
| Claude Opus 5.5, log-odds display, replicate | 1% line | removed | full_strength | 146 | 280 | 10 / 3 | -0.268 [-0.366, -0.168] | -0.282 [-0.382, -0.182] |
| Claude Opus 5.5, log-odds display, replicate | 1% line | visible | valid | 385 | 466 | 31 / 15 | -0.162 [-0.222, -0.108] | -0.194 [-0.260, -0.132] |
| Claude Opus 5.5, log-odds display, replicate | 1% line | visible | full_strength | 224 | 382 | 10 / 3 | -0.316 [-0.410, -0.220] | -0.330 [-0.426, -0.230] |
| Claude Opus 5.5, log-odds display, replicate | matched-strength line | removed | valid | 247 | 387 | 23 / 23 | -0.280 [-0.368, -0.188] | -0.280 [-0.368, -0.188] |
| Claude Opus 5.5, log-odds display, replicate | matched-strength line | removed | full_strength | 130 | 308 | 7 / 5 | -0.356 [-0.450, -0.258] | -0.360 [-0.454, -0.262] |
| Claude Opus 5.5, log-odds display, replicate | matched-strength line | visible | valid | 354 | 467 | 23 / 23 | -0.226 [-0.286, -0.164] | -0.226 [-0.290, -0.164] |
| Claude Opus 5.5, log-odds display, replicate | matched-strength line | visible | full_strength | 205 | 383 | 7 / 5 | -0.356 [-0.450, -0.262] | -0.360 [-0.454, -0.262] |
| Qwen3.5-122B-A10B, log-odds display | 1% line | removed | valid | 152 | 140 | 31 / 15 | +0.024 [-0.038, +0.084] | -0.008 [-0.080, +0.060] |
| Qwen3.5-122B-A10B, log-odds display | 1% line | removed | full_strength | 51 | 68 | 10 / 3 | -0.034 [-0.076, +0.006] | -0.048 [-0.096, -0.006] |
| Qwen3.5-122B-A10B, log-odds display | 1% line | visible | valid | 225 | 115 | 31 / 15 | +0.220 [+0.160, +0.284] | +0.188 [+0.126, +0.258] |
| Qwen3.5-122B-A10B, log-odds display | 1% line | visible | full_strength | 92 | 61 | 10 / 3 | +0.062 [+0.014, +0.112] | +0.048 [-0.004, +0.100] |
| Qwen3.5-122B-A10B, log-odds display | matched-strength line | removed | valid | 121 | 160 | 23 / 23 | -0.078 [-0.140, -0.014] | -0.078 [-0.138, -0.014] |
| Qwen3.5-122B-A10B, log-odds display | matched-strength line | removed | full_strength | 36 | 83 | 7 / 5 | -0.094 [-0.138, -0.052] | -0.098 [-0.148, -0.050] |
| Qwen3.5-122B-A10B, log-odds display | matched-strength line | visible | valid | 185 | 146 | 23 / 23 | +0.078 [+0.018, +0.140] | +0.078 [+0.018, +0.142] |
| Qwen3.5-122B-A10B, log-odds display | matched-strength line | visible | full_strength | 76 | 78 | 7 / 5 | -0.004 [-0.052, +0.044] | -0.008 [-0.058, +0.042] |
| POOLED: log-odds display (pre-registered, amendment 2) | 1% line | removed | valid | 726 | 883 | 31 / 15 | -0.105 [-0.155, -0.055] | -0.137 [-0.195, -0.079] |
| POOLED: log-odds display (pre-registered, amendment 2) | 1% line | removed | full_strength | 401 | 624 | 10 / 3 | -0.149 [-0.204, -0.087] | -0.163 [-0.219, -0.105] |
| POOLED: log-odds display (pre-registered, amendment 2) | 1% line | visible | valid | 1052 | 1028 | 31 / 15 | +0.016 [-0.019, +0.052] | -0.016 [-0.057, +0.025] |
| POOLED: log-odds display (pre-registered, amendment 2) | 1% line | visible | full_strength | 616 | 755 | 10 / 3 | -0.093 [-0.146, -0.036] | -0.107 [-0.161, -0.051] |
| POOLED: log-odds display (pre-registered, amendment 2) | matched-strength line | removed | valid | 636 | 966 | 23 / 23 | -0.220 [-0.269, -0.175] | -0.220 [-0.270, -0.173] |
| POOLED: log-odds display (pre-registered, amendment 2) | matched-strength line | removed | full_strength | 339 | 689 | 7 / 5 | -0.233 [-0.286, -0.180] | -0.237 [-0.290, -0.184] |
| POOLED: log-odds display (pre-registered, amendment 2) | matched-strength line | visible | valid | 960 | 1061 | 23 / 23 | -0.067 [-0.103, -0.031] | -0.067 [-0.106, -0.030] |
| POOLED: log-odds display (pre-registered, amendment 2) | matched-strength line | visible | full_strength | 567 | 773 | 7 / 5 | -0.137 [-0.194, -0.085] | -0.141 [-0.195, -0.088] |
| pooled: old display | 1% line | removed | valid | 739 | 1062 | 31 / 15 | -0.162 [-0.197, -0.130] | -0.194 [-0.237, -0.153] |
| pooled: old display | 1% line | removed | full_strength | 280 | 486 | 10 / 3 | -0.103 [-0.132, -0.075] | -0.117 [-0.149, -0.085] |
| pooled: old display | 1% line | visible | valid | 853 | 1136 | 31 / 15 | -0.141 [-0.174, -0.108] | -0.173 [-0.217, -0.129] |
| pooled: old display | 1% line | visible | full_strength | 351 | 549 | 10 / 3 | -0.099 [-0.130, -0.068] | -0.113 [-0.145, -0.081] |
| pooled: old display | matched-strength line | removed | valid | 665 | 1137 | 23 / 23 | -0.236 [-0.271, -0.203] | -0.236 [-0.277, -0.196] |
| pooled: old display | matched-strength line | removed | full_strength | 244 | 530 | 7 / 5 | -0.143 [-0.175, -0.112] | -0.147 [-0.181, -0.112] |
| pooled: old display | matched-strength line | visible | valid | 715 | 1230 | 23 / 23 | -0.258 [-0.289, -0.223] | -0.257 [-0.297, -0.218] |
| pooled: old display | matched-strength line | visible | full_strength | 278 | 602 | 7 / 5 | -0.162 [-0.198, -0.127] | -0.166 [-0.202, -0.130] |