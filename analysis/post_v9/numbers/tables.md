## Item 1: gap closed (plain-referenced), slope per 10 ECI, 14 writers

| task | line | plain median | B = 16 | B = 0 |
|---|---|---|---|---|
| maths | -6.53 | 10.32 | +0.063 [+0.031, +0.102] | +0.021 [+0.004, +0.040] |
| ordering | -8.06 | 11.96 | +0.099 [+0.062, +0.140] | +0.047 [+0.029, +0.072] |
| belief tracking | -2.39 | 8.03 | +0.121 [+0.068, +0.182] | +0.017 [-0.012, +0.052] |
| acrostic | -2.55 | 8.34 | +0.165 [+0.129, +0.206] | +0.019 [-0.002, +0.052] |
| contradiction | -0.58 | 7.79 | +0.129 [+0.091, +0.178] | -0.009 [-0.032, +0.019] |
| maths, correctness-only validity | | | +0.062 [+0.029, +0.103] | +0.021 [+0.004, +0.040] |

### Per writer, gap closed at B = 16 (mean of 3 chains)

| writer (ECI) | maths | ordering | belief tracking | acrostic | contradiction |
|---|---|---|---|---|---|
| Llama 4 Maverick (132.2) | +0.148 | +0.057 | +0.062 | -0.065 | +0.037 |
| GPT-5 nano (139.4) | +0.062 | +0.064 | +0.009 | -0.037 | -0.005 |
| Qwen3.5-9B (139.5) | +0.005 | +0.077 | +0.096 | -0.026 | +0.024 |
| Claude Haiku 4.5 (142.4) | +0.102 | +0.188 | +0.070 | +0.001 | +0.034 |
| Qwen3.5-397B-A17B (146.7) | +0.071 | +0.152 | +0.218 | +0.040 | +0.036 |
| GPT-5.4 Mini (148.8) | +0.028 | +0.069 | -0.006 | +0.142 | +0.158 |
| Gemini 3.8 Flash (156.7) | +0.089 | +0.204 | +0.322 | +0.467 | +0.399 |
| Claude Opus 4.8 (158.2) | +0.294 | +0.307 | +0.419 | +0.152 | +0.409 |
| GPT-6 Sol (162.7) | +0.176 | +0.275 | +0.396 | +0.340 | +0.384 |
| Claude Fable 5.1 (164.7) | +0.294 | +0.467 | +0.516 | +0.538 | +0.455 |
| Claude Sonnet 5.5 (165.0) | +0.216 | +0.381 | +0.414 | +0.451 | +0.295 |
| GPT-6.1 Sol (166.1) | +0.231 | +0.212 | +0.151 | +0.410 | +0.317 |
| GPT-6 Astra (166.4) | +0.269 | +0.355 | +0.405 | +0.305 | +0.399 |
| Claude Opus 5.5 (167.3) | +0.261 | +0.428 | +0.419 | +0.463 | +0.326 |

### Per writer, gap closed at B = 0 (mean of 3 chains)

| writer (ECI) | maths | ordering | belief tracking | acrostic | contradiction |
|---|---|---|---|---|---|
| Llama 4 Maverick (132.2) | +0.014 | +0.028 | +0.029 | -0.046 | -0.009 |
| GPT-5 nano (139.4) | +0.017 | +0.021 | +0.009 | -0.031 | -0.001 |
| Qwen3.5-9B (139.5) | +0.027 | +0.018 | +0.006 | -0.014 | +0.017 |
| Claude Haiku 4.5 (142.4) | +0.001 | +0.033 | +0.009 | -0.069 | +0.018 |
| Qwen3.5-397B-A17B (146.7) | +0.055 | +0.013 | -0.004 | -0.018 | -0.050 |
| GPT-5.4 Mini (148.8) | +0.012 | +0.012 | -0.008 | -0.065 | -0.012 |
| Gemini 3.8 Flash (156.7) | +0.035 | +0.013 | -0.094 | -0.024 | -0.022 |
| Claude Opus 4.8 (158.2) | +0.024 | +0.252 | +0.003 | -0.040 | +0.037 |
| GPT-6 Sol (162.7) | +0.000 | +0.133 | +0.195 | -0.027 | +0.108 |
| Claude Fable 5.1 (164.7) | +0.072 | +0.207 | +0.094 | +0.064 | -0.090 |
| Claude Sonnet 5.5 (165.0) | +0.041 | +0.095 | -0.058 | +0.001 | -0.045 |
| GPT-6.1 Sol (166.1) | +0.090 | +0.148 | +0.078 | -0.025 | -0.108 |
| GPT-6 Astra (166.4) | +0.149 | +0.123 | +0.092 | +0.079 | -0.027 |
| Claude Opus 5.5 (167.3) | +0.087 | +0.149 | +0.014 | -0.015 | +0.013 |

## Item 2: B = 16, robustness of the trend

| task | all 14 | without the 5 Anthropic writers (9) | lab fixed effects (within-lab) | Spearman rho, perm. p (two-sided / one-sided) | Spearman without Anthropic |
|---|---|---|---|---|---|
| maths | +0.063 [+0.031, +0.102] | +0.049 [-0.004, +0.099] | +0.073 [+0.019, +0.120] | 0.72, p = 0.0043 / 0.0022 | 0.67, p = 0.0582 / 0.0308 |
| ordering | +0.099 [+0.062, +0.140] | +0.075 [+0.033, +0.119] | +0.098 [+0.052, +0.162] | 0.88, p = 0.0001 / 0.0001 | 0.93, p = 0.0007 / 0.0002 |
| belief tracking | +0.121 [+0.068, +0.182] | +0.092 [+0.027, +0.169] | +0.143 [+0.035, +0.233] | 0.72, p = 0.0054 / 0.0029 | 0.70, p = 0.0449 / 0.0219 |
| acrostic | +0.165 [+0.129, +0.206] | +0.149 [+0.102, +0.214] | +0.170 [+0.099, +0.270] | 0.81, p = 0.0006 / 0.0001 | 0.85, p = 0.0065 / 0.0034 |
| contradiction | +0.129 [+0.091, +0.178] | +0.128 [+0.084, +0.192] | +0.132 [-0.016, +0.190] | 0.67, p = 0.0104 / 0.0053 | 0.78, p = 0.0165 / 0.0077 |

## Item 3: pooled slope with task offsets (task fixed effects)

| fit | pooled slope [95%] | per-task slopes | task minus pooled, 95% interval |
|---|---|---|---|
| B0_all14 | +0.019 [+0.008, +0.034] | maths +0.021, ordering +0.047, belief tracking +0.017, acrostic +0.019, contradiction -0.009 | maths [-0.019, +0.017], ordering [+0.011, +0.048], belief tracking [-0.024, +0.024], acrostic [-0.021, +0.025], contradiction [-0.054, -0.003] |
| B0_without_anthropic | +0.020 [-0.005, +0.043] | maths +0.022, ordering +0.037, belief tracking +0.029, acrostic +0.015, contradiction -0.005 | maths [-0.034, +0.026], ordering [-0.006, +0.041], belief tracking [-0.031, +0.049], acrostic [-0.036, +0.040], contradiction [-0.055, +0.021] |
| B0_new_tasks_only | +0.009 [-0.005, +0.029] | belief tracking +0.017, acrostic +0.019, contradiction -0.009 | belief tracking [-0.013, +0.032], acrostic [-0.012, +0.033], contradiction [-0.042, +0.003] |
| B0_maths_ordering_only | +0.034 [+0.022, +0.050] | maths +0.021, ordering +0.047 | maths [-0.028, -0.001], ordering [+0.001, +0.028] |
| B16_all14 | +0.115 [+0.091, +0.148] | maths +0.063, ordering +0.099, belief tracking +0.121, acrostic +0.165, contradiction +0.129 | maths [-0.078, -0.024], ordering [-0.047, +0.008], belief tracking [-0.036, +0.042], acrostic [+0.015, +0.078], contradiction [-0.020, +0.049] |
| B16_without_anthropic | +0.099 [+0.071, +0.137] | maths +0.049, ordering +0.075, belief tracking +0.092, acrostic +0.149, contradiction +0.128 | maths [-0.106, -0.007], ordering [-0.062, +0.006], belief tracking [-0.062, +0.040], acrostic [-0.001, +0.106], contradiction [-0.001, +0.069] |
| B16_new_tasks_only | +0.138 [+0.110, +0.176] | belief tracking +0.121, acrostic +0.165, contradiction +0.129 | belief tracking [-0.055, +0.018], acrostic [-0.007, +0.056], contradiction [-0.039, +0.022] |
| B16_maths_ordering_only | +0.081 [+0.054, +0.117] | maths +0.063, ordering +0.099 | maths [-0.036, +0.003], ordering [-0.003, +0.036] |

## Item 4: crossings at the strict 1% line, B = 16, all 14 writers

| task | pairs with both replies correct | forward (ON under, OFF over) | reverse (OFF under, ON over) | valid ON replies under the line (unpaired) |
|---|---|---|---|---|
| maths | 17,376 | 0 | 0 | 0 |
| ordering | 17,373 | 0 | 0 | 0 |
| belief tracking | 15,484 | 2,971 | 190 | 4,104 |
| acrostic | 17,553 | 244 | 29 | 309 |
| contradiction | 18,528 | 6,565 | 554 | 9,192 |

Net crossing rate per writer = (forward - reverse) / all 1,500 pairs (3 chains x 500); 95% interval resamples chains, then problems. Writers with net > 5% on any task:

| task | writer | net [95%] | forward | reverse | reverse rate |
|---|---|---|---|---|---|
| belief tracking | Claude Sonnet 5.5 | +51.0% [+23.5, +72.3] | 765 | 0 | 0.0% |
| belief tracking | Claude Opus 5.5 | +33.9% [+5.2, +53.9] | 513 | 5 | 0.3% |
| belief tracking | Claude Fable 5.1 | +32.7% [+13.9, +60.4] | 490 | 0 | 0.0% |
| belief tracking | GPT-6 Astra | +29.2% [+12.4, +46.0] | 440 | 2 | 0.1% |
| belief tracking | Claude Opus 4.8 | +20.0% [-2.3, +40.8] | 319 | 19 | 1.3% |
| belief tracking | Gemini 3.8 Flash | +10.7% [+0.0, +21.0] | 160 | 0 | 0.0% |
| belief tracking | GPT-6.1 Sol | -6.3% [-13.2, -0.7] | 30 | 124 | 8.3% |
| acrostic | Claude Fable 5.1 | +7.3% [+1.1, +16.7] | 110 | 1 | 0.1% |
| contradiction | Claude Opus 5.5 | +66.8% [+32.5, +87.3] | 1022 | 20 | 1.3% |
| contradiction | Claude Fable 5.1 | +62.1% [+42.0, +83.1] | 934 | 2 | 0.1% |
| contradiction | Claude Opus 4.8 | +60.0% [+32.5, +78.3] | 916 | 16 | 1.1% |
| contradiction | Claude Sonnet 5.5 | +59.7% [+38.2, +74.7] | 900 | 5 | 0.3% |
| contradiction | GPT-6.1 Sol | +55.2% [+48.3, +62.3] | 828 | 0 | 0.0% |
| contradiction | GPT-6 Astra | +48.5% [+32.7, +62.9] | 773 | 45 | 3.0% |
| contradiction | Gemini 3.8 Flash | +31.5% [+16.0, +44.1] | 473 | 0 | 0.0% |
| contradiction | GPT-5.4 Mini | +16.2% [-17.5, +42.7] | 342 | 99 | 6.6% |
| contradiction | GPT-6 Sol | +10.3% [-11.3, +29.1] | 286 | 132 | 8.8% |
| contradiction | Claude Haiku 4.5 | -3.2% [-8.7, +2.7] | 27 | 75 | 5.0% |
| contradiction | Llama 4 Maverick | -3.8% [-8.1, -0.6] | 4 | 61 | 4.1% |
| contradiction | GPT-5 nano | -6.4% [-13.5, +0.0] | 0 | 96 | 6.4% |

## Item 5: accuracy cost at B = 16 (OFF correct share - ON correct share, percentage points; mean over chains with a template)

Maths: m45's correctness-only re-judge. Other tasks: the task's valid flag.

| writer (ECI) | maths | ordering | belief tracking | acrostic | contradiction |
|---|---|---|---|---|---|
| Llama 4 Maverick (132.2) | +33.1 (99 vs 66) | +0.9 (91 vs 90) | +7.1 (70 vs 63) | +7.5 (96 vs 88) | +0.5 (96 vs 95) |
| GPT-5 nano (139.4) | +3.9 (98 vs 94) | +16.1 (88 vs 72) | +3.4 (70 vs 67) | -2.4 (96 vs 98) | -0.1 (95 vs 95) |
| Qwen3.5-9B (139.5) | +1.2 (93 vs 91) | +2.3 (91 vs 89) | +6.9 (71 vs 64) | +4.4 (98 vs 94) | -0.3 (94 vs 94) |
| Claude Haiku 4.5 (142.4) | +0.5 (100 vs 99) | +10.1 (90 vs 80) | -2.1 (56 vs 58) | +7.2 (87 vs 80) | +1.1 (93 vs 92) |
| Qwen3.5-397B-A17B (146.7) | +2.6 (98 vs 95) | +3.6 (88 vs 85) | +7.7 (66 vs 58) | +20.5 (85 vs 64) | -1.8 (85 vs 87) |
| GPT-5.4 Mini (148.8) | -0.1 (100 vs 100) | +3.3 (90 vs 87) | +3.5 (58 vs 54) | +0.2 (80 vs 79) | -2.1 (85 vs 87) |
| Gemini 3.8 Flash (156.7) | +3.1 (99 vs 96) | -0.8 (93 vs 94) | +2.9 (81 vs 78) | +0.8 (98 vs 97) | +0.6 (96 vs 96) |
| Claude Opus 4.8 (158.2) | +16.7 (100 vs 83) | +5.3 (90 vs 85) | +2.4 (71 vs 69) | +3.7 (92 vs 88) | -4.5 (86 vs 91) |
| GPT-6 Sol (162.7) | +21.8 (100 vs 78) | +0.7 (93 vs 92) | -1.3 (91 vs 92) | +2.3 (97 vs 95) | +0.4 (93 vs 93) |
| Claude Fable 5.1 (164.7) | +20.6 (100 vs 79) | +10.0 (91 vs 81) | -1.5 (95 vs 96) | +11.1 (98 vs 87) | -2.0 (90 vs 92) |
| Claude Sonnet 5.5 (165.0) | +18.8 (97 vs 78) | +2.1 (93 vs 90) | +4.7 (96 vs 91) | +1.5 (98 vs 97) | +3.5 (94 vs 90) |
| GPT-6.1 Sol (166.1) | +30.1 (99 vs 69) | +1.1 (88 vs 87) | +0.1 (99 vs 99) | +5.5 (97 vs 91) | +0.1 (91 vs 91) |
| GPT-6 Astra (166.4) | +22.3 (100 vs 78) | +3.6 (92 vs 89) | +0.6 (94 vs 93) | +9.6 (97 vs 87) | +1.6 (96 vs 94) |
| Claude Opus 5.5 (167.3) | +18.5 (100 vs 81) | +5.5 (89 vs 83) | +7.7 (98 vs 90) | +2.4 (99 vs 96) | +3.3 (94 vs 91) |
| slope of cost per 10 ECI (share) | +0.039 [-0.047, +0.109] | -0.008 [-0.052, +0.015] | -0.010 [-0.031, +0.012] | -0.002 [-0.044, +0.033] | +0.004 [-0.009, +0.019] |
