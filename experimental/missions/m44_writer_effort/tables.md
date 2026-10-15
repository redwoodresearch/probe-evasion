## Baseline: m36's 11 writers only (cross-model part)

| quantity | observed | 95% two-stage |
|---|---|---|
| gamma | +0.150 | [-0.12, +0.40] |
| delta | +0.172 | [+0.12, +0.22] |
| coef_eci_per_10 | +0.171 | [+0.11, +0.24] |
| coef_log10_tokens | +0.011 | [-0.18, +0.23] |
| eci_only_slope_per_10 | +0.172 | [+0.12, +0.22] |

## Effort arms

| arm | chains | S, B=0 | S, B=4 | S, B=8 | S, B=12 | S, B=16 | S chains (B=16) | S_valid (16) | valid share (16) | S_late | eval S late | rounds with instruction | median output tokens / call | output tokens to 16 | $ to 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 (low) | 5 | +0.20 | +0.27 | +0.58 | +0.67 | +0.72 | +0.70, +0.63, +0.73, +0.76, +0.78 | +0.74 | +0.85 | +0.69 | +0.67 | 84 / 85 | 935 | 16544 | $2.11 |
| Claude Opus 5.5 (medium) | 5 | +0.15 | +0.39 | +0.61 | +0.48 | +0.69 | +0.79, +0.80, +0.62, +0.72, +0.52 | +0.73 | +0.76 | +0.59 | +0.66 | 85 / 85 | 1585 | 30196 | $2.68 |
| Claude Opus 5.5 (high) | 5 | +0.30 | +0.57 | +0.62 | +0.66 | +0.70 | +0.76, +0.65, +0.59, +0.71, +0.79 | +0.74 | +0.78 | +0.68 | +0.68 | 85 / 85 | 2430 | 50236 | $3.20 |
| Claude Opus 5.5 (xhigh) | 5 | +0.15 | +0.29 | +0.39 | +0.37 | +0.63 | +0.64, +0.82, +0.59, +0.79, +0.33 | +0.66 | +0.80 | +0.50 | +0.60 | 60 / 85 | 4720 | 160784 | $4.47 |
| Claude Opus 5.5 (max) | 5 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00, +0.00, +0.00, +0.00, +0.00 | - | +0.00 | +0.00 | - | 0 / 85 | 1095 | 60897 | $1.25 |
| Claude Opus 5.5 (high, m36) | 3 | +0.08 | +0.47 | +0.54 | - | +0.74 | +0.79, +0.69, +0.74 | +0.77 | +0.81 | +0.64 | +0.71 | 51 / 51 | 2357 | 51380 | $2.95 |
| Claude Sonnet 5.5 (low) | 5 | +0.13 | +0.19 | +0.29 | +0.29 | +0.43 | +0.35, +0.43, +0.39, +0.59, +0.40 | +0.44 | +0.88 | +0.36 | +0.42 | 85 / 85 | 254 | 4284 | $0.93 |
| Claude Sonnet 5.5 (high) | 5 | +0.09 | +0.25 | +0.12 | +0.36 | +0.55 | +0.43, +0.61, +0.68, +0.46, +0.58 | +0.57 | +0.89 | +0.46 | +0.50 | 85 / 85 | 1538 | 25986 | $1.18 |
| Claude Sonnet 5.5 (high, m36) | 3 | +0.03 | +0.33 | +0.33 | - | +0.59 | +0.60, +0.70, +0.46 | +0.61 | +0.83 | +0.46 | +0.59 | 51 / 51 | 1222 | 20760 | $1.18 |
| GPT-6 Sol (low) | 5 | +0.23 | +0.37 | +0.36 | +0.38 | +0.34 | +0.51, +0.29, +0.47, +0.00, +0.45 | +0.44 | +0.73 | +0.36 | +0.44 | 85 / 85 | 275 | 4821 | $0.65 |
| GPT-6 Sol (medium) | 5 | +0.25 | +0.21 | +0.36 | +0.44 | +0.54 | +0.67, +0.51, +0.43, +0.45, +0.66 | +0.55 | +0.90 | +0.49 | +0.49 | 85 / 85 | 593 | 10611 | $0.63 |
| GPT-6 Sol (high) | 5 | +0.26 | +0.27 | +0.33 | +0.56 | +0.55 | +0.57, +0.38, +0.75, +0.65, +0.40 | +0.56 | +0.87 | +0.55 | +0.51 | 85 / 85 | 1117 | 19050 | $0.71 |
| GPT-6 Sol (medium (default), m36) | 3 | +0.11 | +0.20 | +0.34 | - | +0.51 | +0.66, +0.48, +0.39 | +0.52 | +0.91 | +0.42 | +0.47 | 51 / 51 | 618 | 10608 | $0.58 |
| Gemini 3.8 Flash (default) | 5 | +0.07 | +0.14 | +0.34 | +0.42 | +0.52 | +0.40, +0.57, +0.60, +0.45, +0.57 | +0.53 | +0.91 | +0.47 | +0.48 | 85 / 85 | 719 | 48381 | $1.02 |
| Gemini 3.8 Flash (high) | 5 | +0.07 | +0.20 | +0.32 | +0.40 | +0.44 | +0.49, +0.47, +0.51, +0.39, +0.34 | +0.45 | +0.92 | +0.42 | +0.41 | 85 / 85 | 1398 | 133524 | $1.76 |
| Gemini 3.8 Flash (default, m36) | 3 | +0.04 | +0.24 | +0.28 | - | +0.41 | +0.56, +0.24, +0.44 | +0.42 | +0.93 | +0.35 | +0.35 | 51 / 51 | 1176 | 46254 | $0.55 |

## Refusals per arm (API refusal stops are the API's filter, cut off mid-generation; Haiku-flagged and plain-text are the writer's own text)

| arm | refused / answered attempts | share | API stops | API categories | Haiku-flagged | plain text | chains with a refusal | failed rounds by cause | cut-off attempts |
|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 (low) | 0 / 89 | 0% | 0 | - | 0 | 0 | 0 | {'leak': 1} | 0 |
| Claude Opus 5.5 (medium) | 2 / 90 | 2% | 2 | {'cyber': 2} | 0 | 0 | 2 | - | 0 |
| Claude Opus 5.5 (high) | 9 / 95 | 9% | 9 | {'cyber': 9} | 0 | 0 | 3 | - | 0 |
| Claude Opus 5.5 (xhigh) | 100 / 160 | 62% | 99 | {'cyber': 99} | 1 | 0 | 5 | {'refusal': 25} | 0 |
| Claude Opus 5.5 (max) | 255 / 255 | 100% | 255 | {'cyber': 255} | 0 | 0 | 5 | {'refusal': 85} | 0 |
| Claude Opus 5.5 (high, m36) | 3 / 57 | 5% | 3 | {'not recorded': 3} | 0 | 0 | 2 | - | 0 |
| Claude Sonnet 5.5 (low) | 0 / 85 | 0% | 0 | - | 0 | 0 | 0 | - | 0 |
| Claude Sonnet 5.5 (high) | 0 / 88 | 0% | 0 | - | 0 | 0 | 0 | - | 0 |
| Claude Sonnet 5.5 (high, m36) | 0 / 53 | 0% | 0 | - | 0 | 0 | 0 | - | 0 |
| GPT-6 Sol (low) | 0 / 86 | 0% | 0 | - | 0 | 0 | 0 | - | 0 |
| GPT-6 Sol (medium) | 0 / 85 | 0% | 0 | - | 0 | 0 | 0 | - | 0 |
| GPT-6 Sol (high) | 0 / 86 | 0% | 0 | - | 0 | 0 | 0 | - | 0 |
| GPT-6 Sol (medium (default), m36) | 0 / 51 | 0% | 0 | - | 0 | 0 | 0 | - | 0 |
| Gemini 3.8 Flash (default) | 0 / 85 | 0% | 0 | - | 0 | 0 | 0 | - | 0 |
| Gemini 3.8 Flash (high) | 0 / 86 | 0% | 0 | - | 0 | 0 | 0 | - | 0 |
| Gemini 3.8 Flash (default, m36) | 0 / 51 | 0% | 0 | - | 0 | 0 | 0 | - | 0 |

## Primary effort contrasts (S at B = 16)

| contrast | difference | 95% Welch | 95% bootstrap | chains |
|---|---|---|---|---|
| opus55-high - opus55-low | -0.02 | [-0.12, +0.09] | [-0.09, +0.06] | 5 vs 5 |
| sonnet55-high - sonnet55-low | +0.12 | [-0.02, +0.26] | [+0.01, +0.22] | 5 vs 5 |
| gpt6sol-high - gpt6sol-low | +0.20 | [-0.07, +0.48] | [+0.01, +0.41] | 5 vs 5 |
| gemini38flash-high - gemini38flash-default | -0.08 | [-0.19, +0.04] | [-0.17, +0.01] | 5 vs 5 |
| opus55-max - opus55-low | -0.72 | [-0.79, -0.65] | [-0.76, -0.67] | 5 vs 5 |
| opus55-xhigh - opus55-low | -0.09 | [-0.32, +0.15] | [-0.26, +0.06] | 5 vs 5 |
| gpt6sol-high - gpt6sol-low, without the low chain at 0 | +0.12 | [-0.09, +0.32] | [-0.03, +0.27] | 5 vs 4 |

## Secondary contrasts

| measure: contrast | difference | 95% Welch | 95% bootstrap |
|---|---|---|---|
| S: opus55-high - opus55-low | -0.02 | [-0.12, +0.09] | [-0.09, +0.06] |
| S: sonnet55-high - sonnet55-low | +0.12 | [-0.02, +0.26] | [+0.01, +0.22] |
| S: gpt6sol-high - gpt6sol-low | +0.20 | [-0.07, +0.48] | [+0.01, +0.41] |
| S: gemini38flash-high - gemini38flash-default | -0.08 | [-0.19, +0.04] | [-0.17, +0.01] |
| S: opus55-medium - opus55-low | -0.03 | [-0.17, +0.12] | [-0.13, +0.07] |
| S: opus55-high - opus55-medium | +0.01 | [-0.14, +0.16] | [-0.10, +0.12] |
| S: opus55-xhigh - opus55-high | -0.07 | [-0.31, +0.17] | [-0.24, +0.09] |
| S: gpt6sol-medium - gpt6sol-low | +0.20 | [-0.06, +0.46] | [+0.04, +0.40] |
| S: gpt6sol-high - gpt6sol-medium | +0.00 | [-0.20, +0.21] | [-0.15, +0.16] |
| S_late: opus55-high - opus55-low | -0.01 | [-0.10, +0.08] | [-0.08, +0.06] |
| S_late: sonnet55-high - sonnet55-low | +0.10 | [-0.10, +0.29] | [-0.04, +0.22] |
| S_late: gpt6sol-high - gpt6sol-low | +0.19 | [+0.02, +0.36] | [+0.07, +0.31] |
| S_late: gemini38flash-high - gemini38flash-default | -0.05 | [-0.19, +0.09] | [-0.15, +0.06] |
| S_late: opus55-medium - opus55-low | -0.11 | [-0.39, +0.18] | [-0.30, +0.07] |
| S_late: opus55-high - opus55-medium | +0.10 | [-0.19, +0.38] | [-0.09, +0.29] |
| S_late: opus55-xhigh - opus55-high | -0.18 | [-0.49, +0.13] | [-0.39, +0.01] |
| S_late: gpt6sol-medium - gpt6sol-low | +0.13 | [-0.01, +0.27] | [+0.03, +0.24] |
| S_late: gpt6sol-high - gpt6sol-medium | +0.06 | [-0.11, +0.23] | [-0.07, +0.19] |
| eval_S_late: opus55-high - opus55-low | +0.01 | [-0.08, +0.10] | [-0.05, +0.07] |
| eval_S_late: sonnet55-high - sonnet55-low | +0.08 | [-0.07, +0.22] | [-0.03, +0.18] |
| eval_S_late: gpt6sol-high - gpt6sol-low | +0.07 | [-0.05, +0.20] | [-0.01, +0.16] |
| eval_S_late: gemini38flash-high - gemini38flash-default | -0.07 | [-0.21, +0.07] | [-0.18, +0.03] |
| eval_S_late: opus55-medium - opus55-low | -0.00 | [-0.15, +0.14] | [-0.12, +0.10] |
| eval_S_late: opus55-high - opus55-medium | +0.01 | [-0.13, +0.16] | [-0.07, +0.12] |
| eval_S_late: opus55-xhigh - opus55-high | -0.08 | [-0.30, +0.15] | [-0.24, +0.06] |
| eval_S_late: gpt6sol-medium - gpt6sol-low | +0.05 | [-0.12, +0.22] | [-0.06, +0.17] |
| eval_S_late: gpt6sol-high - gpt6sol-medium | +0.02 | [-0.16, +0.20] | [-0.11, +0.16] |
| S_valid: opus55-high - opus55-low | +0.00 | [-0.07, +0.08] | [-0.05, +0.07] |
| S_valid: sonnet55-high - sonnet55-low | +0.12 | [-0.02, +0.27] | [+0.01, +0.23] |
| S_valid: gpt6sol-high - gpt6sol-low | +0.12 | [-0.08, +0.33] | [-0.02, +0.28] |
| S_valid: gemini38flash-high - gemini38flash-default | -0.08 | [-0.20, +0.04] | [-0.17, +0.01] |
| S_valid: opus55-medium - opus55-low | -0.01 | [-0.14, +0.12] | [-0.11, +0.08] |
| S_valid: opus55-high - opus55-medium | +0.01 | [-0.12, +0.15] | [-0.07, +0.11] |
| S_valid: opus55-xhigh - opus55-high | -0.08 | [-0.30, +0.14] | [-0.24, +0.05] |
| S_valid: gpt6sol-medium - gpt6sol-low | +0.11 | [-0.06, +0.28] | [-0.00, +0.24] |
| S_valid: gpt6sol-high - gpt6sol-medium | +0.01 | [-0.19, +0.22] | [-0.14, +0.17] |

## Replication: the m44 arm repeating m36's setting vs m36's chains (S, B = 16)

| arm | m36 writer | difference | 95% Welch | pooled |
|---|---|---|---|---|
| opus55-high | opus55 | -0.04 | [-0.15, +0.08] | True |
| sonnet55-high | sonnet55 | -0.04 | [-0.27, +0.20] | True |
| gpt6sol-medium | gpt6sol | +0.03 | [-0.24, +0.31] | True |
| gemini38flash-default | gemini38flash | +0.10 | [-0.24, +0.45] | True |

## Sensitivity: primary contrasts without chains that had a refusal

| contrast | difference | 95% Welch | chains |
|---|---|---|---|
| opus55-high - opus55-low | -0.04 | [-0.17, +0.09] | 2 vs 5 |
| sonnet55-high - sonnet55-low | +0.12 | [-0.02, +0.26] | 5 vs 5 |
| gpt6sol-high - gpt6sol-low | +0.20 | [-0.07, +0.48] | 5 vs 5 |
| gemini38flash-high - gemini38flash-default | -0.08 | [-0.19, +0.04] | 5 vs 5 |

## Capability vs compute (S at B = 16)

| quantity | observed | 95% two-stage | draws used |
|---|---|---|---|
| beta_tok | +0.153 | [-0.08, +0.36] | 9932 |
| gamma | +0.156 | [-0.11, +0.40] | 9998 |
| delta | +0.172 | [+0.12, +0.22] | 9998 |
| share_from_compute | +0.139 | [-0.22, +0.45] | 9930 |
| coef_eci_per_10 | +0.153 | [+0.10, +0.23] | 9998 |
| coef_log10_tokens | +0.052 | [-0.11, +0.21] | 9998 |
| eci_only_slope_per_10 | +0.158 | [+0.10, +0.21] | 9998 |

Within-model slope of S on log10 output tokens (effort arms, per model): gemini38flash -0.09, gpt6sol +0.35, opus55 -0.01, sonnet55 +0.18

| setting point | ECI | chains | S (16) | output tokens to 16 | $ to 16 |
|---|---|---|---|---|---|
| llama4maverick | 132.2 | 3 | +0.09 | 2436 | $0.17 |
| gpt5nano | 139.4 | 3 | +0.03 | 20934 | $0.02 |
| qwen35_9b | 139.5 | 3 | +0.20 | 50958 | $0.12 |
| haiku45 | 142.4 | 3 | +0.26 | 43332 | $0.55 |
| qwen35_397b | 146.7 | 3 | +0.44 | 14221 | $0.48 |
| gpt54mini | 148.8 | 3 | +0.31 | 27817 | $0.29 |
| gemini38flash + gemini38flash-default | 156.7 | 8 | +0.48 | 47583 | $0.84 |
| gemini38flash-high | 156.7 | 5 | +0.44 | 133524 | $1.76 |
| opus48 | 158.2 | 3 | +0.54 | 109398 | $4.77 |
| gpt6sol + gpt6sol-medium | 162.7 | 8 | +0.53 | 10610 | $0.62 |
| gpt6sol-high | 162.7 | 5 | +0.55 | 19050 | $0.71 |
| gpt6sol-low | 162.7 | 5 | +0.34 | 4821 | $0.65 |
| sonnet55 + sonnet55-high | 165.0 | 8 | +0.57 | 24026 | $1.18 |
| sonnet55-low | 165.0 | 5 | +0.43 | 4284 | $0.93 |
| opus55 + opus55-high | 167.3 | 8 | +0.72 | 50665 | $3.11 |
| opus55-low | 167.3 | 5 | +0.72 | 16544 | $2.11 |
| opus55-medium | 167.3 | 5 | +0.69 | 30196 | $2.68 |

## Longer searches

| writer | S, B=16 | S, B=24 | S, B=32 | S, B=40 | S, B=48 | S chains (48) | S_valid (48) | valid share (48) | valid under 1% line (48; re-judged ok) | lowest valid score (any B) | $ to 48 per chain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | +0.74 | +0.69 | +0.72 | +0.50 | +0.75 | +0.83, +0.64, +0.78 | +0.81 | +0.71 | 0 | -5.53 | $19.30 |
| GPT-6 Sol | +0.51 | +0.52 | +0.60 | +0.65 | +0.72 | +0.72, +0.77, +0.67 | +0.73 | +0.83 | 0 | -4.82 | $3.61 |
| Gemini 3.8 Flash | +0.41 | +0.41 | +0.25 | +0.39 | +0.52 | +0.60, +0.37, +0.57 | +0.52 | +0.92 | 0 | -6.88 | $2.19 |
| Qwen3.5-397B-A17B | +0.44 | +0.14 | +0.46 | +0.46 | +0.27 | +0.06, +0.34, +0.42 | +0.47 | +0.81 | 0 | -3.80 | $2.11 |

| writer | gain 16 -> 48 [bootstrap 95%] | [paired t 95%] | reading | gain 32 -> 48 [bootstrap] | [paired t] | reading |
|---|---|---|---|---|---|---|
| Claude Opus 5.5 | +0.01 [-0.05, +0.05] | [-0.12, +0.15] | plateau | +0.03 [+0.01, +0.07] | [-0.05, +0.11] | both: small positive gain |
| GPT-6 Sol | +0.21 [+0.05, +0.29] | [-0.12, +0.54] | still improving | +0.11 [+0.00, +0.23] | [-0.16, +0.39] | still improving |
| Gemini 3.8 Flash | +0.10 [+0.04, +0.13] | [-0.02, +0.23] | still improving | +0.26 [+0.05, +0.57] | [-0.42, +0.95] | still improving |
| Qwen3.5-397B-A17B | -0.17 [-0.53, +0.04] | [-0.94, +0.60] | plateau | -0.19 [-0.54, +0.00] | [-0.94, +0.56] | plateau |

Gap (Opus 5.5 minus the mean of the other three): B = 16 +0.28, B = 48 +0.25; change -0.03 [-0.14, +0.09].
ECI slope over the 4 writers (per 10 ECI, descriptive): B=16 +0.127, B=24 +0.261, B=32 +0.138, B=40 +0.055, B=48 +0.245

Re-test of m36's B = 16 instructions and plain (identical replies / compared):
gemini38flash/chain_0@B16 500/500, gemini38flash/chain_1@B16 500/500, gemini38flash/chain_2@B16 500/500, gpt6sol/chain_0@B16 499/500, gpt6sol/chain_1@B16 500/500, gpt6sol/chain_2@B16 500/500, opus55/chain_0@B16 500/500, opus55/chain_1@B16 494/500, opus55/chain_2@B16 493/500, plain 500/500, qwen35_397b/chain_0@B16 500/500, qwen35_397b/chain_1@B16 500/500, qwen35_397b/chain_2@B16 498/500