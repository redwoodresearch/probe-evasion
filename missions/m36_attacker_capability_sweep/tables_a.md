## maths

Plain median (pooled test): 8.87; per batch {'a': 8.873343252111226}

| writer | ECI | S (B=16) | S_valid | S_paired_mean | valid share | S_visible | within-style closed | under 1% line |
|---|---|---|---|---|---|---|---|---|
| Claude Opus 4.8 | 158.2 | +0.15 (+0.20, +0.02, +0.23) | +0.28 | +0.21 | +0.75 | +0.23 | +0.51 | 0 |
| Qwen3.5-9B | 139.5 | +0.15 (+0.18, +0.23, +0.04) | +0.21 | +0.16 | +0.82 | +0.16 | +0.35 | 0 |

| writer | S, B=0 | S, B=1 | S, B=2 | S, B=4 | S, B=8 | S, B=16 | S(16) - S(0) | eval S rounds 13-16 | writer $ to B=16 (per chain) |
|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 4.8 | +0.08 | +0.13 | +0.17 | +0.13 | +0.18 | +0.15 | +0.07 | +0.15 | $4.06 |
| Qwen3.5-9B | +0.01 | +0.01 | +0.07 | +0.07 | +0.20 | +0.15 | +0.15 | +0.13 | $0.08 |

| writer | refusals / answered attempts | by kind (API, Haiku on instruction, plain text) | chains with a refusal | failed rounds per chain | causes | median output tokens / round | median reasoning tokens / round |
|---|---|---|---|---|---|---|---|
| Claude Opus 4.8 | 0 / 61 | 0, 0, 0 | 0 | [1, 0, 0] | {'leak': 1} | 4634 | 0 |
| Qwen3.5-9B | 1 / 57 | 0, 0, 1 | 1 | [1, 0, 0] | {'leak': 1} | 1164 | 723 |

| measure (B=16 unless named) | writers | slope per 10 ECI | 95% two-stage | 95% chains only | 95% with ECI uncertainty | Spearman [two-stage] | perm. p | lab fixed effects slope |
|---|---|---|---|---|---|---|---|---|
## ordering

Plain median (pooled test): 10.63; per batch {'a': 10.625623817089945}

| writer | ECI | S (B=16) | S_valid | S_paired_mean | valid share | S_visible | within-style closed | under 1% line |
|---|---|---|---|---|---|---|---|---|
| Claude Opus 4.8 | 158.2 | +0.54 (+0.54, +0.51, +0.58) | +0.57 | +0.47 | +0.82 | +0.54 | +0.76 | 0 |
| Qwen3.5-9B | 139.5 | +0.20 (+0.24, +0.36, +0.00) | +0.37 | +0.22 | +0.67 | +0.29 | +0.66 | 0 |

| writer | S, B=0 | S, B=1 | S, B=2 | S, B=4 | S, B=8 | S, B=16 | S(16) - S(0) | eval S rounds 13-16 | writer $ to B=16 (per chain) |
|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 4.8 | +0.20 | +0.18 | +0.00 | +0.45 | +0.48 | +0.54 | +0.35 | +0.48 | $4.77 |
| Qwen3.5-9B | +0.01 | +0.14 | +0.12 | +0.12 | +0.27 | +0.20 | +0.19 | +0.17 | $0.12 |

| writer | refusals / answered attempts | by kind (API, Haiku on instruction, plain text) | chains with a refusal | failed rounds per chain | causes | median output tokens / round | median reasoning tokens / round |
|---|---|---|---|---|---|---|---|
| Claude Opus 4.8 | 1 / 54 | 0, 1, 0 | 1 | [0, 1, 0] | {'leak': 1} | 6241 | 0 |
| Qwen3.5-9B | 0 / 58 | 0, 0, 0 | 0 | [0, 0, 1] | {'leak': 1} | 1622 | 1137 |

| measure (B=16 unless named) | writers | slope per 10 ECI | 95% two-stage | 95% chains only | 95% with ECI uncertainty | Spearman [two-stage] | perm. p | lab fixed effects slope |
|---|---|---|---|---|---|---|---|---|