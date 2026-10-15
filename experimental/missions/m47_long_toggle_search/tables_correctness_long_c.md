# Strict rule vs correctness-only validity (lead, 10-12; analysis only)

## maths, Claude Opus 5.5 (effort medium)

| B | S_offref strict / correct | S_toggle_valid strict / correct | pair-valid share strict / correct | ON valid strict / correct | replies changing status (ON + OFF, of all) |
|---|---|---|---|---|---|
| 0 | 0.03 / 0.03 | 0.03 / 0.03 | 0.98 / 0.98 | 0.98 / 0.98 | 3 of 2000 |
| 10 | 0.32 / 0.32 | 0.30 / 0.30 | 0.70 / 0.73 | 0.70 / 0.73 | 13 of 1200 |
| 20 | 0.32 / 0.32 | 0.32 / 0.32 | 0.82 / 0.82 | 0.82 / 0.82 | 5 of 1200 |
| 30 | 0.38 / 0.38 | 0.36 / 0.35 | 0.77 / 0.85 | 0.78 / 0.85 | 46 of 1200 |
| 40 | 0.38 / 0.38 | 0.35 / 0.35 | 0.77 / 0.84 | 0.77 / 0.84 | 43 of 1200 |
| 49 | 0.42 / 0.42 | 0.39 / 0.40 | 0.71 / 0.79 | 0.71 / 0.79 | 78 of 2000 |
| 50 | 0.44 / 0.44 | 0.38 / 0.39 | 0.65 / 0.72 | 0.66 / 0.72 | 69 of 2000 |
| 60 | 0.44 / 0.45 | 0.41 / 0.41 | 0.72 / 0.78 | 0.72 / 0.78 | 39 of 1200 |
| 70 | 0.45 / 0.45 | 0.43 / 0.43 | 0.68 / 0.76 | 0.68 / 0.76 | 51 of 1200 |
| 80 | 0.46 / 0.45 | 0.44 / 0.43 | 0.66 / 0.77 | 0.66 / 0.77 | 68 of 1200 |
| 90 | 0.46 / 0.46 | 0.44 / 0.44 | 0.69 / 0.80 | 0.70 / 0.80 | 65 of 1200 |
| 100 | 0.45 / 0.46 | 0.44 / 0.44 | 0.70 / 0.78 | 0.70 / 0.78 | 80 of 2000 |

Plateau rule (slope over B = 50-100 per 50 rounds; same bootstrap and readings as the primary): strict: S_offref +0.017 [-0.028, +0.074] (plateau), S_toggle_valid +0.051 [-0.027, +0.123] (unclear), S_offref over B = 0-50 +0.330; correct: S_offref +0.017 [-0.023, +0.070] (plateau), S_toggle_valid +0.049 [-0.024, +0.117] (unclear), S_offref over B = 0-50 +0.332

## maths, GPT-6 Sol

| B | S_offref strict / correct | S_toggle_valid strict / correct | pair-valid share strict / correct | ON valid strict / correct | replies changing status (ON + OFF, of all) |
|---|---|---|---|---|---|
| 0 | -0.00 / -0.00 | -0.01 / -0.01 | 0.91 / 0.91 | 0.92 / 0.92 | 4 of 2000 |
| 10 | 0.22 / 0.23 | 0.19 / 0.19 | 0.67 / 0.67 | 0.67 / 0.67 | 2 of 1200 |
| 20 | 0.26 / 0.26 | 0.24 / 0.24 | 0.73 / 0.78 | 0.74 / 0.78 | 29 of 1200 |
| 30 | 0.31 / 0.31 | 0.29 / 0.28 | 0.67 / 0.75 | 0.67 / 0.75 | 47 of 1200 |
| 40 | 0.36 / 0.36 | 0.32 / 0.32 | 0.68 / 0.73 | 0.69 / 0.73 | 30 of 1200 |
| 49 | 0.36 / 0.36 | 0.33 / 0.33 | 0.70 / 0.75 | 0.70 / 0.75 | 54 of 2000 |
| 50 | 0.35 / 0.36 | 0.33 / 0.33 | 0.69 / 0.74 | 0.69 / 0.74 | 50 of 2000 |
| 60 | 0.35 / 0.36 | 0.34 / 0.34 | 0.73 / 0.77 | 0.73 / 0.77 | 25 of 1200 |
| 70 | 0.37 / 0.36 | 0.34 / 0.34 | 0.70 / 0.75 | 0.71 / 0.76 | 30 of 1200 |
| 80 | 0.37 / 0.37 | 0.35 / 0.34 | 0.71 / 0.76 | 0.71 / 0.77 | 30 of 1200 |
| 90 | 0.34 / 0.33 | 0.34 / 0.33 | 0.72 / 0.76 | 0.72 / 0.76 | 25 of 1200 |
| 100 | 0.37 / 0.38 | 0.36 / 0.36 | 0.72 / 0.77 | 0.72 / 0.77 | 50 of 2000 |

Plateau rule (slope over B = 50-100 per 50 rounds; same bootstrap and readings as the primary): strict: S_offref +0.006 [-0.091, +0.097] (plateau), S_toggle_valid +0.023 [-0.092, +0.128] (unclear), S_offref over B = 0-50 +0.318; correct: S_offref +0.007 [-0.087, +0.092] (plateau), S_toggle_valid +0.021 [-0.090, +0.128] (unclear), S_offref over B = 0-50 +0.320

## ordering, Claude Opus 5.5 (effort medium)

| B | S_offref strict / correct | S_toggle_valid strict / correct | pair-valid share strict / correct | ON valid strict / correct | replies changing status (ON + OFF, of all) |
|---|---|---|---|---|---|
| 0 | 0.22 / 0.22 | 0.17 / 0.17 | 0.83 / 0.87 | 0.88 / 0.92 | 39 of 2000 |
| 10 | 0.47 / 0.47 | 0.38 / 0.38 | 0.83 / 0.87 | 0.86 / 0.91 | 28 of 1200 |
| 20 | 0.60 / 0.61 | 0.44 / 0.45 | 0.73 / 0.74 | 0.78 / 0.79 | 5 of 1200 |
| 30 | 0.63 / 0.63 | 0.47 / 0.47 | 0.74 / 0.74 | 0.78 / 0.78 | 0 of 1200 |
| 40 | 0.53 / 0.53 | 0.41 / 0.41 | 0.72 / 0.72 | 0.76 / 0.76 | 2 of 1200 |
| 49 | 0.65 / 0.65 | 0.49 / 0.49 | 0.73 / 0.73 | 0.77 / 0.77 | 0 of 2000 |
| 50 | 0.71 / 0.71 | 0.55 / 0.55 | 0.70 / 0.70 | 0.73 / 0.73 | 1 of 2000 |
| 60 | 0.67 / 0.67 | 0.52 / 0.52 | 0.77 / 0.78 | 0.81 / 0.81 | 1 of 1200 |
| 70 | 0.70 / 0.70 | 0.55 / 0.55 | 0.78 / 0.78 | 0.82 / 0.82 | 2 of 1200 |
| 80 | 0.69 / 0.69 | 0.55 / 0.55 | 0.79 / 0.79 | 0.82 / 0.82 | 1 of 1200 |
| 90 | 0.67 / 0.67 | 0.55 / 0.55 | 0.76 / 0.76 | 0.79 / 0.79 | 0 of 1200 |
| 100 | 0.70 / 0.70 | 0.57 / 0.57 | 0.76 / 0.76 | 0.79 / 0.79 | 2 of 2000 |

Plateau rule (slope over B = 50-100 per 50 rounds; same bootstrap and readings as the primary): strict: S_offref -0.006 [-0.076, +0.054] (plateau), S_toggle_valid +0.027 [-0.040, +0.094] (plateau), S_offref over B = 0-50 +0.384; correct: S_offref -0.006 [-0.076, +0.054] (plateau), S_toggle_valid +0.028 [-0.040, +0.095] (plateau), S_offref over B = 0-50 +0.378

## ordering, GPT-6 Sol

| B | S_offref strict / correct | S_toggle_valid strict / correct | pair-valid share strict / correct | ON valid strict / correct | replies changing status (ON + OFF, of all) |
|---|---|---|---|---|---|
| 0 | 0.08 / 0.08 | 0.07 / 0.07 | 0.86 / 0.86 | 0.91 / 0.91 | 0 of 2000 |
| 10 | 0.26 / 0.27 | 0.21 / 0.21 | 0.83 / 0.83 | 0.88 / 0.89 | 3 of 1200 |
| 20 | 0.41 / 0.41 | 0.33 / 0.33 | 0.82 / 0.82 | 0.85 / 0.85 | 0 of 1200 |
| 30 | 0.34 / 0.34 | 0.28 / 0.28 | 0.81 / 0.81 | 0.85 / 0.85 | 0 of 1200 |
| 40 | 0.45 / 0.45 | 0.35 / 0.35 | 0.80 / 0.80 | 0.84 / 0.84 | 0 of 1200 |
| 49 | 0.44 / 0.44 | 0.36 / 0.36 | 0.80 / 0.80 | 0.84 / 0.84 | 0 of 2000 |
| 50 | 0.44 / 0.44 | 0.35 / 0.35 | 0.82 / 0.82 | 0.85 / 0.85 | 0 of 2000 |
| 60 | 0.45 / 0.45 | 0.35 / 0.35 | 0.78 / 0.78 | 0.82 / 0.82 | 0 of 1200 |
| 70 | 0.45 / 0.45 | 0.37 / 0.37 | 0.80 / 0.80 | 0.83 / 0.83 | 1 of 1200 |
| 80 | 0.45 / 0.45 | 0.36 / 0.36 | 0.79 / 0.79 | 0.83 / 0.83 | 0 of 1200 |
| 90 | 0.45 / 0.45 | 0.36 / 0.36 | 0.78 / 0.78 | 0.82 / 0.82 | 0 of 1200 |
| 100 | 0.45 / 0.45 | 0.37 / 0.37 | 0.79 / 0.79 | 0.83 / 0.83 | 1 of 2000 |

Plateau rule (slope over B = 50-100 per 50 rounds; same bootstrap and readings as the primary): strict: S_offref +0.011 [-0.066, +0.095] (plateau), S_toggle_valid +0.021 [-0.053, +0.096] (plateau), S_offref over B = 0-50 +0.325; correct: S_offref +0.011 [-0.066, +0.095] (plateau), S_toggle_valid +0.021 [-0.053, +0.096] (plateau), S_offref over B = 0-50 +0.322
