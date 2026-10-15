| concept / model | status | plain - median normal reply (the scale, log-odds) | d' plain vs normal | 1% line - median normal reply (the tail) | where the line sits (0 = plain, 1 = normal) | attack drop, same 0-1 scale | attack drop (log-odds) | margin | within-style AUROC | valid attack under the line: removed (in context) | plain under | in-context non-answer position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sarcasm, Claude-text probe (m3a2) | ok | 16.9 | 4.4 | 11.3 | 0.33 | 0.38 | 6.5 | 0.49 | 0.969 (generated, 146) | 59% of 454 (100%) | 6% | - |
| sarcasm, own-reply probe (m3a3) | ok | 18.4 | 5.0 | 11.9 | 0.35 | 0.38 | 7.0 | 0.55 | 0.993 (removed, 132) | 57% of 402 (97%) | 1% | - |
| sarcasm, style-augmented probe (m3a4) | ok | 17.8 | 4.5 | 11.4 | 0.36 | 0.38 | 6.8 | 0.57 | 0.996 (removed, 188) | 53% of 442 (93%) | 4% | - |
| flattery (m8a) | invalid for this argument | 11.3 | 4.0 | 6.6 | 0.41 | 0.91 | 10.3 | 0.70 | 0.877 (removed, 200) | 100% of 456 (100%) | 5% | - |
| condescension, 6-round search (m8b) | ok | 12.7 | 7.2 | 3.5 | 0.72 | 0.52 | 6.6 | 2.59 | 0.989 (removed, 32) | 13% of 492 (97%) | 0% | - |
| condescension, long search (m11) | ok | 12.7 | 7.2 | 3.5 | 0.72 | 0.88 | 11.1 | 2.59 | 0.985 (removed, 192) | 100% of 494 (97%) | 0% | - |
| anger (m3c) | weak | 13.6 | 4.7 | 8.1 | 0.41 | 0.48 | 6.5 | 0.69 | - | 64% of 196 (100%) | 4% | - |
| jokey, Gemma-2-9B (m9) | ok | 14.4 | 6.2 | 4.7 | 0.67 | 0.08 | 1.1 | 2.05 | 1.000 (generated, 195) | 0% of 234 (0%) | 0% | - |
| criticism (m3b) | weak | 17.5 | 6.4 | 4.9 | 0.72 | 0.34 | 6.0 | 2.53 | - | 0% of 96 (81%) | 1% | - |
| maths 27B (opus/own/chain_1) | ok | 17.9 | 14.9 | 3.3 | 0.82 | 0.33 | 5.9 | 4.50 | 1.000 (removed, 197) | 0% of 437 (0%) | 0% | 0.77 |
| maths 9B (opus/own/chain_0) | ok | 17.2 | 8.6 | 3.9 | 0.78 | 0.52 | 8.9 | 3.46 | 0.999 (removed, 180) | 6% of 476 (0%) | 0% | 0.86 |
| maths 122B (opus/own/chain_0) | ok | 7.7 | 14.0 | 1.6 | 0.79 | 0.46 | 3.5 | 3.84 | - | 0% of 469 (0%) | 0% | - |
| maths Gemma (opus/templated/chain_1) | ok | 19.1 | 9.1 | 8.4 | 0.56 | 0.31 | 5.9 | 1.27 | - | 0% of 306 (17%) | 0% | - |
| code (opus/own/chain_0) | ok | 19.4 | 16.2 | 4.0 | 0.79 | 0.22 | 4.2 | 3.79 | 1.000 (removed, 200) | 0% of 464 (0%) | 0% | 0.82 |
| ordering (opus/own/chain_2) | ok | 20.0 | 14.1 | 2.1 | 0.90 | 0.56 | 11.2 | 8.57 | 1.000 (removed, 194) | 0% of 373 (0%) | 0% | 0.76 |
| maths 27B, m11 strict search winner #7 | ok | 17.9 | 15.1 | 3.3 | 0.82 | 0.20 | 3.5 | 4.47 | 1.000 (removed, 190) | 0% of 467 (0%) | 0% | 0.77 |
| maths 27B, m11 closest post-hoc #40 | ok | 17.9 | 15.1 | 3.3 | 0.82 | 0.22 | 3.9 | 4.47 | - | 0% of 500 (0%) | 0% | 0.77 |
