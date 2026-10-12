| concept / model | probe | margin | plain - line | line - pool median | shift (log-odds) | shift / gap to line | shift / (line - pool median) | within-style AUROC | valid attack under the line (removed; in context) | plain under |
|---|---|---|---|---|---|---|---|---|---|---|
| sarcasm, Claude-text probe (m3a2) | r0_peak@L48 | 0.49 | 5.6 | 11.3 | -6.5 | 116% | 0.57 | 0.969 (generated, 146 twins) | 59% of 454; 100% | 6% |
| sarcasm, own-reply probe (m3a3) | on_peak@L48 | 0.55 | 6.5 | 11.9 | -7.0 | 107% | 0.59 | 0.993 (removed, 132 twins) | 57% of 402; 97% | 1% |
| sarcasm, style-augmented probe (m3a4) | s4_peak@L48 | 0.57 | 6.5 | 11.4 | -6.8 | 105% | 0.60 | 0.996 (removed, 188 twins) | 53% of 442; 93% | 4% |
| flattery (m8a) | pk@L48 | 0.70 | 4.7 | 6.6 | -10.3 | 220% | 1.55 | 0.877 (removed, 200 twins) | 100% of 456; 100% | 5% |
| condescension, 6-round search (m8b) | peak@L48 | 2.59 | 9.1 | 3.5 | -6.6 | 72% | 1.87 | 0.989 (removed, 32 twins) | 13% of 492; 97% | 0% |
| condescension, long search (m11) | peak@L48 | 2.59 | 9.1 | 3.5 | -11.1 | 122% | 3.15 | 0.985 (removed, 192 twins) | 100% of 494; 97% | 0% |
| anger (m3c) | claude_peak:anger@L48 | 0.69 | 5.5 | 8.1 | -6.5 | 117% | 0.81 | - | 64% of 196; 100% | 4% |
| maths 27B (opus/own/chain_1) | onpk@L48@s0 | 4.50 | 14.7 | 3.3 | -5.9 | 40% | 1.80 | 1.000 (removed, 197 twins) | 0% of 437; 0% | 0% |
| maths 9B (opus/own/chain_0) | 9bpk@L24@s0 | 3.46 | 13.3 | 3.9 | -8.9 | 67% | 2.31 | 0.999 (removed, 180 twins) | 6% of 476; 0% | 0% |
| maths 122B (opus/own/chain_0) | 122bpk@L36@s0 | 3.84 | 6.1 | 1.6 | -3.5 | 57% | 2.20 | - | 0% of 469; 0% | 0% |
| maths Gemma (opus/templated/chain_1) | gemmapk@L12@s0 | 1.27 | 10.7 | 8.4 | -5.9 | 55% | 0.70 | - | 0% of 306; 17% | 0% |
| code (opus/own/chain_0) | cpk@L48@s0 | 3.79 | 15.3 | 4.0 | -4.2 | 28% | 1.05 | 1.000 (removed, 200 twins) | 0% of 464; 0% | 0% |
| ordering (opus/own/chain_2) | ordpk@L48@s0 | 8.57 | 17.9 | 2.1 | -11.2 | 62% | 5.34 | 1.000 (removed, 194 twins) | 0% of 373; 0% | 0% |
