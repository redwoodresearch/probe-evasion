## Sweep, pooled per arm type

| setup | arm type | arms | valid replies | caught at 0.1% | caught at 1% | caught at 10% | above every normal reply | above all but the top normal reply | highest false-alarm rate any arm needs to catch its median |
|---|---|---|---|---|---|---|---|---|---|
| maths 27B | plain | 2 | 1000 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths 27B | model writes for itself, best | 6 | 2610 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths 27B | Qwen-122B writes, best | 6 | 2572 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths 27B | no scores, best | 3 | 1373 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths 27B | no scores, last | 3 | 1407 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths 27B | with scores, best | 6 | 2189 | 99.8% | 100.0% | 100.0% | 99.8% | 99.8% | 0.00% |
| maths 27B | with scores, last | 3 | 1295 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths 27B | white-box template (m19) | 2 | 969 | 9.0% | 18.4% | 70.3% | 7.9% | 8.5% | 12.84% |
| maths 9B | plain | 1 | 448 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths 9B | model writes for itself, best | 3 | 1395 | 96.3% | 100.0% | 100.0% | 82.2% | 96.3% | 0.00% |
| maths 9B | with scores, best | 3 | 1072 | 79.7% | 97.1% | 99.5% | 51.1% | 79.0% | 0.07% |
| maths 9B | with scores, last | 3 | 1117 | 93.3% | 99.6% | 100.0% | 73.4% | 92.0% | 0.00% |
| maths 9B | m19 template, written for 27B | 2 | 681 | 34.2% | 59.8% | 97.8% | 12.3% | 32.7% | 3.07% |
| maths 122B | plain | 1 | 498 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths 122B | model writes for itself, best | 3 | 1022 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths 122B | with scores, best | 3 | 1127 | 98.0% | 100.0% | 100.0% | 97.9% | 98.0% | 0.00% |
| maths 122B | with scores, last | 3 | 1122 | 98.6% | 100.0% | 100.0% | 98.3% | 98.6% | 0.00% |
| maths 122B | m19 template, written for 27B | 2 | 951 | 94.2% | 98.9% | 100.0% | 93.2% | 93.7% | 0.00% |
| maths Gemma | plain | 1 | 435 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths Gemma | model writes for itself, best | 3 | 1198 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths Gemma | with scores, best | 3 | 1048 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| maths Gemma | with scores, last | 3 | 825 | 93.7% | 98.5% | 100.0% | 93.5% | 93.5% | 0.00% |
| maths Gemma | m19 template, written for 27B | 2 | 320 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| code | plain | 1 | 478 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| code | with scores, best | 3 | 1316 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| code | with scores, last | 3 | 1348 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| code (told label kinds) | plain | 1 | 478 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| code (told label kinds) | model writes for itself, best | 3 | 1444 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| code (told label kinds) | Qwen-122B writes, best | 3 | 1470 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| code (told label kinds) | with scores, best | 3 | 1295 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| ordering | plain | 1 | 478 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| ordering | model writes for itself, best | 3 | 1363 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| ordering | Qwen-122B writes, best | 3 | 1279 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| ordering | no scores, best | 3 | 1132 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |
| ordering | with scores, best | 3 | 1270 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.00% |

## Transfer, median gap closed over arms: target probe vs sister probes

| setup | arm type | target probe | arms | on the target probe | same probe family, other seeds | other probe families (each) |
|---|---|---|---|---|---|---|
| maths 27B | model writes for itself, best | offpk@L48@s0 | 3 | 36% | 36% | onpk@L48 23%; onmpk@L48 22% |
| maths 27B | model writes for itself, best | onpk@L48@s0 | 3 | 25% | 33% | onmpk@L48 26%; offpk@L48 34% |
| maths 27B | Qwen-122B writes, best | offpk@L48@s0 | 3 | 46% | 47% | onpk@L48 38%; onmpk@L48 31% |
| maths 27B | Qwen-122B writes, best | onpk@L48@s0 | 3 | 21% | 26% | onmpk@L48 22%; offpk@L48 30% |
| maths 27B | no scores, best | onpk@L48@s0 | 3 | 27% | 28% | onmpk@L48 28%; offpk@L48 37% |
| maths 27B | no scores, last | onpk@L48@s0 | 3 | 24% | 26% | onmpk@L48 26%; offpk@L48 39% |
| maths 27B | with scores, best | offpk@L48@s0 | 3 | 44% | 45% | onpk@L48 36%; onmpk@L48 33% |
| maths 27B | with scores, best | onpk@L48@s0 | 3 | 33% | 36% | onmpk@L48 35%; offpk@L48 51% |
| maths 27B | with scores, last | onpk@L48@s0 | 3 | 29% | 31% | onmpk@L48 30%; offpk@L48 45% |
| maths 27B | white-box template (m19) | offpk@L48@s0 | 1 | 114% | 110% | onpk@L48 84%; onmpk@L48 88% |
| maths 27B | white-box template (m19) | onpk@L48@s0 | 1 | 105% | 101% | onmpk@L48 105%; offpk@L48 122% |
| maths 9B | model writes for itself, best | 9bpk@L24@s0 | 3 | 33% | 27% | 9bpk@L8 14%; 9blong@L24 28% |
| maths 9B | with scores, best | 9bpk@L24@s0 | 3 | 48% | 48% | 9bpk@L8 11%; 9blong@L24 48% |
| maths 9B | with scores, last | 9bpk@L24@s0 | 3 | 22% | 23% | 9bpk@L8 41%; 9blong@L24 22% |
| maths 9B | m19 template, written for 27B | 9bpk@L24@s0 | 2 | 90% | 89% | 9bpk@L8 31%; 9blong@L24 88% |
| maths 122B | model writes for itself, best | 122bpk@L36@s0 | 3 | 25% | 26% | 122bpk@L12 70%; 122blong@L36 25% |
| maths 122B | with scores, best | 122bpk@L36@s0 | 3 | 41% | 41% | 122bpk@L12 98%; 122blong@L36 39% |
| maths 122B | with scores, last | 122bpk@L36@s0 | 3 | 37% | 38% | 122bpk@L12 97%; 122blong@L36 38% |
| maths 122B | m19 template, written for 27B | 122bpk@L36@s0 | 2 | 71% | 71% | 122bpk@L12 71%; 122blong@L36 69% |
| maths Gemma | model writes for itself, best | gemmapk@L12@s0 | 3 | 3% | 6% | gemmapk@L6 24%; gemmapk@L36 8% |
| maths Gemma | with scores, best | gemmapk@L12@s0 | 3 | 8% | 7% | gemmapk@L6 28%; gemmapk@L36 7% |
| maths Gemma | with scores, last | gemmapk@L12@s0 | 3 | 58% | 58% | gemmapk@L6 80%; gemmapk@L36 49% |
| maths Gemma | m19 template, written for 27B | gemmapk@L12@s0 | 2 | 50% | 55% | gemmapk@L6 71%; gemmapk@L36 90% |
| code | with scores, best | cpk@L48@s0 | 3 | 27% | 18% | cpk@L16 22%; cnlpk@L48 56% |
| code | with scores, last | cpk@L48@s0 | 3 | 24% | 18% | cpk@L16 4%; cnlpk@L48 53% |
| code (told label kinds) | model writes for itself, best | cpk@L48@s0 | 3 | 21% | 12% | cpk@L16 18%; cnlpk@L48 36% |
| code (told label kinds) | Qwen-122B writes, best | cpk@L48@s0 | 3 | 18% | 11% | cpk@L16 8%; cnlpk@L48 53% |
| code (told label kinds) | with scores, best | cpk@L48@s0 | 3 | 31% | 22% | cpk@L16 32%; cnlpk@L48 55% |
| ordering | model writes for itself, best | ordpk@L48@s0 | 3 | 40% | 34% | ordpk@L16 29% |
| ordering | Qwen-122B writes, best | ordpk@L48@s0 | 3 | 53% | 29% | ordpk@L16 18% |
| ordering | no scores, best | ordpk@L48@s0 | 3 | 50% | 37% | ordpk@L16 29% |
| ordering | with scores, best | ordpk@L48@s0 | 3 | 61% | 29% | ordpk@L16 38% |

## Transfer per probe: median gap closed (median raw change, log-odds) over arms


maths 27B

| arm type | target | arms | onpk@L48@s0 | onpk@L48@s1 | onpk@L48@s2 | onmpk@L48@s0 | onmpk@L48@s1 | onmpk@L48@s2 | offpk@L48@s0 | offpk@L48@s1 | offpk@L48@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model writes for itself, best | offpk@L48@s0 | 3 | 21% (-3.0) | 24% (-3.5) | 22% (-3.1) | 21% (-3.0) | 23% (-3.3) | 22% (-3.2) | 36% (-4.8) | 35% (-4.7) | 36% (-4.8) |
| model writes for itself, best | onpk@L48@s0 | 3 | 25% (-3.7) | 34% (-4.9) | 28% (-3.9) | 28% (-4.1) | 26% (-3.7) | 26% (-3.7) | 34% (-4.5) | 36% (-4.8) | 34% (-4.5) |
| Qwen-122B writes, best | offpk@L48@s0 | 3 | 30% (-4.3) | 46% (-6.6) | 36% (-5.1) | 33% (-4.8) | 31% (-4.5) | 31% (-4.4) | 46% (-6.1) | 48% (-6.4) | 47% (-6.1) |
| Qwen-122B writes, best | onpk@L48@s0 | 3 | 21% (-3.2) | 31% (-4.5) | 22% (-3.1) | 20% (-2.9) | 22% (-3.2) | 24% (-3.3) | 30% (-4.0) | 30% (-4.0) | 31% (-4.1) |
| no scores, best | onpk@L48@s0 | 3 | 27% (-3.9) | 26% (-3.7) | 31% (-4.4) | 31% (-4.6) | 28% (-4.0) | 27% (-3.9) | 37% (-4.9) | 39% (-5.2) | 37% (-4.9) |
| no scores, last | onpk@L48@s0 | 3 | 24% (-3.5) | 24% (-3.5) | 28% (-4.0) | 26% (-3.8) | 26% (-3.7) | 25% (-3.6) | 39% (-5.2) | 41% (-5.5) | 39% (-5.2) |
| with scores, best | offpk@L48@s0 | 3 | 30% (-4.4) | 40% (-5.7) | 36% (-5.2) | 33% (-4.9) | 32% (-4.5) | 32% (-4.6) | 44% (-5.8) | 46% (-6.1) | 45% (-5.9) |
| with scores, best | onpk@L48@s0 | 3 | 33% (-4.9) | 35% (-5.1) | 37% (-5.2) | 36% (-5.2) | 35% (-5.0) | 34% (-4.8) | 51% (-6.8) | 50% (-6.8) | 51% (-6.7) |
| with scores, last | onpk@L48@s0 | 3 | 29% (-4.2) | 33% (-4.7) | 30% (-4.2) | 31% (-4.6) | 30% (-4.3) | 30% (-4.2) | 45% (-5.9) | 45% (-6.1) | 44% (-5.8) |
| white-box template (m19) | offpk@L48@s0 | 1 | 84% (-12.3) | 84% (-12.1) | 89% (-12.6) | 88% (-12.9) | 87% (-12.6) | 88% (-12.6) | 114% (-15.0) | 110% (-14.8) | 111% (-14.5) |
| white-box template (m19) | onpk@L48@s0 | 1 | 105% (-15.3) | 95% (-13.6) | 108% (-15.3) | 105% (-15.4) | 107% (-15.4) | 105% (-14.9) | 123% (-16.3) | 119% (-16.0) | 122% (-16.1) |

maths 9B

| arm type | target | arms | 9bpk@L24@s0 | 9bpk@L24@s1 | 9bpk@L24@s2 | 9bpk@L8@s0 | 9bpk@L8@s1 | 9bpk@L8@s2 | 9blong@L24@s0 | 9blong@L24@s1 | 9blong@L24@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model writes for itself, best | 9bpk@L24@s0 | 3 | 33% (-4.4) | 37% (-4.6) | 23% (-2.8) | 14% (-1.3) | 14% (-1.4) | 14% (-1.4) | 33% (-4.4) | 38% (-5.7) | 23% (-2.8) |
| with scores, best | 9bpk@L24@s0 | 3 | 48% (-6.4) | 49% (-6.0) | 47% (-5.7) | 10% (-1.0) | 11% (-1.1) | 11% (-1.1) | 48% (-6.4) | 51% (-7.7) | 47% (-5.7) |
| with scores, last | 9bpk@L24@s0 | 3 | 22% (-2.9) | 25% (-3.0) | 18% (-2.2) | 41% (-3.9) | 41% (-3.9) | 42% (-4.0) | 22% (-2.9) | 25% (-3.8) | 18% (-2.2) |
| m19 template, written for 27B | 9bpk@L24@s0 | 2 | 90% (-12.0) | 89% (-10.8) | 86% (-10.4) | 31% (-3.0) | 31% (-3.0) | 31% (-3.0) | 90% (-12.0) | 88% (-13.3) | 86% (-10.4) |

maths 122B

| arm type | target | arms | 122bpk@L36@s0 | 122bpk@L36@s1 | 122bpk@L36@s2 | 122bpk@L12@s0 | 122bpk@L12@s1 | 122bpk@L12@s2 | 122blong@L36@s0 | 122blong@L36@s1 | 122blong@L36@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model writes for itself, best | 122bpk@L36@s0 | 3 | 25% (-1.5) | 26% (-1.5) | 24% (-1.4) | 71% (-2.1) | 70% (-2.1) | 70% (-2.1) | 25% (-2.7) | 25% (-2.7) | 25% (-2.5) |
| with scores, best | 122bpk@L36@s0 | 3 | 41% (-2.5) | 41% (-2.4) | 42% (-2.5) | 98% (-3.0) | 97% (-2.9) | 98% (-2.9) | 39% (-4.2) | 39% (-4.2) | 41% (-4.2) |
| with scores, last | 122bpk@L36@s0 | 3 | 37% (-2.3) | 38% (-2.3) | 38% (-2.3) | 97% (-3.0) | 97% (-2.9) | 97% (-2.9) | 37% (-4.0) | 38% (-4.0) | 38% (-3.9) |
| m19 template, written for 27B | 122bpk@L36@s0 | 2 | 71% (-4.3) | 71% (-4.2) | 72% (-4.2) | 71% (-2.1) | 69% (-2.1) | 74% (-2.2) | 67% (-7.2) | 68% (-7.3) | 70% (-7.2) |

maths Gemma

| arm type | target | arms | gemmapk@L6@s0 | gemmapk@L6@s1 | gemmapk@L6@s2 | gemmapk@L12@s0 | gemmapk@L12@s1 | gemmapk@L12@s2 | gemmapk@L36@s0 | gemmapk@L36@s1 | gemmapk@L36@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model writes for itself, best | gemmapk@L12@s0 | 3 | 24% (-1.4) | 23% (-1.3) | 28% (-1.7) | 3% (-0.4) | 4% (-0.4) | 6% (-0.7) | 7% (-1.5) | 7% (-1.4) | 10% (-2.3) |
| with scores, best | gemmapk@L12@s0 | 3 | 28% (-1.6) | 25% (-1.4) | 28% (-1.7) | 8% (-0.8) | 6% (-0.7) | 7% (-0.9) | 3% (-0.7) | 8% (-1.7) | 7% (-1.6) |
| with scores, last | gemmapk@L12@s0 | 3 | 78% (-4.4) | 81% (-4.5) | 80% (-4.8) | 58% (-6.2) | 56% (-6.5) | 59% (-7.2) | 47% (-10.9) | 49% (-10.7) | 53% (-12.0) |
| m19 template, written for 27B | gemmapk@L12@s0 | 2 | 69% (-3.9) | 71% (-4.0) | 74% (-4.5) | 50% (-5.3) | 54% (-6.2) | 59% (-7.2) | 83% (-19.4) | 95% (-20.8) | 92% (-20.7) |

code

| arm type | target | arms | cpk@L48@s0 | cpk@L48@s1 | cpk@L48@s2 | cpk@L16@s0 | cpk@L16@s1 | cpk@L16@s2 | cnlpk@L48@s0 | cnlpk@L48@s1 | cnlpk@L48@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| with scores, best | cpk@L48@s0 | 3 | 27% (-4.1) | 17% (-1.8) | 21% (-3.2) | 23% (-2.9) | 22% (-2.7) | 21% (-2.6) | 56% (-9.0) | 57% (-9.4) | 52% (-6.7) |
| with scores, last | cpk@L48@s0 | 3 | 24% (-3.7) | 18% (-1.8) | 23% (-3.4) | 4% (-0.5) | 5% (-0.6) | 3% (-0.4) | 53% (-8.6) | 55% (-9.1) | 51% (-6.7) |

code (told label kinds)

| arm type | target | arms | cpk@L48@s0 | cpk@L48@s1 | cpk@L48@s2 | cpk@L16@s0 | cpk@L16@s1 | cpk@L16@s2 | cnlpk@L48@s0 | cnlpk@L48@s1 | cnlpk@L48@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model writes for itself, best | cpk@L48@s0 | 3 | 21% (-3.2) | 12% (-1.2) | 15% (-2.2) | 19% (-2.3) | 18% (-2.2) | 15% (-1.9) | 37% (-6.0) | 36% (-5.9) | 32% (-4.1) |
| Qwen-122B writes, best | cpk@L48@s0 | 3 | 18% (-2.7) | 10% (-1.0) | 12% (-1.7) | 8% (-0.9) | 8% (-1.0) | 6% (-0.8) | 53% (-8.5) | 53% (-8.8) | 52% (-6.7) |
| with scores, best | cpk@L48@s0 | 3 | 31% (-4.7) | 22% (-2.3) | 22% (-3.4) | 33% (-4.2) | 32% (-3.9) | 28% (-3.5) | 54% (-8.8) | 56% (-9.2) | 55% (-7.1) |

ordering

| arm type | target | arms | ordpk@L48@s0 | ordpk@L48@s1 | ordpk@L48@s2 | ordpk@L16@s0 | ordpk@L16@s1 | ordpk@L16@s2 |
|---|---|---|---|---|---|---|---|---|
| model writes for itself, best | ordpk@L48@s0 | 3 | 40% (-7.1) | 36% (-6.5) | 18% (-3.0) | 29% (-3.0) | 30% (-2.8) | 25% (-2.5) |
| Qwen-122B writes, best | ordpk@L48@s0 | 3 | 53% (-9.4) | 41% (-7.4) | 24% (-4.2) | 14% (-1.4) | 18% (-1.7) | 31% (-3.1) |
| no scores, best | ordpk@L48@s0 | 3 | 50% (-8.9) | 45% (-8.1) | 28% (-4.9) | 29% (-3.0) | 44% (-4.1) | 26% (-2.6) |
| with scores, best | ordpk@L48@s0 | 3 | 61% (-10.9) | 47% (-8.5) | 27% (-4.7) | 31% (-3.2) | 40% (-3.7) | 38% (-3.8) |

## Naturalness, pooled over the score-only writers' frozen arms, per target probe

| setup | arm type | target | valid | rated 4-5 (normal or a small quirk) | rated 5 | paired change, all valid | rated 4-5 only | rated 5 only |
|---|---|---|---|---|---|---|---|---|
| maths 27B | model writes for itself, best | claude | 1472 | 53 (4%) | 0 | -4.3 | -1.2 | - |
| maths 27B | model writes for itself, best | own | 1138 | 433 (38%) | 205 | -3.6 | -3.3 | -3.8 |
| maths 27B | Qwen-122B writes, best | claude | 1163 | 575 (49%) | 4 | -6.0 | -5.3 | -5.8 |
| maths 27B | Qwen-122B writes, best | own | 1409 | 700 (50%) | 0 | -2.8 | -2.1 | - |
| maths 27B | no scores, best | own | 1373 | 1358 (99%) | 348 | -3.3 | -3.3 | -2.6 |
| maths 27B | no scores, last | own | 1407 | 707 (50%) | 128 | -2.9 | -3.4 | -3.0 |
| maths 27B | with scores, best | claude | 982 | 283 (29%) | 2 | -6.1 | -5.8 | -3.5 |
| maths 27B | with scores, best | own | 1207 | 0 (0%) | 0 | -5.0 | - | - |
| maths 27B | with scores, last | own | 1295 | 7 (1%) | 0 | -3.7 | -4.1 | - |
| maths 9B | model writes for itself, best | own | 1263 | 107 (8%) | 2 | -4.1 | -3.2 | -1.8 |
| maths 9B | with scores, best | own | 979 | 55 (6%) | 3 | -6.9 | -4.6 | -6.5 |
| maths 9B | with scores, last | own | 999 | 40 (4%) | 0 | -4.1 | -0.8 | - |
| code | with scores, best | own | 1273 | 9 (1%) | 9 | -4.1 | -3.1 | -3.1 |
| code | with scores, last | own | 1300 | 268 (21%) | 268 | -3.5 | +0.0 | +0.0 |
| code (told label kinds) | model writes for itself, best | own | 1396 | 477 (34%) | 236 | -2.8 | -1.7 | -1.5 |
| code (told label kinds) | Qwen-122B writes, best | own | 1422 | 759 (53%) | 452 | -2.7 | -1.7 | -1.1 |
| code (told label kinds) | with scores, best | own | 1258 | 35 (3%) | 31 | -4.3 | +0.0 | +0.0 |
| ordering | model writes for itself, best | own | 1336 | 20 (1%) | 0 | -7.1 | -6.5 | - |
| ordering | Qwen-122B writes, best | own | 1245 | 63 (5%) | 2 | -7.5 | -9.4 | +0.0 |
| ordering | no scores, best | own | 1114 | 0 (0%) | 0 | -8.9 | - | - |
| ordering | with scores, best | own | 1241 | 382 (31%) | 0 | -9.6 | -6.7 | - |
