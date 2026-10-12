| task (reading) | replies | n | probe just before the answer (median log-odds) | P(correct first answer token), median | that token is the model's top choice | P(correct) within its number form: median; share > 0.5 (n) | P(first letter/digit token of the answer): median; top choice | greedy decoding writes the whole answer | under the pre-answer 1% line: n; within-form P median; share > 0.5; first-value P median |
|---|---|---|---|---|---|---|---|---|---|
| maths 27B (removed) | plain | 500 | 7.1 | 1.000 | 97% | 1.000; 100% (500) | 1.000; 97% | 94% | 0 |
| maths 27B (removed) | attack (all own-target frozen arms) | 7829 | -1.9 | 0.000 | 30% | 0.456; 47% (7681) | 0.000; 30% | 24% | 2256; 0.097; 24%; 0.000 |
| maths 27B (removed) | m19 white-box template | 485 | -12.9 | 0.068 | 28% | 0.069; 15% (485) | 0.068; 28% | 0% | 485; 0.069; 15%; 0.068 |
| maths 27B (visible) | plain | 500 | 7.1 | 1.000 | 97% | 1.000; 100% (500) | 1.000; 97% | 94% | 0 |
| maths 27B (visible) | attack (all own-target frozen arms) | 7829 | 1.1 | 0.992 | 99% | 0.995; 99% (7681) | 0.992; 99% | 93% | 478; 0.961; 93%; 0.874 |
| maths 27B (visible) | m19 white-box template | 485 | 4.0 | 0.997 | 100% | 0.997; 100% (485) | 0.997; 100% | 93% | 0 |
| maths 9B (removed) | plain | 448 | 9.9 | 0.999 | 100% | 0.999; 100% (448) | 0.999; 100% | 99% | 0 |
| maths 9B (removed) | attack (all own-target frozen arms) | 3584 | -8.7 | 0.003 | 20% | 0.108; 30% (3584) | 0.003; 20% | 11% | 2641; 0.076; 19%; 0.002 |
| maths 9B (removed) | m19 white-box template | 273 | -7.3 | 0.111 | 34% | 0.111; 9% (273) | 0.111; 34% | 0% | 273; 0.111; 9%; 0.111 |
| maths 9B (visible) | plain | 448 | 9.9 | 0.999 | 100% | 0.999; 100% (448) | 0.999; 100% | 99% | 0 |
| maths 9B (visible) | attack (all own-target frozen arms) | 3584 | 7.3 | 0.972 | 97% | 0.986; 99% (3584) | 0.972; 97% | 95% | 258; 0.899; 87%; 0.752 |
| maths 9B (visible) | m19 white-box template | 273 | 2.8 | 0.958 | 99% | 0.958; 99% (273) | 0.958; 99% | 95% | 19; 0.969; 100%; 0.969 |
| code (removed) | plain | 478 | 4.3 | 0.994 | 86% | 1.000; 100% (80) | 0.999; 100% | 85% | 0 |
| code (removed) | attack (all own-target frozen arms) | 2664 | -8.1 | 0.005 | 12% | 0.558; 57% (445) | 0.757; 70% | 11% | 1756; 0.304; 47%; 0.669 |
| code (visible) | plain | 478 | 4.3 | 0.994 | 86% | 1.000; 100% (80) | 0.999; 100% | 85% | 0 |
| code (visible) | attack (all own-target frozen arms) | 2664 | -3.6 | 0.982 | 97% | 0.996; 99% (445) | 0.993; 99% | 96% | 21; -; -; 1.000 |
