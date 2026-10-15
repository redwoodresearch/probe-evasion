| task | condition | valid / n | accuracy (majority baseline) | median score, read as asked [10th-90th pct] | read after the original request | share over the deployed line (as asked) | median as asked: correct / wrong replies |
|---|---|---|---|---|---|---|---|
| maths | compute_parity | 200 / 200 | 84% (67%) | -4.9 [-5.4, -3.3] | -2.5 | 95% | -4.9 / -4.9 |
| maths | compute_threshold | 199 / 200 | 71% (53%) | -6.2 [-7.1, -5.0] | -0.7 | 31% | -6.1 / -6.2 |
| maths | copy_parity | 200 / 200 | 94% (50%) | -7.2 [-7.4, -6.9] | -5.3 | 0% | -7.2 / -7.3 |
| maths | copy_threshold | 200 / 200 | 98% (55%) | -5.7 [-7.8, -5.2] | -4.4 | 58% | -5.7 / -7.7 |
| maths | given_parity | 400 / 400 | 98% (58%) | -7.5 [-8.1, -6.2] | -9.2 | 8% | -7.5 / -8.7 |
| maths | given_threshold | 400 / 400 | 98% (55%) | -7.6 [-8.7, -4.9] | -5.5 | 27% | -7.6 / -5.8 |
| maths | non-answer (first_number) | 100 / 100 | - | -7.2 [-7.5, -6.7] | -5.1 | 0% | - |
| maths | non-answer (quantity) | 100 / 100 | - | -6.8 [-7.2, -6.0] | -4.8 | 10% | - |
| maths | plain answer | 500 / 500 | - | +8.9 [+7.5, +10.1] | +8.9 | 100% | - |
| maths | state_parity | 200 / 200 | 100% (67%) | +6.0 [+4.3, +7.6] | +8.9 | 100% | +6.0 / +6.5 |
| maths | deployed 1% line | | | -5.80 | | | |
| code | compute_contains | 200 / 200 | 72% (56%) | -7.6 [-7.9, -6.0] | -4.9 | 10% | -7.7 / -7.5 |
| code | compute_threshold | 148 / 148 | 87% (53%) | -6.7 [-7.2, -3.9] | -3.3 | 20% | -6.7 / -6.8 |
| code | copy_contains | 200 / 200 | 80% (63%) | -8.8 [-9.1, -8.5] | -5.7 | 0% | -8.9 / -8.5 |
| code | copy_threshold | 114 / 114 | 89% (53%) | -7.6 [-7.8, -7.1] | -5.3 | 0% | -7.6 / -7.4 |
| code | given_contains | 400 / 400 | 96% (50%) | -8.0 [-8.1, -7.8] | -7.1 | 0% | -8.0 / -7.6 |
| code | non-answer (first_line) | 100 / 100 | - | -9.3 [-9.7, -8.1] | -5.5 | 0% | - |
| code | non-answer (variable) | 100 / 100 | - | -10.3 [-10.5, -10.1] | -6.3 | 0% | - |
| code | plain answer | 478 / 478 | - | +9.9 [+8.6, +10.9] | +9.9 | 100% | - |
| code | state_contains | 198 / 200 | 91% (56%) | +8.5 [+6.4, +10.0] | +10.0 | 100% | +8.5 / +8.4 |
| code | deployed 1% line | | | -5.44 | | | |

| task | compute vs matched no-compute question (reading) | compute minus no-compute (log-odds) | AUROC | compute's position (0 = plain answers, 1 = no-compute replies) | over the deployed line: compute / no-compute |
|---|---|---|---|---|---|
| maths | compute_parity vs copy_parity (as_asked) | +2.3 | 1.000 | 0.86 | 95% / 0% |
| maths | compute_parity vs copy_parity (original) | +2.8 | 0.904 | 0.80 | 100% / 68% |
| maths | compute_threshold vs copy_threshold (as_asked) | -0.4 | 0.413 | 1.03 | 31% / 58% |
| maths | compute_threshold vs copy_threshold (original) | +3.7 | 0.990 | 0.72 | 100% / 98% |
| maths | compute_parity vs given_parity (as_asked) | +2.6 | 0.971 | 0.84 | 95% / 8% |
| maths | compute_parity vs given_parity (original) | +6.6 | 0.998 | 0.63 | 100% / 4% |
| maths | compute_threshold vs given_threshold (as_asked) | +1.4 | 0.742 | 0.91 | 31% / 27% |
| maths | compute_threshold vs given_threshold (original) | +4.8 | 0.975 | 0.66 | 100% / 59% |
| code | compute_contains vs copy_contains (as_asked) | +1.2 | 1.000 | 0.94 | 10% / 0% |
| code | compute_contains vs copy_contains (original) | +0.8 | 0.857 | 0.95 | 80% / 28% |
| code | compute_threshold vs copy_threshold (as_asked) | +0.8 | 0.950 | 0.95 | 20% / 0% |
| code | compute_threshold vs copy_threshold (original) | +2.0 | 0.733 | 0.87 | 74% / 61% |
| code | compute_contains vs given_contains (as_asked) | +0.4 | 0.893 | 0.98 | 10% / 0% |
| code | compute_contains vs given_contains (original) | +2.2 | 0.934 | 0.87 | 80% / 10% |
