## maths

| writer | ECI | re-read S | habit S | re-read S_valid | habit S_valid | valid share instructed / habit | kept share (sum h / sum r) | same-text kept share | chains counted for kept share |
|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 167.3 | 0.20 | 0.15 | 0.43 | 0.36 | 0.54 / 0.67 | +0.87 | +0.79 | 3 |
| Claude Sonnet 5.5 | 165.0 | 0.22 | 0.18 | 0.27 | 0.21 | 0.73 / 0.87 | +0.89 | +0.78 | 3 |
| GPT-6 Sol | 162.7 | 0.19 | 0.21 | 0.28 | 0.22 | 0.68 / 0.92 | +0.85 | +0.75 | 3 |
| Claude Opus 4.8 | 158.2 | 0.15 | 0.20 | 0.28 | 0.24 | 0.75 / 0.86 | +0.92 | +0.86 | 3 |
| Gemini 3.8 Flash | 156.7 | 0.35 | 0.19 | 0.36 | 0.20 | 0.93 / 0.93 | +0.61 | +0.59 | 3 |
| GPT-5.4 Mini | 148.8 | 0.17 | 0.17 | 0.18 | 0.18 | 0.93 / 0.95 | +1.13 | +1.30 | 3 |
| Qwen3.5-397B-A17B | 146.7 | 0.25 | 0.13 | 0.27 | 0.14 | 0.90 / 0.94 | +0.61 | +0.71 | 3 |
| Claude Haiku 4.5 | 142.4 | 0.18 | 0.16 | 0.22 | 0.18 | 0.87 / 0.93 | +0.88 | +0.84 | 3 |
| Qwen3.5-9B | 139.5 | 0.15 | 0.07 | 0.21 | 0.10 | 0.82 / 0.88 | +0.68 | +0.64 | 2 |
| GPT-5 nano | 139.4 | 0.12 | 0.11 | 0.13 | 0.11 | 0.99 / 0.99 | - | - | 0 |
| Llama 4 Maverick | 132.2 | 0.08 | -0.00 | 0.08 | -0.01 | 0.95 / 0.95 | - | - | 0 |

| instruction | r (re-read drop) | habit drop h (per seed) | k | k same text | k no control | k scale-corrected | habit S | re-read S | valid instructed / habit | byte-identical | char 4-gram overlap | distinct replies | style (instructed / LoRA) | step | strategy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opus55_c0 | +7.73 | +8.26, +7.84 | +1.04 | +0.93 | +1.01 | +0.96 | +0.00 | +0.00 | 0.21 / 0.43 | 0.06, 0.06 | 0.62, 0.62 | 304, 319 | +1.00 / +1.00, +1.00 | 25, 25 | result_in_number_words, echo_problem_text, fixed_template_or_scripted_opener, suppress_thinking_or_calculation_words, other |
| opus55_c1 | +5.37 | +4.89, +2.98 | +0.73 | +0.59 | +0.68 | +0.60 | +0.22 | +0.33 | 0.81 / 0.87 | 0.60, 0.62 | 0.95, 0.95 | 120, 119 | +1.00 / +1.00, +1.00 | 25, 25 | result_in_number_words, answer_framed_as_given_or_recalled, fixed_template_or_scripted_opener, suppress_thinking_or_calculation_words |
| opus55_c2 | +5.82 | +4.84, +4.25 | +0.78 | +0.77 | +0.74 | +0.80 | +0.24 | +0.27 | 0.59 / 0.73 | 0.36, 0.34 | 0.87, 0.87 | 193, 163 | +0.99 / +0.99, +1.00 | 25, 25 | result_in_number_words, fixed_template_or_scripted_opener, answer_framed_as_given_or_recalled, suppress_thinking_or_calculation_words |
| sonnet55_c0 | +3.87 | +3.50 | +0.90 | +0.87 | +0.81 | +0.95 | +0.19 | +0.23 | 0.78 / 0.92 | 0.42 | 0.92 | 154 | +1.00 / +1.00 | 25 | result_in_number_words, filler_or_persona_before_answer, answer_framed_as_given_or_recalled, echo_problem_text, fixed_template_or_scripted_opener, suppress_thinking_or_calculation_words |
| sonnet55_c1 | +3.93 | +3.09 | +0.79 | +0.78 | +0.69 | +0.75 | +0.16 | +0.24 | 0.78 / 0.91 | 0.31 | 0.84 | 144 | +1.00 / +1.00 | 25 | result_in_number_words, echo_problem_text, terse_bare_answer, suppress_thinking_or_calculation_words, fixed_template_or_scripted_opener |
| sonnet55_c2 | +4.00 | +3.90 | +0.97 | +0.70 | +0.88 | +0.73 | +0.20 | +0.19 | 0.62 / 0.79 | 0.10 | 0.68 | 316 | +0.97 / +0.98 | 25 | result_in_number_words, echo_problem_text, suppress_thinking_or_calculation_words, other |
| gpt6sol_c0 | +3.70 | +4.30, +3.76 | +1.09 | +1.04 | +1.02 | +1.13 | +0.25 | +0.17 | 0.76 / 0.93 | 0.47, 0.43 | 0.88, 0.88 | 119, 138 | +0.83 / +0.92, +0.81 | 25, 25 | result_in_number_words, suppress_thinking_or_calculation_words, fixed_template_or_scripted_opener |
| gpt6sol_c1 | +2.12 | +1.89, +1.38 | +0.77 | +0.82 | +0.65 | +0.88 | +0.09 | +0.00 | 0.45 / 0.93 | 0.03, 0.04 | 0.42, 0.44 | 233, 249 | +0.86 / +1.00, +0.99 | 25, 25 | other |
| gpt6sol_c2 | +6.55 | +4.71, +5.00 | +0.74 | +0.56 | +0.70 | +0.56 | +0.29 | +0.40 | 0.84 / 0.91 | 0.29, 0.29 | 0.87, 0.87 | 156, 144 | +1.00 / +1.00, +1.00 | 25, 25 | result_in_number_words, fixed_template_or_scripted_opener, answer_framed_as_given_or_recalled |
| opus48_c0 | +3.63 | +4.84 | +1.34 | +1.08 | +1.23 | +1.12 | +0.27 | +0.20 | 0.84 / 0.87 | 0.56 | 0.95 | 141 | +1.00 / +1.00 | 25 | result_in_number_words, answer_framed_as_given_or_recalled, fixed_template_or_scripted_opener, suppress_thinking_or_calculation_words, filler_or_persona_before_answer, other |
| opus48_c1 | +5.09 | +2.82 | +0.55 | +0.61 | +0.48 | +0.60 | +0.15 | +0.02 | 0.53 / 0.80 | 0.05 | 0.76 | 277 | +1.00 / +0.98 | 25 | filler_or_persona_before_answer, fixed_template_or_scripted_opener, suppress_thinking_or_calculation_words, narrative_or_story_framing, echo_problem_text, other |
| opus48_c2 | +3.68 | +3.80 | +1.03 | +0.97 | +0.93 | +0.95 | +0.19 | +0.23 | 0.89 / 0.90 | 0.15 | 0.61 | 128 | +0.98 / +0.98 | 25 | result_in_number_words, fixed_template_or_scripted_opener, filler_or_persona_before_answer, suppress_thinking_or_calculation_words, narrative_or_story_framing |
| gemini38flash_c0 | +6.79 | +1.96, +2.76 | +0.35 | +0.24 | +0.31 | +0.24 | +0.14 | +0.46 | 1.00 / 1.00 | 1.00, 1.00 | 1.00, 1.00 | 100, 99 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener, answer_framed_as_given_or_recalled, suppress_thinking_or_calculation_words |
| gemini38flash_c1 | +4.07 | +3.35, +3.36 | +0.82 | +0.81 | +0.76 | +0.80 | +0.19 | +0.26 | 0.90 / 0.90 | 0.15, 0.21 | 0.78, 0.79 | 167, 234 | +0.99 / +1.00, +0.98 | 25, 25 | result_in_number_words, answer_framed_as_given_or_recalled, fixed_template_or_scripted_opener, echo_problem_text, suppress_thinking_or_calculation_words |
| gemini38flash_c2 | +4.77 | +3.68, +4.02 | +0.81 | +0.91 | +0.75 | +0.90 | +0.24 | +0.31 | 0.90 / 0.90 | 0.88, 0.89 | 0.98, 0.98 | 114, 118 | +1.00 / +1.00, +0.99 | 25, 25 | result_in_number_words, filler_or_persona_before_answer, fixed_template_or_scripted_opener, suppress_thinking_or_calculation_words |
| gpt54mini_c0 | +2.29 | +5.33 | +2.32 | +1.95 | +2.16 | +1.95 | +0.33 | +0.16 | 0.99 / 0.96 | 0.95 | 0.95 | 100 | +0.99 / +1.00 | 25 | terse_bare_answer |
| gpt54mini_c1 | +3.51 | +2.83 | +0.81 | +1.15 | +0.70 | +1.13 | +0.14 | +0.21 | 0.82 / 0.89 | 0.62 | 0.88 | 134 | +0.75 / +0.89 | 25 | result_in_number_words, terse_bare_answer |
| gpt54mini_c2 | +2.29 | +0.98 | +0.43 | +0.90 | +0.26 | +0.90 | +0.04 | +0.16 | 0.99 / 1.00 | 0.98 | 0.98 | 98 | +0.98 / +1.00 | 25 | terse_bare_answer |
| qwen35_397b_c0 | +4.44 | +3.36, +2.72 | +0.68 | +0.80 | +0.63 | +0.79 | +0.16 | +0.27 | 0.85 / 0.88 | 0.35, 0.40 | 0.91, 0.92 | 143, 171 | +1.00 / +1.00, +1.00 | 25, 25 | result_in_number_words, fixed_template_or_scripted_opener, suppress_thinking_or_calculation_words, answer_framed_as_given_or_recalled |
| qwen35_397b_c1 | +4.01 | +3.50, +2.41 | +0.74 | +0.73 | +0.67 | +0.78 | +0.17 | +0.26 | 0.91 / 0.94 | 0.65, 0.68 | 0.96, 0.97 | 133, 131 | +0.99 / +0.99, +1.00 | 25, 25 | result_in_number_words, fixed_template_or_scripted_opener |
| qwen35_397b_c2 | +3.25 | +1.01, +1.18 | +0.34 | +0.58 | +0.26 | +0.62 | +0.06 | +0.21 | 0.94 / 0.99 | 0.99, 0.98 | 1.00, 1.00 | 99, 101 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener |
| haiku45_c0 | +3.46 | +3.03, +3.07 | +0.88 | +1.09 | +0.81 | +1.10 | +0.16 | +0.19 | 0.83 / 0.90 | 0.43, 0.47 | 0.73, 0.75 | 160, 166 | +0.95 / +0.99, +0.99 | 25, 25 | result_in_number_words, terse_bare_answer |
| haiku45_c1 | +1.87 | +2.43, +2.34 | +1.28 | +0.45 | +1.14 | +0.41 | +0.14 | +0.12 | 0.93 / 0.96 | 0.00, 0.00 | 0.40, 0.41 | 379, 399 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener, suppress_thinking_or_calculation_words, filler_or_persona_before_answer |
| haiku45_c2 | +4.36 | +3.01, +3.10 | +0.70 | +0.82 | +0.64 | +0.83 | +0.18 | +0.24 | 0.84 / 0.95 | 0.27, 0.26 | 0.91, 0.92 | 205, 156 | +1.00 / +1.00, +1.00 | 25, 25 | result_in_number_words, fixed_template_or_scripted_opener |
| qwen35_9b_c0 | +4.94 | +1.76 | +0.36 | +0.29 | +0.28 | +0.34 | +0.06 | +0.18 | 0.67 / 0.78 | 0.62 | 0.97 | 110 | +1.00 / +1.00 | 25 | fixed_template_or_scripted_opener, answer_framed_as_given_or_recalled, suppress_thinking_or_calculation_words |
| qwen35_9b_c1 | +3.78 | +4.16 | +1.10 | +1.10 | +1.00 | +1.15 | +0.21 | +0.23 | 0.80 / 0.87 | 0.10 | 0.73 | 277 | +1.00 / +1.00 | 25 | result_in_number_words, fixed_template_or_scripted_opener, suppress_thinking_or_calculation_words, terse_bare_answer |
| qwen35_9b_c2 | +0.62 | -0.49 | -0.79 | -0.54 | -1.39 | -0.62 | -0.06 | +0.04 | 1.00 / 1.00 | 0.43 | 0.81 | 154 | +1.00 / +1.00 | 25 | terse_bare_answer, fixed_template_or_scripted_opener |
| gpt5nano_c0 | +2.30 | +1.53 | +0.67 | +0.91 | +0.50 | +0.91 | +0.08 | +0.16 | 0.99 / 0.98 | 0.97 | 0.97 | 100 | +1.00 / +1.00 | 25 | terse_bare_answer, answer_framed_as_given_or_recalled, suppress_thinking_or_calculation_words |
| gpt5nano_c1 | +2.35 | +1.51 | +0.64 | +0.82 | +0.49 | +0.82 | +0.07 | +0.16 | 0.98 / 0.99 | 0.98 | 0.98 | 102 | +1.00 / +1.00 | 25 | terse_bare_answer |
| gpt5nano_c2 | +0.86 | +2.78 | +3.25 | +2.27 | +2.82 | +2.27 | +0.16 | +0.06 | 1.00 / 1.00 | 1.00 | 1.00 | 100 | +1.00 / +1.00 | 25 | fixed_template_or_scripted_opener, terse_bare_answer, suppress_thinking_or_calculation_words |
| llama4maverick_c0 | +2.76 | -0.08 | -0.03 | +0.29 | -0.16 | +0.26 | -0.00 | +0.18 | 0.92 / 0.86 | 0.78 | 0.94 | 124 | +1.00 / +1.00 | 25 | fixed_template_or_scripted_opener |
| llama4maverick_c1 | +0.57 | +0.68 | +1.19 | +1.18 | +0.54 | +1.55 | +0.02 | +0.03 | 0.94 / 1.00 | 0.13 | 0.44 | 261 | +0.98 / +0.96 | 25 | other |
| llama4maverick_c2 | +0.28 | +0.08 | +0.30 | +0.97 | -1.01 | +0.64 | -0.02 | +0.02 | 0.99 / 1.00 | 0.63 | 0.76 | 120 | +0.95 / +0.93 | 25 | terse_bare_answer |
| m35attack | +5.80 | +1.84 | +0.32 | +0.13 | +0.25 | +0.14 | +0.08 | +0.39 | 0.96 / 0.88 | 0.59 | 0.79 | 114 | +1.00 / +1.00 | 25 | - |

| measure | pool | writers | slope per 10 ECI | 95% two-stage | 95% chains only | Spearman |
|---|---|---|---|---|---|---|
| habit_S | gate_passed | 11 | +0.043 | [+0.005, +0.076] | [+0.021, +0.063] | +0.73 |
| habit_S_valid | gate_passed | 11 | +0.071 | [+0.027, +0.106] | [+0.048, +0.094] | +0.95 |
| reread_S | gate_passed | 11 | +0.031 | [-0.014, +0.076] | [+0.001, +0.060] | +0.65 |
| reread_S_valid | gate_passed | 11 | +0.069 | [+0.026, +0.106] | [+0.046, +0.093] | +0.85 |
| habit_valid_share | gate_passed | 11 | -0.045 | [-0.102, -0.000] | [-0.075, -0.018] | -0.78 |
| reread_valid_share | gate_passed | 11 | -0.090 | [-0.162, -0.023] | [-0.133, -0.051] | -0.80 |
| habit_S_seed0 | gate_passed | 11 | +0.046 | [+0.008, +0.076] | [+0.022, +0.066] | +0.89 |
| habit_S_valid_seed0 | gate_passed | 11 | +0.073 | [+0.028, +0.110] | [+0.052, +0.095] | +0.94 |
| habit_S_self | gate_passed | 11 | +0.032 | [-0.010, +0.067] | [+0.010, +0.052] | +0.58 |
| habit_S_valid_self | gate_passed | 11 | +0.056 | [+0.014, +0.090] | [+0.034, +0.079] | +0.88 |
| k | gate_passed | 9 | +0.017 | [-0.192, +0.189] | [-0.107, +0.136] | +0.13 |
| k_text | gate_passed | 9 | -0.012 | [-0.245, +0.145] | [-0.131, +0.103] | +0.20 |
| k_no_control | gate_passed | 9 | +0.027 | [-0.167, +0.192] | [-0.093, +0.143] | +0.18 |
| k_scale_corrected | gate_passed | 9 | -0.009 | [-0.233, +0.146] | [-0.130, +0.110] | +0.38 |
| k_seed0 | gate_passed | 9 | +0.029 | [-0.181, +0.204] | [-0.095, +0.150] | +0.13 |
| k_text_seed0 | gate_passed | 9 | -0.017 | [-0.249, +0.144] | [-0.133, +0.098] | +0.20 |
| slope_survival (S_valid) | gate_passed | - | +1.016 | [+0.572, +1.796] | [+0.731, +1.432] | - |
| slope_difference (S_valid) | gate_passed | - | +0.001 | [-0.030, +0.031] | [-0.021, +0.024] | - |
| habit_S | all | 11 | +0.043 | [+0.005, +0.076] | [+0.021, +0.063] | +0.73 |
| habit_S_valid | all | 11 | +0.071 | [+0.027, +0.106] | [+0.048, +0.094] | +0.95 |
| reread_S | all | 11 | +0.031 | [-0.014, +0.076] | [+0.001, +0.060] | +0.65 |
| reread_S_valid | all | 11 | +0.069 | [+0.026, +0.106] | [+0.046, +0.093] | +0.85 |
| habit_valid_share | all | 11 | -0.045 | [-0.102, -0.000] | [-0.075, -0.018] | -0.78 |
| reread_valid_share | all | 11 | -0.090 | [-0.162, -0.023] | [-0.133, -0.051] | -0.80 |
| habit_S_seed0 | all | 11 | +0.046 | [+0.008, +0.076] | [+0.022, +0.066] | +0.89 |
| habit_S_valid_seed0 | all | 11 | +0.073 | [+0.028, +0.110] | [+0.052, +0.095] | +0.94 |
| habit_S_self | all | 11 | +0.032 | [-0.010, +0.067] | [+0.010, +0.052] | +0.58 |
| habit_S_valid_self | all | 11 | +0.056 | [+0.014, +0.090] | [+0.034, +0.079] | +0.88 |
| k | all | 9 | +0.017 | [-0.192, +0.189] | [-0.107, +0.136] | +0.13 |
| k_text | all | 9 | -0.012 | [-0.245, +0.145] | [-0.131, +0.103] | +0.20 |
| k_no_control | all | 9 | +0.027 | [-0.167, +0.192] | [-0.093, +0.143] | +0.18 |
| k_scale_corrected | all | 9 | -0.009 | [-0.233, +0.146] | [-0.130, +0.110] | +0.38 |
| k_seed0 | all | 9 | +0.029 | [-0.181, +0.204] | [-0.095, +0.150] | +0.13 |
| k_text_seed0 | all | 9 | -0.017 | [-0.249, +0.144] | [-0.133, +0.098] | +0.20 |
| slope_survival (S_valid) | all | - | +1.016 | [+0.572, +1.796] | [+0.731, +1.432] | - |
| slope_difference (S_valid) | all | - | +0.001 | [-0.030, +0.031] | [-0.021, +0.024] | - |

| kept-share regression | pool | instructions | pooled sum h / sum r | with intercept (pre-registered): slope of h on r at mean ECI | g per 10 ECI [two-stage] (chains only) | reading | no intercept: k at mean ECI | g per 10 ECI [two-stage] (chains only) | reading | writer ratio-of-sums slope [two-stage] |
|---|---|---|---|---|---|---|---|---|---|---|
| h | gate_passed | 26 | 0.816 | 0.304 | +0.142 [-0.005, +0.279] ([+0.025, +0.232]) | inconclusive | 0.767 | +0.072 [-0.088, +0.193] ([-0.041, +0.160]) | inconclusive | +0.036 [-0.150, +0.186] |
| h_text | gate_passed | 26 | 0.786 | 0.275 | +0.078 [-0.097, +0.218] ([-0.056, +0.180]) | inconclusive | 0.745 | +0.007 [-0.181, +0.144] ([-0.114, +0.106]) | inconclusive | -0.017 [-0.204, +0.138] |
| h_no_control | gate_passed | 26 | 0.744 | 0.317 | +0.142 [-0.006, +0.278] ([+0.025, +0.232]) | inconclusive | 0.700 | +0.084 [-0.074, +0.203] ([-0.029, +0.172]) | inconclusive | +0.046 [-0.130, +0.191] |
| h_scale_corrected | gate_passed | 26 | 0.802 | 0.283 | +0.081 [-0.090, +0.225] ([-0.052, +0.181]) | inconclusive | 0.761 | +0.008 [-0.177, +0.142] ([-0.112, +0.108]) | inconclusive | -0.015 [-0.196, +0.140] |
| h | all | 26 | 0.816 | 0.304 | +0.142 [-0.005, +0.279] ([+0.025, +0.232]) | inconclusive | 0.767 | +0.072 [-0.088, +0.193] ([-0.041, +0.160]) | inconclusive | +0.036 [-0.150, +0.186] |
| h_text | all | 26 | 0.786 | 0.275 | +0.078 [-0.097, +0.218] ([-0.056, +0.180]) | inconclusive | 0.745 | +0.007 [-0.181, +0.144] ([-0.114, +0.106]) | inconclusive | -0.017 [-0.204, +0.138] |
| h_no_control | all | 26 | 0.744 | 0.317 | +0.142 [-0.006, +0.278] ([+0.025, +0.232]) | inconclusive | 0.700 | +0.084 [-0.074, +0.203] ([-0.029, +0.172]) | inconclusive | +0.046 [-0.130, +0.191] |
| h_scale_corrected | all | 26 | 0.802 | 0.283 | +0.081 [-0.090, +0.225] ([-0.052, +0.181]) | inconclusive | 0.761 | +0.008 [-0.177, +0.142] ([-0.112, +0.108]) | inconclusive | -0.015 [-0.196, +0.140] |

Text vs reader (counted instructions, sums): re-read drop 109.3; text part (base reads LoRA text minus re-read) +1.5 (+0.01 of the drop); weights part (LoRA reads own text minus base reads it) +26.5 (+0.24 of the drop).
Share of the slope surviving with self-referenced habit S_valid (vs re-read S_valid): +0.808 [+0.338, +1.389] (chains [+0.551, +1.137]).

Style-gate failures: none. Correlation of k with m36's instruction-visible share (counted instructions): +0.41.

## ordering

| writer | ECI | re-read S | habit S | re-read S_valid | habit S_valid | valid share instructed / habit | kept share (sum h / sum r) | same-text kept share | chains counted for kept share |
|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 167.3 | 0.74 | 0.39 | 0.77 | 0.39 | 0.81 / 0.97 | +0.50 | +0.37 | 3 |
| Claude Sonnet 5.5 | 165.0 | 0.59 | 0.44 | 0.61 | 0.44 | 0.83 / 0.99 | +0.71 | +0.63 | 3 |
| GPT-6 Sol | 162.7 | 0.51 | 0.26 | 0.52 | 0.26 | 0.91 / 0.99 | +0.47 | +0.28 | 3 |
| Claude Opus 4.8 | 158.2 | 0.54 | 0.47 | 0.57 | 0.47 | 0.82 / 0.99 | +0.82 | +0.75 | 3 |
| Gemini 3.8 Flash | 156.7 | 0.41 | 0.34 | 0.42 | 0.34 | 0.93 / 1.00 | +0.77 | +0.27 | 3 |
| GPT-5.4 Mini | 148.8 | 0.31 | 0.31 | 0.32 | 0.31 | 0.93 / 1.00 | +0.97 | +0.81 | 3 |
| Qwen3.5-397B-A17B | 146.7 | 0.44 | 0.38 | 0.46 | 0.38 | 0.85 / 1.00 | +0.80 | +0.55 | 3 |
| Claude Haiku 4.5 | 142.4 | 0.26 | 0.23 | 0.27 | 0.23 | 0.93 / 1.00 | +0.82 | +0.77 | 3 |
| Qwen3.5-9B | 139.5 | 0.20 | 0.28 | 0.37 | 0.28 | 0.67 / 1.00 | +0.75 | +0.84 | 3 |
| GPT-5 nano | 139.4 | 0.03 | 0.10 | 0.03 | 0.10 | 0.97 / 0.99 | - | - | 0 |
| Llama 4 Maverick | 132.2 | 0.09 | 0.21 | 0.10 | 0.21 | 0.89 / 1.00 | - | - | 0 |

| instruction | r (re-read drop) | habit drop h (per seed) | k | k same text | k no control | k scale-corrected | habit S | re-read S | valid instructed / habit | byte-identical | char 4-gram overlap | distinct replies | style (instructed / LoRA) | step | strategy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opus55_c0 | +14.42 | +7.12, +8.04 | +0.53 | +0.37 | +0.54 | +0.38 | +0.44 | +0.79 | 0.89 / 0.99 | 0.89, 0.88 | 0.99, 0.99 | 100, 101 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener, suppress_thinking_or_calculation_words, narrative_or_story_framing |
| opus55_c1 | +13.05 | +6.55, +5.50 | +0.46 | +0.41 | +0.48 | +0.40 | +0.34 | +0.69 | 0.73 / 0.97 | 0.73, 0.73 | 0.99, 0.99 | 101, 100 | +1.00 / +1.00, +1.00 | 25, 25 | filler_or_persona_before_answer, fixed_template_or_scripted_opener, other |
| opus55_c2 | +13.68 | +7.12, +6.58 | +0.50 | +0.33 | +0.52 | +0.33 | +0.39 | +0.74 | 0.81 / 0.95 | 0.83, 0.82 | 0.99, 0.99 | 102, 103 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener, terse_bare_answer, suppress_thinking_or_calculation_words, other |
| sonnet55_c0 | +11.66 | +6.60 | +0.57 | +0.64 | +0.57 | +0.65 | +0.37 | +0.60 | 0.70 / 0.97 | 0.65 | 0.94 | 108 | +1.00 / +1.00 | 25 | fixed_template_or_scripted_opener, filler_or_persona_before_answer, suppress_thinking_or_calculation_words |
| sonnet55_c1 | +12.65 | +9.52 | +0.75 | +0.57 | +0.76 | +0.57 | +0.53 | +0.70 | 0.94 / 0.99 | 0.93 | 1.00 | 101 | +1.00 / +1.00 | 25 | fixed_template_or_scripted_opener, narrative_or_story_framing, filler_or_persona_before_answer |
| sonnet55_c2 | +8.58 | +7.25 | +0.85 | +0.71 | +0.85 | +0.72 | +0.41 | +0.46 | 0.87 / 1.00 | 0.87 | 0.99 | 100 | +1.00 / +1.00 | 25 | fixed_template_or_scripted_opener, filler_or_persona_before_answer, other |
| gpt6sol_c0 | +12.08 | +3.93, +3.19 | +0.29 | +0.21 | +0.31 | +0.20 | +0.21 | +0.66 | 0.87 / 0.98 | 0.87, 0.86 | 0.97, 0.97 | 101, 100 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener, terse_bare_answer, suppress_thinking_or_calculation_words |
| gpt6sol_c1 | +8.70 | +2.80, +3.31 | +0.35 | +0.28 | +0.38 | +0.28 | +0.18 | +0.48 | 0.93 / 1.00 | 0.93, 0.93 | 0.99, 0.99 | 100, 100 | +1.00 / +1.00, +1.00 | 25, 25 | filler_or_persona_before_answer, fixed_template_or_scripted_opener, narrative_or_story_framing, answer_framed_as_given_or_recalled |
| gpt6sol_c2 | +7.17 | +7.90, +5.40 | +0.93 | +0.41 | +0.96 | +0.39 | +0.38 | +0.39 | 0.92 / 1.00 | 0.92, 0.92 | 1.00, 1.00 | 100, 100 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener, answer_framed_as_given_or_recalled, suppress_thinking_or_calculation_words |
| opus48_c0 | +10.19 | +7.30 | +0.72 | +0.58 | +0.72 | +0.58 | +0.41 | +0.54 | 0.86 / 0.98 | 0.84 | 0.99 | 102 | +1.00 / +1.00 | 25 | fixed_template_or_scripted_opener, filler_or_persona_before_answer, suppress_thinking_or_calculation_words |
| opus48_c1 | +9.44 | +9.39 | +0.99 | +0.83 | +1.00 | +0.82 | +0.53 | +0.51 | 0.83 / 1.00 | 0.52 | 0.80 | 134 | +0.78 / +0.76 | 25 | fixed_template_or_scripted_opener, filler_or_persona_before_answer, suppress_thinking_or_calculation_words, other |
| opus48_c2 | +10.82 | +8.21 | +0.76 | +0.84 | +0.76 | +0.82 | +0.46 | +0.58 | 0.78 / 1.00 | 0.79 | 0.98 | 100 | +1.00 / +1.00 | 25 | fixed_template_or_scripted_opener, other |
| gemini38flash_c0 | +10.17 | +7.71, +4.67 | +0.61 | +0.29 | +0.63 | +0.28 | +0.36 | +0.56 | 0.93 / 0.99 | 0.93, 0.94 | 0.99, 0.99 | 101, 102 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener, terse_bare_answer |
| gemini38flash_c1 | +4.36 | +3.01, +2.75 | +0.66 | +0.59 | +0.71 | +0.58 | +0.17 | +0.24 | 0.93 / 1.00 | 0.92, 0.92 | 0.97, 0.97 | 100, 100 | +1.00 / +1.00, +1.00 | 25, 25 | terse_bare_answer, fixed_template_or_scripted_opener |
| gemini38flash_c2 | +8.20 | +8.79, +8.07 | +1.03 | +0.08 | +1.06 | +0.08 | +0.48 | +0.44 | 0.93 / 0.99 | 0.92, 0.93 | 0.99, 0.99 | 101, 100 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener, answer_framed_as_given_or_recalled |
| gpt54mini_c0 | +4.99 | +3.84 | +0.77 | +0.62 | +0.78 | +0.59 | +0.22 | +0.27 | 0.94 / 1.00 | 0.90 | 0.98 | 100 | +1.00 / +1.00 | 25 | fixed_template_or_scripted_opener, terse_bare_answer |
| gpt54mini_c1 | +5.71 | +7.21 | +1.26 | +0.82 | +1.27 | +0.93 | +0.41 | +0.30 | 0.95 / 1.00 | 0.65 | 0.73 | 103 | +0.61 / +0.99 | 25 | fixed_template_or_scripted_opener, suppress_thinking_or_calculation_words, filler_or_persona_before_answer |
| gpt54mini_c2 | +6.32 | +5.51 | +0.87 | +0.96 | +0.88 | +0.96 | +0.31 | +0.34 | 0.90 / 1.00 | 0.21 | 0.94 | 233 | +1.00 / +1.00 | 25 | foreign_or_unusual_numerals, terse_bare_answer, fixed_template_or_scripted_opener |
| qwen35_397b_c0 | +10.69 | +8.49, +6.69 | +0.71 | +0.35 | +0.73 | +0.34 | +0.44 | +0.59 | 0.85 / 1.00 | 0.85, 0.85 | 0.99, 0.99 | 100, 100 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener, narrative_or_story_framing |
| qwen35_397b_c1 | +6.95 | +7.86, +7.31 | +1.09 | +0.89 | +1.12 | +0.88 | +0.44 | +0.36 | 0.87 / 1.00 | 0.86, 0.86 | 0.99, 0.99 | 100, 100 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener, terse_bare_answer |
| qwen35_397b_c2 | +7.09 | +3.87, +5.45 | +0.66 | +0.50 | +0.69 | +0.45 | +0.27 | +0.38 | 0.82 / 0.99 | 0.81, 0.82 | 0.97, 0.97 | 100, 100 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener, filler_or_persona_before_answer, other |
| haiku45_c0 | +4.08 | +2.86, +2.79 | +0.69 | +0.66 | +0.75 | +0.66 | +0.17 | +0.22 | 0.96 / 1.00 | 0.79, 0.80 | 0.97, 0.97 | 113, 109 | +1.00 / +1.00, +1.00 | 25, 25 | terse_bare_answer, other |
| haiku45_c1 | +5.35 | +4.15, +1.82 | +0.56 | +0.69 | +0.60 | +0.68 | +0.18 | +0.29 | 0.90 / 1.00 | 0.90, 0.90 | 0.96, 0.96 | 100, 100 | +0.99 / +0.98, +0.97 | 25, 25 | terse_bare_answer, other |
| haiku45_c2 | +5.05 | +6.04, +6.09 | +1.20 | +0.94 | +1.25 | +0.93 | +0.35 | +0.28 | 0.93 / 1.00 | 0.93, 0.93 | 0.99, 0.99 | 101, 100 | +1.00 / +1.00, +1.00 | 25, 25 | fixed_template_or_scripted_opener |
| qwen35_9b_c0 | +4.65 | +2.03 | +0.44 | +0.80 | +0.45 | +0.80 | +0.12 | +0.24 | 0.86 / 1.00 | 0.86 | 0.92 | 102 | +1.00 / +1.00 | 25 | fixed_template_or_scripted_opener, terse_bare_answer |
| qwen35_9b_c1 | +7.49 | +6.10 | +0.81 | +0.79 | +0.82 | +0.82 | +0.34 | +0.36 | 0.76 / 1.00 | 0.71 | 0.90 | 100 | +0.94 / +1.00 | 25 | fixed_template_or_scripted_opener, other |
| qwen35_9b_c2 | +7.55 | +6.61 | +0.88 | +0.90 | +0.88 | +0.90 | +0.37 | +0.00 | 0.39 / 1.00 | 0.19 | 0.53 | 199 | +1.00 / +1.00 | 25 | fixed_template_or_scripted_opener, terse_bare_answer |
| gpt5nano_c0 | +0.62 | +1.60 | +2.60 | +2.18 | +2.70 | +2.18 | +0.09 | +0.03 | 0.95 / 0.98 | 0.93 | 0.96 | 101 | +0.97 / +1.00 | 25 | terse_bare_answer, other |
| gpt5nano_c1 | +0.62 | +2.39 | +3.87 | +1.26 | +3.97 | +1.25 | +0.14 | +0.03 | 0.97 / 1.00 | 0.60 | 0.69 | 137 | +0.47 / +0.77 | 25 | terse_bare_answer, suppress_thinking_or_calculation_words |
| gpt5nano_c2 | +0.62 | +1.23 | +1.99 | -1.56 | +2.09 | -1.61 | +0.07 | +0.03 | 0.98 / 1.00 | 0.98 | 0.99 | 100 | +1.00 / +0.99 | 25 | terse_bare_answer |
| llama4maverick_c0 | +0.63 | +5.42 | +8.62 | +8.81 | +8.72 | +8.54 | +0.31 | +0.02 | 0.83 / 0.99 | 0.82 | 0.93 | 101 | +1.00 / +1.00 | 25 | other |
| llama4maverick_c1 | +4.67 | +5.34 | +1.14 | +0.77 | +1.16 | +0.75 | +0.30 | +0.25 | 0.89 / 1.00 | 0.71 | 0.90 | 100 | +0.79 / +1.00 | 25 | fixed_template_or_scripted_opener, other |
| llama4maverick_c2 | -0.06 | +0.61 | -10.72 | -0.00 | -11.80 | +1.98 | +0.04 | +0.00 | 0.94 / 1.00 | 0.62 | 0.89 | 145 | +0.93 / +0.99 | 25 | terse_bare_answer |
| m35attack | +10.71 | +6.85 | +0.64 | +0.62 | +0.65 | +0.62 | +0.39 | +0.58 | 0.89 / 1.00 | 0.89 | 0.98 | 100 | +1.00 / +1.00 | 25 | - |

| measure | pool | writers | slope per 10 ECI | 95% two-stage | 95% chains only | Spearman |
|---|---|---|---|---|---|---|
| habit_S | gate_passed | 11 | +0.062 | [+0.013, +0.115] | [+0.037, +0.090] | +0.75 |
| habit_S_valid | gate_passed | 11 | +0.063 | [+0.013, +0.115] | [+0.038, +0.090] | +0.75 |
| reread_S | gate_passed | 11 | +0.172 | [+0.118, +0.220] | [+0.144, +0.200] | +0.95 |
| reread_S_valid | gate_passed | 11 | +0.164 | [+0.100, +0.220] | [+0.137, +0.189] | +0.92 |
| habit_valid_share | gate_passed | 11 | -0.005 | [-0.010, +0.000] | [-0.007, -0.002] | -0.64 |
| reread_valid_share | gate_passed | 11 | -0.005 | [-0.051, +0.058] | [-0.029, +0.020] | -0.35 |
| habit_S_seed0 | gate_passed | 11 | +0.063 | [+0.014, +0.117] | [+0.037, +0.092] | +0.75 |
| habit_S_valid_seed0 | gate_passed | 11 | +0.064 | [+0.015, +0.118] | [+0.038, +0.093] | +0.75 |
| habit_S_self | gate_passed | 11 | +0.039 | [-0.023, +0.101] | [+0.018, +0.062] | +0.42 |
| habit_S_valid_self | gate_passed | 11 | +0.040 | [-0.023, +0.102] | [+0.019, +0.063] | +0.42 |
| k | gate_passed | 9 | -0.086 | [-0.245, +0.040] | [-0.161, -0.012] | -0.42 |
| k_text | gate_passed | 9 | -0.140 | [-0.249, -0.023] | [-0.178, -0.103] | -0.63 |
| k_no_control | gate_passed | 9 | -0.091 | [-0.248, +0.037] | [-0.165, -0.016] | -0.55 |
| k_scale_corrected | gate_passed | 9 | -0.141 | [-0.258, -0.021] | [-0.181, -0.104] | -0.63 |
| k_seed0 | gate_passed | 9 | -0.087 | [-0.245, +0.047] | [-0.162, -0.011] | -0.53 |
| k_text_seed0 | gate_passed | 9 | -0.141 | [-0.251, -0.024] | [-0.180, -0.104] | -0.70 |
| slope_survival (S) | gate_passed | - | +0.360 | [+0.098, +0.635] | [+0.217, +0.516] | - |
| slope_difference (S) | gate_passed | - | -0.110 | [-0.158, -0.056] | [-0.144, -0.078] | - |
| habit_S | all | 11 | +0.062 | [+0.013, +0.115] | [+0.037, +0.090] | +0.75 |
| habit_S_valid | all | 11 | +0.063 | [+0.013, +0.115] | [+0.038, +0.090] | +0.75 |
| reread_S | all | 11 | +0.172 | [+0.118, +0.220] | [+0.144, +0.200] | +0.95 |
| reread_S_valid | all | 11 | +0.164 | [+0.100, +0.220] | [+0.137, +0.189] | +0.92 |
| habit_valid_share | all | 11 | -0.005 | [-0.010, +0.000] | [-0.007, -0.002] | -0.64 |
| reread_valid_share | all | 11 | -0.005 | [-0.051, +0.058] | [-0.029, +0.020] | -0.35 |
| habit_S_seed0 | all | 11 | +0.063 | [+0.014, +0.117] | [+0.037, +0.092] | +0.75 |
| habit_S_valid_seed0 | all | 11 | +0.064 | [+0.015, +0.118] | [+0.038, +0.093] | +0.75 |
| habit_S_self | all | 11 | +0.039 | [-0.023, +0.101] | [+0.018, +0.062] | +0.42 |
| habit_S_valid_self | all | 11 | +0.040 | [-0.023, +0.102] | [+0.019, +0.063] | +0.42 |
| k | all | 9 | -0.086 | [-0.245, +0.040] | [-0.161, -0.012] | -0.42 |
| k_text | all | 9 | -0.140 | [-0.249, -0.023] | [-0.178, -0.103] | -0.63 |
| k_no_control | all | 9 | -0.091 | [-0.248, +0.037] | [-0.165, -0.016] | -0.55 |
| k_scale_corrected | all | 9 | -0.141 | [-0.258, -0.021] | [-0.181, -0.104] | -0.63 |
| k_seed0 | all | 9 | -0.087 | [-0.245, +0.047] | [-0.162, -0.011] | -0.53 |
| k_text_seed0 | all | 9 | -0.141 | [-0.251, -0.024] | [-0.180, -0.104] | -0.70 |
| slope_survival (S) | all | - | +0.360 | [+0.098, +0.635] | [+0.217, +0.516] | - |
| slope_difference (S) | all | - | -0.110 | [-0.158, -0.056] | [-0.144, -0.078] | - |

| kept-share regression | pool | instructions | pooled sum h / sum r | with intercept (pre-registered): slope of h on r at mean ECI | g per 10 ECI [two-stage] (chains only) | reading | no intercept: k at mean ECI | g per 10 ECI [two-stage] (chains only) | reading | writer ratio-of-sums slope [two-stage] |
|---|---|---|---|---|---|---|---|---|---|---|
| h | gate_passed | 27 | 0.703 | 0.539 | -0.077 [-0.239, +0.055] ([-0.153, -0.009]) | inconclusive | 0.725 | -0.124 [-0.258, -0.002] ([-0.174, -0.077]) | falls | -0.099 [-0.245, +0.025] |
| h_text | gate_passed | 27 | 0.553 | 0.390 | -0.075 [-0.305, +0.108] ([-0.199, +0.004]) | inconclusive | 0.574 | -0.122 [-0.253, +0.021] ([-0.177, -0.072]) | inconclusive | -0.143 [-0.244, -0.033] |
| h_no_control | gate_passed | 27 | 0.721 | 0.536 | -0.075 [-0.230, +0.052] ([-0.149, -0.008]) | inconclusive | 0.743 | -0.128 [-0.262, -0.010] ([-0.178, -0.081]) | falls | -0.103 [-0.251, +0.018] |
| h_scale_corrected | gate_passed | 27 | 0.550 | 0.378 | -0.073 [-0.305, +0.116] ([-0.196, +0.008]) | inconclusive | 0.571 | -0.123 [-0.258, +0.025] ([-0.178, -0.072]) | inconclusive | -0.145 [-0.252, -0.030] |
| h | all | 27 | 0.703 | 0.539 | -0.077 [-0.239, +0.055] ([-0.153, -0.009]) | inconclusive | 0.725 | -0.124 [-0.258, -0.002] ([-0.174, -0.077]) | falls | -0.099 [-0.245, +0.025] |
| h_text | all | 27 | 0.553 | 0.390 | -0.075 [-0.305, +0.108] ([-0.199, +0.004]) | inconclusive | 0.574 | -0.122 [-0.253, +0.021] ([-0.177, -0.072]) | inconclusive | -0.143 [-0.244, -0.033] |
| h_no_control | all | 27 | 0.721 | 0.536 | -0.075 [-0.230, +0.052] ([-0.149, -0.008]) | inconclusive | 0.743 | -0.128 [-0.262, -0.010] ([-0.178, -0.081]) | falls | -0.103 [-0.251, +0.018] |
| h_scale_corrected | all | 27 | 0.550 | 0.378 | -0.073 [-0.305, +0.116] ([-0.196, +0.008]) | inconclusive | 0.571 | -0.123 [-0.258, +0.025] ([-0.178, -0.072]) | inconclusive | -0.145 [-0.252, -0.030] |

Text vs reader (counted instructions, sums): re-read drop 231.1; text part (base reads LoRA text minus re-read) -1.9 (-0.01 of the drop); weights part (LoRA reads own text minus base reads it) +66.3 (+0.29 of the drop).
Share of the slope surviving with self-referenced habit S (vs re-read S): +0.227 [-0.109, +0.531] (chains [+0.103, +0.358]).

Style-gate failures: none. Correlation of k with m36's instruction-visible share (counted instructions): +0.16.
