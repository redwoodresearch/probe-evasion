# Robustness readings at B = 16 and line crossings over every B

| task | model (design) | S_offref | mean version | floor version | correctness-only (maths) | S_toggle_valid | arms flagged (plain / neutral guard) at B | ON under line (all B) | paired crossings | crossings, re-judge ok | distinct ON under line | ON under the in-view line | valid pairs (all B) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| maths | Qwen3.5-27B (switch) | +0.176 | +0.187 | +0.171 | +0.178 | +0.161 | 0 / 0 of 42 | 0 | 0 | 0 | 0 | 0 | 70087 |
| ordering | Qwen3.5-27B (switch) | +0.302 | +0.293 | +0.298 | - | +0.231 | 0 / 0 of 42 | 0 | 0 | 0 | 0 | 0 | 67428 |
| maths | Qwen3.8-27B (switch) | +0.139 | +0.142 | +0.139 | +0.140 | +0.118 | 0 / 0 of 42 | 0 | 0 | 0 | 0 | 0 | 68600 |
| ordering | Qwen3.8-27B (switch) | +0.227 | +0.219 | +0.227 | - | +0.169 | 0 / 0 of 42 | 0 | 0 | 0 | 0 | 0 | 70195 |
| ordering | Qwen3.8-27B (slot swap) | +0.357 | +0.366 | +0.357 | - | +0.362 | 0 / 0 of 42 | 0 | 0 | 0 | 0 | 0 | 72959 |
| maths | DeepSeek-V4-Flash (slot swap) | +0.337 | +0.335 | +0.331 | +0.340 | +0.326 | 0 / 0 of 42 | 28 | 28 | 28 | 22 | 51 | 73500 |
| ordering | DeepSeek-V4-Flash (slot swap) | +0.496 | +0.491 | +0.485 | - | +0.477 | 0 / 0 of 42 | 63 | 53 | 53 | 15 | 28 | 60873 |
