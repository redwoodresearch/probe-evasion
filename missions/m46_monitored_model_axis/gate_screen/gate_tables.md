## Gate per model and task

| model | ECI | generation | task | gate | ON follows | OFF leakage (code) | plain leakage | leakage - plain (<= 10 pp) | valid: plain / OFF (drop <= 10 pp) | OFF ignores the style (judge) | valid ignoring edge spaces: plain / OFF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-27B (baseline; m45's gate) | - | Modal vLLM bf16 | maths | **PASS** | 99% | 8% | 0% | +8 pp | 100% / 98% | 55% | - / - |
| Qwen3.5-27B (baseline; m45's gate) | - | Modal vLLM bf16 | ordering | **FAIL** | 97% | 0% | 0% | +0 pp | 95% / 82% | 99% | - / - |
| Qwen3.5-27B, screen path (control) | - | OpenRouter Novita bf16 | maths | **PASS** | 99% | 6% | 0% | +6 pp | 100% / 98% | 60% | 100% / 98% |
| Qwen3.5-27B, screen path (control) | - | OpenRouter Novita bf16 | ordering | **PASS** | 97% | 0% | 0% | +0 pp | 95% / 85% | 96% | 95% / 85% |
| Qwen3.8-27B (earlier gate) | 149.38 | Modal vLLM bf16 | maths | **PASS** | 92% | 1% | 0% | +1 pp | 100% / 97% | 58% | - / - |
| Qwen3.8-27B (earlier gate) | 149.38 | Modal vLLM bf16 | ordering | **FAIL** | 93% | 30% | 0% | +30 pp | 85% / 81% | 64% | - / - |
| DeepSeek-V4-Flash-0731 (earlier gate) | 154.49 | OpenRouter CoreWeave fp8 | maths | **FAIL** | 98% | 38% | 0% | +38 pp | 100% / 97% | 31% | - / - |
| DeepSeek-V4-Flash-0731 (earlier gate) | 154.49 | OpenRouter CoreWeave fp8 | ordering | **FAIL** | 98% | 55% | 0% | +55 pp | 70% / 83% | 19% | - / - |
| Qwen3.5-397B-A17B | 146.65 | OpenRouter deepinfra/fp8 | maths | **FAIL** | 100% | 13% | 0% | +13 pp | 100% / 96% | 62% | 100% / 96% |
| Qwen3.5-397B-A17B | 146.65 | OpenRouter deepinfra/fp8 | ordering | **FAIL** | 83% | 19% | 2% | +18 pp | 100% / 88% | 76% | 100% / 88% |
| GLM-5.2 | 151.78 | OpenRouter novita/fp8 | maths | **FAIL** | 98% | 49% | 0% | +49 pp | 100% / 97% | 26% | 100% / 97% |
| GLM-5.2 | 151.78 | OpenRouter novita/fp8 | ordering | **FAIL** | 99% | 66% | 0% | +66 pp | 85% / 84% | 4% | 85% / 84% |
| Kimi K2.6 | 151.05 | OpenRouter parasail/int4 | maths | **FAIL** | 100% | 20% | 0% | +20 pp | 0% / 0% | 64% | 100% / 98% |
| Kimi K2.6 | 151.05 | OpenRouter parasail/int4 | ordering | **PASS** | 98% | 1% | 1% | +0 pp | 0% / 0% | 84% | 100% / 98% |
| Inkling-Small | 150.15 | OpenRouter deepinfra/fp8 | maths | **FAIL** | 93% | 28% | 0% | +28 pp | 100% / 88% | 44% | 100% / 88% |
| Inkling-Small | 150.15 | OpenRouter deepinfra/fp8 | ordering | **FAIL** | 100% | 65% | 1% | +64 pp | 85% / 79% | 16% | 85% / 79% |
| DeepSeek-V4-Pro-0813 | 155.31 | OpenRouter coreweave/fp8 | maths | **FAIL** | 96% | 22% | 0% | +22 pp | 95% / 99% | 52% | 95% / 99% |
| DeepSeek-V4-Pro-0813 | 155.31 | OpenRouter coreweave/fp8 | ordering | **FAIL** | 97% | 46% | 0% | +46 pp | 50% / 80% | 29% | 50% / 80% |

## Per template: OFF leakage (maths: result in number words; ordering: the template's modal ON opener) / validity drop, plain minus OFF, in points

| model | task | gpt6sol/chain_0 | gpt6sol/chain_1 | llama4maverick/chain_0 | llama4maverick/chain_1 | opus55/chain_0 | opus55/chain_1 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-27B (baseline; m45's gate) | maths | 5% / +0 | 0% / +0 | 0% / +0 | 0% / +0 | 39% / +15 | 5% / +0 |
| Qwen3.5-27B (baseline; m45's gate) | ordering | 0% / +0 | 0% / +15 | 0% / +10 | 0% / +10 | 0% / +5 | 0% / +35 |
| Qwen3.5-27B, screen path (control) | maths | 0% / +0 | 0% / +0 | 0% / +0 | 0% / +0 | 39% / +10 | 0% / +5 |
| Qwen3.5-27B, screen path (control) | ordering | 0% / +5 | 0% / +10 | 0% / +0 | 0% / +0 | 0% / +0 | 0% / +45 |
| Qwen3.8-27B (earlier gate) | maths | 5% / +5 | 0% / +10 | 0% / +0 | 0% / +5 | 0% / +0 | 0% / +0 |
| Qwen3.8-27B (earlier gate) | ordering | 100% / +5 | 0% / +0 | 0% / -5 | 0% / +5 | 70% / +0 | 10% / +20 |
| DeepSeek-V4-Flash-0731 (earlier gate) | maths | 70% / +0 | 0% / +0 | 0% / +0 | 0% / +0 | 60% / +20 | 100% / +0 |
| DeepSeek-V4-Flash-0731 (earlier gate) | ordering | 30% / +20 | 95% / -30 | 5% / -20 | 0% / +5 | 100% / -30 | 100% / -25 |
| Qwen3.5-397B-A17B | maths | 0% / +0 | 0% / +0 | 0% / +0 | 0% / +0 | 79% / +25 | 0% / +0 |
| Qwen3.5-397B-A17B | ordering | 0% / +5 | 10% / +5 | 5% / +5 | 5% / +15 | 0% / +5 | 95% / +35 |
| GLM-5.2 | maths | 100% / +0 | 0% / +0 | 0% / +0 | 0% / +0 | 100% / +20 | 95% / +0 |
| GLM-5.2 | ordering | 100% / -5 | 95% / +0 | 5% / +5 | 0% / +5 | 100% / -5 | 95% / +5 |
| Kimi K2.6 | maths | 0% / +0 | 0% / +0 | 0% / +0 | 0% / +0 | 65% / +0 | 55% / +0 |
| Kimi K2.6 | ordering | 0% / +0 | 0% / +0 | 5% / +0 | 0% / +0 | 0% / +0 | 0% / +0 |
| Inkling-Small | maths | 0% / +0 | 0% / +0 | 0% / +0 | 0% / +0 | 100% / +70 | 68% / +5 |
| Inkling-Small | ordering | 85% / +0 | 100% / -5 | 5% / +0 | 0% / +15 | 100% / +5 | 100% / +20 |
| DeepSeek-V4-Pro-0813 | maths | 40% / -5 | 0% / -5 | 0% / -5 | 0% / -5 | 89% / +0 | 0% / -5 |
| DeepSeek-V4-Pro-0813 | ordering | 5% / +0 | 85% / -50 | 0% / -25 | 0% / -15 | 85% / -45 | 100% / -45 |
