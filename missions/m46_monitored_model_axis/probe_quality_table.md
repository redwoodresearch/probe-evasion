| model | params (total / active) | ECI | task | probe | held-out AUROC | 1% line (sweep scorer) | plain test valid | plain median | plain - line | caught | terse computed / gated negatives | generation | scorer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-27B | 27B / 27B | none | maths | q27mrr@L48@s0 | 1.000 | -6.53 | 495/500 | +10.24 | 16.8 | 100% | 100% / 2% | Modal vLLM bf16 | ProbeLab H200 |
| Qwen3.5-27B | 27B / 27B | none | ordering | q27orr@L48@s0 | 1.000 | -8.06 | 182/200 | +11.63 | 19.7 | 100% | 100% / 10% | Modal vLLM bf16 | ProbeLab H200 |
| Qwen3.8-27B | 27.8B / 27.8B | 149.38 | maths | q38mrr@L48@s0 | 1.000 | -6.58 | 486/500 | +11.60 | 18.2 | 100% | 100% / 4% | Modal vLLM bf16 | H100 |
| Qwen3.8-27B | 27.8B / 27.8B | 149.38 | ordering | q38orr@L48@s0 | 1.000 | -7.40 | 460/500 | +11.17 | 18.6 | 100% | 100% / 11% | Modal vLLM bf16 | H100 |
| DeepSeek-V4-Flash-0731 | 284B / 13B | 154.49 | maths | ds4mrr@L32@s0 | 1.000 | -5.56 | 452/500 | +11.68 | 17.2 | 100% | 100% / 0% | OpenRouter coreweave/fp8 | B200 |
| DeepSeek-V4-Flash-0731 | 284B / 13B | 154.49 | ordering | ds4orr@L32@s0 | 1.000 | -7.48 | 139/200 | +9.19 | 16.7 | 100% | 100% / 5% | OpenRouter coreweave/fp8 | B200 |
