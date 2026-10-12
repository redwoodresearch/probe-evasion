
### maths 27B

| arm | type | target | onpk@L48@s0 | onpk@L48@s1 | onpk@L48@s2 | onmpk@L48@s0 | onmpk@L48@s1 | onmpk@L48@s2 | offpk@L48@s0 | offpk@L48@s1 | offpk@L48@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opus/claude/chain_0 | with scores, best | offpk@L48@s0 | 32% | 52% | 36% | 34% | 33% | 33% | **54%** | 56% | 55% |
| opus/claude/chain_1 | with scores, best | offpk@L48@s0 | 30% | 40% | 36% | 33% | 30% | 30% | **39%** | 42% | 40% |
| opus/claude/chain_2 | with scores, best | offpk@L48@s0 | 30% | 40% | 36% | 33% | 32% | 32% | **44%** | 46% | 45% |
| opus/own/chain_0 | with scores, best | onpk@L48@s0 | **33%** | 22% | 37% | 36% | 35% | 34% | 51% | 50% | 51% |
| opus/own/chain_0@last | with scores, last | onpk@L48@s0 | **29%** | 21% | 34% | 31% | 30% | 30% | 45% | 45% | 44% |
| opus/own/chain_1 | with scores, best | onpk@L48@s0 | **40%** | 53% | 42% | 44% | 42% | 42% | 59% | 62% | 59% |
| opus/own/chain_1@last | with scores, last | onpk@L48@s0 | **21%** | 33% | 28% | 24% | 25% | 23% | 35% | 36% | 34% |
| opus/own/chain_2 | with scores, best | onpk@L48@s0 | **32%** | 35% | 30% | 33% | 31% | 34% | 39% | 40% | 42% |
| opus/own/chain_2@last | with scores, last | onpk@L48@s0 | **31%** | 41% | 30% | 32% | 30% | 32% | 49% | 48% | 50% |
| opus_noscores/own/chain_0 | no scores, best | onpk@L48@s0 | **16%** | 26% | 14% | 18% | 15% | 17% | 25% | 23% | 25% |
| opus_noscores/own/chain_0@last | no scores, last | onpk@L48@s0 | **15%** | 18% | 15% | 15% | 14% | 15% | 15% | 15% | 16% |
| opus_noscores/own/chain_1 | no scores, best | onpk@L48@s0 | **27%** | 43% | 31% | 31% | 28% | 28% | 37% | 39% | 37% |
| opus_noscores/own/chain_1@last | no scores, last | onpk@L48@s0 | **26%** | 32% | 28% | 28% | 27% | 27% | 39% | 41% | 39% |
| opus_noscores/own/chain_2 | no scores, best | onpk@L48@s0 | **28%** | 22% | 33% | 31% | 29% | 27% | 40% | 41% | 40% |
| opus_noscores/own/chain_2@last | no scores, last | onpk@L48@s0 | **24%** | 24% | 30% | 26% | 26% | 25% | 41% | 42% | 39% |
| q122/claude/chain_0 | Qwen-122B writes, best | offpk@L48@s0 | 28% | 46% | 30% | 29% | 30% | 28% | **43%** | 45% | 44% |
| q122/claude/chain_1 | Qwen-122B writes, best | offpk@L48@s0 | 30% | 46% | 36% | 33% | 31% | 31% | **46%** | 48% | 47% |
| q122/claude/chain_2 | Qwen-122B writes, best | offpk@L48@s0 | 38% | 45% | 40% | 39% | 38% | 38% | **52%** | 54% | 52% |
| q122/own/chain_0 | Qwen-122B writes, best | onpk@L48@s0 | **6%** | 3% | 3% | 6% | 6% | 7% | 11% | 10% | 12% |
| q122/own/chain_1 | Qwen-122B writes, best | onpk@L48@s0 | **21%** | 31% | 22% | 20% | 22% | 24% | 30% | 30% | 31% |
| q122/own/chain_2 | Qwen-122B writes, best | onpk@L48@s0 | **32%** | 49% | 38% | 34% | 34% | 33% | 48% | 50% | 50% |
| q27/claude/chain_0 | model writes for itself, best | offpk@L48@s0 | 20% | 23% | 18% | 21% | 20% | 21% | **28%** | 27% | 29% |
| q27/claude/chain_1 | model writes for itself, best | offpk@L48@s0 | 21% | 24% | 23% | 19% | 23% | 22% | **39%** | 40% | 40% |
| q27/claude/chain_2 | model writes for itself, best | offpk@L48@s0 | 24% | 35% | 22% | 25% | 23% | 26% | **36%** | 35% | 36% |
| q27/own/chain_0 | model writes for itself, best | onpk@L48@s0 | **27%** | 34% | 32% | 29% | 29% | 28% | 34% | 36% | 34% |
| q27/own/chain_1 | model writes for itself, best | onpk@L48@s0 | **24%** | 36% | 28% | 28% | 24% | 26% | 35% | 36% | 35% |
| q27/own/chain_2 | model writes for itself, best | onpk@L48@s0 | **25%** | 33% | 27% | 26% | 26% | 25% | 33% | 33% | 33% |
| m19/s2_own | white-box template (m19) | onpk@L48@s0 | **105%** | 95% | 108% | 105% | 107% | 105% | 123% | 119% | 122% |
| m19/s2_claude | white-box template (m19) | offpk@L48@s0 | 84% | 84% | 89% | 88% | 87% | 88% | **114%** | 110% | 111% |

### maths 9B

| arm | type | target | 9bpk@L24@s0 | 9bpk@L24@s1 | 9bpk@L24@s2 | 9bpk@L8@s0 | 9bpk@L8@s1 | 9bpk@L8@s2 | 9blong@L24@s0 | 9blong@L24@s1 | 9blong@L24@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opus/own/chain_0 | with scores, best | 9bpk@L24@s0 | **67%** | 60% | 65% | 10% | 11% | 11% | 67% | 62% | 65% |
| opus/own/chain_1 | with scores, best | 9bpk@L24@s0 | **48%** | 49% | 47% | -8% | -7% | -8% | 48% | 51% | 47% |
| opus/own/chain_2 | with scores, best | 9bpk@L24@s0 | **26%** | 29% | 23% | 58% | 59% | 60% | 26% | 31% | 23% |
| self/own/chain_0 | model writes for itself, best | 9bpk@L24@s0 | **36%** | 39% | 27% | 14% | 14% | 14% | 36% | 40% | 27% |
| self/own/chain_1 | model writes for itself, best | 9bpk@L24@s0 | **33%** | 37% | 23% | 15% | 16% | 15% | 33% | 38% | 23% |
| self/own/chain_2 | model writes for itself, best | 9bpk@L24@s0 | **23%** | 27% | 17% | 5% | 6% | 5% | 23% | 28% | 17% |
| m19/s2_own | m19 template, written for 27B | 9bpk@L24@s0 | **105%** | 96% | 105% | 54% | 54% | 55% | 105% | 96% | 105% |
| m19/s2_claude | m19 template, written for 27B | 9bpk@L24@s0 | **76%** | 81% | 66% | 8% | 8% | 7% | 76% | 81% | 66% |
| opus/own/chain_0@last | with scores, last | 9bpk@L24@s0 | **52%** | 46% | 41% | 41% | 41% | 42% | 52% | 46% | 41% |
| opus/own/chain_1@last | with scores, last | 9bpk@L24@s0 | **16%** | 21% | 13% | -4% | -3% | -2% | 16% | 22% | 13% |
| opus/own/chain_2@last | with scores, last | 9bpk@L24@s0 | **22%** | 25% | 18% | 54% | 53% | 55% | 22% | 25% | 18% |

### maths 122B

| arm | type | target | 122bpk@L36@s0 | 122bpk@L36@s1 | 122bpk@L36@s2 | 122bpk@L12@s0 | 122bpk@L12@s1 | 122bpk@L12@s2 | 122blong@L36@s0 | 122blong@L36@s1 | 122blong@L36@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opus/own/chain_0 | with scores, best | 122bpk@L36@s0 | **57%** | 53% | 57% | 98% | 97% | 98% | 54% | 53% | 55% |
| opus/own/chain_0@last | with scores, last | 122bpk@L36@s0 | **58%** | 53% | 57% | 97% | 97% | 97% | 54% | 53% | 56% |
| opus/own/chain_1 | with scores, best | 122bpk@L36@s0 | **41%** | 41% | 42% | 108% | 108% | 111% | 39% | 39% | 41% |
| opus/own/chain_1@last | with scores, last | 122bpk@L36@s0 | **37%** | 38% | 38% | 112% | 111% | 113% | 37% | 38% | 38% |
| opus/own/chain_2 | with scores, best | 122bpk@L36@s0 | **32%** | 35% | 34% | 80% | 79% | 79% | 33% | 34% | 34% |
| opus/own/chain_2@last | with scores, last | 122bpk@L36@s0 | **35%** | 37% | 37% | 82% | 80% | 82% | 35% | 36% | 36% |
| self/own/chain_0 | model writes for itself, best | 122bpk@L36@s0 | **24%** | 25% | 24% | 71% | 70% | 70% | 24% | 25% | 24% |
| self/own/chain_1 | model writes for itself, best | 122bpk@L36@s0 | **30%** | 30% | 30% | 75% | 73% | 74% | 30% | 29% | 30% |
| self/own/chain_2 | model writes for itself, best | 122bpk@L36@s0 | **25%** | 26% | 24% | 70% | 68% | 68% | 25% | 25% | 25% |
| m19/s2_own | m19 template, written for 27B | 122bpk@L36@s0 | **76%** | 74% | 76% | 82% | 80% | 85% | 71% | 71% | 73% |
| m19/s2_claude | m19 template, written for 27B | 122bpk@L36@s0 | **65%** | 68% | 68% | 60% | 57% | 63% | 63% | 64% | 67% |

### maths Gemma

| arm | type | target | gemmapk@L6@s0 | gemmapk@L6@s1 | gemmapk@L6@s2 | gemmapk@L12@s0 | gemmapk@L12@s1 | gemmapk@L12@s2 | gemmapk@L36@s0 | gemmapk@L36@s1 | gemmapk@L36@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opus/templated/chain_0 | with scores, best | gemmapk@L12@s0 | 28% | 25% | 28% | **-5%** | -3% | 1% | 3% | 8% | 7% |
| opus/templated/chain_1 | with scores, best | gemmapk@L12@s0 | 61% | 64% | 63% | **55%** | 55% | 57% | 38% | 40% | 45% |
| opus/templated/chain_2 | with scores, best | gemmapk@L12@s0 | 23% | 22% | 23% | **8%** | 6% | 7% | 3% | 2% | -1% |
| self/templated/chain_0 | model writes for itself, best | gemmapk@L12@s0 | 24% | 26% | 30% | **3%** | 4% | 6% | 7% | 6% | 10% |
| self/templated/chain_1 | model writes for itself, best | gemmapk@L12@s0 | 24% | 23% | 28% | **22%** | 20% | 24% | 29% | 34% | 39% |
| self/templated/chain_2 | model writes for itself, best | gemmapk@L12@s0 | 22% | 22% | 25% | **3%** | 4% | 5% | 6% | 7% | 8% |
| m19/s2_own | m19 template, written for 27B | gemmapk@L12@s0 | 82% | 87% | 88% | **45%** | 51% | 53% | 100% | 112% | 110% |
| m19/s2_claude | m19 template, written for 27B | gemmapk@L12@s0 | 56% | 55% | 60% | **54%** | 56% | 65% | 66% | 79% | 74% |
| opus/templated/chain_0@last | with scores, last | gemmapk@L12@s0 | 83% | 81% | 80% | **80%** | 79% | 87% | 57% | 60% | 64% |
| opus/templated/chain_1@last | with scores, last | gemmapk@L12@s0 | 78% | 91% | 83% | **58%** | 56% | 59% | 47% | 49% | 53% |
| opus/templated/chain_2@last | with scores, last | gemmapk@L12@s0 | 16% | 18% | 19% | **2%** | 3% | 2% | 1% | 3% | -1% |

### code

| arm | type | target | cpk@L48@s0 | cpk@L48@s1 | cpk@L48@s2 | cpk@L16@s0 | cpk@L16@s1 | cpk@L16@s2 | cnlpk@L48@s0 | cnlpk@L48@s1 | cnlpk@L48@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opus/own/chain_0 | with scores, best | cpk@L48@s0 | **28%** | 17% | 27% | 43% | 42% | 40% | 56% | 57% | 56% |
| opus/own/chain_0@last | with scores, last | cpk@L48@s0 | **29%** | 18% | 28% | 43% | 42% | 39% | 56% | 57% | 57% |
| opus/own/chain_1 | with scores, best | cpk@L48@s0 | **27%** | 17% | 21% | 23% | 22% | 21% | 59% | 57% | 52% |
| opus/own/chain_1@last | with scores, last | cpk@L48@s0 | **4%** | 5% | 3% | 4% | 4% | 3% | 0% | 0% | 0% |
| opus/own/chain_2 | with scores, best | cpk@L48@s0 | **25%** | 16% | 18% | 19% | 15% | 17% | 50% | 50% | 44% |
| opus/own/chain_2@last | with scores, last | cpk@L48@s0 | **24%** | 19% | 23% | 3% | 5% | 2% | 53% | 55% | 51% |

### code (told label kinds)

| arm | type | target | cpk@L48@s0 | cpk@L48@s1 | cpk@L48@s2 | cpk@L16@s0 | cpk@L16@s1 | cpk@L16@s2 | cnlpk@L48@s0 | cnlpk@L48@s1 | cnlpk@L48@s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opus/own/chain_0 | with scores, best | cpk@L48@s0 | **23%** | 13% | 20% | 31% | 32% | 28% | 54% | 56% | 55% |
| opus/own/chain_1 | with scores, best | cpk@L48@s0 | **31%** | 22% | 22% | 42% | 36% | 38% | 46% | 44% | 42% |
| opus/own/chain_2 | with scores, best | cpk@L48@s0 | **35%** | 26% | 31% | 33% | 30% | 28% | 57% | 58% | 55% |
| q122/own/chain_0 | Qwen-122B writes, best | cpk@L48@s0 | **18%** | 10% | 12% | 24% | 25% | 22% | 53% | 53% | 56% |
| q122/own/chain_1 | Qwen-122B writes, best | cpk@L48@s0 | **23%** | 15% | 20% | 8% | 8% | 6% | 55% | 55% | 52% |
| q122/own/chain_2 | Qwen-122B writes, best | cpk@L48@s0 | **11%** | 8% | 8% | 6% | 6% | 5% | 19% | 19% | 20% |
| q27/own/chain_0 | model writes for itself, best | cpk@L48@s0 | **26%** | 13% | 19% | 19% | 18% | 15% | 54% | 54% | 48% |
| q27/own/chain_1 | model writes for itself, best | cpk@L48@s0 | **21%** | 12% | 15% | 23% | 22% | 21% | 0% | 0% | 0% |
| q27/own/chain_2 | model writes for itself, best | cpk@L48@s0 | **15%** | 9% | 9% | 8% | 8% | 8% | 37% | 36% | 32% |

### ordering

| arm | type | target | ordpk@L48@s0 | ordpk@L48@s1 | ordpk@L48@s2 | ordpk@L16@s0 | ordpk@L16@s1 | ordpk@L16@s2 |
|---|---|---|---|---|---|---|---|---|
| opus/own/chain_0 | with scores, best | ordpk@L48@s0 | **36%** | 25% | 3% | 31% | 40% | 18% |
| opus/own/chain_1 | with scores, best | ordpk@L48@s0 | **61%** | 47% | 27% | 47% | 64% | 40% |
| opus/own/chain_2 | with scores, best | ordpk@L48@s0 | **62%** | 53% | 30% | 31% | 36% | 38% |
| opus_noscores/own/chain_0 | no scores, best | ordpk@L48@s0 | **50%** | 44% | 28% | 29% | 44% | 26% |
| opus_noscores/own/chain_1 | no scores, best | ordpk@L48@s0 | **45%** | 45% | 29% | -6% | 7% | 2% |
| opus_noscores/own/chain_2 | no scores, best | ordpk@L48@s0 | **63%** | 52% | 28% | 51% | 63% | 50% |
| q122/own/chain_0 | Qwen-122B writes, best | ordpk@L48@s0 | **53%** | 41% | 24% | 43% | 77% | 34% |
| q122/own/chain_1 | Qwen-122B writes, best | ordpk@L48@s0 | **28%** | 27% | 9% | 4% | 12% | 6% |
| q122/own/chain_2 | Qwen-122B writes, best | ordpk@L48@s0 | **53%** | 45% | 31% | 14% | 18% | 31% |
| q27/own/chain_0 | model writes for itself, best | ordpk@L48@s0 | **40%** | 40% | 45% | 23% | 27% | 25% |
| q27/own/chain_1 | model writes for itself, best | ordpk@L48@s0 | **40%** | 36% | 18% | 29% | 30% | 35% |
| q27/own/chain_2 | model writes for itself, best | ordpk@L48@s0 | **39%** | 32% | 12% | 33% | 50% | 16% |
