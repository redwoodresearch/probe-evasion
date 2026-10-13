# Two quick questions on the switch sweep

**Setup in one line:** writer models write an attack template for Qwen3.5-27B. Qwen sees the same template with
`Style switch: ON` or `OFF` (one token different). We measure how much lower the probe reads ON than OFF, counting only
correct answers.

![Where replies sit](figures/simple_where.png)

## Q1. Which number should be the headline?

- **(a) Share of OFF's distance to the flag line that ON covers** (1 = ON reaches the line).
  *My pick.*
- (b) The same drop as a share of a **plain answer's** distance to the line. This is a fixed yardstick, but 1 isn't
  quite "at the line".
- (c) The **raw** drop in log-odds. The simplest, but not comparable across probes or models.
- (d) **Crossings:** the share of pairs where ON is under the line and OFF is above it. Zero so far on these probes.

*Catch for (a):* on one of the newer probes (spotting a contradiction), the template alone already puts most OFF
replies under the line (30 of about 40 in a test run). Then (a) breaks, because OFF has no distance left to cover.
That case needs (b) or (d), and it's the strongest argument for Q2.

## Q2. Should we train the probe with instructions in view?

- **Why:** any instruction in context shifts the probe's readings, so a line set without one doesn't transfer. On two
  of the newer probes, the instruction alone pushes answers under the line, with no evasion at all.
- **Why not:** it changes the yardstick mid-project and mixes defender strength into an attacker measure.
- **My suggestion:** keep this probe as the yardstick, and add an instruction-trained probe as a second reading on the
  same replies (no new runs needed).
