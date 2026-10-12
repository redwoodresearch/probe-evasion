# Mission m42_adaptive_defender: does a probe retrained on attack styles erase the trend?

## Why

If m36's trend mostly measures how well writers find answer styles the probe never saw in training, a defender who
adds those styles to training would collapse it. That decides whether the eval tracks a lasting capability or a gap in
one probe's training set.

## Design

- **Leave-one-lab-out retraining.** For maths and ordering on Qwen3.5-27B, retrain the layer-48 own-reply probe (m22 /
  m26 recipe and reading) with extra positives and negatives in the sweep's styles: valid attack replies from m36's
  test or eval sets as positives, and same-style no-work twins as negatives (m36's twins; generate more if needed).
  For each lab (Anthropic, OpenAI, Google, Qwen, Meta), train on the other labs' styles only. Recalibrate the 1% line
  on the same ordinary pool (also report the false-alarm rate on the broad chat pool).
- Re-score the held-out lab's m36 test replies (activations already define the reading; reuse saved activations if
  available, otherwise re-read) and recompute S, S_valid and the trend against ECI using held-out-lab scores only.
- **Augmented-style probe.** One probe trained with generic style augmentation (m28's random answer styles plus number
  words), no sweep styles. Same re-scoring.
- **Second probe design.** One max-pooled or attention-pooled probe trained the standard way, deployed with its own
  reading and line, re-scoring m36's test replies.
- Report probe quality (held-out AUROC, plain caught at the line) for every probe.

## Budget

GPU about $500, Claude about $150 (twin validity).

## Rules

- Modal names `probe-evasion_nabil-baugher_m42-<label>_kill-2026-11-30`; use your own Modal volume for state
  (`probe-evasion_nabil-baugher_m42-state_kill-2026-11-30`; the shared volume is near its inode limit). Stop only your
  own apps, by id. Keep scaledown windows short so GPUs do not idle through writer/judge phases.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits and no edits to other missions' modules (ask the
  lead, SendMessage to "main"); code in `src/probe_evasion/missions/m42_adaptive_defender.py`, importing from m36 and earlier modules
  rather than copying. Write only in `missions/m42_adaptive_defender/`, that module and `data/concepts/m42_*`. No repo-changing git. Long
  runs detached in Modal. Log spend in STATUS; tell the lead before exceeding the budget.
- Pre-register the design and metric in STATUS before any run; one critic check on the design, one before the verdict.
  Save everything (every prompt, every reply with per-token scores, every judge call).
- RESULT.md in plain language: numbers, figures, caveats, cost.
