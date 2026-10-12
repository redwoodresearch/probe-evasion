# m29_existing_data_analyses: status

## Plan (2026-10-11)

Saved data only, code in `src/probe_evasion/missions/m29_existing_data_analyses.py`, outputs here.

Data found (all saved, per-token scores and all sister-probe peaks included):
- Frozen tests: `test_rows.jsonl` in m22 `runs/qwen35_27b`, m23 `runs/m23_{9b,122b,gemma}`, m25
  `runs/m25_qwen35_27b_code{,_neutral}`, m26 `runs/qwen35_27b_ordering`. Every row has `peaks` for every probe in the
  monitor (9 for maths 27B), `valid`, `logits`, task metadata.
- Calibration pools: `calibration_scores.jsonl` next to each (peaks for every probe); lines in `lines.json`.
- Zero-shot: only on the eval set (chain round 0 + `zeroshot_*` dirs); no frozen-test zero-shot arms exist
  (except frozen rounds that happened to be round 0).
- In-context non-answers and styled twins: m27 `part_b/<task>/clean.jsonl` and `part_b/summary.json`
  (maths 27B own-probe arms, maths 9B, code m22-description run, ordering; m19 template).
- m22 test arms for the Claude-text probe have no styled twins (m27 did own-probe arms only).

Analyses: 1 false-alarm sweep; 2 transfer to sister probes (all saved, no GPU needed); 3 validity/selection;
4 naturalness judge (Haiku 5.5, cap $25); 5 central table + figure with an m28 hook; 6 (added by the lead) tone vs
task margin comparison.

## Log

- Read brief and m19/m22/m23/m25/m26/m27 RESULTs. Spawned a read-only search for the tone-concept score files (item 6).
- Lead added item 6 (tone vs task margin). Explore agent located tone score files (score caches keyed by
  sha1(prompt NUL reply); m3c embeds scores). Recomputed counts match every RESULT headline (m3a2 268/454 removed,
  m3a3 229/402, m3a4 233/442, m8a 454/456, m8b 63/492, m11 492/494, m3c 126/196).
- Steps done: sweep, transfer, validity, central (+ figure), naturalness (Haiku 5.5, thinking disabled; 11,801 judged
  items, $6.85; validated: plain 100% rated 5 except ordering 99% 4-5; 8 hand-written oddities rated 1-2; agreement
  with m26's own Haiku naturalness tags on 300 replies: Spearman 0.75, within one point 99%), margin (+ figure),
  summaries. No GPU used. m3a's sarcasm headline probe (mean-trained, EMA-read) left out of item 6 by the
  readout rule; m3a2/m3a3/m3a4 peak probes used instead.
- Decision: central table skips "code (told label kinds)" (the first code run deviated from m22's writer prompt);
  it stays in the other tables.
- Next: draft RESULT.md, critic check-in, fix, final.
- Drafted RESULT.md. Self-checks: corrected "paired = unpaired within 0.4" (true for 68/89 arms, max 1.0) and the
  invalid-as-caught summary. m28 hook tested with a fake source built from m22's rows (reproduces the m22 dots); smoke
  and pilot folders are skipped. m28 has no test_rows.jsonl yet.
- Spawned the critic (research-reviewer) on RESULT + code.
