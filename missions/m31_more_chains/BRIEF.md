# Mission m31_more_chains: tighten the "does the writer need probe scores?" comparison with more runs

## Why

The post's Result 3 (probe scores add little over seeing replies and correctness) rests on 3 chains per arm, with chain
spreads that overlap (m22: last-round test means −4.0 with scores vs −3.2 without; m26 ordering: −7.2 vs −6.1 over rounds
1–15). m28 is adding 3+3 chains with a pinned Opus 4.8 writer (no 5.5/4.8 mixing). Nabil approved spending to firm this
up (monthly guideline ~$50k; this is small).

## What to run

With m28's pinned writer (`opus48_pinned` with scores, `opus48_pinned_noscores` without; registered in
`src/probe_evasion/missions/m28_baselines_and_writer_knowledge.py`, see its STATUS "Pinned Opus 4.8 writer: how it is
registered"; import, don't copy or edit) on the m22 runner:
- maths 27B, own-reply probe: 5 more chains with scores and 5 more without, 16 rounds each;
- ordering 27B (m26's Task and Monitor): 5 more chains with scores and 5 more without.
Together with m28's 3+3 that gives 8 chains per arm. Same eval sets, seeds, validity rules and leak/refusal checks as m22,
m26 and m28; m28's writer prompt for the pinned arms (check with m28's STATUS which writer prompt the pinned arms use, and
use the identical one).

Final test (pre-register in STATUS first): best and last round of every new chain, 100 untouched problems × 5 seeds.

## Analysis

Pool your chains with m28's pinned chains (read its saved runs; don't write to its folders). Report per arm:
per-round median change from plain (paired), last-round and best-round test medians with the spread across 8 chains, a
permutation test over chains for "with scores vs without", and the per-round figure with all chains. Plain-language
verdict: is there a detectable benefit from seeing probe scores, and how big?

## Rules

- Modal names `probe-evasion_nabil-baugher_m31-<label>_kill-2026-11-30`; stop only your own apps by id. GPU cap about
  $150, Claude API cap about $300 (judges dominate; m22's cheaper rule far from the line with a full check near it is
  fine). Batch scoring per round.
- Keys via `load_dotenv('.env')`, never printed. No core-module edits; code in
  `src/probe_evasion/missions/m31_more_chains.py`. Don't edit other missions' modules; ask the lead (SendMessage to
  "main") for any change m22's runner needs.
- Write only in `missions/m31_more_chains/`, that module. No repo-changing git. Long runs detached in Modal.
- Critic check-ins: design; before the verdict. Save everything.
