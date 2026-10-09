# Probe registry (m3a2_sarcasm_patch)

All in `results/concept_probes/`. Keys inside each file: `sarcasm@L16`, `sarcasm@L48`. Mission names: `r<k>_peak@L<layer>`, `r<k>_gd@L16`.

| file | data | recipe |
|---|---|---|
| `m3a2_sarcasm_patch_r0_peak.json` | m3a's off-policy rows `data/concepts/m3a_sarcasm_off_sym.jsonl` (4,920 Claude-written replies) | copy of m3a's `m3a_sarcasm_offsym_peak.json` seed 0 (keys `maths@L16@s0`, `maths@L48@s0` there): BCE on the EMA peak, val-loss stopping, max 15 epochs. **Round-0 primary = L48.** |
| `m3a2_sarcasm_patch_r0_gd.json` | same rows | copy of m3a's `m3a_sarcasm_offsym_gd.json` L16 (gd = per-token sigmoid averaged over the reply, Neural Chameleons recipe). m3a's main probe. |
| `m3a2_sarcasm_patch_r1_peak.json` | `data/concepts/m3a2_sarcasm_patch_r1_train.jsonl` = m3a's rows + patch 1 (261 sarcastic + 313 sincere Claude-written replies in the m3a searched prompt's style, Haiku-checked, request word augmented) = 5,494 rows | BCE on the EMA peak, seed 0, val-loss stopping (ran to the 15-epoch cap). **Round-1 primary = L48.** |
| `m3a2_sarcasm_patch_r1_gd.json` | same as r1 | gd at L16 |
| `m3a2_sarcasm_patch_r1x3_peak.json` | r1 rows with every patch row repeated 3x | peak recipe. **Unreliable diagnostic:** row-level validation split puts copies in both training and validation. |
| `m3a2_sarcasm_patch_r1x3_gd.json` | same as r1x3 | gd at L16; same caveat |
| `m3a2_sarcasm_patch_r2b_peak.json` / `_r2b_gd.json` | **Arm B.** `data/concepts/m3a2_sarcasm_patch_r2b_train.jsonl` = m3a's 4,920 rows + broad patch (459 full sarcastic/sincere pairs in 10 registers, Claude Sonnet, Haiku-checked, request word augmented; `..._patch2b_candidates.jsonl`) + 900 shared instruction rows (`..._shared_instruction_rows_v2.jsonl`) = 6,738 | peak (seed 0) L16/L48; gd L16. **Arm B primary = peak L48**, secondary gd L16. |
| `m3a2_sarcasm_patch_r2c_peak.json` / `_r2c_gd.json` | **Arm C.** `..._r2c_train.jsonl` = round-1 rows (5,494) + 900 shared instruction rows = 6,394 | peak L16/L48; gd L16. Ensembles: 2C = r2c peak L48 OR r0 gd L16 (lead's spec); 2C-pk = r2c peak L48 OR r2c peak L16 (readout-matched). |
| `m3a2_sarcasm_patch_r2s_peak.json` / `_r2s_gd.json` | **Control.** `..._r2s_train.jsonl` = m3a's rows + 900 shared instruction rows = 5,820 | peak L16/L48; gd L16. Credits the shared fix alone. |
