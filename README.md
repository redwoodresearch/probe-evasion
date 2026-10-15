# probe-evasion

Can models get past activation-probe monitors while still doing a monitored task, and does that ability grow with model
capability? This repo holds a Redwood Research sprint on that question. Its main result is the post
"More capable models write better probe-evasion prompts": 14 writer models (Epoch Capabilities Index 132–167) write
prompts that make Qwen3.5-27B answer correctly while linear probes on its activations stay quiet, and stronger writers
move the probes further.

**Start at [`mainline/README.md`](mainline/README.md).** It describes the experiment, maps every stage to its code,
lists the probes and data, and has one command that rebuilds the post's numbers and figures.

## Layout

| Path | What it is |
|---|---|
| `mainline/` | Entry point for the post: README, `make_all.py`, the post's figures |
| `src/probe_evasion/` | All code (one Python package). Mainline modules are listed in `mainline/README.md`; the package also holds the exploratory code, which the mainline imports in places for shared helpers |
| `src/probe_evasion/missions/` | One module per experiment ("mission"), `m<number>_<name>.py` |
| `missions/` | Outputs, run logs (`STATUS.md`) and results (`RESULT.md`) of the missions the post uses, plus older missions whose code or data the mainline still reads (m11, m19, m1a, m22, m25–m28, m36, m38, m39, m43, m50) |
| `analysis/` | The post's analysis (`post_v9/`: numbers, figures, appendix examples), the word-classifier baseline (`tfidf_baseline/`), and three helper scripts the figures load (`post_v6/`, `post_v8/`, `for_james/`) |
| `data/concepts/` | Datasets: probe training, calibration and test problems. Mainline files are listed in `mainline/README.md`; the rest belong to experimental missions |
| `experimental/` | Everything exploratory or superseded that the mainline does not read: earlier concepts and designs, side projects, older drafts and reports. See [`experimental/README.md`](experimental/README.md) |
| `tests/` | Unit tests for the arithmetic answer checks |
| `results/`, `cache/`, `external/` | Not in git: probe weights of older probes, judge caches, downloads |

## Setup

`uv sync` (Python 3.12). Put `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` in a `.env` at the repo root (git-ignored).
GPU work runs on [Modal](https://modal.com); app and volume names follow
`probe-evasion_nabil-baugher_<label>_kill-2026-11-30`. The raw outputs of the main sweeps (about 20 GB) are not in git;
see `mainline/README.md`.
