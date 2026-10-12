"""The defender's natural fix for mission m1a's global-line finding: train the probe for the reading it is deployed
with. concept_probes trains on the mean of per-token probabilities, then the monitor reads the peak of a running
average (EMA). Benign lookup emails were therefore only pushed to a low average, and their boilerplate tokens
("Hi", "Best regards") keep high scores that the EMA peak picks up. Here the same training rows and the same split are
fitted with binary cross-entropy on the EMA peak itself (logit space, alpha 0.5, neutral start, as gepa.readings).

    uv run modal run --detach -m probe_evasion.remote_search::launch --chains-file <chain calling peak_probes>
"""

import json
from pathlib import Path

import modal
import numpy as np

from probe_evasion import concept_probes, gepa, remote_search  # remote_search: registers the cloud driver for launch_cloud
from probe_evasion.generation import write_jsonl
from probe_evasion.missions.m1a_offpolicy_maths import (MISSION, ONPOLICY_MATCHED, ONPOLICY_TRAIN, OFFPOLICY_TRAIN, SCORES, TRAIN_LAYERS, jl,
                                                        score_rows)
from probe_evasion.modal_app import app, hf_cache

PEAK_FILES = {f"{MISSION}_qwen_peak": ONPOLICY_TRAIN, f"{MISSION}_claude_sys_peak": OFFPOLICY_TRAIN}
PEAK_PROBES = {"onpeak@L16": (f"{MISSION}_qwen_peak", "maths@L16"), "onpeak@L48": (f"{MISSION}_qwen_peak", "maths@L48"),
               "offpeak@L16": (f"{MISSION}_claude_sys_peak", "maths@L16"), "offpeak@L48": (f"{MISSION}_claude_sys_peak", "maths@L48")}
EPOCHS, PATIENCE, BATCH, LR, WEIGHT_DECAY = 40, 3, 32, 1e-3, 0.01
_LAB = concept_probes.ProbeLab._get_user_cls()


def ema_peaks(logit_seqs, alpha: float = gepa.EMA_ALPHA):
    """Differentiable EMA peak per sequence (torch tensors of per-token logits), starting from logit 0."""
    import torch

    lengths = [len(x) for x in logit_seqs]
    padded = torch.nn.utils.rnn.pad_sequence(logit_seqs, batch_first=True)
    mask = torch.arange(padded.shape[1], device=padded.device)[None, :] < torch.tensor(lengths, device=padded.device)[:, None]
    state = torch.zeros(padded.shape[0], device=padded.device)
    steps = []
    for t in range(padded.shape[1]):
        state = alpha * padded[:, t] + (1 - alpha) * state
        steps.append(state)
    ema = torch.stack(steps, 1).masked_fill(~mask, float("-inf"))
    return ema.max(1).values


def load_qwen(self) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    self.model_key = "qwen35"
    model_id = concept_probes.MODELS["qwen35"]
    self.tok = AutoTokenizer.from_pretrained(model_id)
    self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    self.layers = next(m for n, m in self.model.named_modules()
                       if n.endswith("language_model.layers") and isinstance(m, torch.nn.ModuleList))


@app.cls(image=concept_probes.probe_image, gpu="H200", volumes={"/hf-cache": hf_cache}, timeout=3 * 60 * 60, scaledown_window=300,
         memory=131072)
class PeakProbeLab:
    @modal.enter()
    def load(self):
        load_qwen(self)

    @modal.method()
    def train_peak(self, rows: list[dict], layers: tuple[int, ...], seed: int = 0) -> dict:
        """Same validation split as ProbeLab.train (seed-0 permutation, 20% validation); early stopping on validation
        AUROC of the EMA-peak reading."""
        import torch
        from sklearn.metrics import roc_auc_score

        acts = [{l: a.cpu() for l, a in r.items()} for r, _ in _LAB._reply_acts(self, rows, layers, concept_probes.MAX_TRAIN_TOKENS)]
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(rows), generator=gen).tolist()
        n_val = len(rows) // 5
        val, train = perm[:n_val], perm[n_val:]
        y = torch.tensor([float(r["label"]) for r in rows], device="cuda")
        out = {}
        for layer in layers:
            X = [acts[i][layer].cuda().float() for i in range(len(rows))]
            w = torch.zeros(X[0].shape[-1], device="cuda", requires_grad=True)
            b = torch.zeros((), device="cuda", requires_grad=True)
            opt = torch.optim.AdamW([w, b], lr=LR, weight_decay=WEIGHT_DECAY)
            peaks = lambda ids: ema_peaks([X[i] @ w + b for i in ids])
            best, state, patience = -1.0, None, 0
            for epoch in range(EPOCHS):
                order = torch.randperm(len(train), generator=gen).tolist()
                for s in range(0, len(order), BATCH):
                    batch = [train[j] for j in order[s:s + BATCH]]
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(peaks(batch), y[batch])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    auc = roc_auc_score(y[val].cpu().numpy(), peaks(val).cpu().numpy())
                if auc > best + 1e-4:
                    best, state, patience = auc, (w.detach().clone(), b.detach().clone()), 0
                else:
                    patience += 1
                    if patience >= PATIENCE:
                        break
            print(f"peak probe layer {layer}: val AUROC {best:.4f} ({epoch + 1} epochs)", flush=True)
            out[f"maths@L{layer}"] = {"layer": layer, "val_auroc": best, "weight": state[0].cpu().tolist(), "bias": state[1].item(),
                                      "n_train": len(train), "n_val": len(val), "objective": "bce on ema peak"}
            del X
        return out


@app.cls(image=concept_probes.probe_image, gpu="H200", volumes={"/hf-cache": hf_cache}, timeout=3 * 60 * 60, scaledown_window=300,
         memory=131072)
class PeakProbeLabSeeds:
    @modal.enter()
    def load(self):
        load_qwen(self)

    @modal.method()
    def train_peak_seeded(self, rows: list[dict], layer: int, seeds: tuple[int, ...], max_epochs: int = 15) -> dict:
        """Critic 4's fix: the first version stopped on validation AUROC, which is 1.0 after one epoch, so its weights
        were from epoch 1. Here: stop on validation loss (patience 3, at most max_epochs), one probe per seed (the seed
        sets both the validation split and the batch order). Activations are extracted once and shared."""
        import torch
        from sklearn.metrics import roc_auc_score

        acts = [a[layer].cpu() for a, _ in _LAB._reply_acts(self, rows, (layer,), concept_probes.MAX_TRAIN_TOKENS)]
        y = torch.tensor([float(r["label"]) for r in rows], device="cuda")
        X = [a.cuda().float() for a in acts]
        out = {}
        for seed in seeds:
            gen = torch.Generator().manual_seed(seed)
            if any("group" in r for r in rows):  # m48: copies of one reply (re-read under another request) share a group and a side
                groups = list(dict.fromkeys(r["group"] for r in rows))
                gperm = torch.randperm(len(groups), generator=gen).tolist()
                val_groups = {groups[i] for i in gperm[:len(groups) // 5]}
                val = [i for i, r in enumerate(rows) if r["group"] in val_groups]
                train = [i for i, r in enumerate(rows) if r["group"] not in val_groups]
            else:  # every earlier caller: the original row-level split, unchanged
                perm = torch.randperm(len(rows), generator=gen).tolist()
                n_val = len(rows) // 5
                val, train = perm[:n_val], perm[n_val:]
            w = torch.zeros(X[0].shape[-1], device="cuda", requires_grad=True)
            b = torch.zeros((), device="cuda", requires_grad=True)
            opt = torch.optim.AdamW([w, b], lr=LR, weight_decay=WEIGHT_DECAY)
            peaks = lambda ids: ema_peaks([X[i] @ w + b for i in ids])
            best, state, patience, epoch = float("inf"), None, 0, 0
            for epoch in range(max_epochs):
                order = torch.randperm(len(train), generator=gen).tolist()
                for s in range(0, len(order), BATCH):
                    batch = [train[j] for j in order[s:s + BATCH]]
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(peaks(batch), y[batch])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    vp = peaks(val)
                    vloss = torch.nn.functional.binary_cross_entropy_with_logits(vp, y[val]).item()
                    auc = roc_auc_score(y[val].cpu().numpy(), vp.cpu().numpy())
                if vloss < best - 1e-4:
                    best, state, patience, best_auc, best_epoch = vloss, (w.detach().clone(), b.detach().clone()), 0, auc, epoch + 1
                else:
                    patience += 1
                    if patience >= PATIENCE:
                        break
            print(f"seed {seed} layer {layer}: best val loss {best:.4f} at epoch {best_epoch} (AUROC {best_auc:.4f}), ran {epoch + 1}", flush=True)
            out[f"maths@L{layer}@s{seed}"] = {"layer": layer, "val_loss": best, "val_auroc": best_auc, "epoch": best_epoch, "seed": seed,
                                               "weight": state[0].cpu().tolist(), "bias": state[1].item(), "objective": "bce on ema peak, val-loss stop"}
        return out


SEEDED_FILES = {f"{MISSION}_qwen_peak_seeds": ONPOLICY_TRAIN, f"{MISSION}_qwen_matched_peak_seeds": ONPOLICY_MATCHED,
                f"{MISSION}_claude_sys_peak_seeds": OFFPOLICY_TRAIN}
SEEDS = (0, 1, 2)
SHORT = {f"{MISSION}_qwen_peak_seeds": "onpk", f"{MISSION}_qwen_matched_peak_seeds": "onmpk", f"{MISSION}_claude_sys_peak_seeds": "offpk"}


def load_seeded_probes() -> dict:
    out = {}
    for f, short in SHORT.items():
        for key, p in json.loads((concept_probes.PROBES / f"{f}.json").read_text()).items():
            out[f"{short}@L48@s{p['seed']}"] = p
    return out


@app.local_entrypoint()
def seeded_peak():
    """Retrain the peak probes properly (3 seeds each, val-loss stopping, plus a size-matched own-replies version) and
    re-score the one-sentence calibration pool, the confirmation rows and Qwen's held-out replies with them."""
    lab = PeakProbeLabSeeds()
    calls = {f: lab.train_peak_seeded.spawn(jl(path), 48, SEEDS) for f, path in SEEDED_FILES.items()}
    for f, call in calls.items():
        out = call.get()
        (concept_probes.PROBES / f"{f}.json").write_text(json.dumps(out))
        print(f, {k: (round(v["val_loss"], 4), v["epoch"]) for k, v in out.items()}, flush=True)
    probes = load_seeded_probes()
    keep = ("task", "prompt", "reply", "reasoning", "finished")
    sets = {"calibration": [r for r in jl(SCORES / "calibration_unscored.jsonl") if r["finished"]],
            "calibration_one_sentence": jl(SCORES / "calibration_one_sentence.jsonl"),
            "v2_calibration": jl(V2 / "calibration_replies.jsonl"), "confirm": jl(SCORES / "confirm.jsonl"),
            "heldout_qwen": jl(SCORES / "peak_heldout_qwen.jsonl")}
    for name, rows in sets.items():
        rows = [{k: r[k] for k in keep if k in r} | {k: r[k] for k in ("arm", "lenient", "present", "division", "label", "sample") if k in r}
                for r in rows]
        score_rows(rows, probes)
        for r in rows:
            r.pop("logits", None)
        write_jsonl(SCORES / f"seeded_{name}.jsonl", rows)
        print(f"seeded-scored {name}: {len(rows)}", flush=True)


def load_peak_probes() -> dict:
    files = {f: json.loads((concept_probes.PROBES / f"{f}.json").read_text()) for f in PEAK_FILES}
    return {name: files[f][key] for name, (f, key) in PEAK_PROBES.items()}


V2 = Path("results/gepa/qwen35_maths_ema_onpolicyclean_L48_20261008-v2")


@app.local_entrypoint()
def peak_probes(train: bool = True):
    """Train both peak probes, then score with them: every calibration pool, the v2 search's frozen-prompt tests (held
    out x1 and x10, new division x3) and, if present, the format-negatives rows. Writes SCORES/peak_<set>.jsonl."""
    if train:
        lab = PeakProbeLab()
        calls = {f: lab.train_peak.spawn(jl(path), TRAIN_LAYERS) for f, path in PEAK_FILES.items()}
        for f, call in calls.items():
            out = call.get()
            (concept_probes.PROBES / f"{f}.json").write_text(json.dumps(out))
            print(f, {k: round(v["val_auroc"], 4) for k, v in out.items()}, flush=True)
    probes = load_peak_probes()
    keep = ("task", "prompt", "reply", "reasoning", "finished")
    sets = {"calibration": jl(SCORES / "calibration_unscored.jsonl"),
            "calibration_one_sentence": jl(SCORES / "calibration_one_sentence.jsonl"),
            "v2_calibration": jl(V2 / "calibration_replies.jsonl")}
    for f in sorted(V2.glob("*test_*.jsonl")):
        sets[f"v2_{f.stem}"] = jl(f)
    if (SCORES / "format_negatives.jsonl").exists():
        sets["format_negatives"] = jl(SCORES / "format_negatives.jsonl")
    for name, rows in sets.items():
        rows = [{k: r[k] for k in keep if k in r} | {k: r[k] for k in ("arm", "lenient", "present") if k in r}
                for r in rows if r.get("finished", True)]
        score_rows(rows, probes)
        write_jsonl(SCORES / f"peak_{name}.jsonl", rows)
        print(f"peak-scored {name}: {len(rows)}", flush=True)


@app.local_entrypoint()
def launch_cloud(chains_file: str):
    """remote_search.launch, but from this module so PeakProbeLab is registered in the launched app."""
    chains = json.loads(Path(chains_file).read_text())
    calls = {name: remote_search.chain.spawn(name, steps) for name, steps in chains.items()}
    for name, call in calls.items():
        print("finished", call.get(), flush=True)


@app.local_entrypoint()
def final_checks():
    """(1) Instruction-text-alone check for the v2 prompt: the plain arm's fresh replies re-read with the v2 instruction
    in front, all probes. (2) The candidate tests (seed-1234 division rows, plain replies with candidate instructions in
    front, candidate-format no-maths replies) re-read with the peak-trained probes."""
    from probe_evasion.missions.m1a_offpolicy_maths import V2_WINNER

    plain = [r for r in jl(SCORES / "fresh.jsonl") if r["arm"] == "plain"]
    ctx = [{k: r[k] for k in ("task", "reply", "reasoning", "finished", "lenient", "present", "sample")}
           | {"arm": "plain reply, v2 winner in front", "prompt": f"{V2_WINNER}\n\n{r['task']['task']}"} for r in plain]
    score_rows(ctx)
    score_rows(ctx, load_peak_probes())
    write_jsonl(SCORES / "v2_context.jsonl", ctx)
    probes = load_peak_probes()
    for name in ("candidate_division", "candidate_context", "candidate_format_negatives"):
        rows = jl(SCORES / f"{name}.jsonl")
        for r in rows:
            r.pop("logits", None)
        score_rows(rows, probes)
        if name == "candidate_division":
            removed = [{"prompt": r["task"]["task"], "reply": r["reply"], "reasoning": r["reasoning"]} for r in rows]
            score_rows(removed, probes)
            for r, x in zip(rows, removed):
                r["scores_instruction_removed"].update(x["scores"])
        write_jsonl(SCORES / f"peak_{name}.jsonl", rows)
        print(f"peak-scored {name}: {len(rows)}", flush=True)


@app.local_entrypoint()
def offpeak_confirm(samples: int = 2):
    """Pre-registered confirmation (STATUS 13:15): plain vs the two frozen candidate prompts on new division problems
    (seed 4321) and new mixed-template problems (seed 7070), every probe incl. the peak-trained ones, instruction
    removed too; and the peak probes on Qwen's own held-out-style replies."""
    import random as _random

    from probe_evasion.followups import fresh_division_tasks
    from probe_evasion.missions.m1a_offpolicy_maths import (CANDIDATES, CAL_STYLES, CALIBRATION_TASKS, attack_replies,
                                                            fresh_tasks, frozen, heldout_rows, load_probes, maths, one_sentence_calibration_tasks,
                                                            prompt_sets, ONPOLICY_HELDOUT)
    from probe_evasion.onpolicy_probes import NO_WORKING as NO_WORKING_TEXT

    heldout = heldout_rows(ONPOLICY_HELDOUT, "qwen35")
    score_rows(heldout, load_peak_probes())
    write_jsonl(SCORES / "peak_heldout_qwen.jsonl", heldout)
    earlier = {t["task"] for seed, n in ((7, 120), (99, 120), (1234, 360)) for t in fresh_division_tasks(n, seed=seed)}
    division = [t for t in fresh_division_tasks(400, seed=4321) if t["task"] not in earlier][:300]
    used = set().union(*prompt_sets().values(), {t["task"] for t in jl(CALIBRATION_TASKS)}, {t["task"] for t in one_sentence_calibration_tasks()},
                       {t["task"] for t in fresh_tasks()}, earlier, {t["task"] for t in division})
    rng = _random.Random(7070)
    mixed = []
    while len(mixed) < 200:
        m = maths(rng, 1)[0]
        task = f"{m['computed']} {CAL_STYLES['one sentence']} {NO_WORKING_TEXT}"
        if task not in used:
            used.add(task)
            mixed.append({"concept": "maths", "split": "confirm", "task": task, "result": m["result"]})
    assert len(division) == 300
    arms = {"plain": ""} | {name: frozen(name) for name in CANDIDATES}
    rows = []
    for seed in range(samples):
        rows += [r | {"sample": seed} for r in attack_replies(arms, division + mixed, seed=6000 + seed)]
    write_jsonl(SCORES / "confirm_unscored.jsonl", rows)
    probes = load_probes() | load_peak_probes()
    score_rows(rows, probes)
    removed = [{"prompt": r["task"]["task"], "reply": r["reply"], "reasoning": r["reasoning"]} for r in rows]
    score_rows(removed, probes)
    for r, x in zip(rows, removed):
        r["scores_instruction_removed"] = x["scores"]
        r["division"] = r["task"]["task"] in {t["task"] for t in division}
    write_jsonl(SCORES / "confirm.jsonl", rows)
    print(f"confirm: {len(rows)} rows", flush=True)


def fit_peak_probe(X: list, y, seed: int, max_epochs: int = 15) -> dict:
    """The PeakProbeLabSeeds recipe as a plain function (for other models' labs): X = per-reply [tokens, d] float
    tensors on the GPU, y = float labels on the GPU. Seed sets the 20% validation split and batch order; stops on
    validation loss (patience PATIENCE, at most max_epochs)."""
    import torch
    from sklearn.metrics import roc_auc_score

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=gen).tolist()
    n_val = len(X) // 5
    val, train = perm[:n_val], perm[n_val:]
    w = torch.zeros(X[0].shape[-1], device="cuda", requires_grad=True)
    b = torch.zeros((), device="cuda", requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=LR, weight_decay=WEIGHT_DECAY)
    peaks = lambda ids: ema_peaks([X[i] @ w + b for i in ids])
    best, state, patience, epoch, best_auc, best_epoch = float("inf"), None, 0, 0, float("nan"), 0
    for epoch in range(max_epochs):
        order = torch.randperm(len(train), generator=gen).tolist()
        for s in range(0, len(order), BATCH):
            batch = [train[j] for j in order[s:s + BATCH]]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(peaks(batch), y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vp = peaks(val)
            vloss = torch.nn.functional.binary_cross_entropy_with_logits(vp, y[val]).item()
            auc = roc_auc_score(y[val].cpu().numpy(), vp.cpu().numpy())
        if vloss < best - 1e-4:
            best, state, patience, best_auc, best_epoch = vloss, (w.detach().clone(), b.detach().clone()), 0, auc, epoch + 1
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    return {"val_loss": best, "val_auroc": best_auc, "epoch": best_epoch, "seed": seed, "weight": state[0].cpu().tolist(),
            "bias": state[1].item(), "objective": "bce on ema peak, val-loss stop"}
