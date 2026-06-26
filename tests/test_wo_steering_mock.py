"""CPU mock-execution of the WO#5 GPU cells (82l steering, 82m selectivity).

Execs cells/76_wo_logic.py for the pure helpers, then execs cells/82l_wo_steering.py
and cells/82m_wo_selectivity.py UNMODIFIED against a tiny synthetic HookedTransformer
stand-in — so the actual orchestration (site location, train/test split, probe fit,
inject hook, paired-bootstrap CI, verdict, checkpoint/resume, CSV/JSON deliverables)
runs end-to-end with NO GPU and NO real model. This guards A100 time: a runtime bug
in the cell fails here, not three hours into a run.

The mock model encodes the product B*C along TWO directions: dP, which the bump
readout reads (the "used" subspace), and dDecode, a decodable-but-UNUSED subspace.
On the failing C1 '=' the product lives on dDecode (a probe finds it, R^2 high) while
resid·dP is uninformative -> injecting along the fitted (dDecode) direction does NOT
move the answer -> the pipeline must return CLEAN_NULL. At C4's '=' the product lives
on dP (the model succeeds there), so the COUNTERFACTUAL C4 reference (inject P', score
P') produces a large positive Δ with a CI clear of 0 -> the instrument is proven.

This is the scientifically EXPECTED real-world outcome (decodable-but-failing C1 +
routing breakdown -> a clean null), and it exercises end-to-end BOTH the null-reporting
path and the signal-DETECTION path (via C4). The RECOVERS verdict branch is covered
directly with scalars in tests/test_wo_logic.py — a self-consistent linear-readout mock
cannot manufacture a RECOVERS (injecting an already-decodable value is a no-op: the
ceiling), which is itself a faithful property of the experiment.

Run:  python3 tests/test_wo_steering_mock.py
"""
import os
import re
import sys
import shutil
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "..", "cells")
_fails = []


def check(name, cond):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        _fails.append(name)


# --------------------------------------------------------------------------- mock
class _Cfg:
    def __init__(self, nL, device):
        self.n_layers = nL
        self.device = device


class _Tok:
    """Whitespace tokenizer with a fixed string<->id vocab + leading-space answer ids."""
    def __init__(self, vocab):
        self.vocab = vocab                       # str -> id
        self.inv = {i: s for s, i in vocab.items()}

    def __call__(self, text, add_special_tokens=True):
        # only used as tokenizer(" <int>", add_special_tokens=False) for answer ids
        return {"input_ids": [self.vocab[text]]}

    def decode(self, ids):
        return "".join("" if i == 0 else self.inv.get(int(i), "") for i in ids)


class MockModel:
    """Linear-readout transformer stand-in. Residuals are layer-independent (so a hook
    at any swept layer reaches the readout); the bump readout selects the product token
    whose standardized-coordinate matches resid·dP."""
    def __init__(self, pairs, d=48, n_layers=5, seed=0):
        self.d = d
        self.cfg = _Cfg(n_layers, "cpu")
        rng = np.random.default_rng(seed)
        # orthonormal-ish directions
        M = rng.standard_normal((4, d))
        Q, _ = np.linalg.qr(M.T)
        self.dB, self.dP, self.dDecode, self.dNoiseseed = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
        prods = np.array([B * C for (B, C) in pairs], float)
        Bs = np.array([B for (B, C) in pairs], float)
        self.meanBC, self.stdBC = float(prods.mean()), float(prods.std() + 1e-9)
        self.meanB, self.stdB = float(Bs.mean()), float(Bs.std() + 1e-9)
        # vocab: all prompt tokens + every product answer token (leading space).
        vocab = {"<bos>": 0}
        def _add(s):
            if s not in vocab:
                vocab[s] = len(vocab)
        for (B, C) in pairs:
            for s in self._render_c1(B, C).split(" "):
                _add(s)
            for s in self._render_c4(B, C).split(" "):
                _add(s)
            _add(" " + str(int(B * C)))
        self.tokenizer = _Tok(vocab)
        self.V = len(vocab)
        # product-token id -> standardized coordinate (for the bump readout).
        self.tok_coord = {}
        for (B, C) in pairs:
            tid = vocab[" " + str(int(B * C))]
            self.tok_coord[tid] = (B * C - self.meanBC) / self.stdBC
        self._prompt_by_key = {}                 # token-tuple -> (B, C, surface)

    # -- surfaces (match WO_CONDITIONS C1/C4 exactly) --
    @staticmethod
    def _render_c1(B, C):
        return f"( 0 + {B} ) * {C} ="

    @staticmethod
    def _render_c4(B, C):
        return f"( {B} * {C} ) ="

    def to_tokens(self, prompt, prepend_bos=True):
        ids = [0] + [self.tokenizer.vocab[s] for s in prompt.split(" ")]
        key = tuple(ids)
        m1 = re.match(r"^\( 0 \+ (\d+) \) \* (\d+) =$", prompt)
        m4 = re.match(r"^\( (\d+) \* (\d+) \) =$", prompt)
        if m1:
            self._prompt_by_key[key] = (int(m1.group(1)), int(m1.group(2)), "C1")
        elif m4:
            self._prompt_by_key[key] = (int(m4.group(1)), int(m4.group(2)), "C4")
        return torch.tensor([ids], dtype=torch.long)

    # -- residual field (layer-independent), per position --
    def _resid_seq(self, tokens):
        ids = tokens[0].tolist()
        key = tuple(ids)
        if key in self._prompt_by_key:
            B, C, surface = self._prompt_by_key[key]
        else:                                            # generation/teacher-forcing: prompt is a prefix
            B, C, surface = None, None, "C4"
            for k, v in sorted(self._prompt_by_key.items(), key=lambda kv: -len(kv[0])):
                if len(k) <= len(ids) and tuple(ids[:len(k)]) == k:
                    B, C, surface = v
                    break
            if B is None:
                B, C = 30, 30
        seq = len(ids)
        rng = np.random.default_rng(1000 + B * 97 + C)
        R = 0.005 * rng.standard_normal((seq, self.d))   # small vs product-token z-spacing (clean argmax)
        zP = (B * C - self.meanBC) / self.stdBC
        zB = (B - self.meanB) / self.stdB
        final = seq - 1
        # ')' site: encode operand B on dB (decodable; readout ignores it).
        rp = None
        strs = [self.tokenizer.decode([i]).strip() for i in ids]
        for i, s in enumerate(strs):
            if s == ")":
                rp = i
        if rp is not None:
            R[rp] += 2.0 * zB * self.dB + 1.5 * zP * self.dDecode
        if surface == "C4":
            R[final] += 1.0 * zP * self.dP                 # product on the READ direction (bump peaks at true token)
        else:  # C1: product decodable on dDecode, but the readout (dP) does NOT carry it
            R[final] += 1.5 * zP * self.dDecode            # decodable-but-unused on dDecode
        return torch.tensor(R, dtype=torch.float32), final

    def _readout(self, resid_final):
        r = resid_final.numpy().astype(float)
        coordP = float(r @ self.dP)
        logits = np.full(self.V, -8.0)
        for tid, c in self.tok_coord.items():
            logits[tid] = 2.0 - 6.0 * (coordP - c) ** 2     # bump: peak where resid·dP == coord
        return torch.tensor(logits, dtype=torch.float32)

    def _cache(self, tokens):
        R, final = self._resid_seq(tokens)
        seq = R.shape[0]
        cache = {f"blocks.{L}.hook_resid_post": R.clone().unsqueeze(0) for L in range(self.cfg.n_layers)}
        return cache, final

    def __call__(self, tokens):
        cache, final = self._cache(tokens)
        seq = tokens.shape[1]
        out = torch.zeros((1, seq, self.V), dtype=torch.float32)
        out[0, final] = self._readout(cache[f"blocks.{self.cfg.n_layers - 1}.hook_resid_post"][0, final])
        return out

    def run_with_cache(self, tokens, names_filter=None):
        cache, final = self._cache(tokens)
        seq = tokens.shape[1]
        out = torch.zeros((1, seq, self.V), dtype=torch.float32)
        out[0, final] = self._readout(cache[f"blocks.{self.cfg.n_layers - 1}.hook_resid_post"][0, final])
        return out, cache

    def run_with_hooks(self, tokens, fwd_hooks):
        cache, final = self._cache(tokens)
        seq = tokens.shape[1]
        (name, hook) = fwd_hooks[0]
        L = int(name.split(".")[1])
        rp = cache[name]                          # [1, seq, d]
        hook(rp, None)                            # mutate in place (the additive steer)
        out = torch.zeros((1, seq, self.V), dtype=torch.float32)
        out[0, final] = self._readout(rp[0, final])   # readout from the HOOKED layer
        return out


# ------------------------------------------------------------------ infra stubs
def _make_ns(art_dir):
    import json as _json
    import pickle as _pickle
    ns = {}
    ns["log"] = lambda *a, **k: None
    ns["print"] = print

    def _p(name, ext):
        return os.path.join(art_dir, name if name.endswith(ext) else name + ext)
    EXT = {"json": ".json", "pickle": ".pkl", "text": ".txt"}

    def has_artifact(name, kind=None):
        if kind:
            return os.path.exists(_p(name, EXT[kind]))
        return any(os.path.exists(_p(name, e)) for e in (".json", ".pkl", ".txt"))
    ns["has_artifact"] = has_artifact
    ns["save_json"] = lambda n, o: open(_p(n, ".json"), "w").write(_json.dumps(o, default=str)) or _p(n, ".json")
    ns["load_json"] = lambda n: _json.load(open(_p(n, ".json")))
    ns["save_pickle"] = lambda n, o: _pickle.dump(o, open(_p(n, ".pkl"), "wb")) or _p(n, ".pkl")
    ns["load_pickle"] = lambda n: _pickle.load(open(_p(n, ".pkl"), "rb"))

    class _PathLike:
        def __init__(self, base):
            self.base = base
        def __truediv__(self, other):
            return _PathLike(os.path.join(self.base, str(other)))
        def __str__(self):
            return self.base
        def mkdir(self, *a, **k):
            os.makedirs(self.base, exist_ok=True)
        def exists(self):
            return os.path.exists(self.base)
    res = _PathLike(os.path.join(art_dir, "results"))
    res.mkdir()
    ns["WO_RESULTS"] = res
    ns["wo_save_result"] = lambda fn, text: open(os.path.join(str(res), fn), "w").write(text)
    return ns


def _exec_cell(ns, fname):
    path = os.path.join(CELLS, fname)
    exec(compile(open(path).read(), path, "exec"), ns)


def _run(pairs):
    art = tempfile.mkdtemp(prefix="wo5mock_")
    try:
        ns = _make_ns(art)
        _exec_cell(ns, "76_wo_logic.py")
        # notebook globals the GPU cells expect
        ns["CFG"] = {"wo_steer_tags": ["base"], "wo_steer_seed": 3, "wo_steer_layer_stride": 1,
                     "wo_steer_test_frac": 0.5, "wo_steer_min_test_n": 10, "wo_steer_nboot": 400,
                     "wo_steer_n": len(pairs), "wo_sel_min_n": 10}
        ns["WO_FSPROBE_PAIRS"] = pairs
        ns["WO_FSPROBE_RIDGE"] = 1.0
        ns["WO_FSPROBE_FOLDS"] = 5
        mm = MockModel(pairs, seed=0)
        ns["model"] = mm
        ns["tokenizer"] = mm.tokenizer
        ns["wo_load_model"] = lambda tag: None
        ns["WO_ACTIVE_TAG"] = "base"
        import matplotlib
        matplotlib.use("Agg")
        _exec_cell(ns, "82l_wo_steering.py")
        out = ns["WO_STEER"]["base"]
        # resume: re-exec must reuse the capture pickle + sweep ckpt and reproduce the verdict.
        ns2 = _make_ns(art)
        _exec_cell(ns2, "76_wo_logic.py")
        ns2.update({k: ns[k] for k in ("CFG", "WO_FSPROBE_PAIRS", "WO_FSPROBE_RIDGE",
                                       "WO_FSPROBE_FOLDS", "model", "tokenizer",
                                       "wo_load_model", "WO_ACTIVE_TAG")})
        matplotlib.use("Agg")
        _exec_cell(ns2, "82l_wo_steering.py")
        out2 = ns2["WO_STEER"]["base"]
        # Experiment B on the cached residuals (pure CPU).
        _exec_cell(ns, "82m_wo_selectivity.py")
        selrows = ns["WO_SELECTIVITY"]["base"]["rows"]
        # WO#5.1 calibration (82n) — reuses the WO#5 capture pickle; needs the cell-82
        # overwrite-patch hook (82n asserts it) which the mock provides as a stub.
        def _mk_patch(vec_dev, pos):
            def hook(resid_post, hook):
                resid_post[:, pos, :] = vec_dev.to(resid_post.dtype)
                return resid_post
            return hook
        ns["_wo_mk_patch_hook"] = _mk_patch
        _exec_cell(ns, "82n_wo_steering_calibration.py")
        cal = ns["WO_STEER_CAL"]["base"]
        # WO#5.1b re-metric (82o) — flip-rate + logit-diff, reuses the same cache.
        _exec_cell(ns, "82o_wo_steering_remetric.py")
        remet = ns["WO_STEER_RO"]["base"]
        # WO#5.1c full-product metric (82p) — teacher-forced logprob + greedy decode.
        _exec_cell(ns, "82p_wo_steering_fullproduct.py")
        fp = ns["WO_STEER_FP"]["base"]
        # capture artifact really exists on disk (the CPU path Exp B depends on).
        cap = os.path.exists(os.path.join(art, "wo_steer_resid_base.pkl"))
        csv = os.path.exists(os.path.join(art, "results", "causal_steering_summary.csv"))
        calcsv = os.path.exists(os.path.join(art, "results", "causal_steering_calibration_summary.csv"))
        rocsv = os.path.exists(os.path.join(art, "results", "causal_steering_remetric_summary.csv"))
        fpcsv = os.path.exists(os.path.join(art, "results", "causal_steering_fullproduct_summary.csv"))
        return out, out2, selrows, cap, csv, cal, calcsv, remet, rocsv, fp, fpcsv
    finally:
        shutil.rmtree(art, ignore_errors=True)


def main():
    pairs = []
    rng = np.random.default_rng(0)
    seen = set()
    while len(pairs) < 80:
        B, C = int(rng.integers(20, 50)), int(rng.integers(20, 50))
        if (B, C) not in seen:
            seen.add((B, C)); pairs.append((B, C))

    print("\n[mock end-to-end]  expect CLEAN_NULL on C1, with the C4 reference DETECTING signal")
    out, out2, selrows, cap, csv, cal, calcsv, remet, rocsv, fp, fpcsv = _run(pairs)
    v = out["verdict"]
    check("capture pickle written to disk (Exp B CPU path)", cap)
    check("steering summary CSV written", csv)
    check("C4 reference passes (instrument proven on a routed-product site)", v["c4_ref_ok"])
    check("verdict is CLEAN_NULL (decodable-but-unused at C1 '=')", v["label"] == "CLEAN_NULL")
    check("'=' inject Δ is ~0 (|Δ| <= null tol 0.25)", abs(out["headline"]["inject"]["mean_delta"]) <= 0.25)
    # signal-DETECTION path exercised end-to-end: the C4 counterfactual cell has a large
    # positive Δ AND a CI clear of 0 at its peak layer (a real RECOVERS-style detection).
    c4cells = out["sweep"]["c4_ref"]["inject"]
    peakL = max(c4cells, key=lambda L: (c4cells[L]["mean_delta"] if c4cells[L]["mean_delta"] is not None else -1e9))
    check("C4 ref peak Δ is large + positive (>= recover thr)",
          c4cells[peakL]["mean_delta"] is not None and c4cells[peakL]["mean_delta"] >= out["recover_thr"])
    check("C4 ref peak CI excludes 0 (lo > 0) — detection path works end-to-end",
          c4cells[peakL]["ci"][0] is not None and c4cells[peakL]["ci"][0] > 0.0)
    # controls behave: random ~0 and shuffled does NOT raise the GT logit at '='.
    check("random-direction Δ at '=' is ~0", abs(out["headline"]["random"]["mean_delta"]) <= 0.25)
    check("shuffled-target Δ at '=' is ~0 (wrong product doesn't raise GT)",
          abs(out["headline"]["shuffled"]["mean_delta"]) <= 0.25)
    # erase produced a cell at each site.
    check("erase recorded at both C1 sites", set(out["erase"].keys()) == {"rparen", "equals"})
    # resume reproduced everything from the checkpoints (no recompute drift).
    check("resume reproduces the verdict label", out2["verdict"]["label"] == v["label"])
    check("resume reproduces the headline Δ exactly",
          out2["headline"]["inject"]["mean_delta"] == out["headline"]["inject"]["mean_delta"])
    # Experiment B: product IS decodable at C1 '=' (lives on dDecode); controls behave.
    head = [r for r in selrows if r["site"] == "C1_equals" and r["target"] == "B_times_C"][0]
    check("ExpB: product decodable at C1 '=' (R2_real > 0.6)", head["R2_real"] > 0.6)
    check("ExpB: shuffled-product collapses (R2 < 0.3)", head["R2_shuffled"] < 0.3)
    check("ExpB: selective over the Hewitt–Liang control task (> 0.3)", head["selectivity"] > 0.3)
    check("ExpB: C4 '=' product also decodable (positive anchor)",
          any(r["site"] == "C4_equals" and r["R2_real"] is not None and r["R2_real"] > 0.6 for r in selrows))

    print("\n[WO#5.1 calibration]  82n runs end-to-end on the reused capture + climbs the causal ladder")
    cv = cal["verdict"]
    check("82n: calibration summary CSV written", calcsv)
    check("82n: reused the WO#5 capture (no re-capture)", cal["reused_capture"] is True)
    check("82n: zero-ablation moves the GT logit (hook works, not INSTRUMENT_BROKEN)",
          cal["zeroabl"]["delta"] < 0 and cv["label"] != "INSTRUMENT_BROKEN")
    check("82n: full donor swap raises the donor logit (site is causal)", cal["swap"]["delta"] > 0)
    check("82n: verdict is CALIBRATED (mock readout reads the probe direction)", cv["label"] == "CALIBRATED")
    check("82n: a k_star was recorded", cv["k_star"] is not None)
    check("82n: C1 re-eval ran at k_star", cal["c1_reeval"] is not None)
    check("82n: mock C1 product axis is inert -> C1 not drivable (drives_c1 False)",
          cal["c1_reeval"]["drives_c1"] is False)

    print("\n[WO#5.1b re-metric]  82o re-scores with flip-rate + logit-diff across (layer x k)")
    rv = remet["verdict"]
    check("82o: re-metric summary CSV written", rocsv)
    check("82o: reused the WO#5 capture", remet["reused_capture"] is True)
    check("82o: full swap flips the answer (flip→donor high)", remet["swap"]["flip"] >= 0.5)
    check("82o: probe-direction inject flips at some (layer,k) -> CALIBRATED", rv["label"] == "CALIBRATED")
    check("82o: records a winning (layer*, k*)", rv["layer_star"] is not None and rv["k_star"] is not None)
    check("82o: C1 re-eval ran at the winner", remet["c1_reeval"] is not None)
    check("82o: mock C1 axis inert -> C1 not drivable", remet["c1_reeval"]["drives_c1"] is False)

    print("\n[WO#5.1c full-product]  82p re-scores with teacher-forced logprob + greedy-decode parse")
    check("82p: full-product summary CSV written", fpcsv)
    check("82p: reused the WO#5 capture", fp["reused_capture"] is True)
    check("82p: ran all four interventions (swap + 2 inject + C1)", len(fp["cells"]) == 4)
    check("82p: every cell has a full-product logprob delta + emit-rate",
          all(("fullprod_logprob_delta" in c and "emit_Pprime_rate" in c) for c in fp["cells"].values()))
    check("82p: produced a verdict", fp["verdict"] in
          ("INJECT_WORKS", "DEAD_DIRECTION", "SITE_OR_METRIC_STILL_BROKEN"))
    # mock tokenizer gives each product a DISTINCT token, so the shared-token diagnostic is ~0
    # (the real Llama run is where it ~1.0); the full swap moves the full-product logprob.
    check("82p: mock first-token NOT shared (distinct tokens) -> diagnostic ~0",
          fp["first_tok_shared_frac"] < 0.5)
    # NB the teacher-forced logprob is uniform in the mock (it only fills final-position
    # logits) -> Δ≈0; the REAL model has true intermediate logits. The greedy-decode
    # emit-P' metric IS represented in the mock and is the gold signal here.
    check("82p: full swap makes the model EMIT the full P' (greedy-decode; site is causal)",
          fp["cells"]["swap_C4"]["emit_Pprime_rate"] >= 0.5)
    _c1cell = next(c for name, c in fp["cells"].items() if name.startswith("inject_C1"))
    check("82p: mock C1 axis inert -> C1 emits P' far less than the full swap",
          _c1cell["emit_Pprime_rate"] < fp["cells"]["swap_C4"]["emit_Pprime_rate"])

    print("\n" + ("ALL PASS" if not _fails else f"FAILURES: {_fails}"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    sys.exit(main())
