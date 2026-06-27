"""CPU mock-execution of the WO#6 GPU cells (82r operand-route localization, 82s
path patching + dormant certification).

Execs cells/76_wo_logic.py for the pure helpers, then execs cells/82r and 82s
UNMODIFIED against a tiny TORCH (autograd) HookedTransformer stand-in — so the
whole orchestration runs end-to-end with NO GPU and NO real model: STR item build,
the teacher-forced multi-token logprob-difference metric, the fwd+bwd attribution
passes (this is why the mock must be a real autograd graph, unlike the numpy WO#5
mock), exact activation patching of the role x layer grid + top-K heads + MLPs,
paired bootstrap CIs, the localization verdict, path patching (direct vs mediated),
the Makelov decompose-and-compare, and every checkpoint/resume + JSON/PNG/CSV path.

The mock is engineered to reproduce the EXPECTED science so the assertions test that
the signals flow correctly, not merely that nothing crashes:
  * one mid-layer "mover" head copies the operand-position code to the '=' site,
    so head localization finds a sparse causal locus -> LOCALIZED_OPERAND_ROUTE;
  * the '=' residual carries the product magnitude along a direction ORTHOGONAL to
    the answer-token unembedding span (logit-inert), so the decodable product is a
    DORMANT subspace -> DORMANT_CERTIFIED.

Run:  python3 tests/test_wo6_mock.py
"""
import os
import sys
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
    def __init__(self, nL, nH, dh, d, device):
        self.n_layers, self.n_heads, self.d_head, self.d_model, self.device = nL, nH, dh, d, device


class _Hk:
    def __init__(self, name):
        self.name = name


class _Tok:
    """Char-level tokenizer for '( 0 + B ) * C =' surfaces and ' <int>' answers.
    Structural/digit tokens are fixed ids; each answer integer gets a UNIQUE
    reserved FIRST-token id (so distinct products are distinguishable), decoding to
    ' <first-digit>' with the remaining digits as plain digit tokens."""
    def __init__(self):
        self.sym = {"(": 1, ")": 2, "+": 3, "*": 4, "=": 5}
        self.inv = {0: "", 1: "(", 2: ")", 3: "+", 4: "*", 5: "="}
        for d in range(10):
            self.inv[6 + d] = str(d)               # plain digit tokens 6..15
        self._reserve = {}                         # int value -> reserved first-token id
        self._next = 100

    def _first_id(self, value):
        value = int(value)
        if value not in self._reserve:
            rid = self._next
            self._next += 1
            self._reserve[value] = rid
            self.inv[rid] = " " + str(value)[0]    # decodes to ' <first digit>'
        return self._reserve[value]

    def answer_ids(self, value):
        s = str(int(value))
        return [self._first_id(value)] + [6 + int(c) for c in s[1:]]

    def __call__(self, text, add_special_tokens=False):
        t = text.strip()
        return {"input_ids": self.answer_ids(int(t))}

    def decode(self, ids):
        return "".join(self.inv.get(int(i), "") for i in ids)


class TorchMock(torch.nn.Module):
    """Minimal differentiable HookedTransformer stand-in. resid stream carries a
    product 'code' (the answer's first-token direction) at the operand positions; a
    single mover head at L_MOVER copies it to the '=' site; the bump readout emits
    the nearest-code token. The '=' site also carries the product magnitude along a
    logit-INERT direction (orthogonal to the unembedding span) -> dormant subspace."""
    L_MOVER, H_MOVER = 1, 0

    def __init__(self, d=24, nL=4, nH=4, dh=6, V=512, seed=0):
        super().__init__()
        assert nH * dh == d and dh == 6
        self.cfg = _Cfg(nL, nH, dh, d, "cpu")
        self.tokenizer = _Tok()
        g = torch.Generator().manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        self.Rbasis = Q[:, :6].contiguous()        # 6-dim readout span (= mover W_O image)
        self.u_inert = Q[:, 6].contiguous()        # logit-inert magnitude direction
        c6 = torch.randn(V, 6, generator=g)
        c6 = c6 / c6.norm(dim=1, keepdim=True) * 2.0          # equal-norm token codes
        self.c6 = c6
        self.W_U = (self.Rbasis @ c6.T).contiguous()         # [d, V] unembedding (cols in Rspan)
        self.beta = 2.0
        # per-(layer,head) OV; the mover routes the Rspan code, the rest are inert.
        self.W_V, self.W_O = [], []
        for L in range(nL):
            wv, wo = [], []
            for h in range(nH):
                if L == self.L_MOVER and h == self.H_MOVER:
                    wv.append(self.Rbasis.clone().requires_grad_(True))      # [d,6] project to Rspan
                    wo.append(self.Rbasis.t().clone().requires_grad_(True))  # [6,d] lift back
                else:
                    wv.append((1e-3 * torch.randn(d, dh, generator=g)).requires_grad_(True))
                    wo.append((1e-3 * torch.randn(dh, d, generator=g)).requires_grad_(True))
            self.W_V.append(wv); self.W_O.append(wo)
        self.W_mlp_in = [(1e-3 * torch.randn(d, 16, generator=g)).requires_grad_(True) for _ in range(nL)]
        self.W_mlp_out = [(1e-3 * torch.randn(16, d, generator=g)).requires_grad_(True) for _ in range(nL)]
        self._fwd, self._bwd = [], []

    # ---- TransformerLens-ish hook surface --------------------------------------
    def reset_hooks(self):
        self._fwd, self._bwd = [], []

    def add_hook(self, name, fn, dir="fwd"):
        (self._fwd if dir == "fwd" else self._bwd).append((name, fn))

    @staticmethod
    def _match(hn, name):
        return hn(name) if callable(hn) else (hn == name)

    def _point(self, name, act, call_fwd, names_filter, cache):
        for hn, fn in list(self._fwd) + list(call_fwd or []):
            if self._match(hn, name):
                r = fn(act, _Hk(name))
                if r is not None:
                    act = r
        if cache is not None and (names_filter is None or names_filter(name)):
            cache[name] = act
        for hn, fn in self._bwd:
            if self._match(hn, name) and act.requires_grad:
                def _bh(g, fn=fn, nm=name):
                    fn(g, _Hk(nm)); return None
                act.register_hook(_bh)
        return act

    def to_tokens(self, prompt, prepend_bos=True):
        ids = [0] if prepend_bos else []
        for ch in prompt:
            if ch == " ":
                continue
            if ch in self.tokenizer.sym:
                ids.append(self.tokenizer.sym[ch])
            elif ch.isdigit():
                ids.append(6 + int(ch))
        return torch.tensor([ids], dtype=torch.long)

    @staticmethod
    def _parse(ids):
        ids = [int(i) for i in ids]
        if 5 not in ids:
            return None
        eq = max(i for i, t in enumerate(ids) if t == 5)
        try:
            plus = next(i for i, t in enumerate(ids) if t == 3)
            rparen = max(i for i, t in enumerate(ids[:eq]) if t == 2)
            star = next(i for i in range(rparen + 1, eq) if ids[i] == 4)
        except (StopIteration, ValueError):
            return None
        bdig = [i for i in range(plus + 1, rparen) if 6 <= ids[i] <= 15]
        cdig = [i for i in range(star + 1, eq) if 6 <= ids[i] <= 15]
        if not bdig or not cdig:
            return None
        B = int("".join(str(ids[i] - 6) for i in bdig))
        C = int("".join(str(ids[i] - 6) for i in cdig))
        return {"b_last": bdig[-1], "c_last": cdig[-1], "eq": eq, "B": B, "C": C}

    def _x0(self, ids):
        seq, d = len(ids), self.cfg.d_model
        x = 1e-2 * torch.stack([torch.sin(torch.arange(d) * (0.1 + 0.01 * int(t))) for t in ids])
        p = self._parse(ids)
        if p is not None:
            P = p["B"] * p["C"]
            fa = self.tokenizer.answer_ids(P)[0]
            code = self.W_U[:, fa]                                  # route code (in Rspan)
            x[p["b_last"]] = x[p["b_last"]] + code
            x[p["c_last"]] = x[p["c_last"]] + code
            x[p["eq"]] = x[p["eq"]] + (P / 100.0) * self.u_inert    # logit-inert magnitude (dominant linear P signal)
            self._ops = [p["b_last"], p["c_last"]]
            self._eq = p["eq"]
        else:
            self._ops, self._eq = [], seq - 1
        return x.unsqueeze(0)                                       # [1, seq, d]

    def _forward(self, tokens, call_fwd=None, names_filter=None, want_cache=False):
        ids = tokens[0].tolist()
        seq = len(ids)
        resid = self._x0(ids)                                      # [1, seq, d]
        cache = {} if want_cache else None
        nL, nH, dh, d = self.cfg.n_layers, self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model
        for L in range(nL):
            r2 = resid[0]                                          # [seq, d]
            z = torch.zeros(1, seq, nH, dh)
            zcols = []
            for h in range(nH):
                if L == self.L_MOVER and h == self.H_MOVER and self._ops:
                    moved = r2[self._ops].mean(0, keepdim=True) @ self.W_V[L][h]   # [1, dh]
                    col = torch.zeros(seq, dh)
                    col = col.clone()
                    col[self._eq] = moved[0]
                    zcols.append(col)
                else:
                    zcols.append((r2 @ self.W_V[L][h]) * 0.0)     # inert (keeps graph, ~0)
            z = torch.stack(zcols, dim=1).unsqueeze(0)            # [1, seq, nH, dh]
            z = self._point(f"blocks.{L}.attn.hook_z", z, call_fwd, names_filter, cache)
            attn_out = sum(z[0, :, h, :] @ self.W_O[L][h] for h in range(nH))      # [seq, d]
            resid = resid + attn_out.unsqueeze(0)
            mlp = ((resid[0] @ self.W_mlp_in[L]).relu() @ self.W_mlp_out[L] * 1e-3).unsqueeze(0)
            mlp = self._point(f"blocks.{L}.hook_mlp_out", mlp, call_fwd, names_filter, cache)
            resid = resid + mlp
            resid = self._point(f"blocks.{L}.hook_resid_post", resid, call_fwd, names_filter, cache)
        r6 = resid[0] @ self.Rbasis                              # [seq, 6]
        dist2 = ((r6[:, None, :] - self.c6[None, :, :]) ** 2).sum(-1)             # [seq, V]
        logits = (-self.beta * dist2).unsqueeze(0)              # [1, seq, V]
        return (logits, cache) if want_cache else logits

    def __call__(self, tokens):
        return self._forward(tokens)

    def run_with_hooks(self, tokens, fwd_hooks=None):
        return self._forward(tokens, call_fwd=fwd_hooks)

    def run_with_cache(self, tokens, names_filter=None):
        logits, cache = self._forward(tokens, names_filter=names_filter, want_cache=True)
        return logits, cache


# ------------------------------------------------------------------ infra stubs
def _make_ns(art_dir):
    import json as _json
    import pickle as _pickle
    ns = {"log": lambda *a, **k: None, "print": print}

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

    def _mk_patch_hook(vec_dev, pos):
        def hook(resid_post, hook):
            resid_post[:, pos, :] = vec_dev.to(resid_post.dtype)
            return resid_post
        return hook
    ns["_wo_mk_patch_hook"] = _mk_patch_hook
    return ns


def _exec_cell(ns, fname):
    path = os.path.join(CELLS, fname)
    exec(compile(open(path).read(), path, "exec"), ns)


def _cfg():
    return {
        "wo_loc_tags": ["base"], "wo_loc_n": 12, "wo_loc_targets": ["C", "B"],
        "wo_loc_seed": 1, "wo_loc_head_topk": 4, "wo_loc_nboot": 200,
        "wo_loc_gen_sample": 4, "wo_loc_gen_k": 6, "wo_loc_recover_thr": 0.3,
        "wo_loc_roles": ["plus", "b_last", "rparen", "star", "c_last", "equals"],
        "wo_pp_tags": ["base"], "wo_pp_n": 6, "wo_pp_head_topk": 3,
        "wo_pp_direct_thr": 0.5, "wo_dorm_layers": None,
    }


def _seed_ns(ns, pairs, mm):
    ns["CFG"] = _cfg()
    ns["WO_FSPROBE_PAIRS"] = pairs
    ns["model"] = mm
    ns["tokenizer"] = mm.tokenizer
    ns["wo_load_model"] = lambda tag: None
    ns["WO_ACTIVE_TAG"] = "base"
    import matplotlib
    matplotlib.use("Agg")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    # ~50 distinct two-digit pairs: 82r uses the first wo_loc_n (=12) for the (costly)
    # patching; the FORWARD-only dormant decode (82s) uses all of them (n >> readout dim).
    _rng = np.random.default_rng(0)
    pairs, _seen = [], set()
    while len(pairs) < 50:
        B, C = int(_rng.integers(20, 50)), int(_rng.integers(20, 50))
        if (B, C) not in _seen:
            _seen.add((B, C)); pairs.append((B, C))
    art = tempfile.mkdtemp(prefix="wo6mock_")
    ns = _make_ns(art)
    _exec_cell(ns, "76_wo_logic.py")
    mm = TorchMock(seed=0)
    _seed_ns(ns, pairs, mm)

    print("\n[WO#6 82r]  STR attribution + exact-patch operand-route localization")
    _exec_cell(ns, "82r_wo_operand_localization.py")
    out = ns["WO_LOC"]["base"]
    check("82r: produced a localization output for base", out is not None)
    check("82r: D_clean >> 0 and D_corrupt << 0 (metric is correctly signed)",
          out["D_clean_mean"] > 0 and out["D_corrupt_mean"] < 0)
    check("82r: operand-position exact recovery passes threshold",
          out["exact"]["operand_pos_recovery"] >= ns["CFG"]["wo_loc_recover_thr"])
    check("82r: a sparse causal head set is found -> LOCALIZED_OPERAND_ROUTE",
          out["verdict"]["label"] == "LOCALIZED_OPERAND_ROUTE")
    mover = (mm.L_MOVER, mm.H_MOVER)
    check("82r: the mover head is the best causal head",
          tuple(out["exact"]["best_head"]) == mover)
    check("82r: attribution-vs-exact agreement computed (spearman)",
          out["attribution_vs_exact_agreement"]["spearman"] is not None)
    check("82r: paired operand-vs-equals CI emitted",
          out["exact"]["paired_operand_vs_equals_ci"] is not None)
    check("82r: gold flip-rate computed (float in [0,1])",
          isinstance(out["exact"]["gold_flip_rate"], float)
          and 0.0 <= out["exact"]["gold_flip_rate"] <= 1.0)
    check("82r: heatmap PNG written", os.path.exists(os.path.join(art, "results", "operand_position_patch_base.png")))
    check("82r: per-tag JSON written", os.path.exists(os.path.join(art, "results", "operand_position_patch_base.json")))
    check("82r: summary CSV written", os.path.exists(os.path.join(art, "results", "operand_localization_summary.csv")))

    # resume: re-exec must reuse the build/attr/exact checkpoints and reproduce the verdict.
    ns2 = _make_ns(art)
    _exec_cell(ns2, "76_wo_logic.py")
    _seed_ns(ns2, pairs, TorchMock(seed=0))
    _exec_cell(ns2, "82r_wo_operand_localization.py")
    check("82r: resume reproduces the verdict from checkpoints",
          ns2["WO_LOC"]["base"]["verdict"]["label"] == out["verdict"]["label"])

    print("\n[WO#6 82s]  path patching (direct vs mediated) + Makelov dormant certification")
    _exec_cell(ns, "82s_wo_pathpatch_dormant.py")
    pp = ns["WO_PP"]["base"]
    dorm = ns["WO_DORM"]["base"]
    check("82s: path-patch JSON written", os.path.exists(os.path.join(art, "results", "head_path_patch_base.json")))
    check("82s: every head classified (DIRECT/MEDIATED/INCONCLUSIVE)",
          all(h["classification"] in ("DIRECT", "MEDIATED", "INCONCLUSIVE") for h in pp["heads"]))
    check("82s: the mover head has a real total recovery",
          any(tuple((h["layer"], h["head"])) == mover and (h["total_recovery"] or 0) > 0.3
              for h in pp["heads"]))
    check("82s: dormant-certification JSON written",
          os.path.exists(os.path.join(art, "results", "dormant_certification_base.json")))
    check("82s: decodable product at '=' is certified DORMANT (logit-inert)",
          dorm["headline_verdict"] == "DORMANT_CERTIFIED")
    hl = dorm["per_layer"][-1]
    check("82s: full decode R^2 high but logit-affecting subspace recovers far less",
          (hl["R2_full"] or 0) > 0.5 and (hl["R2_row"] or 0) < (hl["R2_full"] or 0))
    check("82s: decode direction is dominantly logit-inert",
          (hl["inert_share"] or 0) > 0.5)
    check("82s: references the WO#5 nulls (B2 dissociation)",
          "WO5.1d_full_swap" in dorm["references_WO5"])

    # resume path patching.
    ns3 = _make_ns(art)
    _exec_cell(ns3, "76_wo_logic.py")
    _seed_ns(ns3, pairs, TorchMock(seed=0))
    _exec_cell(ns3, "82r_wo_operand_localization.py")
    _exec_cell(ns3, "82s_wo_pathpatch_dormant.py")
    check("82s: resume reproduces the dormant verdict",
          ns3["WO_DORM"]["base"]["headline_verdict"] == dorm["headline_verdict"])

    print("\n" + ("ALL PASS" if not _fails else f"FAILURES: {_fails}"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    sys.exit(main())
