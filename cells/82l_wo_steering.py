# ============================================================================
# Phase 6 / WORK ORDER #5 — EXPERIMENT A: CONTRAST-FREE CAUSAL STEERING (GPU).
# ----------------------------------------------------------------------------
# Turns the correlational decodability result (B decodable at ')', B·C decodable
# at '=' on the FAILING zero-shot C1 surface, R^2≈0.96–0.98) into a CAUSAL test —
# WITHOUT the argmax contrast that returned n_used=0 in the salvage (zero-shot C1
# output is B-invariant, so there was nothing to denoise). The fix: drop argmax,
# score the GROUND-TRUTH product's first-answer-token LOGIT. Every test item then
# yields a number regardless of its argmax, so the failing regime has full n.
#
# WHAT IT DOES, for C1 '( 0 + B ) * C =' at the ')' site (target = operand B) and
# the '=' site (target = product B·C):
#   1. Capture hook_resid_post at both sites over WO_FSPROBE_PAIRS (== WO_PAIRS by
#      default), all layers, plus the C4 '( B * C ) =' '=' residual (the positive
#      reference). CACHE the raw residual matrices to disk -> Experiment B (82m) and
#      every re-run are CPU-only.
#   2. TRAIN/TEST split (seeded). Per (site, target, layer) fit the SAME dual-ridge
#      probe as wo_cv_r2 on the TRAIN half (wo_fit_ridge_probe) -> a unit direction
#      w-hat + a value<->coordinate map. NEVER steer with a direction fit on the
#      steered item.
#   3. On held-out TEST items, ACTIVATION-STEER the clean run at one (layer, site)
#      via run_with_hooks. Because we steer the CLEAN run, resid_post at the site ==
#      the cached clean residual, so the whole steering delta is precomputed on CPU
#      from wo_inject_to_target and the hook is a pure additive patch (mirrors the
#      validated cell-75 / _wo_mk_patch_hook idiom).
#        INJECT (headline): write the correct value in along w-hat. Sweep the layer.
#        ERASE  (LEACE-ish): steer the probe coordinate to the train mean (kills the
#                            probe's predictive variance along w-hat). At the peak layer.
#   4. METRIC: Δ = logit_GT(intervened) − logit_GT(clean-C1 baseline), GT = the
#      product's first answer token (leading-space convention, cell 75). Mean Δ over
#      TEST with a PAIRED bootstrap CI (wo_paired_delta_ci) + flip-rate-to-GT-product.
#   5. CONTROLS (each its own row, swept across layers for the PNG):
#        random-direction (norm-matched) -> Δ ≈ 0  (effect is DIRECTION-specific)
#        shuffled-target  (inject a wrong product) -> GT logit must NOT rise
#        C4 positive reference -> the inject mechanism + per-pair logit metric DO
#          move a routed-product logit at a '=' site. NB the literal "inject the TRUE
#          product at C4's =" has a CEILING (C4 already emits it, Δ≈0 would falsely read
#          as a dead instrument), so the reference injects a COUNTERFACTUAL product at
#          C4's = and shows ITS logit rises — ceiling-free. (The literal version is also
#          reported, for completeness, as c4_true_inject.)
#   6. VERDICT (pure, tested) wo_steering_verdict: RECOVERS / CLEAN_NULL / INCONCLUSIVE.
#
# Deliverables: causal_steering_{base,instruct}.json + causal_steering_summary.csv
# + causal_steering_layersweep_{tag}.png. Checkpointed per (model, site, layer),
# resumable; capture pickle guards the GPU. Honest-null governing rule: a clean null
# with the C4 reference passing IS the finding — do not over-claim.
# ============================================================================
import json
import numpy as np
import torch

assert "WO_FSPROBE_PAIRS" in globals() and "wo_fit_ridge_probe" in globals() \
    and "wo_inject_to_target" in globals() and "wo_logit_diff_gt" in globals() \
    and "wo_steering_verdict" in globals() and "wo_locate_c1_sites" in globals() \
    and "wo_paired_delta_ci" in globals() and "wo_shuffle_control" in globals(), (
    "WO#5 steering needs WO_FSPROBE_PAIRS (cell 82d) + cell-76 WO#5 helpers "
    "(wo_fit_ridge_probe / wo_inject_to_target / wo_logit_diff_gt / wo_steering_verdict).")

# ---- knobs (all CFG so a re-run can retune without editing the cell) ----------
CFG.setdefault("wo_steer_tags", ["base", "instruct"])
CFG.setdefault("wo_steer_seed", 707)
CFG.setdefault("wo_steer_test_frac", 0.5)     # TRAIN/TEST split fraction held out for steering.
CFG.setdefault("wo_steer_n", len(WO_FSPROBE_PAIRS))   # cap captured pairs (default: all shared pairs).
CFG.setdefault("wo_steer_layer_stride", 2)    # intervention layer-sweep stride (every k-th layer).
CFG.setdefault("wo_steer_min_test_n", 40)     # warn/guard if usable TEST items < this.
CFG.setdefault("wo_steer_nboot", 10000)       # paired-bootstrap resamples.
CFG.setdefault("wo_steer_recover_thr", float(globals().get("WO_STEER_RECOVER_THR", 0.5)))
CFG.setdefault("wo_steer_null_tol", float(globals().get("WO_STEER_NULL_TOL", 0.25)))        # |Δ| within this == null.
CFG.setdefault("wo_steer_ci_halfwidth_tol", float(globals().get("WO_STEER_NULL_TOL", 0.25)))  # CLEAN_NULL CI tightness.

_st_renderC1 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C1"]
_st_renderC4 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C4"]
# (site key -> the per-item target VALUE and a label). The metric token is ALWAYS the
# product's first token (does the injected value PROPAGATE to the product output?).
WO_STEER_SITES = [("rparen", "operand_B"), ("equals", "product_BC")]
WO_STEER_PAIRS = WO_FSPROBE_PAIRS[: int(CFG["wo_steer_n"])]
WO_STEER_PAIR_SHA = wo_stim_hash(WO_STEER_PAIRS)
log(f"WO#5 steering: {len(WO_STEER_PAIRS)} pairs (sha {WO_STEER_PAIR_SHA[:12]}), "
    f"test_frac={CFG['wo_steer_test_frac']}, layer stride={CFG['wo_steer_layer_stride']}, "
    f"recover_thr={CFG['wo_steer_recover_thr']}.")


def _st_first_tok_id(value):
    """The id of the FIRST answer token for an integer `value`, using the leading-
    space convention the few-shot battery trains and cell-75 validated (' 1081' ->
    [' 108', '1'] -> first id is the space-led chunk). add_special_tokens=False (no
    BOS in a continuation)."""
    ids = tokenizer(" " + str(int(value)), add_special_tokens=False)["input_ids"]
    return int(ids[0])


# ----------------------------------------------------------------------------
# Phase A — CAPTURE: clean residuals at both C1 sites + the C4 '=' reference, all
#   layers, over the shared pairs. Cached to ONE pickle per tag (Experiment B and
#   every steering re-run read it; the model is only needed to (re)build it).
# ----------------------------------------------------------------------------
@torch.no_grad()
def _st_capture(tag, pairs=None, ck=None):
    # `pairs`/`ck` default to the band-(20,49) WO_STEER_PAIRS capture (backward-compatible);
    # WO#6 band-robustness passes a second band's pairs + a distinct cache key.
    pairs = WO_STEER_PAIRS if pairs is None else list(pairs)
    _sha = wo_stim_hash(pairs)
    ck = (f"wo_steer_resid_{tag}" if ck is None else ck)
    if has_artifact(ck, "pickle"):
        _b = load_pickle(ck)
        if _b.get("pair_sha") == _sha:
            log(f"WO#5 steering[{tag}]: residual capture '{ck}' cached — reused (CPU path).")
            return _b
        log(f"WO#5 steering[{tag}]: cached capture pair_sha {str(_b.get('pair_sha'))[:12]} != "
            f"current {_sha[:12]} — re-capturing (pairs changed).")

    wo_load_model(tag)
    nL = model.cfg.n_layers
    items = []                                  # one record per usable pair (C1 ok AND C4 ok)
    n_skip_c1, n_skip_c4 = 0, 0
    c4_argmax_hits, c4_seen = 0, 0
    for (B, C) in pairs:
        p1 = _st_renderC1(B, C)
        assert p1.endswith("=") and not p1.endswith(" ")            # Hazard #2 (Llama tok).
        tok1 = model.to_tokens(p1)
        strs1 = [tokenizer.decode([t]).strip() for t in tok1[0].tolist()]
        loc = wo_locate_c1_sites(strs1, B, C)
        if not loc["ok"]:
            n_skip_c1 += 1
            continue
        p4 = _st_renderC4(B, C)
        assert p4.endswith("=") and not p4.endswith(" ")
        tok4 = model.to_tokens(p4)
        strs4 = [tokenizer.decode([t]).strip() for t in tok4[0].tolist()]
        eq4 = wo_last_index(strs4, "=")
        if eq4 is None:
            n_skip_c4 += 1
            continue

        gt_id = _st_first_tok_id(B * C)
        # ---- C1 clean pass: site residuals (all layers) + the baseline GT logit ----
        lg1, cache1 = model.run_with_cache(
            tok1, names_filter=lambda nm: nm.endswith("hook_resid_post"))
        resid_rp = np.stack([cache1[f"blocks.{L}.hook_resid_post"][0, loc["rparen"], :]
                             .float().cpu().numpy() for L in range(nL)]).astype(np.float16)
        resid_eq = np.stack([cache1[f"blocks.{L}.hook_resid_post"][0, loc["equals"], :]
                             .float().cpu().numpy() for L in range(nL)]).astype(np.float16)
        base_gt = float(lg1[0, -1, gt_id].item())
        del cache1, lg1
        # ---- C4 clean pass: '=' residuals (all layers) + the clean C4 GT logit ----
        lg4, cache4 = model.run_with_cache(
            tok4, names_filter=lambda nm: nm.endswith("hook_resid_post"))
        resid_c4 = np.stack([cache4[f"blocks.{L}.hook_resid_post"][0, eq4, :]
                             .float().cpu().numpy() for L in range(nL)]).astype(np.float16)
        c4_clean_gt = float(lg4[0, -1, gt_id].item())
        c4_argmax = int(lg4[0, -1].argmax().item())
        del cache4, lg4
        c4_seen += 1
        c4_argmax_hits += int(c4_argmax == gt_id)

        items.append({
            "B": int(B), "C": int(C), "gt_id": gt_id,
            "tok1": [int(t) for t in tok1[0].tolist()], "rparen": int(loc["rparen"]),
            "equals": int(loc["equals"]), "base_gt": base_gt,
            "tok4": [int(t) for t in tok4[0].tolist()], "eq4": int(eq4),
            "c4_clean_gt": c4_clean_gt, "c4_argmax": c4_argmax,
            "resid_rparen": resid_rp, "resid_equals": resid_eq, "resid_c4_eq": resid_c4,
        })

    bundle = {
        "tag": tag, "n_layers": nL, "n_used": len(items),
        "n_skipped_c1": n_skip_c1, "n_skipped_c4": n_skip_c4,
        "c4_firsttok_argmax_match_rate": (c4_argmax_hits / c4_seen) if c4_seen else None,
        "pair_sha": _sha, "items": items,
    }
    save_pickle(ck, bundle)
    log(f"WO#5 steering[{tag}]: captured {len(items)} usable items "
        f"(skip C1={n_skip_c1}, C4={n_skip_c4}); C4 first-tok argmax match="
        f"{bundle['c4_firsttok_argmax_match_rate']} (sanity for the GT-token convention).")
    return bundle


# ----------------------------------------------------------------------------
# Phase B — STEER: per (site, layer) fit TRAIN probes, intervene on TEST items.
# ----------------------------------------------------------------------------
def _st_split(n, seed):
    """Deterministic TRAIN/TEST index split over the captured items."""
    idx = np.random.default_rng(int(seed)).permutation(n)
    n_test = max(1, int(round(n * float(CFG["wo_steer_test_frac"]))))
    return idx[n_test:], idx[:n_test]               # (train_idx, test_idx)


def _st_derange(values, seed):
    """A 'wrong value' control with NO value-level fixed point: starts from
    wo_shuffle_control (reused, as the work order specifies) then swaps out any i where
    the shuffled value equals the original — so a shuffled-target / counterfactual control
    never accidentally injects an item's OWN value (which would leak a true-target effect
    into the control). Deterministic given `seed`."""
    v = np.asarray(values)
    s = np.array(wo_shuffle_control(v, seed=int(seed)))
    n = v.shape[0]
    for _pass in range(3):
        fixed = [i for i in range(n) if s[i] == v[i]]
        if not fixed:
            break
        for i in fixed:
            j = (i + 1) % n
            s[i], s[j] = s[j], s[i]
    return s


def _st_mk_add_hook(delta_dev, pos):
    """resid_post[:, pos, :] += delta  (precomputed steering delta on device). The
    additive form of the validated cell-75 patch hook: because we steer the CLEAN
    run, the live resid at `pos` equals the cached clean residual, so this exactly
    realizes wo_inject_to_target(clean_resid, w-hat, target_coord)."""
    def hook(resid_post, hook):
        resid_post[:, pos, :] = resid_post[:, pos, :] + delta_dev.to(resid_post.dtype)
        return resid_post
    return hook


@torch.no_grad()
def _st_logit_at(tokens_list, hook_layer, delta_vec, pos, tok_id):
    """One steered forward pass: add `delta_vec` (np, d) at (hook_layer, pos) of the
    clean `tokens_list`, return (logit at tok_id, argmax id) at the final position.
    delta_vec=None -> a clean pass (no hook)."""
    tok = torch.tensor([tokens_list], device=model.cfg.device, dtype=torch.long)
    if delta_vec is None:
        lg = model(tok)
    else:
        dv = torch.tensor(np.asarray(delta_vec, dtype=np.float32), device=model.cfg.device)
        lg = model.run_with_hooks(
            tok, fwd_hooks=[(f"blocks.{hook_layer}.hook_resid_post", _st_mk_add_hook(dv, pos))])
    row = lg[0, -1]
    return float(row[int(tok_id)].item()), int(row.argmax().item())


def _st_fit(items, train_idx, resid_key, layer, y_of):
    """Fit the TRAIN probe for one (site, layer): X = train residuals at `layer`,
    y = y_of(item). Returns the wo_fit_ridge_probe dict (or None)."""
    X = np.stack([items[i][resid_key][layer].astype(np.float32) for i in train_idx])
    y = np.array([y_of(items[i]) for i in train_idx], dtype=float)
    return wo_fit_ridge_probe(X, y)


def _st_aggregate(inter, base, gt_ids, nboot, seed):
    """mean Δ, paired-bootstrap CI, flip-rate from per-item (intervened_logit,
    argmax_id) lists `inter` and the per-item baseline logits `base`."""
    a = np.array([t[0] for t in inter], dtype=float)
    b = np.asarray(base, dtype=float)
    flips = np.array([1.0 if t[1] == g else 0.0 for t, g in zip(inter, gt_ids)])
    ci = wo_paired_delta_ci(a, b, n_boot=int(nboot), seed=int(seed))
    return {"mean_delta": float(a.mean() - b.mean()), "ci": [ci[0], ci[1]],
            "flip_rate": float(flips.mean()), "n": int(a.size)}


@torch.no_grad()
def _st_sweep(tag, bundle):
    ck = f"wo_steer_sweep_{tag}"
    nL = bundle["n_layers"]
    items = bundle["items"]
    n = len(items)
    seed = int(CFG["wo_steer_seed"])
    nboot = int(CFG["wo_steer_nboot"])
    thr = float(CFG["wo_steer_recover_thr"])
    train_idx, test_idx = _st_split(n, seed)
    L_grid = sorted(set(range(0, nL, int(CFG["wo_steer_layer_stride"]))) | {nL - 1})
    n_test = len(test_idx)

    # PRE-REGISTER the headline layer per site by TRAIN decodability (the layer where the
    # probe fits the target best on the TRAIN half), so the verdict is judged at a layer
    # chosen WITHOUT peeking at the TEST inject Δ — no winner's-curse / selection bias from
    # taking the argmax-Δ layer on the same data the CI is computed on. The full layer sweep
    # is still reported, but as EXPLORATORY (the PNG + exploratory_peak_layer), not the claim.
    def _train_decod_peak(resid_key, y_train):
        best, bestL = None, None
        for L in L_grid:
            Xtr = np.stack([items[i][resid_key][L].astype(np.float32) for i in train_idx])
            r2 = wo_cv_r2(Xtr, y_train, folds=5, ridge=1.0)
            if r2 is not None and (best is None or r2 > best):
                best, bestL = r2, int(L)
        return (bestL if bestL is not None else int(L_grid[len(L_grid) // 2])), best
    _yB_tr = np.array([items[i]["B"] for i in train_idx], float)
    _yP_tr = np.array([items[i]["B"] * items[i]["C"] for i in train_idx], float)
    regL_rp, regR2_rp = _train_decod_peak("resid_rparen", _yB_tr)
    regL_eq, regR2_eq = _train_decod_peak("resid_equals", _yP_tr)
    regL_c4, regR2_c4 = _train_decod_peak("resid_c4_eq", _yP_tr)   # C4 ref layer, pre-registered too
    reg_layer = {"rparen": regL_rp, "equals": regL_eq, "c4_ref": regL_c4}
    reg_r2 = {"rparen": regR2_rp, "equals": regR2_eq, "c4_ref": regR2_c4}

    # per-item TEST targets / metric tokens (fixed; computed once).
    gt_ids = [items[i]["gt_id"] for i in test_idx]
    base_c1 = [items[i]["base_gt"] for i in test_idx]              # clean-C1 GT logit baseline
    Bv = np.array([items[i]["B"] for i in test_idx], float)
    Cv = np.array([items[i]["C"] for i in test_idx], float)
    prod = Bv * Cv
    # SHUFFLED targets (a wrong value to inject) per site — deranged so no item gets its
    # OWN value (a value-level fixed point would contaminate the control toward inject).
    shuf_operand = _st_derange(Bv, seed + 1)
    shuf_product = _st_derange(prod, seed + 2)
    # C4 COUNTERFACTUAL reference: a permuted product P' to inject at C4's '=' and
    # whose OWN first-token logit we score (ceiling-free positive control).
    pref_vals = _st_derange(prod, seed + 3)
    pref_ids = [_st_first_tok_id(v) for v in pref_vals]
    # clean C4 logit at P' (recompute per item from the model? no — score at capture
    # time would need P' which depends on the split; instead score it here with a
    # single clean C4 pass per test item, cached by layer-independent design).
    # (One clean C4 forward per test item; cheap, cached implicitly via the sweep ckpt.)

    # The cached cells (and c4_ref_base) are computed for a SPECIFIC split + control draws,
    # all seed-derived, on a SPECIFIC pair set. wo_steer_seed / test_frac / the pairs are
    # CFG-tunable, so the resume guard MUST fingerprint them — else a seed retune silently
    # serves stale cells scored against new gt_ids/pref_ids (a correctness drift).
    _fp = {"n_test": n_test, "L_grid": L_grid, "seed": seed,
           "test_frac": float(CFG["wo_steer_test_frac"]), "pair_sha": WO_STEER_PAIR_SHA}
    done = {}
    if has_artifact(ck, "json"):
        prev = load_json(ck)
        if all(prev.get(k) == _fp[k] for k in _fp):
            done = prev.get("rows", {})
            log(f"WO#5 steering[{tag}]: resuming sweep ({len(done)} (site,cond,layer) cells done).")
        else:
            log(f"WO#5 steering[{tag}]: stale sweep ckpt (seed/test_frac/pairs/n_test/L_grid changed) — recompute.")

    def _save():
        save_json(ck, {**_fp, "rows": done})

    # ---- clean C4 logit at P' baseline (per test item), needed for the C4 reference ----
    if "c4_ref_base" in done:
        c4_ref_base = done["c4_ref_base"]
    else:
        c4_ref_base = []
        for j, i in enumerate(test_idx):
            v, _ = _st_logit_at(items[i]["tok4"], 0, None, items[i]["eq4"], pref_ids[j])
            c4_ref_base.append(v)
        done["c4_ref_base"] = c4_ref_base
        _save()

    def _cell(site, cond, L, y_target, metric_ids, base_logits, resid_key, fit_for):
        """Compute (or reuse) one swept cell. `fit_for` returns the probe for (site,L);
        `y_target[j]` is the VALUE injected for test item j; metric_ids[j]/base_logits[j]
        are the token/baseline the Δ is scored against."""
        key = f"{site}|{cond}|{L}"
        if key in done:
            return done[key]
        fit = fit_for(L)
        inter = []
        if fit is None:
            res = {"mean_delta": None, "ci": [None, None], "flip_rate": None, "n": 0, "skipped": "no_fit"}
            done[key] = res
            return res
        what = fit["direction"]
        for j, i in enumerate(test_idx):
            clean = items[i][resid_key][L].astype(np.float32)
            coord = wo_probe_coord_for_value(fit, y_target[j])
            steered = wo_inject_to_target(clean, what, coord)
            delta = steered - clean
            if cond == "random":
                u = np.random.default_rng(seed + 5000 + i).standard_normal(clean.shape[0])
                u = u / (np.linalg.norm(u) + 1e-12)
                delta = float(np.linalg.norm(delta)) * u            # NORM-matched random direction
            elif cond == "erase":
                delta = wo_inject_to_target(clean, what, wo_probe_mean_coord(fit)) - clean
            tok_list = items[i]["tok4"] if site == "c4_ref" else items[i]["tok1"]
            pos = items[i]["eq4"] if site == "c4_ref" else items[i][{"rparen": "rparen", "equals": "equals"}[site]]
            inter.append(_st_logit_at(tok_list, L, delta, pos, metric_ids[j]))
        res = _st_aggregate(inter, base_logits, metric_ids, nboot, seed)
        done[key] = res
        return res

    # site -> per-test target value + metric token + baseline + which residual to steer.
    site_spec = {
        "rparen": dict(y_true=Bv, y_shuf=shuf_operand, metric_ids=gt_ids, base=base_c1,
                       resid_key="resid_rparen"),
        "equals": dict(y_true=prod, y_shuf=shuf_product, metric_ids=gt_ids, base=base_c1,
                       resid_key="resid_equals"),
    }
    fit_cache = {}

    def _fit_for(site, resid_key, y_arr_train_value):
        def f(L):
            k = (site, L)
            if k not in fit_cache:
                fit_cache[k] = _st_fit(items, train_idx, resid_key, L, y_arr_train_value)
            return fit_cache[k]
        return f

    sweep = {}
    # ---- the two C1 sites: inject / random / shuffled across the grid ----
    for site in ("rparen", "equals"):
        sp = site_spec[site]
        y_train_value = (lambda it: it["B"]) if site == "rparen" else (lambda it: it["B"] * it["C"])
        fit_for = _fit_for(site, sp["resid_key"], y_train_value)
        sweep[site] = {"inject": {}, "random": {}, "shuffled": {}}
        for L in L_grid:
            sweep[site]["inject"][L] = _cell(site, "inject", L, sp["y_true"], sp["metric_ids"],
                                             sp["base"], sp["resid_key"], fit_for)
            sweep[site]["random"][L] = _cell(site, "random", L, sp["y_true"], sp["metric_ids"],
                                             sp["base"], sp["resid_key"], fit_for)
            sweep[site]["shuffled"][L] = _cell(site, "shuffled", L, sp["y_shuf"], sp["metric_ids"],
                                              sp["base"], sp["resid_key"], fit_for)
            _save()
            _i = sweep[site]["inject"][L]["mean_delta"]
            log(f"WO#5 steering[{tag}] {site} L{L}/{nL - 1}: inject Δ="
                f"{'n/a' if _i is None else f'{_i:+.3f}'} "
                f"rand Δ={_fmt_delta(sweep[site]['random'][L])} shuf Δ={_fmt_delta(sweep[site]['shuffled'][L])}")

    # ---- C4 counterfactual positive reference (inject P' at C4 '='; score P') ----
    fit_c4 = _fit_for("c4_ref", "resid_c4_eq", lambda it: it["B"] * it["C"])
    sweep["c4_ref"] = {"inject": {}}
    for L in L_grid:
        sweep["c4_ref"]["inject"][L] = _cell("c4_ref", "inject", L, pref_vals, pref_ids,
                                            c4_ref_base, "resid_c4_eq", fit_c4)
        _save()
    # ---- ERASE at the PRE-REGISTERED (train-decodability) layer of each C1 site ----
    erase = {}
    for site in ("rparen", "equals"):
        sp = site_spec[site]
        eraseL = reg_layer[site]
        y_train_value = (lambda it: it["B"]) if site == "rparen" else (lambda it: it["B"] * it["C"])
        fit_for = _fit_for(site, sp["resid_key"], y_train_value)
        erase[site] = {"layer": int(eraseL),
                       "result": _cell(site, "erase", eraseL, sp["y_true"], sp["metric_ids"],
                                       sp["base"], sp["resid_key"], fit_for)}
        _save()

    # exploratory (test-set) argmax-Δ layer per site — REPORTED but NOT used for the verdict.
    expl_peak = {}
    for site in ("rparen", "equals"):
        cand = [(L, sweep[site]["inject"][L]["mean_delta"]) for L in L_grid
                if sweep[site]["inject"][L]["mean_delta"] is not None]
        expl_peak[site] = int(max(cand, key=lambda t: t[1])[0]) if cand else None

    return {"sweep": sweep, "erase": erase, "L_grid": L_grid, "n_test": n_test,
            "n_train": len(train_idx), "train_idx": [int(i) for i in train_idx],
            "test_idx": [int(i) for i in test_idx],
            "reg_layer": reg_layer, "reg_decodability_r2": reg_r2,
            "exploratory_peak_layer": expl_peak}


def _fmt_delta(cell):
    d = cell.get("mean_delta")
    return "n/a" if d is None else f"{d:+.3f}"


# ----------------------------------------------------------------------------
# Phase C — verdict, deliverables (JSON + CSV + layer-sweep PNG), per tag.
# ----------------------------------------------------------------------------
def _st_headline_cells(sweep, regL):
    """The '=' site / product inject + control cells at the PRE-REGISTERED layer regL
    (chosen by TRAIN decodability, not the test Δ). Returns (inject, random, shuffled)."""
    eq = sweep["equals"]
    return eq["inject"].get(regL), eq["random"].get(regL), eq["shuffled"].get(regL)


def _st_c4_peak(sweep):
    cand = [c["mean_delta"] for c in sweep["c4_ref"]["inject"].values() if c["mean_delta"] is not None]
    return max(cand) if cand else None


WO_STEER = {}
for _tag in CFG["wo_steer_tags"]:
    _bundle = _st_capture(_tag)
    if _bundle["n_used"] < int(CFG["wo_steer_min_test_n"]) * 2:
        log(f"WO#5 steering[{_tag}]: only {_bundle['n_used']} usable items — under-powered, "
            f"results reported with a warning.")
    _sw = _st_sweep(_tag, _bundle)
    _sweep = _sw["sweep"]
    _hL = _sw["reg_layer"]["equals"]                  # pre-registered (train-decodability) headline layer
    _inj, _rnd, _shf = _st_headline_cells(_sweep, _hL)
    _c4L = _sw["reg_layer"]["c4_ref"]                  # pre-registered C4 reference layer (no max-bias)
    _c4cell = _sweep["c4_ref"]["inject"].get(_c4L)
    _c4 = _c4cell["mean_delta"] if _c4cell else None   # verdict precondition uses the PRE-REGISTERED layer
    _c4_peak = _st_c4_peak(_sweep)                     # exploratory max over layers (reported only)

    _verdict = wo_steering_verdict(
        delta_inject=(_inj["mean_delta"] if _inj else None),
        ci_inject=(tuple(_inj["ci"]) if _inj and _inj["ci"][0] is not None else None),
        delta_random=(_rnd["mean_delta"] if _rnd else None),
        delta_shuffled=(_shf["mean_delta"] if _shf else None),
        delta_c4_ref=_c4, recover_thr=float(CFG["wo_steer_recover_thr"]),
        null_tol=float(CFG["wo_steer_null_tol"]),
        ci_halfwidth_tol=float(CFG["wo_steer_ci_halfwidth_tol"]))

    _out = {
        "tag": _tag, "experiment": "WO5_A_contrast_free_causal_steering",
        "n_used": _bundle["n_used"], "n_train": _sw["n_train"], "n_test": _sw["n_test"],
        "n_test_ok": bool(_sw["n_test"] >= int(CFG["wo_steer_min_test_n"])),
        "pair_sha": WO_STEER_PAIR_SHA, "L_grid": _sw["L_grid"],
        "c4_firsttok_argmax_match_rate": _bundle["c4_firsttok_argmax_match_rate"],
        "headline_layer": _hL, "headline_layer_basis": "train_decodability_peak",
        "reg_decodability_r2": _sw["reg_decodability_r2"],
        "exploratory_peak_layer": _sw["exploratory_peak_layer"],
        "headline": {"inject": _inj, "random": _rnd, "shuffled": _shf,
                     "c4_ref_delta": _c4, "c4_ref_layer": _c4L,
                     "c4_ref_peak_delta": _c4_peak},
        "sweep": _sweep, "erase": _sw["erase"], "verdict": _verdict,
        "recover_thr": float(CFG["wo_steer_recover_thr"]),
        "null_tol": float(CFG["wo_steer_null_tol"]),
        "note": ("Contrast-free: Δ = GT-product-token logit (intervened − clean C1), every TEST "
                 "item contributes regardless of argmax. Headline layer is PRE-REGISTERED by TRAIN "
                 "decodability (no winner's-curse from the test Δ); the layer sweep is exploratory. "
                 "C4 reference is the ceiling-free COUNTERFACTUAL injection (inject P' at C4 '=', score P')."),
    }
    wo_save_result(f"causal_steering_{_tag}.json", json.dumps(_out, indent=2, default=str))
    WO_STEER[_tag] = _out

# ---- flat summary CSV --------------------------------------------------------
_st_rows = []
for _tag in CFG["wo_steer_tags"]:
    o = WO_STEER[_tag]
    for site, conds in o["sweep"].items():
        for cond, by_layer in conds.items():
            for L, cell in by_layer.items():
                ci = cell.get("ci", [None, None])
                _st_rows.append({"tag": _tag, "site": site, "condition": cond, "layer": int(L),
                                 "mean_delta": cell.get("mean_delta"),
                                 "ci_lo": ci[0], "ci_hi": ci[1],
                                 "flip_rate": cell.get("flip_rate"), "n": cell.get("n")})
    for site, e in o["erase"].items():
        c = e["result"]; ci = c.get("ci", [None, None])
        _st_rows.append({"tag": _tag, "site": site, "condition": "erase", "layer": int(e["layer"]),
                         "mean_delta": c.get("mean_delta"), "ci_lo": ci[0], "ci_hi": ci[1],
                         "flip_rate": c.get("flip_rate"), "n": c.get("n")})
wo_save_result("causal_steering_summary.csv",
               wo_battery_csv(_st_rows, ["tag", "site", "condition", "layer", "mean_delta",
                                        "ci_lo", "ci_hi", "flip_rate", "n"]))

# ---- layer-sweep PNG: one figure per tag, panels = ')' site and '=' site -----
def _st_curve(o, site, cond, Lg):
    return [o["sweep"][site][cond][L]["mean_delta"]
            if o["sweep"][site][cond].get(L, {}).get("mean_delta") is not None else np.nan
            for L in Lg]


for _tag in CFG["wo_steer_tags"]:
    try:
        import matplotlib.pyplot as plt
        o = WO_STEER[_tag]; Lg = o["L_grid"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), squeeze=False)
        for si, (site, ttl) in enumerate([("rparen", "')' site — inject operand B"),
                                          ("equals", "'=' site — inject product B·C")]):
            ax = axes[0][si]
            for cond, style in [("inject", "o-"), ("random", "s--"), ("shuffled", "^:")]:
                ax.plot(Lg, _st_curve(o, site, cond, Lg), style, ms=4, label=cond)
            if site == "equals":
                yc = [o["sweep"]["c4_ref"]["inject"][L]["mean_delta"]
                      if o["sweep"]["c4_ref"]["inject"].get(L, {}).get("mean_delta") is not None else np.nan
                      for L in Lg]
                ax.plot(Lg, yc, "d-.", ms=4, color="green", label="C4 ref (P')")
            ax.axhline(0.0, color="grey", lw=0.8)
            ax.axhline(o["recover_thr"], color="red", ls=":", lw=0.8, label=f"recover thr {o['recover_thr']}")
            ax.set_title(f"{_tag}: {ttl}"); ax.set_xlabel("inject layer")
            ax.set_ylabel("mean Δ GT-product logit"); ax.legend(fontsize=7)
        fig.suptitle(f"WO#5 causal steering — {_tag}  (verdict: {o['verdict']['label']})", fontsize=11)
        fig.tight_layout()
        fig.savefig(str(WO_RESULTS / f"causal_steering_layersweep_{_tag}.png"), dpi=130)
        plt.show()
    except Exception as e:
        log(f"(WO#5 steering layer-sweep PNG [{_tag}] skipped: {e})")

# ---- printed verdicts --------------------------------------------------------
print("\n================= WO#5 EXPERIMENT A — CONTRAST-FREE CAUSAL STEERING =================")
print(f"{'tag':<10}{'n_test':>7}{'hdL':>5}{'inject Δ':>10}{'rand Δ':>9}{'shuf Δ':>9}{'C4ref':>8}  verdict")
for _tag in CFG["wo_steer_tags"]:
    o = WO_STEER[_tag]; h = o["headline"]

    def _d(c):
        return "  n/a" if (c is None or c.get("mean_delta") is None) else f"{c['mean_delta']:+.3f}"
    _c4 = h["c4_ref_delta"]                          # verdict-relevant (pre-registered C4 layer)
    print(f"{_tag:<10}{o['n_test']:>7}{str(o['headline_layer']):>5}{_d(h['inject']):>10}"
          f"{_d(h['random']):>9}{_d(h['shuffled']):>9}"
          f"{('n/a' if _c4 is None else f'{_c4:+.2f}'):>8}  {o['verdict']['label']}")
print("-------------------------------------------------------------------------------------")
for _tag in CFG["wo_steer_tags"]:
    o = WO_STEER[_tag]
    _warn = "" if o["n_test_ok"] else f"  ⚠ n_test<{int(CFG['wo_steer_min_test_n'])} (under-powered)"
    print(f"  [{_tag}] {o['verdict']['label']}{_warn}: {o['verdict']['reason']}")
    if not o["verdict"]["c4_ref_ok"]:
        print(f"         (C4 reference did NOT pass — the steering instrument is unproven at this band; "
              "treat the C1 result as INCONCLUSIVE, not a clean null.)")
print("=====================================================================================")
