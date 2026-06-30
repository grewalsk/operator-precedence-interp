# ============================================================================
# Phase 6 / WORK ORDER #6 (Tier 2) — CI HYGIENE + RUN-RECORD (CPU; zero GPU).
# ----------------------------------------------------------------------------
# Removes whole classes of reviewer complaints with no GPU:
#   (1) A CI on EVERY accuracy. Maps the Wilson interval (wo_acc_ci) across every
#       accuracy column in every summary CSV in WO_RESULTS, emitting a *_ci.csv
#       alongside — wrapper drops, cross-model C0..D1, few-shot rows, etc. Wilson
#       needs only k/n, so this runs post-hoc on committed means (no re-run).
#       [Decodability R^2 CIs come from the analysis cells via wo_r2_bootstrap_ci;
#        delta CIs need the in-memory per-item masks (paired bootstrap) and are
#        emitted where those are live — flagged, not faked, here.]
#   (2) A run-record per run: run_meta.json (seed, band, N, pairs sha, model revision,
#       transformer_lens / torch / transformers versions, the parse + answer-extraction
#       rule, prepend_bos) — the camera-ready repro record the cross-model run lacked.
# Pure CPU (csv + stdlib + wo_acc_ci / wo_run_meta). Idempotent; skips *_ci.csv inputs.
# ============================================================================
import csv as _csv
import io as _io
import json as _json
from pathlib import Path as _CIP

assert "wo_acc_ci" in globals() and "wo_run_meta" in globals(), (
    "WO#6 Tier-2 CI hygiene needs cell-76 helpers wo_acc_ci / wo_run_meta.")

# accuracy-bearing column names across the project's summary CSVs (each over WO_N items).
_CI_ACC_COLS = {"acc", "exact_acc", "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
                "A1", "A2", "D1", "fs_C1_0", "fs_C1_2", "fs_C1_4", "acc_C0_band2", "acc_C4_band2"}


def _ci_annotate_csv(path, n_default):
    """Read a summary CSV, append <col>_ci_lo/<col>_ci_hi (Wilson) for every accuracy
    column, write <stem>_ci.csv. Per-row n from an 'n'/'N' column if present, else
    n_default. Returns the output filename or None (no accuracy columns / empty)."""
    p = _CIP(path)
    try:
        rows = list(_csv.DictReader(open(p)))
    except Exception:
        return None
    if not rows:
        return None
    acc_cols = [h for h in rows[0].keys() if h in _CI_ACC_COLS]
    if not acc_cols:
        return None
    out_hdr = []
    for h in rows[0].keys():
        out_hdr.append(h)
        if h in acc_cols:
            out_hdr += [f"{h}_ci_lo", f"{h}_ci_hi"]
    for r in rows:
        nval = r.get("n") or r.get("N") or n_default
        try:
            nn = int(float(nval))
        except (TypeError, ValueError):
            nn = int(n_default)
        for c in acc_cols:
            v = r.get(c, "")
            try:
                lo, hi = wo_acc_ci(float(v), nn)
            except (TypeError, ValueError):
                lo, hi = (None, None)
            r[f"{c}_ci_lo"] = "" if lo is None else f"{lo:.4f}"
            r[f"{c}_ci_hi"] = "" if hi is None else f"{hi:.4f}"
    buf = _io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=out_hdr)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in out_hdr})
    out_name = p.stem + "_ci.csv"
    wo_save_result(out_name, buf.getvalue())
    return out_name


def _ci_ver(m):
    try:
        v = getattr(__import__(m), "__version__", None)
        if v:
            return v
    except Exception:
        pass
    try:
        import importlib.metadata as _im
        return _im.version(m)
    except Exception:
        return None


# (1) annotate every accuracy summary CSV in the results dir.
_ci_done = []
for _csvp in sorted(_CIP(str(WO_RESULTS)).glob("*.csv")):
    if _csvp.stem.endswith("_ci"):
        continue
    _out = _ci_annotate_csv(_csvp, globals().get("WO_N", 400))
    if _out:
        _ci_done.append(_out)

# (2) emit the run-record.
_ci_meta = wo_run_meta(
    model_tag=globals().get("WO_ACTIVE_TAG"),
    model_revision=(globals().get("WO_MODEL_REVISIONS") or {}).get(globals().get("WO_ACTIVE_TAG")),
    tl_version=_ci_ver("transformer_lens"), torch_version=_ci_ver("torch"),
    transformers_version=_ci_ver("transformers"),
    prepend_bos=CFG.get("wo_prepend_bos"), pairs_sha=globals().get("WO_PAIRS_HASH"),
    extra={"using_fallback": globals().get("USING_FALLBACK"),
           "model_registry": {k: v for k, v in globals().get("WO_MODEL_REGISTRY", {}).items()},
           "ci_annotated_csvs": _ci_done})
wo_save_result("run_meta.json", _json.dumps(_ci_meta, indent=2, default=str))

print("\n========== WO#6 (Tier 2) — CI HYGIENE + RUN-RECORD ==========")
print(f"  Wilson CIs written for {len(_ci_done)} accuracy summary CSV(s): {_ci_done}")
print(f"  run_meta.json: band={_ci_meta['band']} N={_ci_meta['N']} seed={_ci_meta['seed']} "
      f"pairs_sha={str(_ci_meta.get('pairs_sha'))[:12]} TL={_ci_meta['transformer_lens']} torch={_ci_meta['torch']}")
print("  (Decodability R^2 CIs come from the analysis cells via wo_r2_bootstrap_ci; delta CIs")
print("   are paired-bootstrapped where the per-item masks are live.)")
print("=============================================================")
