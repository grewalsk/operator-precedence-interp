# ============================================================================
# Phase -1 / Gate G_INFRA — CONFIG + CHECKPOINT/RESUME INFRASTRUCTURE
# ----------------------------------------------------------------------------
# This is the FIRST cell. Every later cell depends on the globals/helpers it
# defines. It is fully idempotent: safe to re-run after a GPU/kernel disconnect.
# It does NO GPU work and loads NO model (model is reloaded in Phase 0).
# ============================================================================

import os
import io
import sys
import json
import time
import pickle
import random
import datetime
import platform
import pathlib
from pathlib import Path

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch must exist on the GPU box, but be safe
    torch = None
    _HAS_TORCH = False


# ----------------------------------------------------------------------------
# 1. Persistent artifact directory ART (survives kernel/GPU disconnects)
# ----------------------------------------------------------------------------
# Priority:
#   (a) Google Colab + mountable Drive  -> /content/drive/MyDrive/opprec_interp
#   (b) Local persistent dir under HOME  -> ~/opprec_interp_artifacts
# Both branches are wrapped so a failure degrades gracefully to the local path.
# (b) is the expected branch on a dedicated cloud GPU box where HOME is the
# persistent user volume; only Colab uses the Drive branch.

ART_DRIVE_DEFAULT = "/content/drive/MyDrive/opprec_interp"
ART_LOCAL_DEFAULT = str(Path.home() / "opprec_interp_artifacts")


def _in_colab() -> bool:
    """True iff we appear to be running inside a Google Colab kernel."""
    if "google.colab" in sys.modules:
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_art_dir() -> Path:
    """Pick and create a persistent artifact directory; return its Path."""
    # (a) Try Colab Drive first.
    if _in_colab():
        try:
            from google.colab import drive  # type: ignore
            # Mount is idempotent: if already mounted, this is a no-op.
            if not os.path.ismount("/content/drive"):
                drive.mount("/content/drive", force_remount=False)
            art = Path(ART_DRIVE_DEFAULT)
            art.mkdir(parents=True, exist_ok=True)
            # Confirm we can actually write (Drive sometimes mounts read-only).
            _probe = art / ".write_probe"
            _probe.write_text("ok")
            _probe.unlink(missing_ok=True)
            return art
        except Exception as e:
            print(f"[ART] Colab Drive unavailable ({type(e).__name__}: {e}); "
                  f"falling back to local persistent dir.")

    # (b) Local persistent directory under the home dir.
    art = Path(ART_LOCAL_DEFAULT)
    art.mkdir(parents=True, exist_ok=True)
    return art


ART = _resolve_art_dir()
print(f"[ART] Persistent artifact directory: {ART}")


# ----------------------------------------------------------------------------
# 6. log(msg): timestamped print   (defined early so later setup can use it)
# ----------------------------------------------------------------------------
def log(msg: str) -> None:
    """Timestamped stdout print, flushed immediately."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 3. CFG dict — all knobs as PARAMETERS (no hardcoded magic deep in later cells)
# ----------------------------------------------------------------------------
def _auto_device() -> str:
    if _HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


_DEVICE = _auto_device()
_DTYPE = (torch.bfloat16 if _HAS_TORCH else None)  # bf16: nondeterminism recorded, not fought

CFG = {
    # --- model / runtime ---
    "model_name": "meta-llama/Llama-3.1-8B",
    "seed": 0,
    "device": _DEVICE,
    "dtype": _DTYPE,
    "dtype_name": "bfloat16",
    # bf16 matmul is nondeterministic across runs; we RECORD this, we do not fight it.
    "determinism_note": (
        "bf16 matmul accumulation order is nondeterministic on GPU; small "
        "logit jitter is expected and is recorded, not eliminated."
    ),

    # --- operand-band params (parameterized span so Phase 3 can SEARCH over it) ---
    # The task studies operand-recognition / operand-precedence in arithmetic of
    # the form  a OP (b digits) OP (c digits). Digit counts are given as inclusive
    # [min, max] ranges so Phase 3 can sweep band widths rather than hardcoding.
    "b_digits": {"min": 1, "max": 3},   # inclusive digit-count band for operand b
    "c_digits": {"min": 1, "max": 3},   # inclusive digit-count band for operand c
    "a_digits": {"min": 1, "max": 1},   # leading operand (kept narrow by default)
    "operators": ["+", "-", "*"],        # operators considered in the prompt family
    # Phase 3 search grid over band widths (each entry is a (min,max) digit band).
    "digit_band_grid": [[1, 1], [1, 2], [1, 3], [2, 3], [3, 3]],

    # --- padding sweep (left-pad lengths probed for position robustness) ---
    "pad_lengths": [0, 2, 4, 8, 16],

    # --- dataset sizing ---
    "n_per_factor": 2000,   # target examples per experimental factor / band cell
    "max_new_tokens": 8,    # generation budget when checking answers
    "batch_size": 32,       # default eval batch (later cells may override per-mem)

    # --- reproducibility / bookkeeping ---
    "tl_from_pretrained": True,  # prefer transformer_lens HookedTransformer in Phase 0
    "hf_fallback": True,         # allow HF wrapper exposing run_with_cache/run_with_hooks
}

# --- derive all paths from ART (single source of truth) ---
CFG["paths"] = {
    "art": str(ART),
    "gate_status": str(ART / "gate_status.json"),
    "dataset": str(ART / "dataset.pkl"),
    "cache": str(ART / "cache"),
    "figures": str(ART / "figures"),
    "logs": str(ART / "logs"),
}
# Make sure the derived subdirectories exist.
for _sub in ("cache", "figures", "logs"):
    Path(CFG["paths"][_sub]).mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 4. Seeding helper (python / numpy / torch / cuda). bf16 nondeterminism noted.
# ----------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    """Seed python-random, numpy, and torch (+cuda). Idempotent."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # We deliberately do NOT force torch.use_deterministic_algorithms(True):
        # bf16 matmul nondeterminism is RECORDED (see CFG['determinism_note']),
        # not fought, because forcing determinism would change/limit kernels.
    log(f"set_all_seeds({seed}) applied "
        f"(torch={'yes' if _HAS_TORCH else 'no'}, "
        f"cuda={'yes' if (_HAS_TORCH and torch.cuda.is_available()) else 'no'}).")


set_all_seeds(CFG["seed"])


# ----------------------------------------------------------------------------
# 5. Artifact helpers — all read/write UNDER ART.
#    JSON for small/structured; pickle for tensors/arrays/objects; text for raw.
# ----------------------------------------------------------------------------
def _art_path(name: str, ext: str) -> Path:
    """Map an artifact name to ART/<name><ext>, allowing names with subdirs."""
    p = ART / (name if name.endswith(ext) else f"{name}{ext}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically (tmp + os.replace) so a disconnect mid-write
    cannot leave a half-written artifact that later cells would trust.

    Hardened vs. the original:
      * try/finally unlinks a stray tmp file if os.replace() ever raises, so
        failed writes don't accumulate orphan .tmp.<pid> files in ART.
      * parent-directory fsync after os.replace makes the rename itself durable
        against a true host crash (best-effort; skipped where unsupported)."""
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # fsync the directory entry so the rename survives a crash, not just the
        # file contents. Not all platforms allow opening a dir fd; ignore if not.
        try:
            _dfd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(_dfd)
            finally:
                os.close(_dfd)
        except (OSError, AttributeError):
            pass
    finally:
        # If we crashed before/at os.replace, tmp may still exist — clean it up.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# Map the public 'kind' names to their on-disk extensions so existence checks
# can be matched to the loader that will actually be used.
_EXT_BY_KIND = {"json": ".json", "pickle": ".pkl", "text": ".txt"}


def has_artifact(name: str, kind: str = None) -> bool:
    """True iff an artifact with this base name exists on disk.

    IMPORTANT: pass `kind` ('json' | 'pickle' | 'text') to check for the SAME
    type you will load with, e.g.

        if has_artifact('cache', 'pickle'): cache = load_pickle('cache')

    Without `kind` this returns True if ANY of {.json,.pkl,.txt} exists, which
    is convenient but UNSAFE in the standard resilience idiom: an artifact saved
    via save_pickle would make `has_artifact('x')` True while `load_json('x')`
    raises FileNotFoundError. Prefer the kind-aware form to pair the existence
    check with its loader."""
    if kind is not None:
        if kind not in _EXT_BY_KIND:
            raise ValueError(f"has_artifact kind must be one of {sorted(_EXT_BY_KIND)}, got {kind!r}")
        return _art_path(name, _EXT_BY_KIND[kind]).exists()
    for ext in (".json", ".pkl", ".txt"):
        if _art_path(name, ext).exists():
            return True
    return False


def save_json(name: str, obj) -> str:
    path = _art_path(name, ".json")
    data = json.dumps(obj, indent=2, default=str).encode("utf-8")
    _atomic_write_bytes(path, data)
    return str(path)


def load_json(name: str):
    path = _art_path(name, ".json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(name: str, obj) -> str:
    path = _art_path(name, ".pkl")
    _atomic_write_bytes(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    return str(path)


def load_pickle(name: str):
    path = _art_path(name, ".pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_text(name: str, s: str) -> str:
    path = _art_path(name, ".txt")
    _atomic_write_bytes(path, str(s).encode("utf-8"))
    return str(path)


def load_text(name: str) -> str:
    path = _art_path(name, ".txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ----------------------------------------------------------------------------
# 7. GATE-STATUS ledger persisted to ART/gate_status.json
#    Lets the final dashboard reconstruct G0..G4 across sessions.
# ----------------------------------------------------------------------------
_GATE_FILE = Path(CFG["paths"]["gate_status"])


def get_gates() -> dict:
    """Return the full gate ledger {gate: {passed, detail, ts}}; {} if none."""
    if _GATE_FILE.exists():
        try:
            with open(_GATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"[gates] WARNING: could not read gate ledger ({e}); treating as empty.")
            return {}
    return {}


def set_gate(gate: str, passed: bool, detail: str = "") -> dict:
    """Record a gate result (read-modify-write the on-disk ledger) so re-running
    this infra cell — or any phase — never clobbers other gates' results."""
    gates = get_gates()
    gates[str(gate)] = {
        "passed": bool(passed),
        "detail": str(detail),
        "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_write_bytes(
        _GATE_FILE,
        json.dumps(gates, indent=2, default=str).encode("utf-8"),
    )
    status = "PASS" if passed else "FAIL"
    log(f"[gate {gate}] {status} — {detail}")
    return gates


# ----------------------------------------------------------------------------
# 8. MODEL-RELOAD GUARD PATTERN (note only — no GPU work here)
# ----------------------------------------------------------------------------
# The HookedTransformer 'model' and 'tokenizer' CANNOT be pickled across GPU
# disconnects, so they are NOT artifacts. Phase 0 reloads them, guarded like:
#
#     if "model" not in globals():
#         model, tokenizer = load_model_phase0(CFG)   # defined in Phase 0 cell
#
# Re-running Phase 0 after a reconnect rebuilds them in-memory; everything else
# (datasets, caches, gate ledger) is restored from ART via the helpers above.
# This cell intentionally does NONE of that — it only sets up the scaffolding.

# ----------------------------------------------------------------------------
# Record an environment snapshot + mark the infra gate as passed.
# ----------------------------------------------------------------------------
_env_snapshot = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "in_colab": _in_colab(),
    "has_torch": _HAS_TORCH,
    "torch_version": (torch.__version__ if _HAS_TORCH else None),
    "cuda_available": (bool(torch.cuda.is_available()) if _HAS_TORCH else False),
    "cuda_device": (torch.cuda.get_device_name(0)
                    if (_HAS_TORCH and torch.cuda.is_available()) else None),
    "numpy": np.__version__,
    "art": str(ART),
    "device": CFG["device"],
    "dtype": CFG["dtype_name"],
    "seed": CFG["seed"],
    "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
save_json("env_snapshot", _env_snapshot)

log(f"Infra ready. device={CFG['device']} dtype={CFG['dtype_name']} "
    f"seed={CFG['seed']} model={CFG['model_name']}")
log(f"Existing gates on disk: {list(get_gates().keys()) or '(none yet)'}")

# Self-check: prove the artifact round-trip works before any later cell trusts it.
# Use the kind-aware has_artifact so the existence check matches the loader.
_RT_KEY = "_infra_selfcheck"
save_json(_RT_KEY, {"ok": True, "seed": CFG["seed"]})
_rt_ok = has_artifact(_RT_KEY, "json") and load_json(_RT_KEY).get("ok") is True
set_gate("G_INFRA", _rt_ok,
         f"artifact round-trip + seeding OK; ART={ART}; device={CFG['device']}")
print("PASS: infra cell" if _rt_ok else "FAIL: infra cell artifact round-trip")