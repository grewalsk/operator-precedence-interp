# ============================================================================
# SETUP — deps + Drive HF cache + HF auth + version/RAM readout. Run first.
# Mirrors the main notebook's setup so the load calls below are apples-to-apples.
# ============================================================================
import os
import sys
import subprocess


def _pip(*a):
    subprocess.run([sys.executable, "-m", "pip", *a], check=False)


# ABI GUARD (mirror the main notebook's setup): Colab ships a prebuilt torchaudio
# whose compiled .so is ABI-locked to Colab's torch. Installing transformer_lens can
# shift torch, after which `import torchaudio` raises
#   OSError: _torchaudio.abi3.so: undefined symbol: torch_library_impl
# and transformers 5.x imports torchaudio (loss_rnnt) at EVERY from_pretrained -> ALL
# model loads die. We don't use audio/vision, so UNINSTALL them (absent -> transformers
# degrades gracefully) and PIN torch so the install can't move it. Unconditional so a
# resumed runtime with a half-shifted torch is also repaired (then restart the runtime).
print("stripping torchaudio/torchvision (ABI-fragile, unused) + pinning torch ...")
_pip("uninstall", "-q", "-y", "torchaudio", "torchvision")
_torch_pin = []
try:
    import torch as _t
    _torch_pin = [f"torch=={_t.__version__.split('+')[0]}"]
except Exception:
    pass
try:
    import transformer_lens  # noqa: F401
    print("transformer_lens already installed.")
except Exception:
    print("installing transformer_lens (one-time per session)...")
    _pip("install", "-q", "transformer_lens", *_torch_pin)

# Reuse the main run's weight cache on Drive so nothing re-downloads unnecessarily.
try:
    from google.colab import drive  # type: ignore
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    print("HF_HOME ->", os.environ["HF_HOME"])
except Exception as e:
    print("(Drive cache skipped:", repr(e), "— weights go to ephemeral /root)")

# HF token: Colab secret 'HF_TOKEN' -> env HF_TOKEN -> env HUGGINGFACE_TOKEN.
_tok = None
try:
    from google.colab import userdata  # type: ignore
    try:
        _tok = userdata.get("HF_TOKEN")
    except Exception:
        _tok = None
except Exception:
    pass
_tok = _tok or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if _tok:
    os.environ["HF_TOKEN"] = _tok
    os.environ["HUGGINGFACE_TOKEN"] = _tok
    try:
        from huggingface_hub import login, whoami
        login(token=_tok, add_to_git_credential=False)
        print("HF logged in as:", whoami(_tok).get("name", "(unknown)"))
    except Exception as e:
        print("HF login set (whoami unavailable:", repr(e), ")")
else:
    print("!! No HF token found — gated repos will 401. Set HF_TOKEN and re-run.")

# Versions + hardware (so we can correlate a failure with a version/RAM).
import torch


def _ver(m):
    try:
        return __import__(m).__version__
    except Exception:
        try:
            import importlib.metadata as _im
            return _im.version(m)
        except Exception:
            return "n/a"


print("\n--- environment ---")
print("python            :", sys.version.split()[0])
print("torch             :", torch.__version__)
print("transformers      :", _ver("transformers"))
print("transformer_lens  :", _ver("transformer_lens"))
print("accelerate        :", _ver("accelerate"))
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0)
    print("GPU               :", _p.name, f"({_p.total_memory / 1e9:.0f} GB)")
else:
    print("GPU               : NONE (this diagnostic needs a GPU runtime)")
try:
    import psutil
    _m = psutil.virtual_memory()
    print("system RAM        :", f"{_m.total / 1e9:.0f} GB total, {_m.available / 1e9:.0f} GB available")
    if _m.total < 30e9:
        print("  ⚠ LOW system RAM (<30 GB) — a 7-9B HF load can OOM on CPU before reaching GPU.")
except Exception:
    pass
print("HF_HOME           :", os.environ.get("HF_HOME", "(default ~/.cache)"))
