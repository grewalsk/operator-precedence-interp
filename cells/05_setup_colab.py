# ============================================================================
# SETUP — install deps + authenticate to the gated model. RUN THIS FIRST.
# Idempotent: re-running skips the install and re-uses the token.
# Uses subprocess (not %pip) so the cell is valid Python everywhere.
# ============================================================================
import os, sys, subprocess

# ----------------------------------------------------------------------------
# 1) Dependency: transformer_lens (Colab has torch/transformers, not this).
#
#    GOTCHA this guards against: installing transformer_lens can shift `torch`,
#    which breaks Colab's PREBUILT torchaudio/torchvision (their compiled .so is
#    ABI-locked to the original torch) -> on import you get
#        OSError: .../_torchaudio.abi3.so: undefined symbol: torch_library_impl
#    We use neither audio nor vision, so we (a) remove them to sidestep the ABI
#    clash entirely, and (b) PIN torch to Colab's version so nothing else shifts.
# ----------------------------------------------------------------------------
def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", *args], check=False)

def _import_tl():
    # fresh import attempt, clearing any half-loaded modules from a prior failure
    for _m in [k for k in list(sys.modules)
               if k.split(".")[0] in ("transformer_lens", "torchaudio", "torchvision")]:
        del sys.modules[_m]
    import transformer_lens  # noqa: F401

try:
    import transformer_lens  # noqa: F401
    print("transformer_lens already installed.")
except Exception:
    print("Installing transformer_lens (one-time per session)...")
    # Remove the unused, ABI-fragile audio/vision libs so a torch shift can't break import.
    _pip("uninstall", "-q", "-y", "torchaudio", "torchvision")
    # Pin torch to whatever Colab already has, so the install doesn't move it.
    _torch_pin = []
    try:
        import torch
        _torch_pin = [f"torch=={torch.__version__.split('+')[0]}"]
    except Exception:
        pass
    _pip("install", "-q", "transformer_lens", *_torch_pin)
    try:
        _import_tl()
        print("transformer_lens installed and imported OK.")
    except Exception as e:
        print("!! transformer_lens import still failing:", repr(e))
        print("   CLEANEST FIX: Runtime > Disconnect and delete runtime, reopen the notebook,")
        print("   then Run all. (Your current runtime is in a half-changed state; a pristine")
        print("   one installs cleanly.)")

# ----------------------------------------------------------------------------
# 2) (Recommended on Colab) cache the 16 GB weights on Drive so you don't
#    re-download every session — saves time AND idle-GPU credits. MUST be set
#    BEFORE Phase 0 downloads the model.
# ----------------------------------------------------------------------------
USE_DRIVE_HF_CACHE = True
if USE_DRIVE_HF_CACHE:
    try:
        import google.colab  # noqa: F401  (only succeeds inside Colab)
        from google.colab import drive
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
        os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)
        print("HF_HOME ->", os.environ["HF_HOME"], "(model caches to Drive; download is one-time)")
    except Exception as e:
        print("(Drive HF cache skipped:", repr(e), "— weights go to ephemeral /root this session)")

# ----------------------------------------------------------------------------
# 3) Hugging Face auth for the GATED repo meta-llama/Llama-3.1-8B.
#    Precedence: Colab Secret 'HF_TOKEN' -> env HF_TOKEN -> env HUGGINGFACE_TOKEN.
# ----------------------------------------------------------------------------
_GATED_REPO = "meta-llama/Llama-3.1-8B"   # keep in sync with CFG['model_name']
_tok = None
try:
    from google.colab import userdata       # Colab "Secrets" (🔑 in left sidebar)
    try:
        _tok = userdata.get("HF_TOKEN")
    except Exception:
        _tok = None                          # secret missing or notebook-access off
except Exception:
    pass
_tok = _tok or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

if _tok:
    os.environ["HF_TOKEN"] = _tok            # Phase 0 reads these
    os.environ["HUGGINGFACE_TOKEN"] = _tok
    from huggingface_hub import login
    login(token=_tok, add_to_git_credential=False)
    try:
        from huggingface_hub import whoami
        print("HF logged in as:", whoami(_tok).get("name", "(unknown)"))
    except Exception as e:
        print("HF login set (whoami unavailable:", repr(e), ")")
    # Token present != access granted. Pre-check so the failure is obvious HERE,
    # not 30 frames deep inside Phase 0.
    try:
        from huggingface_hub import auth_check
        auth_check(_GATED_REPO, token=_tok)
        print(f"✓ access to {_GATED_REPO} confirmed — Phase 0 will load.")
    except ImportError:
        print("(huggingface_hub too old to pre-check access; Phase 0 will surface any 401.)")
    except Exception as e:
        print(f"✗ token works but gated-repo access NOT granted yet for {_GATED_REPO}:")
        print("  Request it (usually instant):", f"https://huggingface.co/{_GATED_REPO}")
        print("  Detail:", repr(e))
else:
    print("✗ No HF token found — Llama-3.1-8B is GATED, so Phase 0 will 401 without one.")
    print("  1) Request access:", f"https://huggingface.co/{_GATED_REPO}")
    print("  2) Create a READ token: https://huggingface.co/settings/tokens")
    print("  3) Colab: 🔑 Secrets (left sidebar) -> add HF_TOKEN -> enable 'Notebook access'")
    print("     -> re-run this cell.  (Or run: import os; os.environ['HF_TOKEN']='hf_...')")
