# ============================================================================
# SETUP — install deps + authenticate to the gated model. RUN THIS FIRST.
# Idempotent: re-running skips the install and re-uses the token.
# Uses subprocess (not %pip) so the cell is valid Python everywhere.
# ============================================================================
import os, sys, subprocess

# ----------------------------------------------------------------------------
# 1) Dependency: transformer_lens (Colab has torch/transformers, not this).
# ----------------------------------------------------------------------------
def _pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

try:
    import transformer_lens  # noqa: F401
    print("transformer_lens already installed.")
except ModuleNotFoundError:
    print("Installing transformer_lens (one-time per session)...")
    _pip_install("transformer_lens")
    try:
        import transformer_lens  # noqa: F401
        print("transformer_lens installed OK.")
    except ModuleNotFoundError:
        print("!! transformer_lens still not importable. Do: Runtime > Restart session, "
              "then re-run this cell. (A transformers upgrade can require one restart.)")

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
