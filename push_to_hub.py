"""
push_to_hub.py
──────────────
Run this from Kaggle (or locally) after training to upload model
checkpoints + config to HuggingFace Hub.

Usage (Kaggle notebook cell):
    !python /kaggle/working/Lpatchtst/push_to_hub.py

Or set HF_TOKEN as a Kaggle Secret instead of hard-coding it here.
"""

import os
import sys
import json
import shutil
import importlib
import torch

# ── Config ───────────────────────────────────────────────────────────────────
HF_USERNAME   = "gulnawaz123"
REPO_NAME     = "Full_Tokenizer_30m"          # change freely
REPO_ID       = f"{HF_USERNAME}/{REPO_NAME}"

# Token: read from environment variable (Kaggle Secret or shell export).
# Set via:  os.environ["HF_TOKEN"] = "hf_..."  in a prior cell, or
#           Add a Kaggle Secret named HF_TOKEN.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set. "
                           "Add it as a Kaggle Secret or export it in your shell.")

# Paths to look for checkpoints (adjust if your working dir differs)
SEARCH_DIRS = [
    "/kaggle/working/Lpatchtst",
    "/kaggle/working",
    ".",
]
CHECKPOINT_FILES = [
    "pretrain_best.pth",
    "model.safetensors",
]
CONFIG_FILES = [
    "config.py",
    "model.py",
    "loss.py",
    "tokenizer.py",
    "features.py",
    "oracle.py",
    "data_loader.py",
]

# ── Install huggingface_hub if missing ───────────────────────────────────────
try:
    from huggingface_hub import HfApi, create_repo, upload_folder
except ImportError:
    print("Installing huggingface_hub …")
    os.system(f"{sys.executable} -m pip install -q huggingface_hub")
    from huggingface_hub import HfApi, create_repo, upload_folder

# ── Resolve source directory ─────────────────────────────────────────────────
def _find_file(fname):
    for d in SEARCH_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None

src_dir = None
for d in SEARCH_DIRS:
    if os.path.exists(os.path.join(d, "config.py")):
        src_dir = d
        break

if src_dir is None:
    raise FileNotFoundError("Cannot locate config.py in any search directory.")

print(f"Source directory : {src_dir}")

# ── Stage files into a temp upload folder ────────────────────────────────────
upload_dir = "/tmp/lpatchtst_hf_upload"
os.makedirs(upload_dir, exist_ok=True)

copied = []

# Copy checkpoints
for ckpt in CHECKPOINT_FILES:
    path = _find_file(ckpt)
    if path:
        shutil.copy2(path, os.path.join(upload_dir, ckpt))
        copied.append(ckpt)
        print(f"  ✓ {ckpt}  ({os.path.getsize(path)/1e6:.1f} MB)")
    else:
        print(f"  ✗ {ckpt} not found — skipping")

if not copied:
    raise FileNotFoundError(
        "No checkpoint files found. Train the model first.\n"
        f"Expected one of: {CHECKPOINT_FILES}"
    )

# Copy source files
for fname in CONFIG_FILES:
    path = _find_file(fname)
    if path:
        shutil.copy2(path, os.path.join(upload_dir, fname))
        print(f"  ✓ {fname}")

# ── Write a minimal model card ───────────────────────────────────────────────
model_card = f"""---
language:
  - en
tags:
  - time-series
  - finance
  - pytorch
  - patchtst
license: mit
---

# LPatchTST — NIFTY 50 Trading Model

A patch-based Transformer (LPatchTST) trained on NIFTY 50 30-minute bars
for directional signal prediction.

## Files
| File | Description |
|------|-------------|
| `best_model_lpatchtst.pth` | Best fine-tuned checkpoint |
| `pretrained_lpatchtst.pth` | Pre-trained backbone checkpoint |
| `config.py` | Hyperparameters & architecture config |
| `model.py` | Model definition |

## Loading the model
```python
import torch
import sys
sys.path.insert(0, ".")   # ensure local modules are importable

import config
from model import LPatchTST

net = LPatchTST(
    input_mode=config.INPUT_MODE,
    seq_len=config.LOOKBACK_WINDOW,
    n_features=0,          # set to your feature count
    s1_bits=config.TOKENIZER_S1_BITS,
    s2_bits=config.TOKENIZER_S2_BITS,
    d_model=config.D_MODEL,
    patch_len=config.PATCH_LEN,
    stride=config.STRIDE,
    n_heads=config.N_HEADS,
    n_layers=config.N_LAYERS,
    lstm_layers=config.LSTM_LAYERS,
    dropout=config.DROPOUT,
    aggregation=config.AGGREGATION_MODE,
)
state = torch.load("best_model_lpatchtst.pth", map_location="cpu")
net.load_state_dict(state)
net.eval()
```
"""
with open(os.path.join(upload_dir, "README.md"), "w") as f:
    f.write(model_card)

# ── Create / ensure repo exists ───────────────────────────────────────────────
api = HfApi(token=HF_TOKEN)

print(f"\nCreating / verifying repo: {REPO_ID} …")
create_repo(
    repo_id=REPO_ID,
    token=HF_TOKEN,
    repo_type="model",
    exist_ok=True,
    private=False,      # set True if you want a private repo
)
print(f"  ✓ Repo ready: https://huggingface.co/{REPO_ID}")

# ── Upload ────────────────────────────────────────────────────────────────────
print(f"\nUploading {len(os.listdir(upload_dir))} files …")
api.upload_folder(
    folder_path=upload_dir,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload LPatchTST checkpoint and source",
)

print(f"\n✅ Upload complete!")
print(f"   View at: https://huggingface.co/{REPO_ID}")
