# audit_gradclip.py
# Run ONE training step manually and check what clip_grad_norm_ actually sees
import torch, config, model
from model import LPatchTST, InputMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build a dummy model
net = LPatchTST(
    input_mode=config.INPUT_MODE,
    seq_len=config.LOOKBACK_WINDOW,
    n_features=0,  # tokens_only
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
).to(device)

# Simulate exploded gradients
for p in net.parameters():
    p.grad = torch.randn_like(p) * 100.0  # fake large grads

# Check what names clip_grad_norm_ sees vs named_parameters
all_param_names = [n for n, _ in net.named_parameters()]
params_with_grad = [p for p in net.parameters() if p.grad is not None]

pre_clip_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=9999.0)  # no-op clip

print(f"Total params seen by clip_grad_norm_: {len(params_with_grad)}")
print(f"Total named params: {len(all_param_names)}")
print(f"Pre-clip total grad norm: {pre_clip_norm:.2f}")

# Now check if _orig_mod. or module. prefixed params exist (torch.compile / DDP artifacts)
prefixed = [n for n in all_param_names if "_orig_mod." in n or "module." in n]
print(f"Params with DDP/compile prefix (may escape clip): {len(prefixed)}")
if prefixed:
    print("  Examples:", prefixed[:5])