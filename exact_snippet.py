import torch

def asymmetric_number_line_loss(pred, target):
    # This is a dummy function to match the snippet's structure if needed, 
    # but the snippet actually calculates it manually.
    pass

tgt = torch.tensor([0.5,  0.5,  0.5,  -0.5, -0.5,  0.0,  0.0])
pred = torch.tensor([0.1,  0.9, -0.3,  -0.1, -0.9,  0.3, -0.3])

# Expected: wrong-dir (-0.3 vs +0.5) highest, flat bars equal, overshoot gentlest
for p, t in zip(pred, tgt):
    g = p - t
    side = "undershoot" if (g * t).item() < 0 else "overshoot"
    mw = t.abs().pow(0.5).clamp(min=0.3).item()
    scale = 2.0 if side == "undershoot" else 0.8
    l = scale * g.item()**2 * mw
    print(f"pred={p:+.1f} tgt={t:+.1f}  → {side:10s}  loss={l:.4f}")
