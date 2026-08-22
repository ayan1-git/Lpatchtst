import torch
from loss import asymmetric_number_line_loss

tgt = torch.tensor([0.5,  0.5,  0.5,  -0.5, -0.5,  0.0,  0.0])
pred = torch.tensor([0.1,  0.9, -0.3,  -0.1, -0.9,  0.3, -0.3])

for p, t in zip(pred, tgt):
    # Use the function for a single element to get the specific loss
    l = asymmetric_number_line_loss(p.unsqueeze(0), t.unsqueeze(0)).item()
    
    # For reporting, we still want to know the side
    g = p - t
    side = "undershoot" if (g * t).item() < 0 else "overshoot"
    print(f"pred={p:+.1f} tgt={t:+.1f}  → {side:10s}  loss={l:.4f}")
