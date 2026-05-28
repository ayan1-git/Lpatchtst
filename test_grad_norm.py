
import torch
import math
import numpy as np
from train import _grad_norm

def test_grad_norm():
    print("Testing _grad_norm function...")
    
    # Case 1: Normal gradients
    model = torch.nn.Linear(10, 1)
    with torch.no_grad():
        model.weight.grad = torch.ones_like(model.weight) * 1.0
        model.bias.grad = torch.ones_like(model.bias) * 1.0
    
    # total = sum(grad.norm(2)^2) = (10*1)^2 + (1*1)^2 = 10 + 1 = 11? 
    # No, norm(2) of ones(10) is sqrt(10). norm(2)^2 is 10.
    # total = 10 + 1 = 11. sqrt(11) approx 3.3166
    gn = _grad_norm(model)
    print(f"Case 1 (Normal): Expected ~3.3166, Got: {gn:.4f}")
    assert math.isfinite(gn) and gn > 0

    # Case 2: No gradients
    model_no_grad = torch.nn.Linear(10, 1)
    # no grads set
    gn = _grad_norm(model_no_grad)
    print(f"Case 2 (No Grad): Expected 0.0, Got: {gn:.4f}")
    assert gn == 0.0

    # Case 3: Inf gradients
    model_inf = torch.nn.Linear(10, 1)
    with torch.no_grad():
        model_inf.weight.grad = torch.full_like(model_inf.weight, float('inf'))
        model_inf.bias.grad = torch.zeros_like(model_inf.bias)
    gn = _grad_norm(model_inf)
    print(f"Case 3 (Inf Grad): Expected 0.0 (fixed), Got: {gn:.4f}")
    assert gn == 0.0

    # Case 4: NaN gradients
    model_nan = torch.nn.Linear(10, 1)
    with torch.no_grad():
        model_nan.weight.grad = torch.full_like(model_nan.weight, float('nan'))
        model_nan.bias.grad = torch.zeros_like(model_nan.bias)
    gn = _grad_norm(model_nan)
    print(f"Case 4 (NaN Grad): Expected 0.0 (fixed), Got: {gn:.4f}")
    assert gn == 0.0

    print("\n✅ All tests passed!")

if __name__ == '__main__':
    test_grad_norm()
