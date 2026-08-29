"""noaux_tc Mixture of Experts.

Following DeepSeek-V3's router: sigmoid gate + per-expert bias used
for top-k selection, with `n_group`/`topk_group` group scoring
(degenerate to no group scoring when both are 1).

The output is `routed_out + shared_out`, where:
    gate = sigmoid(W_gate x) + e_score_correction_bias
    topk_w, topk_idx = gate.topk(topk, dim=-1)
    topk_w = topk_w / topk_w.sum(-1, keepdim=True) * routed_scaling_factor
    routed_out = sum_k topk_w[k] * Expert_k(x)

Each Expert is a SwiGLU MLP with the GLM-style `swiglu_limit=10`
clamping on each side of the gate and up projections.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ExpertSwiGLU(nn.Module):
    """A single expert: W_down( silu(W_gate x) * W_up x )."""

    def __init__(self, d_model: int, expert_dim: int, swiglu_limit: float = 10.0):
        super().__init__()
        self.W_gate = nn.Linear(d_model, expert_dim, bias=False)
        self.W_up = nn.Linear(d_model, expert_dim, bias=False)
        self.W_down = nn.Linear(expert_dim, d_model, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.W_gate(x)).clamp(-self.swiglu_limit, self.swiglu_limit)
        up = F.silu(self.W_up(x)).clamp(-self.swiglu_limit, self.swiglu_limit)
        return self.W_down(gate * up)


class NoAuxMoE(nn.Module):
    """MoE with sigmoid router, noaux_tc selection, and shared expert."""

    def __init__(self, d_model: int, n_experts: int, topk: int,
                 expert_dim: int, n_shared: int = 1,
                 routed_scaling_factor: float = 2.5,
                 swiglu_limit: float = 10.0,
                 n_group: int = 1, topk_group: int = 1,
                 router_dtype: torch.dtype = torch.float32):
        super().__init__()
        if n_experts < topk:
            raise ValueError("n_experts must be >= topk")
        self.d_model = d_model
        self.n_experts = n_experts
        self.topk = topk
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor

        # Router: linear producing n_experts logits per token.
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        # Per-expert bias added after sigmoid; used for the group scoring
        # in noaux_tc. Stored as fp32 for stability.
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(n_experts, dtype=router_dtype),
        )
        # Routed experts.
        self.W_gate = nn.Parameter(torch.empty(n_experts, d_model, expert_dim))
        self.W_up = nn.Parameter(torch.empty(n_experts, d_model, expert_dim))
        self.W_down = nn.Parameter(torch.empty(n_experts, expert_dim, d_model))
        for p in (self.W_gate, self.W_up, self.W_down):
            nn.init.normal_(p, std=0.02)
        # Shared expert(s) — each is a full SwiGLU MLP.
        self.shared_experts = nn.ModuleList([
            _ExpertSwiGLU(d_model, expert_dim, swiglu_limit)
            for _ in range(n_shared)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model] -> [B, T, d_model].

        The router's last forward pass is stashed on `self.last_router_logits`
        so the training loop can read it (e.g. for a load-balancing aux
        loss). It is a no-op attribute for inference.
        """
        B, T, H = x.shape
        x_flat = x.reshape(B * T, H)
        # 1) Router.
        gate_logits = self.gate(x_flat)                    # [B*T, E]
        gate_sig = torch.sigmoid(gate_logits.float())      # [B*T, E] in fp32
        gate_for_select = gate_sig + self.e_score_correction_bias.to(gate_sig.dtype)
        # 2) noaux_tc group scoring. With n_group=1 and topk_group=1,
        #    group scoring is a no-op: we just take the global top-k.
        if self.n_group == 1 and self.topk_group == 1:
            topk_w, topk_idx = torch.topk(gate_for_select, self.topk, dim=-1)
        else:
            # Reshape [B*T, n_group, experts_per_group] and take the
            # top-`topk_group` groups by sum, then top-`topk` within.
            experts_per_group = self.n_experts // self.n_group
            grouped = gate_for_select.view(B * T, self.n_group, experts_per_group)
            group_scores = grouped.sum(dim=-1)             # [B*T, n_group]
            _, group_idx = torch.topk(group_scores, self.topk_group, dim=-1)
            # Mask: keep only the selected groups.
            group_mask = torch.zeros_like(grouped)
            group_mask.scatter_(1, group_idx.unsqueeze(-1).expand(-1, -1, experts_per_group), 1.0)
            masked = grouped.masked_fill(group_mask == 0, float("-inf"))
            topk_w, topk_idx_local = torch.topk(masked, self.topk, dim=-1)
            # Convert local (group, in-group) indices to global expert ids.
            group_offset = (
                torch.arange(self.n_group, device=grouped.device) * experts_per_group
            ).view(1, self.n_group, 1)
            topk_idx = topk_idx_local + group_offset
            topk_idx = topk_idx.view(B * T, self.topk)
            topk_w = topk_w.view(B * T, self.topk)
        # 3) Renormalize weights and apply scaling factor.
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-6)
        topk_w = topk_w * self.routed_scaling_factor
        # 4) Gather expert weights and apply.
        # x_flat: [B*T, H], topk_idx: [B*T, topk]
        sel_gate = self.W_gate[topk_idx]                   # [B*T, topk, H, D]
        sel_up = self.W_up[topk_idx]
        sel_down = self.W_down[topk_idx]
        # Per-token matmul. We compute expert outputs one token at a
        # time using the selected experts only (efficient at our scale).
        routed = torch.zeros(B * T, H, device=x.device, dtype=x.dtype)
        for k in range(self.topk):
            e_idx = topk_idx[:, k]                          # [B*T]
            w_g = sel_gate[:, k]                            # [B*T, H, D]
            w_u = sel_up[:, k]
            w_d = sel_down[:, k]                            # [B*T, D, H]
            gate = F.silu(torch.bmm(x_flat.unsqueeze(1), w_g).squeeze(1))
            gate = gate.clamp(-10.0, 10.0)
            up = F.silu(torch.bmm(x_flat.unsqueeze(1), w_u).squeeze(1))
            up = up.clamp(-10.0, 10.0)
            out = torch.bmm((gate * up).unsqueeze(1), w_d).squeeze(1)
            routed = routed + topk_w[:, k].to(x.dtype).unsqueeze(-1) * out
        routed = routed.view(B, T, H)
        # 5) Shared expert(s).
        shared = x
        for se in self.shared_experts:
            shared = se(shared)
        # Stash router stats for the training loop.
        self.last_router_logits = gate_for_select.detach()
        return routed + shared
