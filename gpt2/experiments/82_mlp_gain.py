"""Broadcast across depth (paper §4.3): do MLP blocks preferentially amplify
J-space-aligned directions? Gain of MLP block l+1 on unit directions:
centered lens vectors vs the same MLP's own output-weight rows vs isotropic
random (gain 1 by normalization).

Run:  python experiments/82_mlp_gain.py [model]
"""

import json
import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
kit = core.Kit(MODEL)
torch.manual_seed(0)
N = 512

blocks = kit.model.layers  # gpt2 blocks: .mlp with ln_2 pre-norm


@torch.no_grad()
def mlp_out(l: int, V: torch.Tensor) -> torch.Tensor:
    """Norm of MLP_{l}'s output on unit inputs V [n, d] (applied after the
    block's own pre-LN, as in the forward pass)."""
    blk = blocks[l]
    return blk.mlp(blk.ln_2(V)).norm(dim=1)


out = {}
print(f"{MODEL}: MLP gain at block l+1 (normalized: random isotropic = 1)")
print(f"{'L':>3} {'lens(cen)':>10} {'mlp_out rows':>13}")
for l in kit.layers:
    if l + 1 >= kit.n_layers:
        break
    idx = torch.randperm(kit.U_eff.shape[0])[:N].tolist()
    Vlens = kit.vectors(l, idx, centered=True)
    Vlens = Vlens / Vlens.norm(dim=1, keepdim=True).clamp_min(1e-8)
    W = blocks[l].mlp.c_proj.weight.detach().float()  # [d_mlp? rows x d]
    W = W if W.shape[1] == kit.d_model else W.T
    ridx = torch.randperm(W.shape[0])[:N]
    Vmlp = W[ridx].cuda()
    Vmlp = Vmlp / Vmlp.norm(dim=1, keepdim=True).clamp_min(1e-8)
    R = torch.randn(N, kit.d_model, device="cuda")
    R = R / R.norm(dim=1, keepdim=True)
    g_rand = mlp_out(l + 1, R).median()
    g_lens = (mlp_out(l + 1, Vlens).median() / g_rand).item()
    g_mlp = (mlp_out(l + 1, Vmlp).median() / g_rand).item()
    out[l] = dict(lens=g_lens, mlp=g_mlp)
    print(f"{l:>3} {g_lens:>10.2f} {g_mlp:>13.2f}")
json.dump(out, open(f"/tiny-jlens/gpt2/results/mlp_gain_{MODEL}.json", "w"))
