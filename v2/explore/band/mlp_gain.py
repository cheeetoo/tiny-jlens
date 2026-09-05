"""MLP gain of J-lens directions (paper Fig 32; run from v2/).  For source layer L, the output
norm of block L+1's MLP on a unit direction v (after that block's pre-LN), normalized by the
median over isotropic random unit directions.  2000 J-lens vectors per layer, raw and centered.
Writes mlp_gain.json."""
import sys, torch, json, os
sys.path.insert(0,'.')
from jl.model import Lensed
torch.set_num_threads(1); torch.manual_seed(0)
lm=Lensed(device=os.environ.get('DEVICE','cpu'))
blocks=lm.m.layers  # HF GPT2Block modules
def mlp_out(block, V):            # V [n, d] unit directions
    return block.mlp(block.ln_2(V)).norm(dim=-1)
out={}
idx=torch.randperm(lm.vocab)[:2000]
with torch.no_grad():
    for L in range(11):
        blk=blocks[L+1]
        rnd=torch.randn(2000,lm.d); rnd=rnd/rnd.norm(dim=-1,keepdim=True)
        base=mlp_out(blk,rnd).median()
        Vc=lm.V(L)[idx]; Vc=Vc/Vc.norm(dim=-1,keepdim=True)
        Vr=(lm.U@lm.J[L])[idx]; Vr=Vr/Vr.norm(dim=-1,keepdim=True)
        g_c=(mlp_out(blk,Vc)/base).median().item(); g_r=(mlp_out(blk,Vr)/base).median().item()
        # neuron output directions of the same block as a second control (paper's comparison set)
        W=blk.mlp.c_proj.weight  # [d_ff, d] in HF Conv1D
        Wn=W[torch.randperm(W.shape[0])[:2000]]; Wn=Wn/Wn.norm(dim=-1,keepdim=True)
        g_n=(mlp_out(blk,Wn)/base).median().item()
        out[L]=dict(centered=g_c,raw=g_r,neuron=g_n); print(f'L{L:2d}  centered J-lens {g_c:.2f}   raw J-lens {g_r:.2f}   neuron dirs {g_n:.2f}',flush=True)
json.dump(out,open('explore/band/mlp_gain.json','w'))
