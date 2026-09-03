"""Dictionary similarity between layers, controlling for the dominant direction (run from v2/).
Even after vocabulary-mean centering, one principal component carries 26-37% of each layer's
dictionary variance at layers 0-10 (2% at 11), so plain linear CKA is ~1 everywhere.  Variants:
drop the top-5/20 PCs, or mean squared canonical correlation between top-r subspaces.
Result: smooth drift with depth, no block structure.  Writes cka.json."""
import sys, torch, json
sys.path.insert(0,'.')
from jl.model import Lensed
torch.set_num_threads(1); torch.manual_seed(0)
import os
lm=Lensed(device=os.environ.get('DEVICE','cpu')); LAYERS=list(range(12))
idx=torch.randperm(lm.vocab)[:3000]
D={}; SV={}
for L in LAYERS:
    V=lm.V(L)[idx]; V=V-V.mean(0); D[L]=V; SV[L]=torch.linalg.svd(V,full_matrices=False)
def cka(A,B):   # linear CKA, feature-space form
    return float((A.T@B).norm()**2/((A.T@A).norm()*(B.T@B).norm()))
def whiten(L,r): return SV[L][0][:,:r]
def drop_top(L,k): U,S,Vh=SV[L]; S=S.clone(); S[:k]=0; return U@torch.diag(S)@Vh
W={r:{L:whiten(L,r) for L in LAYERS} for r in [50,100,300]}; T5={L:drop_top(L,5) for L in LAYERS}; T20={L:drop_top(L,20) for L in LAYERS}
out={}
for name,f in [('linear',lambda a,b:cka(D[a],D[b])),('drop_top5',lambda a,b:cka(T5[a],T5[b])),('drop_top20',lambda a,b:cka(T20[a],T20[b])),
               ('mean_cca_r50',lambda a,b:cka(W[50][a],W[50][b])),('mean_cca_r100',lambda a,b:cka(W[100][a],W[100][b])),('mean_cca_r300',lambda a,b:cka(W[300][a],W[300][b]))]:
    M=[[f(a,b) for b in LAYERS] for a in LAYERS]; out[name]=M
    print('\n'+name,flush=True); [print(f'{a:2d} '+' '.join(f'{M[a][b]:.2f}' for b in LAYERS),flush=True) for a in LAYERS]
print('\nlayer: top-1 PC share, top-5 PC share, #PCs for 90% var')
for L in LAYERS:
    s=SV[L][1]**2; c=s.cumsum(0)/s.sum(); print(L, f'{float(s[0]/s.sum()):.2f}', f'{float(s[:5].sum()/s.sum()):.2f}', int((c<0.9).sum())+1)
json.dump(out,open('explore/band/cka.json','w'))
