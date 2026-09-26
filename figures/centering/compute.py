"""Geometry of raw vs centered J-lens vectors, all layers, for the figure."""
import numpy as np, torch, jl, pathlib
OUT = pathlib.Path(__file__).resolve().parent
torch.manual_seed(0)
lm = jl.Lensed(device="cpu")
V_ = lm.vocab
sub = torch.randperm(V_)[:4000]                      # vocab subsample for pairwise stats
labels = [" the", " of", ",", " Paris", " apple", " seven", " dog"]
lab_ids = [lm.tid(s) for s in labels]
out = {}
per_layer = []
for L in range(12):
    V = lm.U @ lm.J[L]                               # raw: row t = J_L^T w_t   [vocab, d]
    mu = V.mean(0)
    C = V - mu
    # cosine of each raw vector to the mean direction; centered vectors likewise
    cos_mu_raw = (V @ mu) / (V.norm(dim=1) * mu.norm())
    cos_mu_cen = (C @ mu) / (C.norm(dim=1) * mu.norm())
    # pairwise cosines on the subsample
    def pw(M):
        Mn = torch.nn.functional.normalize(M[sub], dim=1)
        G = Mn @ Mn.T
        iu = torch.triu_indices(len(sub), len(sub), 1)
        return G[iu[0], iu[1]]
    pw_raw, pw_cen = pw(V), pw(C)
    # PCA of the raw subsample (uncentered SVD gives the mean direction as PC1; use centered PCA basis
    # for both so the two panels share axes)
    Cs = C[sub]
    Uc, Sc, Wc = torch.linalg.svd(Cs, full_matrices=False)
    basis = Wc[:2].T                                  # [d, 2]
    proj_raw = V[sub] @ basis
    proj_cen = Cs @ basis
    # also: basis with PC1 = mean direction, PC2 = top centered PC orthogonal to it
    e1 = mu / mu.norm()
    Cperp = Cs - (Cs @ e1)[:, None] * e1                # centered vectors, mean direction removed
    _, _, Wp = torch.linalg.svd(Cperp, full_matrices=False)
    e2 = Wp[0]; e2 = e2 - (e2 @ e1) * e1; e2 = e2 / e2.norm()
    basis_mu = torch.stack([e1, e2], 1)
    proj_raw_mu = V[sub] @ basis_mu
    proj_cen_mu = Cs @ basis_mu
    lab_raw_mu = V[lab_ids] @ basis_mu
    lab_cen_mu = C[lab_ids] @ basis_mu
    var_pc1 = float(Sc[0]**2 / (Sc**2).sum())
    d = dict(L=L, mu_norm=float(mu.norm()), mean_raw_norm=float(V.norm(dim=1).mean()),
             mean_cen_norm=float(C.norm(dim=1).mean()),
             pw_raw_mean=float(pw_raw.mean()), pw_cen_mean=float(pw_cen.mean()),
             cos_mu_raw_mean=float(cos_mu_raw.mean()), var_pc1_centered=var_pc1)
    per_layer.append(d); print(d, flush=True)
    out[f"L{L}_cos_mu_raw"] = cos_mu_raw.numpy(); out[f"L{L}_cos_mu_cen"] = cos_mu_cen.numpy()
    out[f"L{L}_pw_raw"] = pw_raw[::50].numpy(); out[f"L{L}_pw_cen"] = pw_cen[::50].numpy()
    out[f"L{L}_proj_raw_mu"] = proj_raw_mu.numpy(); out[f"L{L}_proj_cen_mu"] = proj_cen_mu.numpy()
    out[f"L{L}_lab_raw_mu"] = lab_raw_mu.numpy(); out[f"L{L}_lab_cen_mu"] = lab_cen_mu.numpy()
    out[f"L{L}_raw_norm"] = V.norm(dim=1).numpy(); out[f"L{L}_cen_norm"] = C.norm(dim=1).numpy()
out["labels"] = np.array([str(x) for x in labels]); out["sub"] = sub.numpy()
import json; json.dump(per_layer, open(f"{OUT}/per_layer.json", "w"), indent=1)
np.savez(f"{OUT}/geom.npz", **out)
print("saved")
