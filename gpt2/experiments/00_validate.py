"""Validation: our readout == reference apply(); the lens-vector identity;
exact centering-invariance of readouts; a smoke-test swap on gpt2-small.

Run:  python experiments/00_validate.py
"""

import sys

sys.path.insert(0, "/tiny-jlens/gpt2")

import torch

import core

kit = core.Kit("gpt2")
prompt = "The capital of the country where Polish is the primary language is"
ids = kit.encode(prompt)

# --- 1. exact match with the reference implementation -----------------------
ref_logits, model_logits, ref_ids = kit.lens.apply(kit.model, prompt)
assert torch.equal(ref_ids.cpu(), ids.cpu())
resid = kit.residuals(ids)
worst = 0.0
for l in kit.layers:
    ours = kit.lens_logits(resid[l], l).cpu()
    worst = max(worst, (ours - ref_logits[l]).abs().max().item())
print(f"[1] max |ours - reference apply()| over all layers/positions: {worst:.2e}")
assert worst < 1e-3

# --- 2. the vector identity: logit = <v,h>/sigma + beta ---------------------
l = 8
h = resid[l]
y = h @ kit.lens.jacobians[l].T
sigma = y.std(dim=-1, unbiased=False, keepdim=True)  # LN uses biased std
Vfull = kit.V(l)  # [vocab, d]
recon = (h @ Vfull.T) / sigma + (kit.beta[None, :] if kit.beta is not None else 0.0)
exact = kit.lens_logits(h, l)
err = (recon - exact).abs().max().item()
print(f"[2] max |<v,h>/sigma + beta  -  exact logits| at L{l}: {err:.2e}")
assert err < 1e-2

# --- 3. centering invariance of rankings (exact theorem) --------------------
vbar = kit.vbar(l)
shifted = ((h @ (Vfull - vbar[None, :]).T)) / sigma + (kit.beta[None, :] if kit.beta is not None else 0.0)
# same rankings <=> difference is constant across vocab at each position
diff = (recon - shifted)  # should equal <vbar,h>/sigma broadcast: constant per row
row_spread = (diff - diff.mean(dim=1, keepdim=True)).abs().max().item()
print(f"[3] centering shifts every logit by a per-position constant; spread: {row_spread:.2e}")
assert row_spread < 1e-3

# --- 4. smoke test: intermediate in lens + clamped swap flips answer --------
warsaw, madrid = kit.first_content_id(" Warsaw"), kit.first_content_id(" Madrid")
poland, spain = kit.tok_id(" Poland"), kit.tok_id(" Spain")
base = kit.model_logits(ids)[-1]
print(f"[4] clean top-1: {kit.decode([base.argmax().item()])!r}  "
      f"(rank Warsaw {int((base > base[warsaw]).sum())}, Madrid {int((base > base[madrid]).sum())})")
band = [7, 8, 9, 10]
r = kit.ranks(resid[8], 8, [poland])[-1, 0].item()
print(f"    ' Poland' lens rank at final position, L8: {r}")
edits = core.swap_clamped(kit, ids, band, [poland], [spain], alpha=1.0)
swapped = core.logits_with(kit, ids, edits)[-1]
print(f"    after Poland->Spain clamped swap (L7-10, raw): top-1 {kit.decode([swapped.argmax().item()])!r}  "
      f"(rank Madrid {int((swapped > swapped[madrid]).sum())})")
edits_c = core.swap_clamped(kit, ids, band, [poland], [spain], alpha=1.0, centered=True)
swapped_c = core.logits_with(kit, ids, edits_c)[-1]
print(f"    same swap, centered coordinates:            top-1 {kit.decode([swapped_c.argmax().item()])!r}  "
      f"(rank Madrid {int((swapped_c > swapped_c[madrid]).sum())})")
print("all validation checks passed")
