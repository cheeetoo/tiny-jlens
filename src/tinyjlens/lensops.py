"""Core J-lens operations for experiments.

Conventions
-----------
- `layer` indexes the residual stream AT THE OUTPUT of block `layer`
  (matching the reference implementation's ActivationRecorder).
- The lens readout at layer l is softmax(unembed(J_l @ h)) where unembed is
  the model's final norm + LM head (reference `model.unembed`).
- The *J-lens vector* for token t at layer l is the residual-space direction
  v_t = J_l^T @ u_t, where u_t is the token's effective unembedding row with
  the final norm's elementwise gain folded in (and, for LayerNorm models,
  mean-centred to account for the centring step). Inner products <v_t, h>
  then approximate lens logits up to the shared normalisation factor
  (paper §Methods, "per-token probe").

All heavy tensors live on GPU in float32.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

import jlens
from jlens.hooks import ActivationRecorder


def effective_unembedding(model) -> torch.Tensor:
    """[vocab, d_model] float32 GPU: LM-head rows with final-norm gain folded in.

    For RMSNorm(weight w): row_t <- u_t * w.
    For LayerNorm(weight w, bias b): row_t <- (u_t * w) centred over dims
    (the centring of LN makes logits invariant to the all-ones component).
    The bias only shifts logits per-token and does not affect directions.
    """
    lm_head: nn.Module = model._lm_head
    norm: nn.Module = model._final_norm
    U = lm_head.weight.detach().float()  # [vocab, d]
    w = getattr(norm, "weight", None)
    if w is not None:
        U = U * w.detach().float()[None, :]
    if isinstance(norm, nn.LayerNorm):
        U = U - U.mean(dim=1, keepdim=True)
    return U


@dataclass
class LensKit:
    """Bundle of model + lens + precomputed pieces used by every experiment."""

    model: object  # jlens.HFLensModel
    lens: jlens.JacobianLens
    U_eff: torch.Tensor  # [vocab, d] float32 GPU

    @classmethod
    def build(cls, model, lens) -> "LensKit":
        U = effective_unembedding(model).cuda()
        for l in lens.source_layers:
            lens.jacobians[l] = lens.jacobians[l].float().cuda()
        return cls(model=model, lens=lens, U_eff=U)

    # ---------- readout ----------

    @torch.no_grad()
    def residuals(self, input_ids: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
        """Residual stream [seq, d] float32 at each requested layer for a
        single prompt (input_ids [1, seq])."""
        record = sorted(set(layers))
        with ActivationRecorder(self.model.layers, at=record) as rec:
            self.model.forward(input_ids)
            return {l: rec.activations[l][0].detach().float() for l in record}

    @torch.no_grad()
    def lens_logits(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Full lens readout logits [..., vocab] for residuals h [..., d] at `layer`.
        Uses the model's real unembed (norm included), matching the reference."""
        transported = h @ self.lens.jacobians[layer].T
        return self.model.unembed(transported).float()

    @torch.no_grad()
    def lens_ranks_of(self, h: torch.Tensor, layer: int, token_ids: list[int]) -> torch.Tensor:
        """Rank (0 = top) of each token id in the lens readout at each position.
        h: [seq, d] -> returns [seq, len(token_ids)] int64."""
        logits = self.lens_logits(h, layer)  # [seq, vocab]
        # rank of token = number of tokens with strictly higher logit
        target = logits[:, token_ids]  # [seq, k]
        return (logits[:, None, :] > target[:, :, None]).sum(dim=2)

    # ---------- J-lens vectors ----------

    @torch.no_grad()
    def jlens_vector(self, layer: int, token_id: int) -> torch.Tensor:
        """Residual-space J-lens vector v_t = J_l^T u_t (not normalised)."""
        return self.lens.jacobians[layer].T @ self.U_eff[token_id]

    @torch.no_grad()
    def jlens_vectors(self, layer: int, token_ids: list[int]) -> torch.Tensor:
        """[len(ids), d]"""
        return self.U_eff[token_ids] @ self.lens.jacobians[layer]

    @torch.no_grad()
    def all_jlens_matrix(self, layer: int) -> torch.Tensor:
        """[vocab, d] — full J-lens vector matrix W_U_eff @ J_l. ~110MB fp32."""
        return self.U_eff @ self.lens.jacobians[layer]

    # ---------- sparse decomposition (gradient pursuit) ----------

    @torch.no_grad()
    def gradient_pursuit(
        self,
        x: torch.Tensor,
        layer: int,
        k: int,
        *,
        candidate_ids: torch.Tensor | None = None,
        n_candidates: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sparse non-negative decomposition of residual-space x [d] into k
        J-lens vectors at `layer` (paper §Methods "sparse decomposition").

        Greedy: at each step pick the candidate token whose (unit) lens vector
        has max positive inner product with the residual r, then refit
        non-negative least squares over the selected set.

        For tractability the candidate set defaults to the top `n_candidates`
        tokens of the lens readout of x (inner-product ranking); pass
        `candidate_ids` to override.

        Returns (token_ids [k], coeffs [k], reconstruction [d]).
        """
        Jl = self.lens.jacobians[layer]
        if candidate_ids is None:
            scores = self.U_eff @ (Jl @ x)
            candidate_ids = scores.topk(n_candidates).indices
        V = (self.U_eff[candidate_ids] @ Jl)  # [C, d]
        Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-8)

        selected: list[int] = []
        coeffs = torch.zeros(0, device=x.device)
        r = x.clone()
        for _ in range(k):
            corr = Vn @ r
            if selected:
                corr[torch.tensor(selected, device=x.device)] = -torch.inf
            best = int(corr.argmax())
            if corr[best] <= 0:
                break
            selected.append(best)
            A = V[selected]  # [s, d]
            # non-negative least squares via projected gradient (small s)
            G = A @ A.T
            b = A @ x
            c = torch.linalg.lstsq(G + 1e-6 * torch.eye(len(selected), device=x.device), b.unsqueeze(1)).solution.squeeze(1)
            c = c.clamp_min(0.0)
            step = 1.0 / (torch.linalg.matrix_norm(G, 2) + 1e-6)
            prev = None
            for _ in range(400):
                grad = G @ c - b
                c = (c - step * grad).clamp_min(0.0)
                if prev is not None and (c - prev).norm() < 1e-6 * (c.norm() + 1e-9):
                    break
                prev = c.clone()
            coeffs = c
            r = x - A.T @ c
        if not selected:
            return (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0),
                torch.zeros_like(x),
            )
        token_ids = candidate_ids[torch.tensor(selected, device=candidate_ids.device)]
        recon = V[selected].T @ coeffs
        return token_ids.cpu(), coeffs.cpu(), recon
