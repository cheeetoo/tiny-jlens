# The cone is gauge: diagnosis of the GPT-2 J-lens dictionary

*(experiments/10_cone.py → results/cone_gpt2.json; gpt2-small, authors'
released lens. 2026-08-17.)*

## The problem

At GPT-2 scale the 50,257 J-lens vectors are packed into d=768 dimensions and
previous work found them near-parallel ("the cone"): sparse decomposition
captured ~0.2% of probe variance, occupancy measured ~1, top-k ablation had no
dose axis, and matched-norm J/non-J privilege splits compressed. The question:
is this a fact about GPT-2's workspace, or an artifact of how the dictionary
is parameterized?

## The gauge argument

The exact lens readout is

    logit_t(h) = ⟨v_t, h⟩ / σ(J h) + β_t

(σ = final-LayerNorm std, β_t = unembedding bias term; verified to 2e-4
against the reference implementation). Replace every dictionary row by
a·v_t + u for any fixed vector u and any a > 0: every logit shifts by the same
per-position constant ⟨u,h⟩/σ and rescales by a — and softmax is invariant to
both. **No readout the lens ever produces can distinguish the dictionary from
any of its gauge transforms.** Verified numerically: the shift is constant
across the vocabulary to 1e-4.

So pairwise cosines, spans, pseudoinverse coordinates, and projections of the
raw dictionary are not properties of the lens until a gauge is fixed. The
canonical representative is the centered dictionary (vocabulary mean removed).

## Diagnosis (per layer L0–L10)

| quantity | value |
|---|---|
| share of dictionary second moment in the mean v̄ | **0.97–0.99** |
| cos(top principal axis, v̄) | **1.000** at every layer |
| mean pairwise \|cos\|, raw | 0.97–0.99 |
| mean pairwise \|cos\|, centered | 0.07–0.10 (random baseline ≈ 0.03) |
| top-PC share, centered | 6–9% (no dominant direction) |
| uniformity of v̄'s logit profile (std/\|mean\| over 50k tokens) | **0.013–0.022** (random directions: 0.08–0.8) |

The "cone axis" is not merely aligned with the mean — it *is* the mean, and
the mean writes the same logit to every vocabulary token to within ~2%. It is
pure gauge: a direction the softmax readout cannot see.

**Provenance.** v̄ = Jᵀū exactly, where ū is the mean effective unembedding
row. Two factors compound: (i) GPT-2's unembedding itself is degenerate — ū
carries 83% of the unembedding's second moment; (ii) J amplifies the ū
direction 2.6→6.9× across depth, vs ~1–1.7× for random directions. The model
really does transport a large shared "output energy" component forward — it is
just not token-discriminative content, and at d=768 it dwarfs the content.

## Consequences (measured on natural activations)

| operation | raw dictionary | centered |
|---|---|---|
| gradient pursuit k=16, variance explained | 0.0–0.6% | **3–16%** (L8–9: 13–16% — the paper's own range for Claude probes) |
| top-k ablation, norm removed k=1 → k=10 | 2–7% → 3–11% (no dose axis: every "atom" is v̄) | 7–13% → 10–27% (graded) |

## Method decision

All geometry (decomposition, occupancy, ablation, matched-norm splits,
coordinate reads) is computed in the centered gauge. Readouts are provably
unchanged. Interventions along *differences* of lens vectors (the paper's
coordinate swap moves h along v_s − v_t, in which v̄ cancels) are already
gauge-invariant in direction — which retroactively explains why swaps
transferred to tiny models while decompositions failed. The paper's §3.1
projection swap is *not* gauge-invariant (its coefficient ⟨h, v̂_s⟩ measures
mostly the shared axis in the raw gauge), predicting the previously observed
anti-directional behavior on GPT-2; the centered form is the meaningful one.

At Claude scale the correction is presumably negligible (the spread dwarfs
the mean); at GPT-2 scale it is 99% of the dictionary. Every experiment
downstream reports raw-gauge numbers alongside where they differ.

## Addendum: the cone across the GPT-2 family (2026-08-18)

gpt2-medium (d=1024): identical structure to small — v̄ share 0.98–0.99,
cos(axis, v̄)=1.000, uniformity ~0.02, centered |cos| 0.09–0.18.

gpt2-large (d=1280): **the cone is gone** — v̄ share 0.03–0.16 across layers,
J no longer amplifies the ū pullback (0.4–0.9× vs random ~1×), raw ≈ centered
on every measure (pairwise cos, PC shares, pursuit variance). The gauge
correction converges to a no-op exactly where the artifact disappears, which
is the behavior one wants from a principled fix: it makes 124M–355M
commensurable with 774M+ and costs nothing where it is not needed. With real
geometry at large, occupancy becomes 5–9 atoms (vs 2–3 at small/medium),
extending the capacity ladder toward the paper's ~25.
