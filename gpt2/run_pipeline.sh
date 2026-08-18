#!/bin/bash
# Self-driving overnight pipeline:
# 1. wait for the medium n=1000 lens -> run the medium CONFIRMATORY suite
# 2. wait for the large fit's checkpoint to reach ~275 prompts -> materialize
#    interim large lens -> run the large exploration chain
set -x
# band-matched medium C5b variant first (cheap)
python experiments/71_ablation.py gpt2-medium 1 19:22 > results/c5b_gpt2-medium_k1_L19-22.out 2>&1

until [ -f ../lenses/gpt2-medium/lens_meta.json ]; do sleep 180; done
export TJL_CONFIRM=1
M=gpt2-medium
python experiments/20_capability.py $M            > results/confirm_med_capability.out 2>&1
python experiments/50_reasoning.py readout $M     > results/confirm_med_c3_readout.out 2>&1
python experiments/50_reasoning.py swap $M        > results/confirm_med_c3_swap.out 2>&1
python experiments/50_reasoning.py crossfn $M     > results/confirm_med_c3_crossfn.out 2>&1
python experiments/50_reasoning.py probe $M       > results/confirm_med_c3_probe.out 2>&1
python experiments/30_report.py corr $M           > results/confirm_med_c1_corr.out 2>&1
python experiments/30_report.py swap $M           > results/confirm_med_c1_swap.out 2>&1
python experiments/30_report.py inject $M         > results/confirm_med_c1_inject.out 2>&1
python experiments/40_modulation.py $M            > results/confirm_med_c2.out 2>&1
python experiments/42_imagine.py $M               > results/confirm_med_c2_imagine.out 2>&1
python experiments/70_selectivity.py same_latent $M > results/confirm_med_c5a.out 2>&1
python experiments/71_ablation.py $M 1            > results/confirm_med_c5b_k1.out 2>&1
python experiments/71_ablation.py $M 3            > results/confirm_med_c5b_k3.out 2>&1
python experiments/60_flexibility.py $M           > results/confirm_med_c4.out 2>&1
unset TJL_CONFIRM
echo MEDIUM CONFIRM DONE

until python - <<'PY'
import torch, sys, os
p="/tiny-jlens/lenses/gpt2-large/lens_ckpt.pt"
sys.exit(0 if os.path.exists(p) and torch.load(p,map_location="cpu",weights_only=True)["n_done"]>=275 else 1)
PY
do sleep 300; done
python - <<'PY'
import torch, jlens
ck = torch.load("/tiny-jlens/lenses/gpt2-large/lens_ckpt.pt", map_location="cpu", weights_only=True)
n = ck["n_done"]
lens = jlens.JacobianLens(jacobians={l: s / n for l, s in ck["jacobian_sum"].items()}, n_prompts=n, d_model=1280)
lens.save("/tiny-jlens/lenses/gpt2-large/lens.pt")
print("interim large lens:", lens)
PY
M=gpt2-large
python experiments/10_cone.py $M              > results/cone_$M.out 2>&1
python experiments/80_structure.py band $M    > results/band_$M.out 2>&1
python experiments/50_reasoning.py readout $M > results/c3_readout_$M.out 2>&1
python experiments/50_reasoning.py swap $M    > results/c3_swap_$M.out 2>&1
python experiments/50_reasoning.py crossfn $M > results/c3_crossfn_$M.out 2>&1
python experiments/50_reasoning.py probe $M   > results/c3_probe_$M.out 2>&1
python experiments/30_report.py corr $M       > results/c1_corr_$M.out 2>&1
python experiments/30_report.py swap $M       > results/c1_swap_$M.out 2>&1
python experiments/30_report.py inject $M     > results/c1_inject_$M.out 2>&1
python experiments/40_modulation.py $M        > results/c2_$M.out 2>&1
python experiments/70_selectivity.py same_latent $M > results/c5a_$M.out 2>&1
python experiments/71_ablation.py $M 1        > results/c5b_${M}_k1.out 2>&1
python experiments/60_flexibility.py $M       > results/c4_$M.out 2>&1
python experiments/80_structure.py occupancy $M > results/occupancy_$M.out 2>&1
echo LARGE EXPLORATION DONE
