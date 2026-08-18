#!/bin/bash
# Full exploration chain on gpt2-medium (interim lens, 275 prompts).
set -x
M=gpt2-medium
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
python experiments/42_imagine.py $M           > results/c2_imagine_$M.out 2>&1
python experiments/70_selectivity.py same_latent $M > results/c5a_$M.out 2>&1
python experiments/71_ablation.py $M 1        > results/c5b_${M}_k1.out 2>&1
python experiments/71_ablation.py $M 3        > results/c5b_${M}_k3.out 2>&1
python experiments/60_flexibility.py $M       > results/c4_$M.out 2>&1
python experiments/80_structure.py occupancy $M > results/occupancy_$M.out 2>&1
echo MEDIUM CHAIN DONE
