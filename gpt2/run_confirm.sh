#!/bin/bash
# Confirmatory suite, gpt2-small, per CONFIRMED.md (frozen). TJL_CONFIRM=1.
set -x
export TJL_CONFIRM=1
M=gpt2
python experiments/20_capability.py $M            > results/confirm_capability.out 2>&1
python experiments/50_reasoning.py readout $M     > results/confirm_c3_readout.out 2>&1
python experiments/50_reasoning.py swap $M        > results/confirm_c3_swap.out 2>&1
python experiments/50_reasoning.py crossfn $M     > results/confirm_c3_crossfn.out 2>&1
python experiments/50_reasoning.py probe $M       > results/confirm_c3_probe.out 2>&1
python experiments/30_report.py corr $M           > results/confirm_c1_corr.out 2>&1
python experiments/30_report.py swap $M           > results/confirm_c1_swap.out 2>&1
python experiments/30_report.py inject $M         > results/confirm_c1_inject.out 2>&1
python experiments/40_modulation.py $M            > results/confirm_c2.out 2>&1
python experiments/42_imagine.py $M               > results/confirm_c2_imagine.out 2>&1
python experiments/70_selectivity.py same_latent $M > results/confirm_c5a.out 2>&1
python experiments/71_ablation.py $M 1            > results/confirm_c5b_k1.out 2>&1
python experiments/71_ablation.py $M 3            > results/confirm_c5b_k3.out 2>&1
python experiments/60_flexibility.py $M           > results/confirm_c4.out 2>&1
echo CONFIRM SUITE DONE
