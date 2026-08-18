#!/bin/bash
# Wait for the large n=1000 lens, then run its confirmatory + follow-ups.
set -x
python experiments/42_imagine.py gpt2-large > results/c2_imagine_gpt2-large.out 2>&1
python experiments/43_demand.py gpt2-large  > results/c2_demand_gpt2-large.out 2>&1
until [ -f ../lenses/gpt2-large/lens_meta.json ]; do sleep 300; done
export TJL_CONFIRM=1
M=gpt2-large
python experiments/20_capability.py $M            > results/confirm_lg_capability.out 2>&1
python experiments/50_reasoning.py readout $M     > results/confirm_lg_c3_readout.out 2>&1
python experiments/50_reasoning.py swap $M        > results/confirm_lg_c3_swap.out 2>&1
python experiments/50_reasoning.py crossfn $M     > results/confirm_lg_c3_crossfn.out 2>&1
python experiments/50_reasoning.py probe $M       > results/confirm_lg_c3_probe.out 2>&1
python experiments/30_report.py corr $M           > results/confirm_lg_c1_corr.out 2>&1
python experiments/30_report.py swap $M           > results/confirm_lg_c1_swap.out 2>&1
python experiments/30_report.py inject $M         > results/confirm_lg_c1_inject.out 2>&1
python experiments/40_modulation.py $M            > results/confirm_lg_c2.out 2>&1
python experiments/42_imagine.py $M               > results/confirm_lg_c2_imagine.out 2>&1
python experiments/70_selectivity.py same_latent $M > results/confirm_lg_c5a.out 2>&1
python experiments/71_ablation.py $M 1            > results/confirm_lg_c5b_k1.out 2>&1
python experiments/71_ablation.py $M 1 31:34      > results/confirm_lg_c5b_k1_L31-34.out 2>&1
python experiments/60_flexibility.py $M           > results/confirm_lg_c4.out 2>&1
echo LARGE CONFIRM DONE
