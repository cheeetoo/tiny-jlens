#!/bin/bash
# Confirmatory suite per docs/CONFIRMATORY.md (frozen 2026-08-17)
set -x
cd /tiny-jlens
L=runs/smollm2-135m-it/lens.pt
S=1234
python3 scripts/c3_reasoning.py --lens $L --band 19:26 --alphas 1,2 --seed $S --phases filter,readout,swap,crossfn,timing,probe --k-gp 25 --out runs/confirm_c3.json > runs/confirm_c3.log 2>&1
python3 scripts/c3_reasoning.py --lens $L --band 19:26 --alphas 1 --seed $S --phases filter,probe --k-gp 50 --out runs/confirm_c3_k50.json > runs/confirm_c3_k50.log 2>&1
python3 scripts/c3_reasoning.py --lens $L --band 17:26 --alphas 1 --seed $S --phases filter,swap --out runs/confirm_c3_b17.json > runs/confirm_c3_b17.log 2>&1
python3 scripts/c3_reasoning.py --lens $L --band 21:26 --alphas 1 --seed $S --phases filter,swap --out runs/confirm_c3_b21.json > runs/confirm_c3_b21.log 2>&1
python3 scripts/c1_report.py --lens $L --band 19:26 --swap-mode proj --alpha 1 --seed $S --confirm --k-gp 16 --out runs/confirm_c1.json > runs/confirm_c1.log 2>&1
python3 scripts/c1_report.py --lens $L --band 19:26 --swap-mode coord --alpha 1 --seed $S --confirm --k-gp 50 --out runs/confirm_c1_coord_k50.json > runs/confirm_c1_coord_k50.log 2>&1
python3 scripts/c1d_introspection.py --lens $L --band 19:26 --strengths 0,8,16 --n-concepts 40 --out runs/confirm_c1d.json > runs/confirm_c1d.log 2>&1
python3 scripts/c5a_selectivity_language.py --lens $L --band 19:26 --single-pair --alt-map German:Spanish,Italian:French --confirm --out runs/confirm_c5a.json > runs/confirm_c5a.log 2>&1
python3 scripts/c5b_ablation.py --lens $L --band 19:26 --k 10 --confirm --out runs/confirm_c5b.json > runs/confirm_c5b.log 2>&1
python3 scripts/c2_modulation.py --lens $L --band 19:26 --confirm --out runs/confirm_c2.json > runs/confirm_c2.log 2>&1
python3 scripts/c2_variants.py --lens $L --out runs/confirm_c2v.json > runs/confirm_c2v.log 2>&1
python3 scripts/c2c_modulation_privilege.py --lens $L --band 19:26 --out runs/confirm_c2c.json > runs/confirm_c2c.log 2>&1
python3 scripts/c4_flexibility.py --lens $L --band 19:26 --alphas 1,2 --union --out runs/confirm_c4.json > runs/confirm_c4.log 2>&1
echo CONFIRMATORY_SUITE_DONE
