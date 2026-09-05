"""The library behind the five criteria: model + lens, residual-stream edits, interventions.

    jl.model             gpt2-small + the released J-lens: loading, J-lens vectors, exact readout
    jl.hooks             residual-stream edits as forward hooks
    jl.interventions     swap / coordinate swap / clamp / pursuit / loading / J-space ablation
    jl.stats             the summary statistics the criteria share
    jl.c1_report         one module per criterion: prompt material, the run, the summary
    jl.c2_modulation
    jl.c3_reasoning
    jl.c4_generalization
    jl.c5_selectivity
    jl.band              the structural statistics behind the choice of band
"""
from .model import BAND, LENS_PATH, REF_DATA, RESULTS, Lensed, ranks_of, results_dir
from .hooks import Edit, Session
from .interventions import (ablation_edits, ablation_select, clamp_edits, coord_swap_edits,
                            delta_edits, loading, pursuit, swap_edits, unit)
from .stats import median, sign_test, spearman, wilson

__all__ = ["BAND", "LENS_PATH", "REF_DATA", "RESULTS", "Lensed", "ranks_of", "results_dir",
           "Edit", "Session",
           "ablation_edits", "ablation_select", "clamp_edits", "coord_swap_edits", "delta_edits",
           "loading", "pursuit", "swap_edits", "unit",
           "median", "sign_test", "spearman", "wilson"]
