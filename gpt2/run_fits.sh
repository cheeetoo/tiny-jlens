#!/bin/bash
# Sequential background lens fits: gpt2-medium then gpt2-large (Neuronpedia recipe).
set -x
python fit_lens.py gpt2-medium ../lenses/gpt2-medium --n-prompts 1000 > ../lenses/gpt2-medium/fit.log 2>&1
python fit_lens.py gpt2-large  ../lenses/gpt2-large  --n-prompts 1000 > ../lenses/gpt2-large/fit.log 2>&1
