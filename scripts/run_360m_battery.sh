#!/bin/bash
# SmolLM2-360M ladder battery: band first, then C2-threshold question + core ladder points.
cd /tiny-jlens
M=HuggingFaceTB/SmolLM2-360M-Instruct
L=runs/smollm2-360m-it/lens.pt
python3 scripts/band_analysis.py --model $M --lens $L --n-prompts 16 --out runs/band_360m.json > runs/band_360m.log 2>&1
# band parsed by the caller; default heuristic below is replaced after inspection
