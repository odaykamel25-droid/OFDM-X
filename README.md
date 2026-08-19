# OFDM-X — Reproducibility Package

This repository contains the validated source code and supplementary numerical data supporting the manuscript.

## Structure

- `figures/` — final figure-generation scripts
- `fcae/` — FCAE V8 scripts
- `rl/` — final Actor-Critic adaptive waveform selection
- `benchmarks/` — PAPR/complexity benchmark
- `rsma_isac/` — RSMA and ISAC evaluation
- `data/` — validated numerical results
- `supplementary/` — supplementary material
- `legacy/` — development and intermediate scripts retained for traceability

## Validated operating points

- Vehicular: 200 Hz, 1 path, P(AFDM)=0.102, selected OFDM
- LEO: 1000 Hz, 3 paths, P(AFDM)=0.967, selected AFDM
- ISAC: 450 Hz, 7 paths, P(AFDM)=0.963, selected AFDM

The `legacy/` directory is not used as the source of final reported results.
