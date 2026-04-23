# Intelligent Software Engineering Coursework

## My Project
Improving Configuration Performance Tuning with Guided Local Search under a Limited Measurement Budget

## Overview
My project compares a Guided Local Search method against a Random Search baseline for configuration performance tuning using two real-world datasets from Lab 3:
- STORM: minimise latency
- x264: maximise performance

## Methods
- Random Search
- Guided Local Search using configuration similarity and random restarts

## Experimental Setup
- Budget: 50 measurements
- Runs: 30 per method per dataset
- Statistics: median, mean, standard deviation, Wilcoxon signed-rank test

## Repository Structure
- `data/` datasets
- `src/` source code
- `results/` raw and summary outputs

## Reproducibility
Run:
```bash
source venv/bin/activate
python3 src/run_experiments.py
