

# MOF-LENS: Latent Evolutionary Navigation System for Smart MOF Discovery and Optimization by the Lotus Effect Algorithm

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)  


**MOF-LENS** is a **drug-agnostic, AI-powered platform** for designing **pH-responsive metal-organic frameworks (MOFs)** as **intelligent nanocarriers**.  
It integrates:
- **Latent-space kNN mapping** from 6D structural/chemical features to real MOFs,
- **Lotus Effect Algorithm (LEA)** — a bio-inspired optimizer with **adaptive Lévy flights** and **self-cleaning diversity control**,
- **Multi-objective fitness** balancing **pore size**, **drug compatibility**, **pH stability**, and **biocompatibility**.

Fully validated with:
- **Molecular docking** (pH 7.4 vs 5.5)
- **Paclitaxel (PTX) retargeting** (3-line config change)
- **30-run statistical benchmarking** vs. PSO, GA, Random Search


<p align="center">
  <img src="MOF-LENS.png" alt="MOF-LENS Overview" width="500" height="600">
</p>

---

## Repository Structure

```bash
MOF-LENS/
│
├── data/
│   └── MOF.csv                    # 10,000+ MOFs: Refcode, PLD, ASA, void_fraction, linker SMILES, etc.
│
├── results/                       # DOX optimization outputs
│   ├── raw/                       # Per-run results (LEA_run_0.csv, ...)
│   ├── plots/                     # Convergence, SHAP, sensitivity, diversity
│   ├── top5_frequency_lea.csv     # Top 5 MOFs by frequency + IUPAC names
│   └── top5_with_docking.csv      # + ΔG (pH 7.4 & 5.5), ΔΔG, docking poses
│
├── results_ptx/                   # Paclitaxel retargeting demo
│   └── top5_ptx.csv               # PTX-optimized MOFs (PLD ~21 Å)
│
├── src/
│   ├── algorithms.py              # LEA, PSO, GA, Random Search, Filter
│   ├── fitness.py                 # Fitness: PLD, chem_sim, pH, NH₂, toxicity, hydrophobicity
│   ├── data_preprocessing.py      # kNN latent space, normalization, SMILES sanitization
│   ├── experiments.py             # Full pipeline: 30 runs, stats, plots
│   ├── validation/docking.py      # AutoDock Vina (pH 7.4 & 5.5)
│   └── analysis.py                # SHAP, weight sensitivity, diversity ablation
│
├── config.yaml                    # DOX optimization (baseline)
├── config_ptx.yaml                # Paclitaxel retargeting (3 changes)
├── main.py                        # Run full DOX experiments
├── demo_retargeting.py            # Proof-of-concept: PTX in <1 min
├── ListTop5.py                    # Generate top 5 + IUPAC + docking merge
│
├── requirements.txt               # pip install -r requirements.txt
├── README.md                      # This file
└── LICENSE
