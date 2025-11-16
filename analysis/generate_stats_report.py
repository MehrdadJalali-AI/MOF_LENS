
# analysis/generate_stats_report.py
import os
import numpy as np
import pandas as pd
from typing import Dict, List

def generate_stats_report(all_results: Dict[str, List]) -> None:
    stats = []
    for name, runs in all_results.items():
        best_fits = [r['best_fitness'] for r in runs]
        final_fits = [r['final_fitness'] for r in runs]
        runtimes = [r['runtime'] for r in runs]
        diversities = [r['diversity'] for r in runs]

        stats.append({
            'Method': name,
            'Best Fitness (mean ± std)': f"{np.mean(best_fits):.4f} ± {np.std(best_fits):.4f}",
            'Final Fitness (mean ± std)': f"{np.mean(final_fits):.4f} ± {np.std(final_fits):.4f}",
            'Runtime (s, mean ± std)': f"{np.mean(runtimes):.2f} ± {np.std(runtimes):.2f}",
            'Diversity (mean ± std)': f"{np.mean(diversities):.4f} ± {np.std(diversities):.4f}",
        })

    df = pd.DataFrame(stats)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/summary_stats.csv", index=False)
    print("Stats report → results/summary_stats.csv")
