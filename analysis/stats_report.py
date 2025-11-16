# analysis/stats_report.py
"""
30-run statistics + p-values + effect sizes
"""

from scipy.stats import mannwhitneyu
import numpy as np
import pandas as pd

def generate_stats_report(all_results, config):
    lea_fits = [r['best_fitness'] for r in all_results.get('LEA', [])]
    summary = []

    for name, runs in all_results.items():
        fits = [r['best_fitness'] for r in runs]
        mean, sd = np.mean(fits), np.std(fits)
        ci_low, ci_high = np.percentile(fits, [2.5, 97.5])
        u, p = mannwhitneyu(lea_fits, fits, alternative='greater') if 'LEA' in all_results else (np.nan, np.nan)
        summary.append({
            'Method': name,
            'Mean': f"{mean:.4f}",
            'SD': f"{sd:.4f}",
            '95%CI': f"[{ci_low:.3f}, {ci_high:.3f}]",
            'p_vs_LEA': f"{p:.2e}" if p < 0.05 else "n.s.",
            'Runtime_s': np.mean([r.get('runtime', 0) for r in runs])
        })

    df = pd.DataFrame(summary)
    df.to_csv("results/summary_stats.csv", index=False)
    print("Stats report → results/summary_stats.csv")