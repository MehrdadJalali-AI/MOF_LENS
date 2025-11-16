# analysis/diversity_ablation.py
"""
Diversity ablation: vary lambda_div and measure fitness + diversity.
Fixes silhouette_score error when only 1 cluster.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import silhouette_score
from fitness import fitness_function_drug_delivery_batch

def diversity_ablation(config, df_norm, ranges, nbrs, df, ref_fp, num_feat):
    lambdas = [0.0, 0.01, 0.03, 0.10]
    results = []

    for lam in lambdas:
        config_copy = config.copy()
        config_copy['fitness']['lambda_div'] = lam

        # Generate 30 random solutions
        n_samples = 30
        population = np.random.rand(n_samples, len(num_feat))
        results_batch = fitness_function_drug_delivery_batch(
            population,
            df_norm,
            ranges,
            nbrs,
            df,
            [],
            ref_fp,
            num_feat,
            config=config_copy
        )
        fits = [r[0] for r in results_batch]

        # === DIVERSITY METRIC ===
        X = population
        if len(X) < 2:
            sil = 0.0
        else:
            # Simulate 2 clusters (e.g., high vs low fitness)
            labels = (np.array(fits) > np.median(fits)).astype(int)
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                sil = 0.0  # all same fitness → no diversity
            else:
                sil = silhouette_score(X, labels)

        results.append({
            "lambda_div": lam,
            "mean_fitness": round(np.mean(fits), 6),
            "std_fitness": round(np.std(fits), 6),
            "silhouette": round(sil, 6),
            "n_clusters": len(unique_labels)
        })

    # Save
    df_out = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df_out.to_csv("results/diversity_ablation.csv", index=False)
    print("Diversity ablation → results/diversity_ablation.csv")