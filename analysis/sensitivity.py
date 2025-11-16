
# analysis/sensitivity.py
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.neighbors import NearestNeighbors
from fitness import fitness_function_drug_delivery_batch

def sensitivity_analysis(
    config: Dict,
    df_norm: pd.DataFrame,
    ranges: Dict,
    nbrs: NearestNeighbors,
    df: pd.DataFrame,
    ref_fp: Any,
    num_feat: List[str],
) -> None:
    X_6d = df_norm[num_feat].values
    nbrs_6d = NearestNeighbors(n_neighbors=1).fit(X_6d)

    np.random.seed(42)
    n_samples = 100
    base_solutions = np.random.rand(n_samples, len(num_feat))

    base_fits = []
    for sol in base_solutions:
        fit, _, _ = fitness_function_drug_delivery_batch(
            sol.reshape(1, -1),
            df_norm,
            ranges,
            nbrs_6d,
            df,
            [],
            ref_fp,
            num_feat,
            config=config,
        )[0]
        base_fits.append(fit)
    base_fits = np.array(base_fits)

    weights = config['fitness']['weights']
    results = []

    for w_key, orig_w in list(weights.items()):
        for delta in [-0.10, -0.05, 0.05, 0.10]:
            new_w = orig_w + delta
            if new_w < 0:
                continue

            weights[w_key] = new_w
            perturbed_fits = []
            for sol in base_solutions:
                fit, _, _ = fitness_function_drug_delivery_batch(
                    sol.reshape(1, -1),
                    df_norm,
                    ranges,
                    nbrs_6d,
                    df,
                    [],
                    ref_fp,
                    num_feat,
                    config=config,
                )[0]
                perturbed_fits.append(fit)

            impact = np.mean(np.abs(np.array(perturbed_fits) - base_fits))
            results.append({
                'weight': w_key,
                'delta': delta,
                'impact': round(impact, 6),
                'new_weight': round(new_w, 3),
            })
            weights[w_key] = orig_w

    df_out = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df_out.to_csv("results/weight_sensitivity.csv", index=False)
    print("Sensitivity analysis → results/weight_sensitivity.csv")
