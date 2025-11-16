
# baselines/deterministic_filter.py
import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.neighbors import NearestNeighbors
from fitness import fitness_function_drug_delivery_batch


class DeterministicFilter:
    def __init__(
        self,
        *,
        population_size: int = 30,
        max_iterations: int = 100,
        df_norm: pd.DataFrame,
        ranges: Dict[str, Tuple[float, float]],
        df: pd.DataFrame,
        nbrs: NearestNeighbors,
        numerical_features: List[str],
        reference_fp_rdkit: Any,
        top_k: int = 5,
        config: Dict[str, Any],
    ) -> None:
        self.df_norm = df_norm
        self.ranges = ranges
        self.df = df
        self.nbrs = nbrs
        self.num_feat = numerical_features
        self.ref_fp = reference_fp_rdkit
        self.top_k = top_k
        self.config = config

        self.fitness_history = []
        self.diversity_score = 0.0

    def optimize(self) -> Tuple[List[np.ndarray], List[float], List[str], List[Dict], List[float]]:
        start = time.time()

        # Filter: PLD > 12, ASA > 1000
        mask = (self.df["pld (A)"] > 12.0) & (self.df["asa (A^2)"] > 1000.0)
        candidates = self.df[mask].index[:3000]

        scored = []
        for idx in candidates:
            sol = self.df_norm.loc[idx, self.num_feat].values
            fit, pen, chem = fitness_function_drug_delivery_batch(
                sol.reshape(1, -1), self.df_norm, self.ranges, self.nbrs,
                self.df, [], self.ref_fp, self.num_feat, config=self.config
            )[0]
            scored.append((fit, pen, chem, self.df.loc[idx, "Refcode"], sol))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:self.top_k]

        self.top_sols = [t[4] for t in top]
        self.top_fit = [t[0] for t in top]
        self.top_mofs = [t[3] for t in top]
        self.top_pen = [t[1] for t in top]
        self.top_chem = [t[2] for t in top]
        self.fitness_history = [t[0] for t in scored[:100]]

        self.runtime = time.time() - start
        return self.top_sols, self.top_fit, self.top_mofs, self.top_pen, self.top_chem
