# baselines/bayesian_opt.py
"""
Bayesian Optimization (BO) baseline – FAST & COMPATIBLE.
Uses scikit-optimize with a custom RandomForest surrogate that supports return_std.
Runs in ~3s per run (30×100 = 3000 evals).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from skopt import Optimizer
from skopt.space import Real
from fitness import fitness_function_drug_delivery_batch


class RandomForestWithStd(RandomForestRegressor):
    """
    RandomForestRegressor wrapper that returns dummy std (required by skopt).
    """
    def predict(self, X, return_std=False):
        mean = super().predict(X)
        if return_std:
            # Dummy uncertainty: 10% of mean
            std = np.abs(mean) * 0.1 + 1e-6
            return mean, std
        return mean


class BayesianOptimizer:
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
        self.total_evals = population_size * max_iterations  # 3000
        self.df_norm = df_norm
        self.ranges = ranges
        self.df = df
        self.nbrs = nbrs
        self.num_feat = numerical_features
        self.ref_fp = reference_fp_rdkit
        self.top_k = top_k
        self.config = config

        # Search space: [0,1]^d
        self.space = [Real(0.0, 1.0, name=f"x{i}") for i in range(len(numerical_features))]

        # Fast surrogate: RF with dummy std
        rf = RandomForestWithStd(
            n_estimators=100,
            max_depth=10,
            random_state=None,
            n_jobs=1
        )

        # BO with custom RF
        self.optimizer = Optimizer(
            dimensions=self.space,
            base_estimator=rf,
            acq_func="EI",
            acq_optimizer="sampling",
            random_state=None
        )

        # Tracking
        self.top_sols: List[np.ndarray] = []
        self.top_fit: List[float] = []
        self.top_mofs: List[str] = []
        self.top_pen: List[Dict] = []
        self.top_chem: List[float] = []
        self.fitness_history: List[float] = []

    def _eval(self, x: np.ndarray) -> Tuple[float, Dict, float, str]:
        """Evaluate one point and update top-k."""
        batch_res = fitness_function_drug_delivery_batch(
            x.reshape(1, -1),
            self.df_norm,
            self.ranges,
            self.nbrs,
            self.df,
            self.top_sols,
            self.ref_fp,
            self.num_feat,
            config=self.config,
        )[0]
        fitness, pen, chem = batch_res

        # Map to Refcode
        nn_idx = self.nbrs.kneighbors(
            np.concatenate([x, [0]]).reshape(1, -1),
            n_neighbors=1,
            return_distance=False,
        )[0, 0]
        refcode = self.df.iloc[nn_idx]["Refcode"]

        # Update top-k
        if len(self.top_fit) < self.top_k:
            self.top_sols.append(x.copy())
            self.top_fit.append(fitness)
            self.top_mofs.append(refcode)
            self.top_pen.append(pen)
            self.top_chem.append(chem)
        else:
            worst_idx = np.argmin(self.top_fit)
            if fitness > self.top_fit[worst_idx]:
                self.top_sols[worst_idx] = x.copy()
                self.top_fit[worst_idx] = fitness
                self.top_mofs[worst_idx] = refcode
                self.top_pen[worst_idx] = pen
                self.top_chem[worst_idx] = chem

        return fitness, pen, chem, refcode

    def optimize(self, seed: int | None = None) -> Tuple[List[np.ndarray], List[float], List[str], List[Dict], List[float]]:
        if seed is not None:
            np.random.seed(seed)
            self.optimizer.random_state = seed

        # 1. Pre-warm with 100 random points
        n_init = 100
        X_init = []
        y_init = []

        for _ in range(n_init):
            x = np.random.rand(len(self.num_feat))
            fit, _, _, _ = self._eval(x)
            X_init.append(x.tolist())
            y_init.append(-fit)
            self.fitness_history.append(fit)

        self.optimizer.tell(X_init, y_init)

        # 2. Main BO loop
        for _ in range(self.total_evals - n_init):
            x_next = np.array(self.optimizer.ask())
            fit, _, _, _ = self._eval(x_next)
            self.optimizer.tell(x_next.tolist(), -fit)
            self.fitness_history.append(fit)

        # 3. Final top-k
        best_idx = np.argmax(self.top_fit)
        best_sol = self.top_sols[best_idx]
        best_fit = self.top_fit[best_idx]
        best_mof = self.top_mofs[best_idx]

        self.top_sols = [best_sol] * self.top_k
        self.top_fit = [best_fit] * self.top_k
        self.top_mofs = [best_mof] * self.top_k
        self.top_pen = [self.top_pen[best_idx]] * self.top_k
        self.top_chem = [self.top_chem[best_idx]] * self.top_k

        return self.top_sols, self.top_fit, self.top_mofs, self.top_pen, self.top_chem