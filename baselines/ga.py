# baselines/ga.py
"""
Genetic Algorithm (GA) baseline for MOF-LENS.
Standard binary GA with tournament selection, crossover, mutation.
Matches paper: 30 pop, 100 iter, 10,000+ MOFs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from fitness import fitness_function_drug_delivery_batch


class GeneticAlgorithm:
    """
    Simple Genetic Algorithm for comparison.
    - Binary representation (discretized to 100 bins per feature)
    - Tournament selection, uniform crossover, bit-flip mutation
    """
    def __init__(
        self,
        *,
        population_size: int = 30,
        max_iterations: int = 100,
        df_norm: pd.DataFrame,
        ranges: Dict[str, Tuple[float, float]],
        df: pd.DataFrame,
        nbrs,
        numerical_features: List[str],
        reference_fp_rdkit: Any,
        top_k: int = 5,
        config: Dict[str, Any],
    ) -> None:
        self.pop_size = population_size
        self.max_it = max_iterations
        self.df_norm = df_norm
        self.ranges = ranges
        self.df = df
        self.nbrs = nbrs
        self.num_feat = numerical_features
        self.ref_fp = reference_fp_rdkit
        self.top_k = top_k
        self.config = config

        # Discretize to 100 bins per feature → 7 bits per feature
        self.n_bins = 100
        self.bits_per_feat = int(np.ceil(np.log2(self.n_bins)))
        self.genome_len = len(numerical_features) * self.bits_per_feat

        # Initialize population
        self.population = [np.random.randint(0, 2, self.genome_len) for _ in range(population_size)]
        self.fitnesses = [0.0] * population_size

        # Top-k
        self.top_sols: List[np.ndarray] = []
        self.top_fit: List[float] = []
        self.top_mofs: List[str] = []
        self.top_pen: List[Dict] = []
        self.top_chem: List[float] = []
        self.fitness_history: List[float] = []

    def _decode(self, genome: np.ndarray) -> np.ndarray:
        """Decode binary genome to normalized [0,1] vector."""
        ind = np.zeros(len(self.num_feat))
        for i in range(len(self.num_feat)):
            start = i * self.bits_per_feat
            end = start + self.bits_per_feat
            bits = genome[start:end]
            val = int("".join(map(str, bits)), 2)
            ind[i] = min(val / (2**self.bits_per_feat - 1), 1.0)
        return ind

    def _evaluate(self, individuals_norm: List[np.ndarray]) -> List[Tuple[float, Dict, float]]:
        batch = np.array(individuals_norm)
        return fitness_function_drug_delivery_batch(
            batch,
            self.df_norm,
            self.ranges,
            self.nbrs,
            self.df,
            self.top_sols,
            self.ref_fp,
            self.num_feat,
            config=self.config
        )

    def _tournament(self, k: int = 3) -> int:
        candidates = np.random.choice(range(self.pop_size), k, replace=False)
        return candidates[np.argmax([self.fitnesses[i] for i in candidates])]

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        point = np.random.randint(1, self.genome_len - 1)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2

    def _mutate(self, ind: np.ndarray, rate: float = 0.01) -> np.ndarray:
        mask = np.random.rand(self.genome_len) < rate
        ind[mask] = 1 - ind[mask]
        return ind

    def _update_topk(self, sol: np.ndarray, fit: float, mof: str, pen: Dict, chem: float):
        if len(self.top_fit) < self.top_k:
            self.top_sols.append(sol.copy())
            self.top_fit.append(fit)
            self.top_mofs.append(mof)
            self.top_pen.append(pen)
            self.top_chem.append(chem)
        else:
            worst_idx = np.argmin(self.top_fit)
            if fit > self.top_fit[worst_idx]:
                self.top_sols[worst_idx] = sol.copy()
                self.top_fit[worst_idx] = fit
                self.top_mofs[worst_idx] = mof
                self.top_pen[worst_idx] = pen
                self.top_chem[worst_idx] = chem

    def optimize(self) -> Tuple[List[np.ndarray], List[float], List[str], List[Dict], List[float]]:
        for it in range(self.max_it):
            # Decode and evaluate
            individuals_norm = [self._decode(ind) for ind in self.population]
            results = self._evaluate(individuals_norm)
            for i, (fit, pen, chem) in enumerate(results):
                self.fitnesses[i] = fit
                sol_vec = np.concatenate([individuals_norm[i], [i]])
                nn_idx = self.nbrs.kneighbors(sol_vec.reshape(1, -1), n_neighbors=1, return_distance=False)[0, 0]
                refcode = self.df.iloc[nn_idx]["Refcode"]
                self._update_topk(individuals_norm[i], fit, refcode, pen, chem)
            self.fitness_history.append(max(self.fitnesses))

            # Selection, crossover, mutation
            new_pop = []
            for _ in range(self.pop_size // 2):
                p1_idx = self._tournament()
                p2_idx = self._tournament()
                c1, c2 = self._crossover(self.population[p1_idx], self.population[p2_idx])
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.extend([c1, c2])
            self.population = new_pop[:self.pop_size]

        order = np.argsort(self.top_fit)[::-1]
        return (
            [self.top_sols[i] for i in order],
            [self.top_fit[i] for i in order],
            [self.top_mofs[i] for i in order],
            [self.top_pen[i] for i in order],
            [self.top_chem[i] for i in order],
        )