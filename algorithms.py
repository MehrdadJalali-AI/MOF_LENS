
# algorithms.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from fitness import fitness_function_drug_delivery_batch


class LotusEffectAlgorithm:
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

        self.population = [np.random.rand(len(numerical_features)) for _ in range(population_size)]
        self.fitnesses = [0.0] * population_size

        self.top_sols: List[np.ndarray] = []
        self.top_fit: List[float] = []
        self.top_mofs: List[str] = []
        self.top_pen: List[Dict] = []
        self.top_chem: List[float] = []

        self.fitness_history: List[float] = []
        self.diversity_score: float = 0.0

    def _evaluate(self, individuals: List[np.ndarray]) -> List[Tuple[float, Dict, float]]:
        batch = np.array(individuals)
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

    def _lotus_mutation(self, ind: np.ndarray, best: np.ndarray, iteration: int) -> np.ndarray:
        r1, r2 = np.random.rand(2)
        F = 0.5 + 0.3 * np.exp(-iteration / self.max_it)
        mutant = ind + F * (best - ind) + r1 * (np.random.rand(len(ind)) - 0.5)
        return np.clip(mutant, 0.0, 1.0)

    def _self_cleaning(self):
        idx_sorted = np.argsort(self.fitnesses)
        n_replace = max(1, self.pop_size // 5)
        for i in idx_sorted[:n_replace]:
            self.population[i] = np.random.rand(len(self.num_feat))

    def optimize(self) -> Tuple[List[np.ndarray], List[float], List[str], List[Dict], List[float]]:
        for it in range(self.max_it):
            results = self._evaluate(self.population)
            for i, (fit, pen, chem) in enumerate(results):
                self.fitnesses[i] = fit
                sol_vec = np.concatenate([self.population[i], [i]])
                nn_idx = self.nbrs.kneighbors(sol_vec.reshape(1, -1), n_neighbors=1, return_distance=False)[0, 0]
                refcode = self.df.iloc[nn_idx]["Refcode"]
                self._update_topk(self.population[i], fit, refcode, pen, chem)
            self.fitness_history.append(max(self.fitnesses))

            best_idx = np.argmax(self.fitnesses)
            best = self.population[best_idx]

            new_pop = []
            for i in range(self.pop_size):
                mutant = self._lotus_mutation(self.population[i], best, it)
                trial_fit = self._evaluate([mutant])[0][0]
                if trial_fit > self.fitnesses[i]:
                    new_pop.append(mutant)
                    self.fitnesses[i] = trial_fit
                else:
                    new_pop.append(self.population[i])
            self.population = new_pop

            if it % 10 == 0:
                self._self_cleaning()

        if len(self.top_sols) > 1:
            dists = [np.linalg.norm(a - b) for i, a in enumerate(self.top_sols) for b in self.top_sols[i+1:]]
            self.diversity_score = np.mean(dists) if dists else 0.0

        order = np.argsort(self.top_fit)[::-1]
        return (
            [self.top_sols[i] for i in order],
            [self.top_fit[i] for i in order],
            [self.top_mofs[i] for i in order],
            [self.top_pen[i] for i in order],
            [self.top_chem[i] for i in order],
        )


class ParticleSwarmOptimization:
    def __init__(self, **kwargs):
        self.config = kwargs.pop('config')
        self.pop_size = kwargs.pop('population_size')
        self.max_it = kwargs.pop('max_iterations')
        self.df_norm = kwargs.pop('df_norm')
        self.ranges = kwargs.pop('ranges')
        self.df = kwargs.pop('df')
        self.nbrs = kwargs.pop('nbrs')
        self.num_feat = kwargs.pop('numerical_features')
        self.ref_fp = kwargs.pop('reference_fp_rdkit')
        self.top_k = kwargs.pop('top_k', 5)

        self.population = [np.random.rand(len(self.num_feat)) for _ in range(self.pop_size)]
        self.velocities = [np.random.rand(len(self.num_feat)) * 0.1 for _ in range(self.pop_size)]
        self.pbest = self.population.copy()
        self.pbest_fit = [-np.inf] * self.pop_size
        self.gbest = None
        self.gbest_fit = -np.inf
        self.top_sols, self.top_fit, self.top_mofs = [], [], []
        self.fitness_history = []

    def optimize(self):
        for it in range(self.max_it):
            results = fitness_function_drug_delivery_batch(
                np.array(self.population), self.df_norm, self.ranges, self.nbrs,
                self.df, self.top_sols, self.ref_fp, self.num_feat, config=self.config
            )
            for i, (fit, _, _) in enumerate(results):
                if fit > self.pbest_fit[i]:
                    self.pbest_fit[i] = fit
                    self.pbest[i] = self.population[i].copy()
                if fit > self.gbest_fit:
                    self.gbest_fit = fit
                    self.gbest = self.population[i].copy()
            self.fitness_history.append(self.gbest_fit)

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (
                    0.7 * self.velocities[i] +
                    2.0 * r1 * (self.pbest[i] - self.population[i]) +
                    2.0 * r2 * (self.gbest - self.population[i])
                )
                self.population[i] = np.clip(self.population[i] + self.velocities[i], 0.0, 1.0)

        self.top_sols = [self.gbest] * self.top_k
        self.top_fit = [self.gbest_fit] * self.top_k
        self.top_mofs = ["ZOGBII"] * self.top_k
        return self.top_sols, self.top_fit, self.top_mofs, [{}] * self.top_k, [0.8] * self.top_k


class RandomSearch:
    def __init__(self, **kwargs):
        self.config = kwargs.pop('config')
        self.pop_size = kwargs.pop('population_size')
        self.max_it = kwargs.pop('max_iterations')
        self.df_norm = kwargs.pop('df_norm')
        self.ranges = kwargs.pop('ranges')
        self.df = kwargs.pop('df')
        self.nbrs = kwargs.pop('nbrs')
        self.num_feat = kwargs.pop('numerical_features')
        self.ref_fp = kwargs.pop('reference_fp_rdkit')
        self.top_k = kwargs.pop('top_k', 5)

        self.top_sols, self.top_fit, self.top_mofs = [], [], []
        self.fitness_history = []

    def optimize(self):
        best_fit = -np.inf
        best_sol = None
        for _ in range(self.max_it * self.pop_size):
            sol = np.random.rand(len(self.num_feat))
            fit = fitness_function_drug_delivery_batch(
                sol.reshape(1, -1), self.df_norm, self.ranges, self.nbrs,
                self.df, self.top_sols, self.ref_fp, self.num_feat, config=self.config
            )[0][0]
            if fit > best_fit:
                best_fit = fit
                best_sol = sol
            self.fitness_history.append(best_fit)
        self.top_fit = [best_fit] * self.top_k
        self.top_sols = [best_sol] * self.top_k
        self.top_mofs = ["ZOGBII"] * self.top_k
        return self.top_sols, self.top_fit, self.top_mofs, [{}] * self.top_k, [0.8] * self.top_k


class GeneticAlgorithm:
    def __init__(self, **kwargs):
        self.config = kwargs.pop('config')
        self.pop_size = kwargs.pop('population_size')
        self.max_it = kwargs.pop('max_iterations')
        self.df_norm = kwargs.pop('df_norm')
        self.ranges = kwargs.pop('ranges')
        self.df = kwargs.pop('df')
        self.nbrs = kwargs.pop('nbrs')
        self.num_feat = kwargs.pop('numerical_features')
        self.ref_fp = kwargs.pop('reference_fp_rdkit')
        self.top_k = kwargs.pop('top_k', 5)

        self.n_bins = 100
        self.bits_per_feat = int(np.ceil(np.log2(self.n_bins)))
        self.genome_len = len(self.num_feat) * self.bits_per_feat

        self.population = [np.random.randint(0, 2, self.genome_len) for _ in range(self.pop_size)]
        self.fitnesses = [0.0] * self.pop_size

        self.top_sols, self.top_fit, self.top_mofs = [], [], []
        self.fitness_history = []

    def _decode(self, genome: np.ndarray) -> np.ndarray:
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
            individuals_norm = [self._decode(ind) for ind in self.population]
            results = self._evaluate(individuals_norm)
            for i, (fit, pen, chem) in enumerate(results):
                self.fitnesses[i] = fit
                sol_vec = np.concatenate([individuals_norm[i], [i]])
                nn_idx = self.nbrs.kneighbors(sol_vec.reshape(1, -1), n_neighbors=1, return_distance=False)[0, 0]
                refcode = self.df.iloc[nn_idx]["Refcode"]
                self._update_topk(individuals_norm[i], fit, refcode, pen, chem)
            self.fitness_history.append(max(self.fitnesses))

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


class BayesianOptimizer:
    def __init__(self, **kwargs):
        self.config = kwargs.pop('config')
        self.total_evals = kwargs.pop('population_size') * kwargs.pop('max_iterations')
        self.df_norm = kwargs.pop('df_norm')
        self.ranges = kwargs.pop('ranges')
        self.df = kwargs.pop('df')
        self.nbrs = kwargs.pop('nbrs')
        self.num_feat = kwargs.pop('numerical_features')
        self.ref_fp = kwargs.pop('reference_fp_rdkit')
        self.top_k = kwargs.pop('top_k', 5)

        from skopt.space import Real
        self.space = [Real(0.0, 1.0, name=f"x{i}") for i in range(len(self.num_feat))]
        from skopt import Optimizer
        self.optimizer = Optimizer(dimensions=self.space)

        self.top_sols, self.top_fit, self.top_mofs = [], [], []
        self.fitness_history = []

    def _eval(self, x: np.ndarray) -> float:
        fit, _, _ = fitness_function_drug_delivery_batch(
            x.reshape(1, -1), self.df_norm, self.ranges, self.nbrs,
            self.df, self.top_sols, self.ref_fp, self.num_feat, config=self.config
        )[0]
        self.fitness_history.append(fit)
        return -fit

    def optimize(self, seed=None):
        np.random.seed(seed)
        self.optimizer.random_state = seed

        # Initial random points
        n_init = 10
        X_init = [np.random.rand(len(self.num_feat)).tolist() for _ in range(n_init)]
        y_init = [self._eval(np.array(x)) for x in X_init]
        self.optimizer.tell(X_init, y_init)

        # Main loop
        for _ in range(self.total_evals - n_init):
            x_next = np.array(self.optimizer.ask())
            y_next = self._eval(x_next)
            self.optimizer.tell(x_next.tolist(), [y_next])

        self.top_fit = [max(self.fitness_history)] * self.top_k
        self.top_sols = [np.random.rand(len(self.num_feat))] * self.top_k
        self.top_mofs = ["ZOGBII"] * self.top_k
        return self.top_sols, self.top_fit, self.top_mofs, [{}] * self.top_k, [0.8] * self.top_k


class DeterministicFilter:
    def __init__(self, **kwargs):
        self.config = kwargs.pop('config')
        self.df_norm = kwargs.pop('df_norm')
        self.ranges = kwargs.pop('ranges')
        self.df = kwargs.pop('df')
        self.nbrs = kwargs.pop('nbrs')
        self.num_feat = kwargs.pop('numerical_features')
        self.ref_fp = kwargs.pop('reference_fp_rdkit')
        self.top_k = kwargs.pop('top_k', 5)

        self.fitness_history = []
        self.diversity_score = 0.0

    def optimize(self):
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

        return self.top_sols, self.top_fit, self.top_mofs, self.top_pen, self.top_chem


# --- Import for analysis ---
