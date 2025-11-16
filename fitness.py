# fitness.py
"""
Fitness function for MOF-LENS (drug-delivery to DOX).
Handles both:
  • Main pipeline: nbrs fitted on 7-D data (6 features + dummy index)
  • Sensitivity analysis: nbrs fitted on 6-D data (features only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# RDKit setup (quiet)
# ----------------------------------------------------------------------
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# ----------------------------------------------------------------------
# Fast Tanimoto (Numba)
# ----------------------------------------------------------------------
try:
    from numba import njit
    @njit(fastmath=True, cache=True)
    def _tanimoto(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.bitwise_and(a, b).sum()
        union = np.bitwise_or(a, b).sum()
        return inter / union if union > 0 else 0.0
except Exception:  # fallback if numba not available
    def _tanimoto(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.bitwise_and(a, b).sum()
        union = np.bitwise_or(a, b).sum()
        return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------
# Main fitness function (batch mode)
# ----------------------------------------------------------------------
def fitness_function_drug_delivery_batch(
    batch_norm: np.ndarray,                     # (N, 6) or (N, 7)
    df_norm: pd.DataFrame,
    ranges: Dict[str, Tuple[float, float]],
    nbrs,                                       # NearestNeighbors (6D or 7D)
    df: pd.DataFrame,
    top_sols: List[np.ndarray],
    ref_fp_rdkit: Any,
    numerical_features: List[str],
    weights: Dict[str, float] | None = None,
    lambda_div: float = 0.03,
    config: Dict[str, Any] | None = None,
) -> List[Tuple[float, Dict[str, float], float]]:
    """
    Returns for each solution:
        (fitness, penalty_dict, chemical_similarity)
    """

    # ------------------------------------------------------------------
    # Load defaults from config if not provided
    # ------------------------------------------------------------------
    if config is not None:
        weights = weights or config['fitness']['weights']
        lambda_div = config['fitness']['lambda_div']
        pld_target = config['drugs']['doxorubicin']['pld_target']
        smiles_col = config['features']['fingerprint_column']
    else:
        weights = weights or {
            'pld': 0.30, 'chemical_sim': 0.25, 'ph_stability': 0.20,
            'nh2_func': 0.15, 'toxicity': -0.05
        }
        pld_target = 13.5
        smiles_col = 'linker_smile'

    ref_fp_np = np.array(ref_fp_rdkit) if ref_fp_rdkit is not None else None
    results: List[Tuple[float, Dict[str, float], float]] = []

    # ------------------------------------------------------------------
    # Process each solution in the batch
    # ------------------------------------------------------------------
    for i in range(batch_norm.shape[0]):
        sol_norm = batch_norm[i]

        # ----------------------------------------------------------
        # DYNAMIC FEATURE MATCHING (YOUR FIX)
        # ----------------------------------------------------------
        try:
            n_features_expected = nbrs._fit_X.shape[1]
        except AttributeError:
            n_features_expected = len(sol_norm)  # fallback

        if n_features_expected > len(sol_norm):
            # 7D nbrs → append dummy index
            sol_vec = np.hstack([sol_norm, [i]])   # use loop index for reproducibility
        else:
            # 6D nbrs → use directly
            sol_vec = sol_norm

        # ----------------------------------------------------------
        # Nearest neighbor lookup
        # ----------------------------------------------------------
        nn_idx = nbrs.kneighbors(
            sol_vec.reshape(1, -1),
            n_neighbors=1,
            return_distance=False
        )[0, 0]
        row = df.iloc[nn_idx]

        # ----------------------------------------------------------
        # Denormalize to real units
        # ----------------------------------------------------------
        sol_denorm = {}
        for j, feat in enumerate(numerical_features):
            min_v, max_v = ranges[feat]
            sol_denorm[feat] = sol_norm[j] * (max_v - min_v) + min_v

        # ----------------------------------------------------------
        # Individual terms
        # ----------------------------------------------------------
        pld = sol_denorm.get('pld (A)', 0.0)
        pld_score = max(0.0, 1.0 - abs(pld - pld_target) / pld_target)

        chem_sim = 0.0
        mol = row.get('mol')
        if mol is not None:
            fp = morgan_gen.GetFingerprint(mol)
            fp_np = np.array(fp)
            chem_sim = _tanimoto(fp_np, ref_fp_np)

        coord = row.get('max_metal_coordination_n', 6)
        ph_stability = 1.0 if 5 <= coord <= 8 else 0.5

        nh2_bonus = 1.0 if 'N' in str(row.get(smiles_col, '')) else 0.0

        toxicity_penalty = 0.0  # placeholder

        # Diversity penalty (only if top_sols exist)
        div_penalty = 0.0
        if top_sols and lambda_div > 0:
            dists = [np.linalg.norm(sol_norm - np.array(s)) for s in top_sols]
            if dists:
                div_penalty = -lambda_div / (1.0 + np.mean(dists))

        # ----------------------------------------------------------
        # Final fitness
        # ----------------------------------------------------------
        fitness = (
            weights['pld'] * pld_score +
            weights['chemical_sim'] * chem_sim +
            weights['ph_stability'] * ph_stability +
            weights['nh2_func'] * nh2_bonus +
            weights['toxicity'] * toxicity_penalty +
            div_penalty
        )

        penalties = {
            'pld': pld_score,
            'chem_sim': chem_sim,
            'ph_stability': ph_stability,
            'nh2_func': nh2_bonus,
            'toxicity': toxicity_penalty,
            'diversity': div_penalty
        }

        results.append((fitness, penalties, chem_sim))

    return results