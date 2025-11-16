# data_preprocessing.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, List, Optional
from rdkit.Chem import Mol  # ← ADD THIS

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def _sanitize_smiles(smiles: str) -> Optional[Mol]:  # ← FIXED
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

def load_and_preprocess_data(config: Dict) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]], pd.DataFrame, NearestNeighbors, List[str], np.ndarray]:
    df = pd.read_csv(config['paths']['data'])

    desired = config['features']['numerical']
    available = df.columns.tolist()
    num_feat = [col for col in desired if col in available]

    if not num_feat:
        raise ValueError(f"No numerical features found!\n  Config wants: {desired}\n  CSV has: {available}")

    df_norm = df[num_feat].copy()
    ranges = {}
    for col in num_feat:
        min_v, max_v = df[col].min(), df[col].max()
        if max_v > min_v:
            df_norm[col] = (df[col] - min_v) / (max_v - min_v)
        else:
            df_norm[col] = 0.0
        ranges[col] = (min_v, max_v)

    smiles_col = config['features']['fingerprint_column']
    print("Sanitizing linker SMILES...")
    df['mol'] = df[smiles_col].apply(_sanitize_smiles)
    valid_mask = df['mol'].notna()
    n_valid = valid_mask.sum()
    n_invalid = len(df) - n_valid

    df_valid = df[valid_mask].copy()
    df_norm_valid = df_norm[valid_mask].copy()

    if n_valid == 0:
        raise ValueError("No valid organic linker SMILES after sanitization!")

    dox_smiles = config['drugs']['doxorubicin']['smiles']
    dox_mol = Chem.MolFromSmiles(dox_smiles)
    ref_fp = morgan_gen.GetFingerprint(dox_mol)
    ref_fp_np = np.array(ref_fp)

    X_index = df_norm_valid[num_feat].values
    X_with_idx = np.hstack([X_index, np.arange(len(df_valid)).reshape(-1, 1)])
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_with_idx)

    print(f"Data loaded: {len(df)} MOFs")
    print(f"  → Valid SMILES: {n_valid} ({n_valid/len(df)*100:.1f}%)")
    print(f"  → Invalid SMILES: {n_invalid} (filtered out)")
    print(f"Numerical features ({len(num_feat)}): {num_feat}")
    print(f"Linker SMILES column: {smiles_col}")
    print(f"DOX target PLD: {config['drugs']['doxorubicin']['pld_target']} Å")
    print(f"DOX fingerprint: {dox_smiles[:50]}... (Morgan, r=2, 256 bits)")

    return df_norm_valid, ranges, df_valid, nbrs, num_feat, ref_fp_np
