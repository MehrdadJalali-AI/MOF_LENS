# --------------------------------------------------------------
# ListTop5.py
# 1. Count top-5 frequency of LEA
# 2. Merge with full MOF table
# 3. Resolve linker SMILES → IUPAC name (PubChem)
# 4. (Optional) Merge docking scores (ΔG pH 7.4 / 5.5)
# 5. Save CSV + console summary
# --------------------------------------------------------------

import os
import ast
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm

# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------
MOF_CSV          = "data/MOF.csv"
LEA_ALL_CSV      = "results/raw/LEA_all.csv"
DOCKING_CSV      = "results/docking/docking_results.csv"
OUTPUT_DIR       = "results"
FREQ_CSV         = os.path.join(OUTPUT_DIR, "top5_frequency_lea.csv")
FINAL_CSV        = os.path.join(OUTPUT_DIR, "top5_with_docking.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------
# 1. Load raw data
# --------------------------------------------------------------
df_mofs = pd.read_csv(MOF_CSV)
df_lea  = pd.read_csv(LEA_ALL_CSV)

# --------------------------------------------------------------
# 2. Parse serialized top_mofs list
# --------------------------------------------------------------
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []                     # corrupted → ignore

df_lea["top_mofs"] = df_lea["top_mofs"].apply(safe_literal_eval)

# Explode → one row per Refcode per run
df_exploded = df_lea.explode("top_mofs").copy()
df_exploded = df_exploded.rename(columns={"top_mofs": "Refcode"})

# --------------------------------------------------------------
# 3. Frequency table
# --------------------------------------------------------------
freq = (
    df_exploded["Refcode"]
    .value_counts()
    .reset_index()
    .rename(columns={"count": "top5_count"})
)

# --------------------------------------------------------------
# 4. Merge with full MOF table
# --------------------------------------------------------------
df_top = df_mofs.merge(freq, on="Refcode", how="inner")

# --------------------------------------------------------------
# 5. SMILES → IUPAC name (PubChem)
# --------------------------------------------------------------
SMILES_COL = "linker_smile"
IUPAC_COL  = "Linker_IUPAC"

smiles_cache = {}

def smiles_to_iupac(smiles: str) -> str:
    if pd.isna(smiles) or not smiles:
        return "Missing SMILES"
    if smiles in smiles_cache:
        return smiles_cache[smiles]

    try:
        compounds = pcp.get_compounds(smiles, "smiles", record_type="2d")
        name = compounds[0].iupac_name if compounds and compounds[0].iupac_name else "Unrecognized"
    except Exception:
        name = "Unrecognized"

    smiles_cache[smiles] = name
    return name

print("\nResolving linker SMILES → IUPAC names...")
unique_smiles = df_top[SMILES_COL].dropna().unique()
for smi in tqdm(unique_smiles, desc="PubChem lookup", unit="SMILES"):
    smiles_to_iupac(smi)                     # fill cache

df_top[IUPAC_COL] = df_top[SMILES_COL].apply(smiles_to_iupac)

# --------------------------------------------------------------
# 6. Tidy columns & save frequency table
# --------------------------------------------------------------
df_top = df_top.rename(columns={
    "asa (A^2)": "ASA",
    "pld (A)":   "PLD",
    "void_fraction": "VoidFraction",
    "metals":    "Metal"
})

df_top = df_top.sort_values("top5_count", ascending=False).reset_index(drop=True)
df_top.to_csv(FREQ_CSV, index=False)

# --------------------------------------------------------------
# 7. (Optional) Merge docking scores – BEST SCORE PER (Refcode, pH)
# --------------------------------------------------------------
if os.path.exists(DOCKING_CSV):
    print("\nMerging docking results...")
    dock = pd.read_csv(DOCKING_CSV)

    # --- Detect score column ---
    score_col = None
    for candidate in ["BindingEnergy_kcal_mol", "Docking_Score", "Score", "Affinity"]:
        if candidate in dock.columns:
            score_col = candidate
            break
    if score_col is None:
        raise KeyError(f"No score column found. Found: {dock.columns.tolist()}")

    required = {"Refcode", "pH"}
    if not required.issubset(dock.columns):
        raise KeyError(f"Docking CSV must contain {required}. Found: {dock.columns.tolist()}")

    # --- Keep BEST (lowest = most negative) score per (Refcode, pH) ---
    print(f"   → Found {len(dock)} docking entries (multiple poses per condition).")
    dock_best = (
        dock
        .sort_values(score_col, ascending=True)  # lower = better
        .drop_duplicates(subset=["Refcode", "pH"], keep="first")
    )
    print(f"   → Kept best score per (Refcode, pH): {len(dock_best)} entries")

    # --- Pivot safely ---
    dock_pivot = dock_best.pivot(index="Refcode", columns="pH", values=score_col)
    dock_pivot = dock_pivot.rename(columns={7.4: "ΔG_pH7.4", 5.5: "ΔG_pH5.5"})

    # --- Compute ΔΔG ---
    if {"ΔG_pH7.4", "ΔG_pH5.5"}.issubset(dock_pivot.columns):
        dock_pivot["ΔΔG"] = dock_pivot["ΔG_pH5.5"] - dock_pivot["ΔG_pH7.4"]
    else:
        dock_pivot["ΔΔG"] = pd.NA

    # --- Merge ---
    final = df_top.merge(dock_pivot, on="Refcode", how="left")
    final.to_csv(FINAL_CSV, index=False)

    # --- Console report ---
    print(f"\nDocking merge complete! → {FINAL_CSV}")
    print(f"   • Score column: '{score_col}'")
    print(f"   • pH 7.4 entries: {dock_pivot['ΔG_pH7.4'].notna().sum()}")
    print(f"   • pH 5.5 entries: {dock_pivot['ΔG_pH5.5'].notna().sum()}")

    if not final.empty:
        top = final.iloc[0]
        print(f"\nTop MOF: {top['Refcode']} ({top['top5_count']} times)")
        print(f"   Linker: {top['Linker_IUPAC']}")
        print(f"   ΔG (pH 7.4) = {top['ΔG_pH7.4']:.2f} kcal/mol")
        print(f"   ΔG (pH 5.5) = {top['ΔG_pH5.5']:.2f} kcal/mol")
        print(f"   ΔΔG = {top['ΔΔG']:.2f} kcal/mol")

    # Show top 5
    cols = ["Refcode", "top5_count", "Linker_IUPAC", "PLD", "ASA", "ΔG_pH7.4", "ΔG_pH5.5", "ΔΔG"]
    print("\nTop-5 MOFs with best docking scores:")
    print(final[[c for c in cols if c in final.columns]].head(5).round(3).to_string(index=False))
else:
    print(f"\nWarning: Docking file not found: {DOCKING_CSV}")
    print(f"   → Only frequency table saved: {FREQ_CSV}")