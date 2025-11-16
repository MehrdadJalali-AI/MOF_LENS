# File: analysis.py
# Handles analysis, visualization, and exports

from utils import *
from data_preprocessing import *
from rdkit.Chem import Draw

def analyze_diversity(top_solutions, df_norm, numerical_features, method_name):
    kmeans = KMeans(n_clusters=min(4, len(top_solutions)), n_init=10, random_state=42)
    clusters = kmeans.fit_predict(top_solutions)
    distances = np.mean([np.linalg.norm(top_solutions[i] - top_solutions[j]) 
                         for i in range(len(top_solutions)) 
                         for j in range(i + 1, len(top_solutions))])
    print(f"Average Euclidean Distance ({method_name}): {distances:.2f}")
    print(f"Number of Clusters ({method_name}): {len(np.unique(clusters))}")
    
    plt.figure(figsize=(8, 5))
    plt.scatter([s[2] for s in top_solutions], [s[0] for s in top_solutions], c=clusters, cmap='viridis')
    plt.xlabel("PLD (Normalized)")
    plt.ylabel("Void Fraction (Normalized)")
    plt.title(f"Clustering of Top MOF Solutions ({method_name})")
    plt.savefig(os.path.join(RESULTS_FOLDER, f'{method_name}_mof_clusters.png'))
    plt.close()
    
    cluster_data = pd.DataFrame({
        'PLD': [s[2] for s in top_solutions],
        'Void_Fraction': [s[0] for s in top_solutions],
        'Cluster': clusters
    })
    cluster_data.to_csv(os.path.join(RESULTS_FOLDER, f'{method_name}_mof_clusters.csv'), index=False)

def visualize_tradeoffs(top_solutions, top_fitness, df_norm, numerical_features, top_mofs, method_name):
    porosity_scores = [sum(df_norm[df_norm['Refcode'] == mof][['void_fraction', 'asa (A^2)', 'pld (A)']].values[0]) 
                       for mof in top_mofs]
    stability_scores = [sum(df_norm[df_norm['Refcode'] == mof][['max_metal_coordination_n', 'n_sbu_point_of_extension']].values[0]) 
                        for mof in top_mofs]
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(porosity_scores, stability_scores, c=top_fitness, cmap='viridis', s=100)
    plt.colorbar(scatter, label='Fitness Score')
    for i, mof in enumerate(top_mofs):
        plt.annotate(mof, (porosity_scores[i], stability_scores[i]))
    plt.xlabel("Porosity Score (Normalized)")
    plt.ylabel("Stability Score (Normalized)")
    plt.title(f"Trade-offs: Porosity vs Stability for Top MOFs ({method_name})")
    plt.savefig(os.path.join(RESULTS_FOLDER, f'{method_name}_tradeoffs.png'))
    plt.close()
    
    tradeoffs_data = pd.DataFrame({
        'Porosity_Score': porosity_scores,
        'Stability_Score': stability_scores,
        'Fitness_Score': top_fitness
    })
    tradeoffs_data.to_csv(os.path.join(RESULTS_FOLDER, f'{method_name}_tradeoffs.csv'), index=False)

def export_top_mofs(top_mofs, top_fitness, df, df_norm, top_penalties, top_chem_scores, filename):
    results = []
    for mof, fitness, penalties, chem_score in zip(top_mofs, top_fitness, top_penalties, top_chem_scores):
        row = df[df['Refcode'] == mof].iloc[0].to_dict()
        row['Fitness'] = fitness
        row['Chemical_Score'] = chem_score
        row.update({f"Penalty_{k}": v for k, v in penalties.items()})
        results.append(row)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_FOLDER, filename), index=False)
    print(f"Top MOFs exported to '{os.path.join(RESULTS_FOLDER, filename)}'")

def export_mof_smiles_images(df, selected_smiles_column, output_folder='mof_smiles_images', top_n=5):
    output_path = os.path.join(RESULTS_FOLDER, output_folder)
    os.makedirs(output_path, exist_ok=True)
    for i, row in df.head(top_n).iterrows():
        refcode = row['Refcode']
        smiles = row.get(selected_smiles_column, '') if selected_smiles_column else ''
        if not smiles or pd.isna(smiles):
            print(f"[SKIP] MOF {refcode} has no valid SMILES.")
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            file_path = os.path.join(output_path, f"{refcode}.png")
            Draw.MolToFile(mol, file_path, size=(300, 300))
            print(f"[OK] Saved structure for {refcode} to {file_path}")
        else:
            print(f"[ERROR] Invalid SMILES for {refcode}: {smiles}")

def plot_convergence(lea_history, pso_history, rs_history):
    plt.figure(figsize=(10, 6))
    plt.plot(lea_history, label="LEA Optimization")
    plt.plot(pso_history, label="PSO Optimization")
    plt.plot(rs_history, label="Random Search")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Score (Higher = Better)")
    plt.title("MOF Optimization for Drug Delivery - Convergence Comparison")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'convergence_comparison.png'))
    plt.close()
    
    convergence_data = pd.DataFrame({
        'Iteration': range(max(len(lea_history), len(pso_history), len(rs_history))),
        'LEA_Fitness': lea_history + [np.nan] * (max(len(lea_history), len(pso_history), len(rs_history)) - len(lea_history)),
        'PSO_Fitness': pso_history + [np.nan] * (max(len(lea_history), len(pso_history), len(rs_history)) - len(pso_history)),
        'RS_Fitness': rs_history + [np.nan] * (max(len(lea_history), len(pso_history), len(rs_history)) - len(rs_history))
    })
    convergence_data.to_csv(os.path.join(RESULTS_FOLDER, 'convergence_comparison.csv'), index=False)

def plot_pld_distribution(df, lea_top_mofs, pso_top_mofs, rs_top_mofs):
    plt.figure(figsize=(8, 5))
    plt.hist(df['pld (A)'], bins=30, alpha=0.7, label="All MOFs")
    for mof in lea_top_mofs:
        plt.axvline(df[df['Refcode'] == mof]['pld (A)'].values[0], color='red', linestyle='dashed', alpha=0.5)
    plt.axvline(df[df['Refcode'] == lea_top_mofs[0]]['pld (A)'].values[0], color='red', linestyle='dashed', label="Top LEA MOFs")
    for mof in pso_top_mofs:
        plt.axvline(df[df['Refcode'] == mof]['pld (A)'].values[0], color='green', linestyle='dashed', alpha=0.5)
    plt.axvline(df[df['Refcode'] == pso_top_mofs[0]]['pld (A)'].values[0], color='green', linestyle='dashed', label="Top PSO MOFs")
    for mof in rs_top_mofs:
        plt.axvline(df[df['Refcode'] == mof]['pld (A)'].values[0], color='blue', linestyle='dashed', alpha=0.5)
    plt.axvline(df[df['Refcode'] == rs_top_mofs[0]]['pld (A)'].values[0], color='blue', linestyle='dashed', label="Top RS MOFs")
    plt.xlabel("Pore Limiting Diameter (PLD, Å)")
    plt.ylabel("Frequency")
    plt.title("PLD Distribution in MOF Dataset")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'pld_distribution.png'))
    plt.close()
    
    pld_data = pd.DataFrame({
        'PLD': df['pld (A)']
    })
    pld_data.to_csv(os.path.join(RESULTS_FOLDER, 'pld_distribution.csv'), index=False)# File: analysis.py
# Handles analysis, visualization, and exports

from utils import *
from data_preprocessing import *
from rdkit.Chem import Draw

def analyze_diversity(top_solutions, df_norm, numerical_features, method_name):
    kmeans = KMeans(n_clusters=min(4, len(top_solutions)), n_init=10, random_state=42)
    clusters = kmeans.fit_predict(top_solutions)
    distances = np.mean([np.linalg.norm(top_solutions[i] - top_solutions[j]) 
                         for i in range(len(top_solutions)) 
                         for j in range(i + 1, len(top_solutions))])
    print(f"Average Euclidean Distance ({method_name}): {distances:.2f}")
    print(f"Number of Clusters ({method_name}): {len(np.unique(clusters))}")
    
    plt.figure(figsize=(8, 5))
    plt.scatter([s[2] for s in top_solutions], [s[0] for s in top_solutions], c=clusters, cmap='viridis')
    plt.xlabel("PLD (Normalized)")
    plt.ylabel("Void Fraction (Normalized)")
    plt.title(f"Clustering of Top MOF Solutions ({method_name})")
    plt.savefig(os.path.join(RESULTS_FOLDER, f'{method_name}_mof_clusters.png'))
    plt.close()
    
    cluster_data = pd.DataFrame({
        'PLD': [s[2] for s in top_solutions],
        'Void_Fraction': [s[0] for s in top_solutions],
        'Cluster': clusters
    })
    cluster_data.to_csv(os.path.join(RESULTS_FOLDER, f'{method_name}_mof_clusters.csv'), index=False)

def visualize_tradeoffs(top_solutions, top_fitness, df_norm, numerical_features, top_mofs, method_name):
    porosity_scores = [sum(df_norm[df_norm['Refcode'] == mof][['void_fraction', 'asa (A^2)', 'pld (A)']].values[0]) 
                       for mof in top_mofs]
    stability_scores = [sum(df_norm[df_norm['Refcode'] == mof][['max_metal_coordination_n', 'n_sbu_point_of_extension']].values[0]) 
                        for mof in top_mofs]
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(porosity_scores, stability_scores, c=top_fitness, cmap='viridis', s=100)
    plt.colorbar(scatter, label='Fitness Score')
    for i, mof in enumerate(top_mofs):
        plt.annotate(mof, (porosity_scores[i], stability_scores[i]))
    plt.xlabel("Porosity Score (Normalized)")
    plt.ylabel("Stability Score (Normalized)")
    plt.title(f"Trade-offs: Porosity vs Stability for Top MOFs ({method_name})")
    plt.savefig(os.path.join(RESULTS_FOLDER, f'{method_name}_tradeoffs.png'))
    plt.close()
    
    tradeoffs_data = pd.DataFrame({
        'Porosity_Score': porosity_scores,
        'Stability_Score': stability_scores,
        'Fitness_Score': top_fitness
    })
    tradeoffs_data.to_csv(os.path.join(RESULTS_FOLDER, f'{method_name}_tradeoffs.csv'), index=False)

def export_top_mofs(top_mofs, top_fitness, df, df_norm, top_penalties, top_chem_scores, filename):
    results = []
    for mof, fitness, penalties, chem_score in zip(top_mofs, top_fitness, top_penalties, top_chem_scores):
        row = df[df['Refcode'] == mof].iloc[0].to_dict()
        row['Fitness'] = fitness
        row['Chemical_Score'] = chem_score
        row.update({f"Penalty_{k}": v for k, v in penalties.items()})
        results.append(row)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_FOLDER, filename), index=False)
    print(f"Top MOFs exported to '{os.path.join(RESULTS_FOLDER, filename)}'")

def export_mof_smiles_images(df, selected_smiles_column, output_folder='mof_smiles_images', top_n=5):
    output_path = os.path.join(RESULTS_FOLDER, output_folder)
    os.makedirs(output_path, exist_ok=True)
    for i, row in df.head(top_n).iterrows():
        refcode = row['Refcode']
        smiles = row.get(selected_smiles_column, '') if selected_smiles_column else ''
        if not smiles or pd.isna(smiles):
            print(f"[SKIP] MOF {refcode} has no valid SMILES.")
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            file_path = os.path.join(output_path, f"{refcode}.png")
            Draw.MolToFile(mol, file_path, size=(300, 300))
            print(f"[OK] Saved structure for {refcode} to {file_path}")
        else:
            print(f"[ERROR] Invalid SMILES for {refcode}: {smiles}")

def plot_convergence(lea_history, pso_history, rs_history):
    plt.figure(figsize=(10, 6))
    plt.plot(lea_history, label="LEA Optimization")
    plt.plot(pso_history, label="PSO Optimization")
    plt.plot(rs_history, label="Random Search")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Score (Higher = Better)")
    plt.title("MOF Optimization for Drug Delivery - Convergence Comparison")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'convergence_comparison.png'))
    plt.close()
    
    convergence_data = pd.DataFrame({
        'Iteration': range(max(len(lea_history), len(pso_history), len(rs_history))),
        'LEA_Fitness': lea_history + [np.nan] * (max(len(lea_history), len(pso_history), len(rs_history)) - len(lea_history)),
        'PSO_Fitness': pso_history + [np.nan] * (max(len(lea_history), len(pso_history), len(rs_history)) - len(pso_history)),
        'RS_Fitness': rs_history + [np.nan] * (max(len(lea_history), len(pso_history), len(rs_history)) - len(rs_history))
    })
    convergence_data.to_csv(os.path.join(RESULTS_FOLDER, 'convergence_comparison.csv'), index=False)

def plot_pld_distribution(df, lea_top_mofs, pso_top_mofs, rs_top_mofs):
    plt.figure(figsize=(8, 5))
    plt.hist(df['pld (A)'], bins=30, alpha=0.7, label="All MOFs")
    for mof in lea_top_mofs:
        plt.axvline(df[df['Refcode'] == mof]['pld (A)'].values[0], color='red', linestyle='dashed', alpha=0.5)
    plt.axvline(df[df['Refcode'] == lea_top_mofs[0]]['pld (A)'].values[0], color='red', linestyle='dashed', label="Top LEA MOFs")
    for mof in pso_top_mofs:
        plt.axvline(df[df['Refcode'] == mof]['pld (A)'].values[0], color='green', linestyle='dashed', alpha=0.5)
    plt.axvline(df[df['Refcode'] == pso_top_mofs[0]]['pld (A)'].values[0], color='green', linestyle='dashed', label="Top PSO MOFs")
    for mof in rs_top_mofs:
        plt.axvline(df[df['Refcode'] == mof]['pld (A)'].values[0], color='blue', linestyle='dashed', alpha=0.5)
    plt.axvline(df[df['Refcode'] == rs_top_mofs[0]]['pld (A)'].values[0], color='blue', linestyle='dashed', label="Top RS MOFs")
    plt.xlabel("Pore Limiting Diameter (PLD, Å)")
    plt.ylabel("Frequency")
    plt.title("PLD Distribution in MOF Dataset")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'pld_distribution.png'))
    plt.close()
    
    pld_data = pd.DataFrame({
        'PLD': df['pld (A)']
    })
    pld_data.to_csv(os.path.join(RESULTS_FOLDER, 'pld_distribution.csv'), index=False)