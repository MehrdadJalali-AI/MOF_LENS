# File: utils.py
# Contains utility functions and constants

import numpy as np
import pandas as pd
from scipy.special import gamma
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import os
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import csr_matrix
from numba import jit
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.error')

# Constants
RESULTS_FOLDER = "EvaluationResults"
FILE_PATH = "MOF.csv"
EXPECTED_FEATURES = ['Refcode', 'void_fraction', 'asa (A^2)', 'pld (A)', 'max_metal_coordination_n', 
                     'n_sbu_point_of_extension']
SMILES_COLUMNS = ['ligand_smile', 'linker_smile', 'metal_sbu_smile', 'metal_cluster_smile']
TOXIC_METALS = ['Pb', 'Cd', 'Cr', 'Ni', 'Hg']

# Toxicity estimation
def estimate_toxicity(metal=None):
    if metal is None:
        return 0.0
    toxicity_scores = {'Zn': 0.2, 'Cu': 0.3, 'Fe': 0.1}
    return toxicity_scores.get(metal, 0.5)

# Lévy flight
def levy_flight(size, amplitude=1.0):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = np.random.normal(0, sigma, size=size)
    v = np.random.normal(0, 1, size=size)
    steps = u / np.abs(v)**(1 / beta)
    return amplitude * steps

# Tanimoto similarity
@jit(nopython=True)
def tanimoto_similarity(fp1, fp2):
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    return intersection / union if union > 0 else 0.0

# Create results folder
def create_results_folder():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)