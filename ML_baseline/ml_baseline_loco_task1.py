"""
LOCO (Leave-One-Compound-Out) ML baseline for SoluBench Task 1.

For each unique test solute S_i:
  - Train = all BigSolDB records where SMILES_Solute != S_i
  - Fit LightGBM to predict LogS(mol/L)
  - Predict LogS for (S_i, Solvent_A) and (S_i, Solvent_B)
  - Answer = A if LogS_A_pred > LogS_B_pred else B

Features (21 total):
  - 10 RDKit descriptors for solute
  - 10 RDKit descriptors for solvent
  - Temperature_K

Usage:
    python ml_baseline_loco_task1.py \
        --bigsoldb BigSolDBv2_0.csv \
        --task1    task1_results.csv \
        --output   task1_loco_results.csv
"""

import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import lightgbm as lgb
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')


# ── Argument parsing ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--bigsoldb', default='BigSolDBv2_0.csv')
parser.add_argument('--task1',    default='task1_results.csv')
parser.add_argument('--output',   default='task1_loco_results.csv')
args = parser.parse_args()


# ── Load data ────────────────────────────────────────────────────────────────

print("Loading data...")
df_big = pd.read_csv(args.bigsoldb)
df_t1  = pd.read_csv(args.task1)

df_big = df_big.dropna(subset=['LogS(mol/L)', 'SMILES_Solute', 'Solvent',
                                'SMILES_Solvent', 'Temperature_K'])

print(f"BigSolDB: {len(df_big)} records, {df_big['SMILES_Solute'].nunique()} unique solutes")
print(f"Task 1:   {len(df_t1)} questions, {df_t1['SMILES'].nunique()} unique solutes")


# ── Descriptors ──────────────────────────────────────────────────────────────

def get_desc(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumHeterocycles(mol),
        ]
    except:
        return None


# Precompute descriptors
print("\nComputing descriptors...")

solute_desc_cache = {}
for s in tqdm(set(df_big['SMILES_Solute'].unique()) | set(df_t1['SMILES'].unique()),
              desc="Solute descriptors"):
    d = get_desc(s)
    if d is not None:
        solute_desc_cache[s] = d

solvent_smiles_map = (df_big[['Solvent', 'SMILES_Solvent']]
                      .drop_duplicates()
                      .set_index('Solvent')['SMILES_Solvent']
                      .to_dict())

solvent_desc_cache = {}
for solvent, smiles in tqdm(solvent_smiles_map.items(), desc="Solvent descriptors"):
    d = get_desc(smiles)
    if d is not None:
        solvent_desc_cache[solvent] = d

print(f"Solute descriptors:  {len(solute_desc_cache)}")
print(f"Solvent descriptors: {len(solvent_desc_cache)}")

solvents_t1 = set(df_t1['Solvent_A'].unique()) | set(df_t1['Solvent_B'].unique())
missing = solvents_t1 - set(solvent_desc_cache.keys())
if missing:
    print(f"WARNING: {len(missing)} Task 1 solvents missing descriptors: {missing}")
else:
    print(f"All {len(solvents_t1)} Task 1 solvents have descriptors.")


# ── Build BigSolDB feature matrix ────────────────────────────────────────────

print("\nBuilding BigSolDB feature matrix...")
X_big, y_big, smiles_big = [], [], []

for _, row in tqdm(df_big.iterrows(), total=len(df_big), desc="Building matrix"):
    sd = solute_desc_cache.get(row['SMILES_Solute'])
    sv = solvent_desc_cache.get(row['Solvent'])
    if sd is None or sv is None:
        continue
    X_big.append(sd + sv + [row['Temperature_K']])
    y_big.append(row['LogS(mol/L)'])
    smiles_big.append(row['SMILES_Solute'])

X_big      = np.array(X_big)
y_big      = np.array(y_big)
smiles_big = np.array(smiles_big)

print(f"Feature matrix: {X_big.shape}")


# ── LOCO prediction ──────────────────────────────────────────────────────────

test_solutes = df_t1['SMILES'].unique()
n_solutes    = len(test_solutes)

print(f"\nStarting LOCO: {n_solutes} solutes")
print(f"Model: LightGBM (default params)")
print("-" * 60)

pred_lgbm  = [None] * len(df_t1)
t_start    = time.time()

for solute in tqdm(test_solutes, desc="LOCO progress"):
    # Train: exclude current test solute
    mask = smiles_big != solute
    X_tr = X_big[mask]
    y_tr = y_big[mask]

    if len(X_tr) == 0:
        continue

    # LightGBM (default params, verbosity=-1 to suppress output)
    lgbm = lgb.LGBMRegressor(verbosity=-1, random_state=42)
    lgbm.fit(X_tr, y_tr)

    # Predict for all Task 1 questions with this solute
    q_idx = df_t1.index[df_t1['SMILES'] == solute].tolist()
    for qi in q_idx:
        row  = df_t1.loc[qi]
        sd   = solute_desc_cache.get(row['SMILES'])
        sv_a = solvent_desc_cache.get(row['Solvent_A'])
        sv_b = solvent_desc_cache.get(row['Solvent_B'])
        if sd is None or sv_a is None or sv_b is None:
            continue
        T = row['Temperature_K']
        feat_a = np.array([sd + sv_a + [T]])
        feat_b = np.array([sd + sv_b + [T]])

        # LightGBM
        logS_a_l = lgbm.predict(feat_a)[0]
        logS_b_l = lgbm.predict(feat_b)[0]
        pred_lgbm[qi] = 'A' if logS_a_l > logS_b_l else 'B'

print(f"\nDone in {time.time()-t_start:.0f}s")


# ── Evaluate ─────────────────────────────────────────────────────────────────

df_t1['pred_loco_lgbm']  = pred_lgbm

def evaluate(preds, answers, label):
    valid   = pd.Series(preds).notna()
    n_valid = valid.sum()
    acc     = (pd.Series(preds)[valid] == pd.Series(answers)[valid]).mean()
    se      = np.sqrt(acc * (1 - acc) / n_valid)
    ci      = 1.96 * se
    print(f"{label:<30} {acc*100:.1f}% +/- {ci*100:.1f}%  (n={n_valid})")
    return acc, ci

print(f"\n{'='*55}")
print(f"Task 1 LOCO Results")
print(f"{'='*55}")
print(f"{'Random baseline':<30} 50.0%")
acc_l, ci_l = evaluate(pred_lgbm,  df_t1['Answer'].tolist(), 'LOCO LightGBM (default)')
print(f"")
print(f"LLM results for comparison:")
print(f"  {'Gemini 3 Flash':<28} 90.6% +/- 0.9%")
print(f"  {'Claude Opus 4.6':<28} 85.6% +/- 1.1%")
print(f"  {'Qwen3.5 397B (best open-src)':<28} 81.7% +/- 1.2%")


# ── Save ─────────────────────────────────────────────────────────────────────

df_t1.to_csv(args.output, index=False)
print(f"\nPredictions saved to: {args.output}")

summary = pd.DataFrame([
    {'Model': 'Random baseline',              'Accuracy (%)': 50.0,              'CI (+/-%)': 0.0,            'Type': 'Baseline'},
    {'Model': 'LOCO LightGBM (default)',      'Accuracy (%)': round(acc_l*100,1),'CI (+/-%)': round(ci_l*100,1),'Type': 'ML'},
    {'Model': 'Gemini 3 Flash',               'Accuracy (%)': 90.6,              'CI (+/-%)': 0.9,            'Type': 'LLM'},
    {'Model': 'Claude Opus 4.6',              'Accuracy (%)': 85.6,              'CI (+/-%)': 1.1,            'Type': 'LLM'},
    {'Model': 'Qwen3.5 397B (best open-src)', 'Accuracy (%)': 81.7,              'CI (+/-%)': 1.2,            'Type': 'LLM'},
])
summary_path = args.output.replace('.csv', '_summary.csv')
summary.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")
