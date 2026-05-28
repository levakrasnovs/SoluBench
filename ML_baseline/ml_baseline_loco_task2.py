"""
LOCO ML baseline for SoluBench Task 2 — descriptors only (no Morgan fingerprints).

Features (21 total per record):
  - 10 RDKit physicochemical descriptors for solute
  - 10 RDKit physicochemical descriptors for solvent
  - Temperature_K (1)

Usage:
    python ml_baseline_loco_task2_desc_only.py \
        --bigsoldb BigSolDBv2_0.csv \
        --task2    task2_results.csv \
        --output   task2_loco_desc_results.csv
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

parser = argparse.ArgumentParser()
parser.add_argument('--bigsoldb', default='BigSolDBv2_0.csv')
parser.add_argument('--task2',    default='task2_results.csv')
parser.add_argument('--output',   default='task2_loco_desc_results.csv')
args = parser.parse_args()

print("Loading data...")
df_big = pd.read_csv(args.bigsoldb)
df_t2  = pd.read_csv(args.task2)

df_big = df_big.dropna(subset=['LogS(mol/L)', 'SMILES_Solute', 'Solvent',
                                'SMILES_Solvent', 'Temperature_K'])

print(f"BigSolDB: {len(df_big)} records, {df_big['SMILES_Solute'].nunique()} unique solutes")
print(f"Task 2:   {len(df_t2)} questions, {df_t2['SMILES'].nunique()} unique solutes")


def get_features(smiles):
    """10 RDKit physicochemical descriptors"""
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


print("\nComputing molecular features (descriptors only)...")

solute_feat_cache = {}
for s in tqdm(set(df_big['SMILES_Solute'].unique()) | set(df_t2['SMILES'].unique()),
              desc="Solute features"):
    f = get_features(s)
    if f is not None:
        solute_feat_cache[s] = f

solvent_smiles_map = (df_big[['Solvent', 'SMILES_Solvent']]
                      .drop_duplicates()
                      .set_index('Solvent')['SMILES_Solvent']
                      .to_dict())

solvent_feat_cache = {}
for solvent, smiles in tqdm(solvent_smiles_map.items(), desc="Solvent features"):
    f = get_features(smiles)
    if f is not None:
        solvent_feat_cache[solvent] = f

print(f"Solute features:  {len(solute_feat_cache)} molecules, 10 dims each")
print(f"Solvent features: {len(solvent_feat_cache)} molecules, 10 dims each")

all_solvents_t2 = set()
for solvs in df_t2['All_solvents']:
    all_solvents_t2.update(solvs.split(';'))
missing = all_solvents_t2 - set(solvent_feat_cache.keys())
if missing:
    print(f"WARNING: {len(missing)} solvents missing: {missing}")
else:
    print(f"All {len(all_solvents_t2)} Task 2 solvents have features.")

print("\nBuilding BigSolDB feature matrix...")
X_big, y_big, smiles_big = [], [], []

for _, row in tqdm(df_big.iterrows(), total=len(df_big), desc="Building matrix"):
    sf = solute_feat_cache.get(row['SMILES_Solute'])
    sv = solvent_feat_cache.get(row['Solvent'])
    if sf is None or sv is None:
        continue
    X_big.append(sf + sv + [row['Temperature_K']])
    y_big.append(row['LogS(mol/L)'])
    smiles_big.append(row['SMILES_Solute'])

X_big      = np.array(X_big, dtype=np.float32)
y_big      = np.array(y_big, dtype=np.float32)
smiles_big = np.array(smiles_big)

print(f"Feature matrix: {X_big.shape}")

test_solutes = df_t2['SMILES'].unique()
print(f"\nStarting LOCO: {len(test_solutes)} solutes | LightGBM, descriptors only")
print("-" * 60)

predictions = [None] * len(df_t2)
t_start     = time.time()

for solute in tqdm(test_solutes, desc="LOCO progress"):
    mask = smiles_big != solute
    X_tr = X_big[mask]
    y_tr = y_big[mask]
    if len(X_tr) == 0:
        continue

    model = lgb.LGBMRegressor(verbosity=-1, random_state=42)
    model.fit(X_tr, y_tr)

    q_idx = df_t2.index[df_t2['SMILES'] == solute].tolist()
    for qi in q_idx:
        row      = df_t2.loc[qi]
        sf       = solute_feat_cache.get(row['SMILES'])
        solvents = row['All_solvents'].split(';')
        T        = row['Temperature_K']
        if sf is None:
            continue

        best_letter = None
        best_logS   = -np.inf
        all_ok      = True

        for i, solvent in enumerate(solvents):
            sv = solvent_feat_cache.get(solvent)
            if sv is None:
                all_ok = False
                break
            feat = np.array([sf + sv + [T]], dtype=np.float32)
            logS_pred = model.predict(feat)[0]
            if logS_pred > best_logS:
                best_logS   = logS_pred
                best_letter = chr(ord('A') + i)

        if all_ok and best_letter is not None:
            predictions[qi] = best_letter

print(f"\nDone in {time.time()-t_start:.0f}s")

df_t2['pred_loco_lgbm_desc'] = predictions

valid   = df_t2['pred_loco_lgbm_desc'].notna()
n_valid = valid.sum()
acc     = (df_t2.loc[valid, 'pred_loco_lgbm_desc'] == df_t2.loc[valid, 'Answer']).mean()
se      = np.sqrt(acc * (1 - acc) / n_valid)
ci      = 1.96 * se

print(f"\n{'='*55}")
print(f"Task 2 LOCO LightGBM descriptors only (n={n_valid}/{len(df_t2)})")
print(f"{'='*55}")
print(f"{'Random baseline (~1/N)':<35} ~17.1%")
print(f"{'LOCO LightGBM (desc only)':<35} {acc*100:.1f}% +/- {ci*100:.1f}%")
print(f"")
print(f"LLM results:")
print(f"  {'Gemini 3 Flash':<33} 66.2% +/- 3.1%")
print(f"  {'Claude Opus 4.5':<33} 60.1% +/- 3.2%")
print(f"  {'Qwen3.5 397B':<33} 52.3% +/- 3.3%")

df_t2.to_csv(args.output, index=False)
print(f"\nSaved to: {args.output}")

summary = pd.DataFrame([
    {'Model': 'Random baseline',             'Accuracy (%)': 17.1,              'CI (+/-%)': 0.0,             'Type': 'Baseline'},
    {'Model': 'LOCO LightGBM (desc only)',   'Accuracy (%)': round(acc*100, 1), 'CI (+/-%)': round(ci*100, 1),'Type': 'ML'},
    {'Model': 'Gemini 3 Flash',              'Accuracy (%)': 66.2,              'CI (+/-%)': 3.1,             'Type': 'LLM'},
    {'Model': 'Claude Opus 4.5',             'Accuracy (%)': 60.1,              'CI (+/-%)': 3.2,             'Type': 'LLM'},
    {'Model': 'Qwen3.5 397B',               'Accuracy (%)': 52.3,              'CI (+/-%)': 3.3,             'Type': 'LLM'},
])
summary_path = args.output.replace('.csv', '_summary.csv')
summary.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")
