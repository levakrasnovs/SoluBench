"""
LOCO ML baseline for SoluBench Task 4.

For each question (Solute_A, Solute_B, Solvent):
  - Train = all BigSolDB records where SMILES_Solute not in {Solute_A, Solute_B}
  - Fit LightGBM to predict LogS(mol/L)
  - Predict LogS(Solute_A, Solvent) and LogS(Solute_B, Solvent)
  - Answer = A if LogS_A_pred > LogS_B_pred else B

Features (21 total per record):
  - 10 RDKit physicochemical descriptors for solute
  - 10 RDKit physicochemical descriptors for solvent
  - Temperature_K

Usage:
    python ml_baseline_loco_task4.py \
        --bigsoldb BigSolDBv2_0.csv \
        --task4    task4_results.csv \
        --output   task4_loco_results.csv
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
parser.add_argument('--task4',    default='task4_results.csv')
parser.add_argument('--output',   default='task4_loco_results.csv')
args = parser.parse_args()

print("Loading data...")
df_big = pd.read_csv(args.bigsoldb)
df_t4  = pd.read_csv(args.task4)

df_big = df_big.dropna(subset=['LogS(mol/L)', 'SMILES_Solute', 'Solvent',
                                'SMILES_Solvent', 'Temperature_K'])

print(f"BigSolDB: {len(df_big)} records, {df_big['SMILES_Solute'].nunique()} unique solutes")
print(f"Task 4:   {len(df_t4)} questions")
print(f"Unique solute pairs: {df_t4[['SMILES_A','SMILES_B']].apply(lambda r: tuple(sorted([r['SMILES_A'],r['SMILES_B']])), axis=1).nunique()}")


def get_features(smiles):
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


print("\nComputing molecular features...")

all_solute_smiles = (set(df_big['SMILES_Solute'].unique()) |
                     set(df_t4['SMILES_A'].unique()) |
                     set(df_t4['SMILES_B'].unique()))

solute_feat_cache = {}
for s in tqdm(all_solute_smiles, desc="Solute features"):
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

print(f"Solute features:  {len(solute_feat_cache)}")
print(f"Solvent features: {len(solvent_feat_cache)}")

# Check Task 4 solvent coverage
missing = set(df_t4['Solvent'].unique()) - set(solvent_feat_cache.keys())
if missing:
    print(f"WARNING: {len(missing)} solvents missing: {missing}")
else:
    print(f"All {df_t4['Solvent'].nunique()} Task 4 solvents have features.")

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

print(f"\nStarting LOCO: {len(df_t4)} questions | LightGBM default params")
print("Note: each question trains a separate model excluding both test solutes")
print("-" * 60)

predictions = [None] * len(df_t4)
t_start     = time.time()

for qi, row in tqdm(df_t4.iterrows(), total=len(df_t4), desc="LOCO progress"):
    smiles_a = row['SMILES_A']
    smiles_b = row['SMILES_B']
    solvent  = row['Solvent']
    T        = row['Temperature_K']

    sf_a = solute_feat_cache.get(smiles_a)
    sf_b = solute_feat_cache.get(smiles_b)
    sv   = solvent_feat_cache.get(solvent)

    if sf_a is None or sf_b is None or sv is None:
        continue

    # Train: exclude both test solutes
    mask = (smiles_big != smiles_a) & (smiles_big != smiles_b)
    X_tr = X_big[mask]
    y_tr = y_big[mask]

    if len(X_tr) == 0:
        continue

    model = lgb.LGBMRegressor(verbosity=-1, random_state=42)
    model.fit(X_tr, y_tr)

    logS_a = model.predict(np.array([sf_a + sv + [T]], dtype=np.float32))[0]
    logS_b = model.predict(np.array([sf_b + sv + [T]], dtype=np.float32))[0]
    predictions[qi] = 'A' if logS_a > logS_b else 'B'

print(f"\nDone in {time.time()-t_start:.0f}s")

df_t4['pred_loco_lgbm'] = predictions

valid   = df_t4['pred_loco_lgbm'].notna()
n_valid = valid.sum()
acc     = (df_t4.loc[valid, 'pred_loco_lgbm'] == df_t4.loc[valid, 'Answer']).mean()
se      = np.sqrt(acc * (1 - acc) / n_valid)
ci      = 1.96 * se

print(f"\n{'='*55}")
print(f"Task 4 LOCO LightGBM Results (n={n_valid}/{len(df_t4)})")
print(f"{'='*55}")
print(f"{'Random baseline':<35} 50.0%")
print(f"{'LOCO LightGBM (default)':<35} {acc*100:.1f}% +/- {ci*100:.1f}%")
print(f"")
print(f"LLM results (SMILES, no reasoning):")
print(f"  {'Gemini 3 Flash':<33} 63.2% +/- 1.7%")
print(f"  {'GPT-5.2':<33} 58.9% +/- 1.8%")
print(f"  {'GLM-5':<33} 50.1% +/- 1.8%")

df_t4.to_csv(args.output, index=False)
print(f"\nSaved to: {args.output}")

summary = pd.DataFrame([
    {'Model': 'Random baseline',         'Accuracy (%)': 50.0,              'CI (+/-%)': 0.0,             'Type': 'Baseline'},
    {'Model': 'LOCO LightGBM (default)', 'Accuracy (%)': round(acc*100, 1), 'CI (+/-%)': round(ci*100, 1),'Type': 'ML'},
    {'Model': 'Gemini 3 Flash (SMILES)', 'Accuracy (%)': 63.2,              'CI (+/-%)': 1.7,             'Type': 'LLM'},
    {'Model': 'GPT-5.2 (SMILES)',        'Accuracy (%)': 58.9,              'CI (+/-%)': 1.8,             'Type': 'LLM'},
    {'Model': 'GLM-5 (SMILES)',          'Accuracy (%)': 50.1,              'CI (+/-%)': 1.8,             'Type': 'LLM'},
])
summary_path = args.output.replace('.csv', '_summary.csv')
summary.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")
