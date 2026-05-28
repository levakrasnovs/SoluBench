"""
LOCO ML baseline for SoluBench Task 3 — Delta approach.

Instead of predicting absolute LogS and comparing two predictions,
we directly predict the DIRECTION of co-solvent effect:
  - Target: sign(LogS_mixture - LogS_pure_Solvent1)
  - This is computed from matched pairs in MixtureSolDB

For each unique test solute S_i:
  - Train = all matched pairs where SMILES_Solute != S_i
  - Fit LightGBM classifier to predict direction (A=enhance / B=reduce)
  - Features: solute_desc(10) + solvent1_desc(10) + solvent2_desc(10)
              + Fraction_Solvent1(1) + Temperature_K(1) = 32 features

Usage:
    python ml_baseline_loco_task3_delta.py \
        --mixturedb MixtureSolDB__4_.csv \
        --task3     task3_results.csv \
        --output    task3_loco_delta_results.csv
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
parser.add_argument('--mixturedb', default='MixtureSolDB__4_.csv')
parser.add_argument('--task3',     default='task3_results.csv')
parser.add_argument('--output',    default='task3_loco_delta_results.csv')
args = parser.parse_args()


# ── Load and prepare MixtureSolDB ────────────────────────────────────────────

print("Loading data...")
df_mix = pd.read_csv(args.mixturedb)
df_t3  = pd.read_csv(args.task3)

df_mix = df_mix.dropna(subset=['LogS(mole_fraction)', 'SMILES_Solute',
                                'SMILES_Solvent1', 'SMILES_Solvent2',
                                'Fraction_Solvent1', 'Temperature_K'])

print(f"MixtureSolDB: {len(df_mix)} records, {df_mix['SMILES_Solute'].nunique()} unique solutes")
print(f"Task 3:       {len(df_t3)} questions")
print()

# Build matched pairs: mixed record + pure Solvent1 endpoint
pure_sv1 = df_mix[df_mix['Fraction_Solvent1'] == 1.0].copy()
mixed    = df_mix[df_mix['IsPureSolventEndpoint'] == False].copy()

keys = ['SMILES_Solute', 'Solvent1', 'Solvent2', 'Temperature_K', 'Fraction_Type']
pairs = mixed.merge(
    pure_sv1[keys + ['LogS(mole_fraction)']].rename(
        columns={'LogS(mole_fraction)': 'LogS_pure'}),
    on=keys, how='inner'
)
pairs['delta_LogS'] = pairs['LogS(mole_fraction)'] - pairs['LogS_pure']
pairs['direction']  = (pairs['delta_LogS'] > 0).astype(int)  # 1=enhance, 0=reduce

print(f"Matched pairs: {len(pairs)}, unique solutes: {pairs['SMILES_Solute'].nunique()}")
print(f"Direction enhance (1): {pairs['direction'].sum()}")
print(f"Direction reduce  (0): {(pairs['direction']==0).sum()}")
print()


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


print("Computing descriptors...")

all_solute_smiles = (set(pairs['SMILES_Solute'].unique()) |
                     set(df_t3['SMILES'].unique()))
solute_cache = {}
for s in tqdm(all_solute_smiles, desc="Solute descriptors"):
    d = get_desc(s)
    if d is not None:
        solute_cache[s] = d

sv1_map = (df_mix[['Solvent1', 'SMILES_Solvent1']]
           .drop_duplicates().set_index('Solvent1')['SMILES_Solvent1'].to_dict())
sv2_map = (df_mix[['Solvent2', 'SMILES_Solvent2']]
           .drop_duplicates().set_index('Solvent2')['SMILES_Solvent2'].to_dict())
solvent_smiles_map = {**sv1_map, **sv2_map}

solvent_cache = {}
for name, smiles in tqdm(solvent_smiles_map.items(), desc="Solvent descriptors"):
    d = get_desc(smiles)
    if d is not None:
        solvent_cache[name] = d

print(f"Solute descriptors:  {len(solute_cache)}")
print(f"Solvent descriptors: {len(solvent_cache)}")

t3_solvents = set(df_t3['Base_solvent'].unique()) | set(df_t3['Added_solvent'].unique())
missing_sv = t3_solvents - set(solvent_cache.keys())
if missing_sv:
    print(f"WARNING: {len(missing_sv)} Task 3 solvents missing: {missing_sv}")
else:
    print(f"All {len(t3_solvents)} Task 3 solvents have descriptors.")
print()


# ── Build feature matrix from matched pairs ──────────────────────────────────

print("Building feature matrix from matched pairs...")
X_pairs, y_pairs, smiles_pairs = [], [], []

for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Building matrix"):
    sd  = solute_cache.get(row['SMILES_Solute'])
    sv1 = solvent_cache.get(row['Solvent1'])
    sv2 = solvent_cache.get(row['Solvent2'])
    if sd is None or sv1 is None or sv2 is None:
        continue
    X_pairs.append(sd + sv1 + sv2 + [row['Fraction_Solvent1'], row['Temperature_K']])
    y_pairs.append(row['direction'])
    smiles_pairs.append(row['SMILES_Solute'])

X_pairs      = np.array(X_pairs, dtype=np.float32)
y_pairs      = np.array(y_pairs, dtype=np.int32)
smiles_pairs = np.array(smiles_pairs)

print(f"Feature matrix: {X_pairs.shape}")
print()


# ── LOCO classification ──────────────────────────────────────────────────────

test_solutes = df_t3['SMILES'].unique()
print(f"Starting LOCO: {len(test_solutes)} unique test solutes")
print("Model: LightGBM classifier, predicting direction of co-solvent effect")
print("-" * 60)

predictions  = [None] * len(df_t3)
skipped      = 0
t_start      = time.time()

for solute in tqdm(test_solutes, desc="LOCO progress"):
    mask = smiles_pairs != solute
    X_tr = X_pairs[mask]
    y_tr = y_pairs[mask]
    if len(X_tr) == 0 or len(np.unique(y_tr)) < 2:
        skipped += 1
        continue

    model = lgb.LGBMClassifier(verbosity=-1, random_state=42)
    model.fit(X_tr, y_tr)

    q_idx = df_t3.index[df_t3['SMILES'] == solute].tolist()
    for qi in q_idx:
        row = df_t3.loc[qi]
        sd  = solute_cache.get(row['SMILES'])
        sv1 = solvent_cache.get(row['Base_solvent'])
        sv2 = solvent_cache.get(row['Added_solvent'])
        if sd is None or sv1 is None or sv2 is None:
            skipped += 1
            continue
        T         = row['Temperature_K']
        frac_base = row['Fraction_base']

        feat = np.array([sd + sv1 + sv2 + [frac_base, T]], dtype=np.float32)
        pred = model.predict(feat)[0]
        predictions[qi] = 'A' if pred == 1 else 'B'

print(f"\nDone in {time.time()-t_start:.0f}s")
if skipped:
    print(f"Skipped: {skipped}")


# ── Evaluate ─────────────────────────────────────────────────────────────────

df_t3['pred_loco_lgbm_delta'] = predictions

valid   = df_t3['pred_loco_lgbm_delta'].notna()
n_valid = valid.sum()
acc     = (df_t3.loc[valid, 'pred_loco_lgbm_delta'] == df_t3.loc[valid, 'Answer']).mean()
se      = np.sqrt(acc * (1 - acc) / n_valid)
ci      = 1.96 * se

print(f"\n{'='*55}")
print(f"Task 3 LOCO LightGBM (delta/classifier) (n={n_valid}/{len(df_t3)})")
print(f"{'='*55}")
print(f"{'Random baseline':<35} 50.0%")
print(f"{'LOCO LightGBM classifier':<35} {acc*100:.1f}% +/- {ci*100:.1f}%")
print()

for ftype in ['mole', 'mass']:
    mask_f = valid & (df_t3['Fraction_type'] == ftype)
    if mask_f.sum() > 0:
        a    = (df_t3.loc[mask_f, 'pred_loco_lgbm_delta'] == df_t3.loc[mask_f, 'Answer']).mean()
        se_f = np.sqrt(a*(1-a)/mask_f.sum())
        print(f"  [{ftype}]: {a*100:.1f}% +/- {se_f*1.96*100:.1f}% (n={mask_f.sum()})")

print()
print(f"LLM results for comparison:")
print(f"  {'Gemini 3 Flash':<33} 86.4% +/- 1.4%")
print(f"  {'Claude Opus 4.5':<33} 84.5% +/- 1.5%")
print(f"  {'Qwen3.5 397B':<33} 83.5% +/- 1.5%")
print(f"  {'DeepSeek V3.2':<33} 55.2% +/- 2.0%")


# ── Save ─────────────────────────────────────────────────────────────────────

df_t3.to_csv(args.output, index=False)
print(f"\nSaved to: {args.output}")

summary = pd.DataFrame([
    {'Model': 'Random baseline',             'Accuracy (%)': 50.0,              'CI (+/-%)': 0.0,             'Type': 'Baseline'},
    {'Model': 'LOCO LightGBM (classifier)',  'Accuracy (%)': round(acc*100, 1), 'CI (+/-%)': round(ci*100, 1),'Type': 'ML'},
    {'Model': 'Gemini 3 Flash',              'Accuracy (%)': 86.4,              'CI (+/-%)': 1.4,             'Type': 'LLM'},
    {'Model': 'Claude Opus 4.5',             'Accuracy (%)': 84.5,              'CI (+/-%)': 1.5,             'Type': 'LLM'},
    {'Model': 'Qwen3.5 397B',               'Accuracy (%)': 83.5,              'CI (+/-%)': 1.5,             'Type': 'LLM'},
    {'Model': 'DeepSeek V3.2',               'Accuracy (%)': 55.2,              'CI (+/-%)': 2.0,             'Type': 'LLM'},
])
summary_path = args.output.replace('.csv', '_summary.csv')
summary.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")
