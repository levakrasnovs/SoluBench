"""
Randomized SMILES robustness test for SoluBench Task 1.

For each question, replaces the canonical SMILES of the solute with a
randomized SMILES (different atom ordering) and re-runs the model.
Compares accuracy against the original canonical SMILES run.

Usage:
    python run_randomized_smiles.py \
        --task1     task1_results.csv \
        --model     gemini-3-flash \
        --col_name  gemini-3-flash-rand \
        --output    task1_randomized_smiles_results.csv

Requires: rdkit, openai, python-dotenv, tqdm
"""

import argparse
import os
import sys
import time
import random

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rdkit import Chem
from tqdm import tqdm

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    sys.exit("Error: OPENROUTER_API_KEY not set")

MODEL_REGISTRY = {
    "gemini-3-flash":    "google/gemini-3-flash-preview",
    "gemini-2.5-flash":  "google/gemini-2.5-flash",
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "gpt-5.2":           "openai/gpt-5.2",
}

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


def randomize_smiles(smiles: str, seed: int = None) -> str:
    """Generate a randomized SMILES string for the same molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles  # fallback to original
        if seed is not None:
            random.seed(seed)
        rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
        # Verify round-trip
        mol_check = Chem.MolFromSmiles(rand_smiles)
        if mol_check is None:
            return smiles
        return rand_smiles
    except Exception:
        return smiles


def build_prompt(row, rand_smiles: str) -> str:
    return f"""You are a chemistry expert.

A compound has the following SMILES:
{rand_smiles}

Two solvents are considered:
A. {row["Solvent_A"]}
B. {row["Solvent_B"]}

Question:
In which solvent is this compound expected to have higher solubility?

IMPORTANT:
- Answer with a single letter only: A or B.
- Do NOT include any explanation or additional text.

Answer:"""


def call_model(prompt: str, model: str, max_retries: int = 5, retry_delay: float = 5.0) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                input=prompt,
                model=model,
                temperature=0.0,
            )
            return resp.output_text.strip().replace("Answer: ", "")
        except Exception as e:
            if attempt == max_retries:
                print(f"\n[ERROR] All {max_retries} attempts failed: {e}")
                return "ERROR"
            wait = retry_delay * attempt
            print(f"\n[WARN] Attempt {attempt} failed ({e}). Retrying in {wait:.0f}s...")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1",     default="task1_results.csv")
    parser.add_argument("--model",     default="gemini-3-flash")
    parser.add_argument("--col_name",  default="gemini-3-flash-rand")
    parser.add_argument("--canonical_col", default="gemini-3-flash",
                        help="Column with original canonical SMILES results for comparison")
    parser.add_argument("--output",    default="task1_randomized_smiles_results.csv")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_delay", type=float, default=5.0)
    args = parser.parse_args()

    model_str = MODEL_REGISTRY.get(args.model, args.model)
    print(f"Model:          {model_str}")
    print(f"Column:         {args.col_name}")
    print(f"Canonical col:  {args.canonical_col}")
    print(f"Random seed:    {args.seed}")

    df = pd.read_csv(args.task1)
    print(f"Loaded {len(df)} rows from {args.task1}")

    # Pre-generate all randomized SMILES
    print("Generating randomized SMILES...")
    rand_smiles_list = []
    n_changed = 0
    for _, row in df.iterrows():
        rs = randomize_smiles(row["SMILES"], seed=args.seed)
        rand_smiles_list.append(rs)
        if rs != row["SMILES"]:
            n_changed += 1
    df["SMILES_randomized"] = rand_smiles_list
    print(f"SMILES changed: {n_changed}/{len(df)} ({n_changed/len(df)*100:.1f}%)")

    # Add result column if missing
    if args.col_name not in df.columns:
        df[args.col_name] = ""

    to_run = df.index if args.overwrite else df[df[args.col_name].isna() | (df[args.col_name] == "")].index
    if args.limit:
        to_run = to_run[:args.limit]
    print(f"Running {len(to_run)} rows\n")

    for idx in tqdm(to_run, desc="Randomized SMILES"):
        row = df.loc[idx]
        prompt = build_prompt(row, row["SMILES_randomized"])
        answer = call_model(prompt, model_str, args.max_retries, args.retry_delay)
        df.at[idx, args.col_name] = answer
        df.to_csv(args.output, index=False)

    # Evaluate
    import numpy as np
    valid = df[args.col_name].notna() & (df[args.col_name] != "") & (df[args.col_name] != "ERROR")
    n = valid.sum()

    acc_rand = (df.loc[valid, args.col_name] == df.loc[valid, "Answer"]).mean()
    se_rand = np.sqrt(acc_rand * (1 - acc_rand) / n)

    print(f"\n{'='*55}")
    print(f"Randomized SMILES results (n={n})")
    print(f"{'='*55}")
    print(f"Randomized SMILES accuracy: {acc_rand*100:.1f}% ± {1.96*se_rand*100:.1f}%")

    if args.canonical_col in df.columns:
        acc_can = (df.loc[valid, args.canonical_col] == df.loc[valid, "Answer"]).mean()
        se_can = np.sqrt(acc_can * (1 - acc_can) / n)
        print(f"Canonical SMILES accuracy:  {acc_can*100:.1f}% ± {1.96*se_can*100:.1f}%")
        print(f"Difference:                 {(acc_can - acc_rand)*100:.1f}%")

        # McNemar test
        from scipy.stats import chi2
        b = ((df.loc[valid, args.canonical_col] == df.loc[valid, "Answer"]) &
             (df.loc[valid, args.col_name] != df.loc[valid, "Answer"])).sum()
        c = ((df.loc[valid, args.canonical_col] != df.loc[valid, "Answer"]) &
             (df.loc[valid, args.col_name] == df.loc[valid, "Answer"])).sum()
        if b + c > 0:
            stat = (abs(b - c) - 1) ** 2 / (b + c)
            p = 1 - chi2.cdf(stat, df=1)
            print(f"McNemar test: b={b}, c={c}, p={p:.4f}")

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
