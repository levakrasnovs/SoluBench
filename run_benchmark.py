"""
run_benchmark.py — SoluBench evaluation script

Usage:
    python run_benchmark.py --task 1 --model gemini/gemini-2.5-flash
    python run_benchmark.py --task 2 --model openai/gpt-4.1 --col_name gpt-4.1
    python run_benchmark.py --task 3 --model x-ai/grok-4.1-fast --reasoning none
    python run_benchmark.py --task 4 --model z-ai/glm-5 --use_name

Supported tasks: 1, 2, 3, 4
Results are appended as a new column to the existing CSV.
Progress is saved after each row (safe to interrupt and resume).
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    sys.exit("Error: OPENROUTER_API_KEY not set in environment or .env file")

# ── Model registry ─────────────────────────────────────────────────────────────
# Maps CSV column names → OpenRouter model strings.
# Pass either the short key or the full OpenRouter string to --model.
MODEL_REGISTRY: dict[str, str] = {
    "claude-sonnet-4.6":        "anthropic/claude-sonnet-4.6",
    "claude-sonnet-4.5":        "anthropic/claude-sonnet-4.5",
    "claude-opus-4.6":          "anthropic/claude-opus-4.6",
    "claude-opus-4.5":          "anthropic/claude-opus-4.5",
    "claude-haiku-4.5":         "anthropic/claude-haiku-4.5",
    "gemini-3-flash":           "google/gemini-3-flash-preview",
    "gemini-3-flash-name":      "google/gemini-3-flash-preview",
    "gemini-2.5-flash":         "google/gemini-2.5-flash",
    "gemini-3.1-pro":           "google/gemini-3.1-pro-preview",
    "gemini-3.1-pro(high)":     "google/gemini-3.1-pro-preview",
    "qwen3-235b":               "qwen/qwen3-235b-a22b-2507",
    "qwen3.5-397b-a17b":        "qwen/qwen3.5-397b-a17b",
    "qwen-2.5-7b-instruct":     "qwen/qwen-2.5-7b-instruct",
    "llama-3.1-8b-instruct":    "meta-llama/llama-3.1-8b-instruct",
    "llama-3.3-70b-instruct":   "meta-llama/llama-3.3-70b-instruct",
    "grok-4.1-fast":            "x-ai/grok-4.1-fast",
    "gemma-3-27b-it":           "google/gemma-3-27b-it",
    "deepseek-v3.2":            "deepseek/deepseek-v3.2",
    "glm-5":                    "z-ai/glm-5",
    "glm-5-name":               "z-ai/glm-5",
    "glm-4.7":                  "z-ai/glm-4.7",
    "gpt-5.2":                  "openai/gpt-5.2",
    "gpt-5.2-name":             "openai/gpt-5.2",
    "gpt-4.1":                  "openai/gpt-4.1",
    "kimi-k2.5":                "moonshotai/kimi-k2.5",
}

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# ── Prompt builders ────────────────────────────────────────────────────────────

def build_prompt_task1(row) -> str:
    return f"""You are a chemistry expert.

A compound has the following SMILES:
{row["SMILES"]}

Two solvents are considered:
A. {row["Solvent_A"]}
B. {row["Solvent_B"]}

Question:
In which solvent is this compound expected to have higher solubility?

IMPORTANT:
- Answer with a single letter only: A or B.
- Do NOT include any explanation or additional text.

Answer:"""


def build_prompt_task2(row) -> str:
    solvents = row["All_solvents"].split(";") if isinstance(row["All_solvents"], str) else row["All_solvents"]
    options_block = "\n".join(f"{chr(65+i)}. {s}" for i, s in enumerate(solvents))
    return f"""You are a chemistry expert.

A compound has the following SMILES:
{row["SMILES"]}

The solubility of this compound was experimentally measured in several solvents at the same temperature.

From the solvents listed below, select the single solvent in which this compound is expected to have the highest solubility.

Solvents:
{options_block}

IMPORTANT:
- Answer with ONE letter only.
- Do NOT include any explanation or additional text.

Answer:"""


def build_prompt_task3(row) -> str:
    return f"""You are a chemistry expert.

A compound has the following SMILES:
{row["SMILES"]}

Compare the solubility of this compound in:
1. Pure {row["Base_solvent"]}
2. In {row["New_composition"]} mixture with corresponding {row["Fraction_type"]} fractions.

The solubility was experimentally measured at the same temperature.

Question:
Is the solubility in the mixture {row["New_composition"]} enhanced or reduced compared to pure {row["Base_solvent"]}?

Options:
A. Enhanced (higher solubility in mixture)
B. Reduced (lower solubility in mixture)

IMPORTANT:
- Answer with a single letter only: A or B
- Do NOT include any explanation or additional text

Answer:"""


def build_prompt_task4(row, use_name: bool = False) -> str:
    if use_name:
        compound_a = row.get("Compound_name_A", row["SMILES_A"])
        compound_b = row.get("Compound_name_B", row["SMILES_B"])
    else:
        compound_a = row["SMILES_A"]
        compound_b = row["SMILES_B"]
    return f"""You are a chemistry expert.

A solvent is given:
{row["Solvent"]}

Two compounds are considered:
A. {compound_a}
B. {compound_b}

Question:
Which compound is expected to have higher solubility in this solvent?

IMPORTANT:
- Answer with a single letter only: A or B.
- Do NOT include any explanation or additional text.

Answer:"""


PROMPT_BUILDERS = {
    1: build_prompt_task1,
    2: build_prompt_task2,
    3: build_prompt_task3,
    4: build_prompt_task4,
}

DATA_PATHS = {
    1: "data/task1_pairwise_solvent_comparison",
    2: "data/task2_best_solvent_selection",
    3: "data/task3_cosolvent_effect_prediction",
    4: "data/task4_pairwise_compound_comparison",
}

# ── API call ───────────────────────────────────────────────────────────────────

def call_model(
    prompt: str,
    model: str,
    reasoning: str | None,
    max_retries: int = 5,
    retry_delay: float = 5.0,
) -> str:
    """Call OpenRouter API with retry logic. Returns raw answer string."""
    kwargs = dict(model=model, temperature=0.0)

    # reasoning parameter (OpenRouter convention)
    if reasoning is not None:
        if reasoning == "none":
            kwargs["reasoning"] = {"effort": "none"}
        elif reasoning == "low":
            kwargs["reasoning"] = {"effort": "low"}
        elif reasoning == "medium":
            kwargs["reasoning"] = {"effort": "medium"}
        elif reasoning == "high":
            kwargs["reasoning"] = {"effort": "high"}

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(input=prompt, **kwargs)
            return resp.output_text.strip().replace('Answer: ', '')

        except Exception as e:
            if attempt == max_retries:
                print(f"\n[ERROR] All {max_retries} attempts failed: {e}")
                return "ERROR"
            wait = retry_delay * attempt
            print(f"\n[WARN] Attempt {attempt} failed ({e}). Retrying in {wait:.0f}s…")
            time.sleep(wait)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SoluBench benchmark runner")
    parser.add_argument("--task",      type=int,  required=True,  choices=[1, 2, 3, 4],
                        help="Task number (1–4)")
    parser.add_argument("--model",     type=str,  required=True,
                        help="OpenRouter model string, e.g. 'openai/gpt-4.1'")
    parser.add_argument("--col_name",  type=str,  default=None,
                        help="Column name to write results into (default: last part of model string)")
    parser.add_argument("--use_name",  action="store_true",
                        help="Use compound names instead of SMILES in prompts (Task 4 only by default)")
    parser.add_argument("--reasoning", type=str,  default="none",
                        choices=["none", "low", "medium", "high"],
                        help="Reasoning effort (for models that support it)")
    parser.add_argument("--data",      type=str,  default=None,
                        help="Override default data path (without extension)")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_delay", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run rows that already have a result")
    parser.add_argument("--limit",     type=int,  default=None,
                        help="Process only first N rows (for testing)")
    args = parser.parse_args()

    # Resolve model string and column name
    model_str = MODEL_REGISTRY.get(args.model, args.model)  # short key or full OR string
    col_name  = args.col_name or args.model.split("/")[-1]
    print(f"Model : {model_str}")
    print(f"Column: {col_name}")

    # Load data (CSV or JSONL)
    base_path = args.data or DATA_PATHS[args.task]
    csv_path  = base_path + ".csv"
    jsonl_path = base_path + ".jsonl"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        save_path = csv_path
        print(f"Loaded {len(df)} rows from {csv_path}")
    elif os.path.exists(jsonl_path):
        import json as _json
        with open(jsonl_path) as f:
            df = pd.DataFrame([_json.loads(l) for l in f])
        save_path = jsonl_path.replace(".jsonl", "_results.csv")
        print(f"Loaded {len(df)} rows from {jsonl_path}")
        print(f"Results will be saved to {save_path}")
    else:
        sys.exit(f"Error: no data file found at {base_path}.csv or {base_path}.jsonl")

    # Add column if missing
    if col_name not in df.columns:
        df[col_name] = ""
        print(f"Added new column: '{col_name}'")
    else:
        filled = df[col_name].notna() & (df[col_name] != "")
        print(f"Column '{col_name}' exists — {filled.sum()}/{len(df)} rows already filled")

    build_prompt = PROMPT_BUILDERS[args.task]

    # Run
    to_run = df.index if args.overwrite else df[df[col_name].isna() | (df[col_name] == "")].index
    if args.limit:
        to_run = to_run[:args.limit]
    print(f"Running {len(to_run)} rows with model '{model_str}' (col='{col_name}')\n")

    for idx in tqdm(to_run, desc=f"Task {args.task}"):
        row = df.loc[idx]
        use_name = args.use_name if args.task == 4 else False
        prompt = build_prompt(row, use_name=use_name) if args.task == 4 else build_prompt(row)
        answer = call_model(
            prompt, model_str, args.reasoning,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        df.at[idx, col_name] = answer
        # Save after every row — safe to interrupt
        df.to_csv(save_path, index=False)

    # Final accuracy report
    if "Answer" in df.columns:
        total = len(df)
        correct = (df[col_name] == df["Answer"]).sum()
        errors  = (df[col_name] == "ERROR").sum()
        print(f"\nDone. Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
        if errors:
            print(f"  Errors (API failures): {errors}")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()