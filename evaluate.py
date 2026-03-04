"""
SoluBench Evaluation Script
============================
Computes accuracy metrics for Tasks 1–4 from results CSV files.

Usage:
    python evaluate.py
    python evaluate.py --results_dir path/to/results/
    python evaluate.py --tasks 1 3 4
    python evaluate.py --summary

Output:
    - Per-task accuracy for each model (overall + by |ΔlogS| bins)
    - Summary table across all tasks
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict


# ── Task configuration ─────────────────────────────────────────────────────────

TASK_CONFIG = {
    1: {
        "file":       "task1_results.csv",
        "answer_col": "Answer",
        "delta_col":  "Delta_LogS",
        "baseline":   50.0,
        "skip_cols":  {"id", "DOI", "SMILES", "Temperature_K", "Solvent_A", "Solvent_B",
                       "LogS_A", "LogS_B", "Delta_LogS", "Answer", "canary"},
        "delta_bins":   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf],
        "delta_labels": ["0.5–1.0", "1.0–1.5", "1.5–2.0", "2.0–2.5", "2.5–3.0", "≥3.0"],
    },
    2: {
        "file":       "task2_results.csv",
        "answer_col": "Answer",
        "delta_col":  "Delta_LogS_best_minus_second",
        "baseline":   17.1,
        "skip_cols":  {"id", "DOI", "SMILES", "Temperature_K", "Best_solvent", "Best_LogS",
                       "Second_best_solvent", "Second_best_LogS", "Delta_LogS_best_minus_second",
                       "All_solvents", "All_LogS", "Answer", "canary"},
        "delta_bins":   [0.1, 0.2, 0.4, 0.7, 1.2, np.inf],
        "delta_labels": ["0.1–0.2", "0.2–0.4", "0.4–0.7", "0.7–1.2", "≥1.2"],
    },
    3: {
        "file":       "task3_results.csv",
        "answer_col": "Answer",
        "delta_col":  "Delta_LogS_new_minus_base",
        "baseline":   50.0,
        "skip_cols":  {"id", "DOI", "SMILES", "Temperature_K", "Base_solvent", "Added_solvent",
                       "Fraction_base", "Fraction_added", "New_composition", "Fraction_type",
                       "Delta_LogS_new_minus_base", "Answer", "canary"},
        "delta_bins":   [0.1, 0.25, 0.5, 0.75, 1.0, np.inf],
        "delta_labels": ["0.1–0.25", "0.25–0.5", "0.5–0.75", "0.75–1.0", "≥1.0"],
    },
    4: {
        "file":       "task4_results.csv",
        "answer_col": "Answer",
        "delta_col":  "delta_logS",
        "baseline":   50.0,
        "skip_cols":  {"id", "Solvent", "Temperature_K", "SMILES_A", "SMILES_B",
                       "Compound_name_A", "Compound_name_B", "LogS_A", "LogS_B",
                       "DOI_A", "DOI_B", "delta_logS", "Answer", "canary"},
        "delta_bins":   [0.5, 1.0, 2.0, np.inf],
        "delta_labels": ["0.5–1.0", "1.0–2.0", "≥2.0"],
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_model_cols(row_keys, skip_cols):
    return [c for c in row_keys if c not in skip_cols]


def accuracy(rows, model_col, answer_col):
    filled = [r for r in rows if r[model_col].strip()]
    if not filled:
        return None
    correct = sum(1 for r in filled if r[model_col].strip() == r[answer_col].strip())
    return round(correct / len(filled) * 100, 1)


def accuracy_by_bins(rows, model_col, answer_col, delta_col, bins, labels):
    result = {}
    for i in range(len(labels)):
        lo = bins[i]
        hi = bins[i + 1]
        subset = [r for r in rows
                  if r[delta_col].strip()
                  and lo <= abs(float(r[delta_col])) < hi]
        result[labels[i]] = accuracy(subset, model_col, answer_col)
    return result


def fmt(v):
    return f"{v:.1f}" if v is not None else "—"


def print_table(headers, rows, col_widths):
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("─" * len(header_line))
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate_task(task_id, results_dir):
    cfg = TASK_CONFIG[task_id]
    fpath = Path(results_dir) / cfg["file"]

    if not fpath.exists():
        print(f"  [Task {task_id}] File not found: {fpath}")
        return {}

    with open(fpath, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    models = get_model_cols(rows[0].keys(), cfg["skip_cols"])
    n = len(rows)

    print(f"\n{'='*65}")
    print(f"  Task {task_id}  |  n = {n}  |  random baseline = {cfg['baseline']:.1f}%")
    print(f"{'='*65}")

    # Overall accuracy
    overall = {}
    for m in models:
        acc = accuracy(rows, m, cfg["answer_col"])
        if acc is not None:
            overall[m] = acc

    sorted_models = sorted(overall, key=lambda m: overall[m], reverse=True)

    print(f"\n  {'Model':<35} {'Accuracy (%)':>12}  {'n_filled':>8}")
    print(f"  {'─'*35}  {'─'*12}  {'─'*8}")
    for m in sorted_models:
        filled = sum(1 for r in rows if r[m].strip())
        print(f"  {m:<35} {overall[m]:>12.1f}  {filled:>8}")

    # Accuracy by |ΔlogS| bins
    print(f"\n  Accuracy by |ΔlogS| bins:")
    bin_labels = cfg["delta_labels"]
    header = ["Model"] + bin_labels
    col_w = [35] + [10] * len(bin_labels)
    bin_rows = []
    for m in sorted_models:
        bd = accuracy_by_bins(rows, m, cfg["answer_col"],
                              cfg["delta_col"], cfg["delta_bins"], bin_labels)
        bin_rows.append([m] + [fmt(bd[lbl]) for lbl in bin_labels])

    print()
    print_table(["  " + header[0]] + header[1:],
                [["  " + r[0]] + r[1:] for r in bin_rows],
                [col_w[0] + 2] + col_w[1:])

    return {"overall": overall, "models": sorted_models, "n": n}


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(all_results):
    print(f"\n{'='*65}")
    print("  SUMMARY — Overall accuracy (%) by task")
    print(f"{'='*65}\n")

    # Collect all models
    all_models = []
    seen = set()
    for task_id in sorted(all_results):
        res = all_results[task_id]
        for m in res.get("models", []):
            if m not in seen:
                all_models.append(m)
                seen.add(m)

    task_ids = sorted(all_results.keys())
    headers = ["Model"] + [f"Task {t}" for t in task_ids]
    col_w = [35] + [10] * len(task_ids)

    print_table(headers, [], col_w)  # header only
    for m in all_models:
        row = [m]
        for t in task_ids:
            v = all_results[t].get("overall", {}).get(m)
            row.append(fmt(v))
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SoluBench evaluation script")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing task result CSV files (default: current dir)")
    parser.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Tasks to evaluate, e.g. --tasks 1 2 4 (default: all)")
    parser.add_argument("--summary", action="store_true",
                        help="Print only the summary table")
    args = parser.parse_args()

    print(f"SoluBench Evaluation")
    print(f"Results dir : {Path(args.results_dir).resolve()}")
    print(f"Tasks       : {args.tasks}")

    all_results = {}
    for task_id in args.tasks:
        if task_id not in TASK_CONFIG:
            print(f"  Unknown task: {task_id}, skipping.")
            continue
        if args.summary:
            # light pass — just compute overall
            cfg = TASK_CONFIG[task_id]
            fpath = Path(args.results_dir) / cfg["file"]
            if fpath.exists():
                with open(fpath, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                models = get_model_cols(rows[0].keys(), cfg["skip_cols"])
                overall = {m: accuracy(rows, m, cfg["answer_col"]) for m in models}
                overall = {m: v for m, v in overall.items() if v is not None}
                sorted_m = sorted(overall, key=lambda m: overall[m], reverse=True)
                all_results[task_id] = {"overall": overall, "models": sorted_m, "n": len(rows)}
        else:
            all_results[task_id] = evaluate_task(task_id, args.results_dir)

    print_summary(all_results)


if __name__ == "__main__":
    main()