"""
SoluBench — plot_figures.py
============================
Generates all publication figures from the four results CSV files.

Usage:
    python plot_figures.py                        # all figures
    python plot_figures.py --tasks 1 3            # specific tasks
    python plot_figures.py --results_dir results/ --out_dir figures/

Output files (in out_dir):
    figure_task1_overall.pdf/png
    figure_task1_polarity.pdf/png
    figure_task2_overall.pdf/png
    figure_task2_mae.pdf/png
    figure_task3_overall.pdf/png
    figure_task3_delta.pdf/png
    figure_task4_overall.pdf/png
    figure_task4_heatmap_solvent.pdf/png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Shared config
# ══════════════════════════════════════════════════════════════════════════════

RCPARAMS_STANDARD = {
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.linewidth':    0.6,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'legend.fontsize':   8,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
}

# Apply standard rcParams globally at import time
plt.rcParams.update(RCPARAMS_STANDARD)

RCPARAMS_SMALL = {**RCPARAMS_STANDARD,
    'font.size':         7,
    'axes.labelsize':    7,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'legend.fontsize':   5.5,
    'legend.frameon':    True,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '#cccccc',
}

C_OPEN = '#4A9B6F'
C_PROP = '#2E6E8E'
C_RAND = '#999999'

PROP_MODELS = [
    'gemini-3-flash', 'gemini-3-flash-preview',
    'gemini-2.5-flash', 'gpt-4.1', 'gpt-5.2',
    'claude-sonnet-4.5', 'claude-sonnet-4.6', 'claude-haiku-4.5',
    'claude-opus-4.5', 'claude-opus-4.6', 'grok-4.1-fast',
    'gemini-3.1-pro', 'gemini-3.1-pro(high)',
]
OPEN_MODELS = [
    'qwen3-235b', 'qwen3.5-397b-a17b', 'gemma-3-27b-it',
    'llama-3.3-70b-instruct', 'llama-3.1-8b-instruct',
    'deepseek-v3.2', 'glm-4.7', 'glm-5', 'kimi-k2.5',
    'qwen-2.5-7b-instruct',
]

NAME_MAP = {
    'gemini-3-flash':           'Gemini 3 Flash',
    'gemini-3-flash-preview':   'Gemini 3 Flash',
    'gemini-3-flash-name':      'Gemini 3 Flash\n(name)',
    'gemini-2.5-flash':         'Gemini 2.5 Flash',
    'gpt-4.1':                  'GPT-4.1',
    'gpt-5.2':                  'GPT-5.2',
    'gpt-5.2-name':             'GPT-5.2\n(name)',
    'claude-sonnet-4.5':        'Claude Sonnet 4.5',
    'claude-sonnet-4.6':        'Claude Sonnet 4.6',
    'claude-haiku-4.5':         'Claude Haiku 4.5',
    'claude-opus-4.5':          'Claude Opus 4.5',
    'claude-opus-4.6':          'Claude Opus 4.6',
    'grok-4.1-fast':            'Grok 4.1 Fast',
    'gemini-3.1-pro':           'Gemini 3.1 Pro (low)',
    'gemini-3.1-pro(high)':     'Gemini 3.1 Pro (high)',
    'qwen3-235b':               'Qwen3 235B',
    'qwen3.5-397b-a17b':        'Qwen3.5 397B',
    'gemma-3-27b-it':           'Gemma 3 27B',
    'llama-3.3-70b-instruct':   'LLaMA 3.3 70B',
    'llama-3.1-8b-instruct':    'LLaMA 3.1 8B',
    'deepseek-v3.2':            'DeepSeek V3.2',
    'glm-4.7':                  'GLM-4.7',
    'glm-5':                    'GLM-5',
    'glm-5-name':               'GLM-5\n(name)',
    'kimi-k2.5':                'Kimi K2.5',
    'qwen-2.5-7b-instruct':     'Qwen2.5 7B',
    'random_baseline':          'Random baseline',
}

# Name map for Task 4 — includes SMILES/name suffixes for bar and heatmap
# Flat names for Task 4 overall bar chart
BAR_NAME_MAP_T4 = {
    'gemini-3-flash':         'Gemini 3 Flash (SMILES)',
    'gemini-3-flash-name':    'Gemini 3 Flash (name)',
    'gpt-5.2':                'GPT-5.2 (SMILES)',
    'gpt-5.2-name':           'GPT-5.2 (name)',
    'glm-5':                  'GLM-5 (SMILES)',
    'glm-5-name':             'GLM-5 (name)',
    'gemini-3.1-pro':         'Gemini 3.1 Pro (low)',
    'gemini-3.1-pro(high)':   'Gemini 3.1 Pro (high)',
    'random_baseline':        'Random baseline',
}

# Multiline names for Task 4 heatmap x-axis labels
HEATMAP_NAME_MAP = {
    'gemini-3-flash':         'Gemini 3\nFlash\n(SMILES)',
    'gemini-3-flash-name':    'Gemini 3\nFlash\n(name)',
    'gpt-5.2':                'GPT-5.2\n(SMILES)',
    'gpt-5.2-name':           'GPT-5.2\n(name)',
    'glm-5':                  'GLM-5\n(SMILES)',
    'glm-5-name':             'GLM-5\n(name)',
    'gemini-3.1-pro':         'Gemini 3.1\nPro (low)',
    'gemini-3.1-pro(high)':   'Gemini 3.1\nPro (high)',
}

# Models to exclude per task (e.g. incomplete runs)
TASK_EXCLUDE = {
    'task3': {'grok-4.1-fast'},
    'task4': {'claude-sonnet-4.5'},
}

META_COLS = {
    'task1': {'id','DOI','SMILES','Temperature_K','Solvent_A','Solvent_B',
              'LogS_A','LogS_B','Delta_LogS','Answer','canary'},
    'task2': {'id','DOI','SMILES','Temperature_K','Best_solvent','Best_LogS',
              'Second_best_solvent','Second_best_LogS','Delta_LogS_best_minus_second',
              'All_solvents','All_LogS','Answer','canary'},
    'task3': {'id','DOI','SMILES','Temperature_K','Base_solvent','Added_solvent',
              'Fraction_base','Fraction_added','New_composition','Fraction_type',
              'Delta_LogS_new_minus_base','Answer','canary','MolWt'},
    'task4': {'id','Solvent','Temperature_K','SMILES_A','SMILES_B',
              'Compound_name_A','Compound_name_B','LogS_A','LogS_B',
              'DOI_A','DOI_B','delta_logS','Answer','canary'},
}

ANSWER_COL = {
    'task1': 'Answer', 'task2': 'Answer', 'task3': 'Answer', 'task4': 'Answer',
}

# Line colors for multi-line plots
LINE_COLORS_PROP = ['#1f77b4','#17becf','#2ca02c','#d62728','#e377c2',
                    '#ff9896','#9467bd','#8c564b','#bcbd22','#98df8a']
LINE_COLORS_OPEN = ['#ff7f0e','#2ca02c','#7f7f7f','#9467bd',
                    '#8c564b','#e377c2','#bcbd22','#17becf',
                    '#d62728','#aec7e8']


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load(results_dir, task):
    fname = {'task1': 'task1_results.csv', 'task2': 'task2_results.csv',
             'task3': 'task3_results.csv', 'task4': 'task4_results.csv'}[task]
    return pd.read_csv(Path(results_dir) / fname)


def model_cols(df, task):
    return [c for c in df.columns if c not in META_COLS[task]]


def accuracy(df, model, answer_col):
    filled = df[model].dropna()
    filled = filled[filled.str.strip() != '']
    if len(filled) == 0:
        return None
    return (filled == df.loc[filled.index, answer_col]).mean() * 100


def overall_results(df, models, answer_col, prop_set, open_set, baseline, baseline_label='Random baseline'):
    results = []
    for m in models:
        if m not in df.columns:
            continue
        acc = accuracy(df, m, answer_col)
        if acc is None:
            continue
        if m in prop_set:
            cat = 'prop'
        elif m in open_set:
            cat = 'open'
        else:
            cat = 'open'  # fallback
        results.append((m, acc, cat))
    if baseline is not None:
        results.append(('random_baseline', baseline, 'rand'))
    return sorted(results, key=lambda x: x[1])


def bar_colors(results):
    colors = []
    for _, _, cat in results:
        if cat == 'rand':   colors.append(C_RAND)
        elif cat == 'prop': colors.append(C_PROP)
        else:               colors.append(C_OPEN)
    return colors


def overall_bar(results, xlim, title, out_path, figsize=(4.5, None)):
    plt.rcParams.update(RCPARAMS_STANDARD)
    n = len(results)
    h = figsize[1] or max(3.5, 0.35 * n + 1.0)
    fig, ax = plt.subplots(figsize=(figsize[0], h))

    y = np.arange(n)
    values = [a for _, a, _ in results]
    names  = [NAME_MAP.get(m, m) for m, _, _ in results]
    colors = bar_colors(results)

    ax.barh(y, values, height=0.7, color=colors, alpha=0.85,
            linewidth=0.4, edgecolor='white')
    for i, val in enumerate(values):
        ax.text(val + 0.5, i, f'{val:.1f}%', va='center', ha='left', fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(*xlim)
    ax.set_xlabel('Accuracy (%)', labelpad=4)

    legend_elements = [
        Patch(facecolor=C_RAND, label='Random baseline'),
        Patch(facecolor=C_OPEN, label='Open-source'),
        Patch(facecolor=C_PROP, label='Proprietary'),
    ]
    # Only show Random if present
    if not any(c == 'rand' for _, _, c in results):
        legend_elements = legend_elements[1:]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7.5,
              frameon=True, framealpha=0.9, edgecolor='#cccccc')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linewidth=0.3, color='#ccc', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.06, top=0.98)
    _save(fig, out_path)


def _save(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'  Saved: {out_path.stem}')


# ══════════════════════════════════════════════════════════════════════════════
# Task 1
# ══════════════════════════════════════════════════════════════════════════════

SOLVENT_POLARITY = {
    'water':'protic','methanol':'protic','ethanol':'protic','n-propanol':'protic',
    'isopropanol':'protic','n-butanol':'protic','sec-butanol':'protic',
    'tert-butanol':'protic','isobutanol':'protic','n-pentanol':'protic',
    'isopentanol':'protic','n-hexanol':'protic','n-heptanol':'protic',
    'n-octanol':'protic','n-nonanol':'protic','n-decanol':'protic',
    '2-butoxyethanol':'protic','2-methoxyethanol':'protic','2-ethoxyethanol':'protic',
    '2-propoxyethanol':'protic','ethylene glycol':'protic','propylene glycol':'protic',
    'transcutol':'protic','acetic acid':'protic','formic acid':'protic',
    'propionic acid':'protic','n-butyric acid':'protic','formamide':'protic',
    'DMSO':'aprotic','DMF':'aprotic','DMAc':'aprotic','NMP':'aprotic',
    'acetonitrile':'aprotic','acetone':'aprotic','2-butanone':'aprotic',
    'MIBK':'aprotic','cyclohexanone':'aprotic','cyclopentanone':'aprotic',
    'gamma-butyrolactone':'aprotic','THF':'aprotic','1,4-dioxane':'aprotic',
    'ethyl acetate':'aprotic','methyl acetate':'aprotic','n-propyl acetate':'aprotic',
    'n-butyl acetate':'aprotic','n-pentyl acetate':'aprotic',
    'isopropyl acetate':'aprotic','isobutyl acetate':'aprotic',
    'ethyl formate':'aprotic','dimethyl carbonate':'aprotic',
    'n-hexane':'nonpolar','n-heptane':'nonpolar','n-octane':'nonpolar',
    'n-pentane':'nonpolar','n-dodecane':'nonpolar','cyclohexane':'nonpolar',
    'benzene':'nonpolar','toluene':'nonpolar','ethylbenzene':'nonpolar',
    'm-xylene':'nonpolar','o-xylene':'nonpolar','p-xylene':'nonpolar',
    'chlorobenzene':'nonpolar','anisole':'nonpolar','chloroform':'nonpolar',
    'dichloromethane':'nonpolar','1,2-dichloroethane':'nonpolar',
    'tetrachloromethane':'nonpolar','diethyl ether':'nonpolar','MTBE':'nonpolar',
}
SHORT_LABELS = {
    'aprotic vs aprotic':  'Apr–Apr',
    'aprotic vs nonpolar': 'Apr–Non',
    'aprotic vs protic':   'Apr–Pro',
    'nonpolar vs nonpolar':'Non–Non',
    'nonpolar vs protic':  'Non–Pro',
    'protic vs protic':    'Pro–Pro',
}


def plot_task1(df, out_dir):
    print('\nTask 1:')
    models = model_cols(df, 'task1')
    prop = [m for m in models if m in PROP_MODELS]
    open_ = [m for m in models if m in OPEN_MODELS]

    # Overall bar
    results = overall_results(df, models, 'Answer', set(PROP_MODELS), set(OPEN_MODELS), 50.0)
    overall_bar(results, (45, 101), 'Task 1', out_dir / 'figure_task1_overall')

    # Delta LogS line chart
    plot_task1_delta(df, out_dir)

    # MolWt line chart
    plot_task1_molwt(df, out_dir)

    # Polarity combo chart
    def classify(row):
        a = SOLVENT_POLARITY.get(row['Solvent_A'], 'unknown')
        b = SOLVENT_POLARITY.get(row['Solvent_B'], 'unknown')
        if 'unknown' in (a, b):
            return 'unknown'
        return ' vs '.join(sorted([a, b]))

    plt.rcParams.update(RCPARAMS_STANDARD)
    df = df.copy()
    df['combo'] = df.apply(classify, axis=1)
    counts = df['combo'].value_counts()
    combos = sorted([c for c in counts.index if counts[c] >= 50 and c != 'unknown'])
    cats = [SHORT_LABELS.get(c, c) for c in combos]
    x = np.arange(len(cats)) * 1.5

    acc_prop, acc_open = {}, {}
    for m in prop:
        df['correct'] = (df[m] == df['Answer']).astype(int)
        acc_prop[m] = df.groupby('combo', observed=True)['correct'].mean()
    for m in open_:
        df['correct'] = (df[m] == df['Answer']).astype(int)
        acc_open[m] = df.groupby('combo', observed=True)['correct'].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    w_p, w_o = 0.12, 0.11

    for i, m in enumerate(prop):
        if m not in acc_prop: continue
        vals = [acc_prop[m].get(c, np.nan) * 100 for c in combos]
        offset = (i - len(prop)/2 + 0.5) * w_p
        ax1.bar(x + offset, vals, w_p, label=NAME_MAP.get(m, m),
                color=LINE_COLORS_PROP[i % len(LINE_COLORS_PROP)], alpha=0.85, linewidth=0)

    for i, m in enumerate(open_):
        if m not in acc_open: continue
        vals = [acc_open[m].get(c, np.nan) * 100 for c in combos]
        offset = (i - len(open_)/2 + 0.5) * w_o
        ax2.bar(x + offset, vals, w_o, label=NAME_MAP.get(m, m),
                color=LINE_COLORS_OPEN[i % len(LINE_COLORS_OPEN)], alpha=0.85, linewidth=0)

    combo_n = [counts[c] for c in combos]
    for ax, title in [(ax1, 'Proprietary'), (ax2, 'Open-source')]:
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=7)
        ax.set_ylim(50, 100)
        ax.set_yticks(np.arange(50, 105, 10))
        ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        for i, n in enumerate(combo_n):
            ax.text(i * 1.5, -0.08, f'$n$={n}', ha='center', va='top',
                    fontsize=7, transform=ax.get_xaxis_transform())
        ax.legend(loc='lower right', ncol=2, borderpad=0.4,
                  handlelength=0.8, handletextpad=0.4, labelspacing=0.2,
                  columnspacing=0.6, fontsize=7, frameon=True,
                  framealpha=0.9, edgecolor='#cccccc')
    ax1.set_ylabel('Accuracy (%)', labelpad=4)
    ax2.tick_params(axis='y', length=0)
    plt.tight_layout()
    _save(fig, out_dir / 'figure_task1_polarity')



def _molwt_line_chart(df, task_key, answer_col, out_dir, out_name, exclude=None):
    """Generic MolWt line chart for any task that has a SMILES column."""
    if not RDKIT_OK:
        print(f'  [skip] {out_name} — rdkit not available')
        return
    plt.rcParams.update(RCPARAMS_STANDARD)
    exclude = (exclude or set()) | TASK_EXCLUDE.get(task_key, set())
    models = [m for m in model_cols(df, task_key) if m not in exclude]
    prop_m = [m for m in models if m in PROP_MODELS]
    open_m = [m for m in models if m in OPEN_MODELS]

    df = df.copy()
    df['MolWt'] = df['SMILES'].apply(
        lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else np.nan
    )

    bins    = [0, 150, 250, 350, 500, np.inf]
    blabels = ['<150', '150–250', '250–350', '350–500', '≥500']
    x = list(range(len(blabels)))

    df['mw_bin'] = pd.cut(df['MolWt'], bins=bins, labels=blabels)
    bin_counts   = df['mw_bin'].value_counts().sort_index()
    xticklabels  = [f'{l}\n$n$={bin_counts[l]}' for l in blabels]

    acc_data = {}
    for m in models:
        if m not in df.columns: continue
        df['correct'] = (df[m] == df[answer_col]).astype(int)
        acc_data[m]   = df.groupby('mw_bin', observed=True)['correct'].mean().values * 100

    prop_m = [m for m in prop_m if m in acc_data]
    open_m = [m for m in open_m if m in acc_data]

    col_p = {m: LINE_COLORS_PROP[i % len(LINE_COLORS_PROP)] for i, m in enumerate(prop_m)}
    col_o = {m: LINE_COLORS_OPEN[i % len(LINE_COLORS_OPEN)] for i, m in enumerate(open_m)}

    fig, (ax_p, ax_o) = plt.subplots(1, 2, figsize=(8.0, 3.6),
                                      sharey=True, gridspec_kw={'wspace': 0.08})

    def style_ax(ax, ylabel=True):
        ax.set_xlim(-0.5, len(blabels) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=7)
        ax.set_xlabel('Molecular Weight (Da)', labelpad=6)
        if ylabel: ax.set_ylabel('Accuracy (%)', labelpad=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--')
        ax.set_axisbelow(True)

    for m in prop_m:
        ax_p.plot(x, acc_data[m], color=col_p[m], linewidth=0.9,
                  marker='o', markersize=3.0, label=NAME_MAP.get(m, m))
    style_ax(ax_p, ylabel=True)
    ax_p.set_title('Proprietary', fontsize=9, fontweight='bold', pad=4)
    ax_p.legend(loc='best', ncol=2, borderpad=0.4, handlelength=1.3,
                handletextpad=0.4, labelspacing=0.2, columnspacing=0.8,
                fontsize=7, frameon=True, framealpha=0.9, edgecolor='#cccccc')

    for m in open_m:
        ax_o.plot(x, acc_data[m], color=col_o[m], linewidth=0.9,
                  marker='s', markersize=3.0, linestyle='--', label=NAME_MAP.get(m, m))
    style_ax(ax_o, ylabel=False)
    ax_o.set_title('Open-source', fontsize=9, fontweight='bold', pad=4)
    ax_o.legend(loc='best', ncol=2, borderpad=0.4, handlelength=1.3,
                handletextpad=0.4, labelspacing=0.2, columnspacing=0.8,
                fontsize=7, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax_o.tick_params(axis='y', length=0)

    # Auto y-range with padding
    all_vals = [v for m in prop_m + open_m for v in acc_data[m] if not np.isnan(v)]
    if all_vals:
        lo = max(0,  min(all_vals) - 5)
        hi = min(100, max(all_vals) + 5)
        ax_p.set_ylim(lo, hi)

    plt.tight_layout()
    _save(fig, out_dir / out_name)


def plot_task1_molwt(df, out_dir):
    """Line chart: accuracy by molecular weight bins — Task 1."""
    _molwt_line_chart(df, 'task1', 'Answer', out_dir, 'figure_task1_molwt')


def plot_task1_delta(df, out_dir):
    """Line chart: accuracy by |ΔlogS| bins, two subplots (prop / open)."""
    plt.rcParams.update(RCPARAMS_STANDARD)
    models = model_cols(df, 'task1')
    prop_m = [m for m in models if m in PROP_MODELS]
    open_m = [m for m in models if m in OPEN_MODELS]

    bins   = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
    blabels = ['0.5–1.0', '1.0–1.5', '1.5–2.0', '2.0–2.5', '2.5–3.0', '≥3.0']
    x = list(range(len(blabels)))

    df = df.copy()
    df['abs_delta'] = df['Delta_LogS'].abs()
    df['bin'] = pd.cut(df['abs_delta'], bins=bins, labels=blabels)
    bin_counts = df['bin'].value_counts().sort_index()
    xticklabels = [f'{l}\n$n$={bin_counts[l]}' for l in blabels]

    acc_data = {}
    for m in models:
        if m not in df.columns: continue
        df['correct'] = (df[m] == df['Answer']).astype(int)
        acc_data[m] = df.groupby('bin', observed=True)['correct'].mean().values * 100

    prop_m = [m for m in prop_m if m in acc_data]
    open_m = [m for m in open_m if m in acc_data]

    col_p = {m: LINE_COLORS_PROP[i % len(LINE_COLORS_PROP)] for i, m in enumerate(prop_m)}
    col_o = {m: LINE_COLORS_OPEN[i % len(LINE_COLORS_OPEN)] for i, m in enumerate(open_m)}

    fig, (ax_p, ax_o) = plt.subplots(1, 2, figsize=(8.0, 3.6),
                                      sharey=True, gridspec_kw={'wspace': 0.08})

    def style_ax(ax, ylabel=True):
        ax.set_xlim(-0.5, len(blabels) - 0.5)
        ax.set_ylim(50, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=7)
        ax.set_yticks(np.arange(55, 105, 5))
        ax.set_yticklabels([str(int(v)) for v in np.arange(55, 105, 5)])
        ax.set_xlabel(r'$|\Delta\log S|$', labelpad=6)
        if ylabel: ax.set_ylabel('Accuracy (%)', labelpad=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--')
        ax.set_axisbelow(True)

    for m in prop_m:
        ax_p.plot(x, acc_data[m], color=col_p[m], linewidth=0.9,
                  marker='o', markersize=3.0, label=NAME_MAP.get(m, m))
    style_ax(ax_p, ylabel=True)
    ax_p.set_title('Proprietary', fontsize=9, fontweight='bold', pad=4)
    ax_p.legend(loc='lower right', ncol=2, borderpad=0.4, handlelength=1.3,
                handletextpad=0.4, labelspacing=0.2, columnspacing=0.8,
                fontsize=7, frameon=True, framealpha=0.9, edgecolor='#cccccc')

    for m in open_m:
        ax_o.plot(x, acc_data[m], color=col_o[m], linewidth=0.9,
                  marker='s', markersize=3.0, linestyle='--', label=NAME_MAP.get(m, m))
    style_ax(ax_o, ylabel=False)
    ax_o.set_title('Open-source', fontsize=9, fontweight='bold', pad=4)
    ax_o.legend(loc='lower right', ncol=2, borderpad=0.4, handlelength=1.3,
                handletextpad=0.4, labelspacing=0.2, columnspacing=0.8,
                fontsize=7, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax_o.tick_params(axis='y', length=0)

    plt.tight_layout()
    _save(fig, out_dir / 'figure_task1_delta')

# ══════════════════════════════════════════════════════════════════════════════
# Task 2
# ══════════════════════════════════════════════════════════════════════════════

def plot_task2(df, out_dir):
    print('\nTask 2:')
    models = model_cols(df, 'task2')

    # Overall bar
    results = overall_results(df, models, 'Answer', set(PROP_MODELS), set(OPEN_MODELS), 17.1)
    overall_bar(results, (10, 75), 'Task 2', out_dir / 'figure_task2_overall')

    # Delta LogS line chart
    plot_task2_delta(df, out_dir)

    # MolWt line chart
    _molwt_line_chart(df, 'task2', 'Answer', out_dir, 'figure_task2_molwt')

    # MAE bar
    plt.rcParams.update(RCPARAMS_STANDARD)
    df = df.copy()
    df['solvents_list'] = df['All_solvents'].apply(lambda s: s.split(';') if isinstance(s, str) else [])
    df['LogS_list'] = df['All_LogS'].apply(lambda s: [float(x) for x in s.split(';')] if isinstance(s, str) else [])

    mae_results = []
    for m in models:
        if m not in df.columns: continue
        errors = []
        for _, row in df.iterrows():
            pred = row[m]
            logS = row['LogS_list']
            best = row['Best_LogS']
            try:
                idx = ord(str(pred).strip()) - ord('A')
                if 0 <= idx < len(logS):
                    errors.append(abs(best - logS[idx]))
            except:
                pass
        if errors:
            cat = 'prop' if m in PROP_MODELS else 'open'
            mae_results.append((m, np.mean(errors), cat))

    mae_results.sort(key=lambda x: x[1], reverse=True)
    n = len(mae_results)
    fig, ax = plt.subplots(figsize=(4.5, max(3.5, 0.35 * n + 1.0)))
    y = np.arange(n)
    maes   = [v for _, v, _ in mae_results]
    names  = [NAME_MAP.get(m, m) for m, _, _ in mae_results]
    colors = [C_PROP if c == 'prop' else C_OPEN for _, _, c in mae_results]

    ax.barh(y, maes, height=0.7, color=colors, alpha=0.85,
            linewidth=0.4, edgecolor='white')
    for i, v in enumerate(maes):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center', ha='left', fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, max(maes) * 1.18)
    ax.set_xlabel('Mean Absolute Error (log S units)', labelpad=4)
    ax.legend(handles=[Patch(facecolor=C_OPEN, label='Open-source'),
                        Patch(facecolor=C_PROP, label='Proprietary')],
              loc='upper right', fontsize=7.5, frameon=True,
              framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linewidth=0.3, color='#ccc', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.30, right=0.97, bottom=0.08, top=0.98)
    _save(fig, out_dir / 'figure_task2_mae')



def plot_task2_delta(df, out_dir):
    """Line chart: accuracy by |ΔlogS| bins (best vs 2nd best solvent) — Task 2."""
    plt.rcParams.update(RCPARAMS_STANDARD)
    models = [m for m in model_cols(df, 'task2') if m not in TASK_EXCLUDE.get('task2', set())]
    prop_m = [m for m in models if m in PROP_MODELS]
    open_m = [m for m in models if m in OPEN_MODELS]

    df = df.copy()
    bins    = [0.0, 0.2, 0.4, 0.7, 1.2, np.inf]
    blabels = ['0.1–0.2', '0.2–0.4', '0.4–0.7', '0.7–1.2', '≥1.2']
    x = list(range(len(blabels)))

    df['bin'] = pd.cut(df['Delta_LogS_best_minus_second'], bins=bins, labels=blabels)
    bin_counts  = df['bin'].value_counts().sort_index()
    xticklabels = [f'{l}\n$n$={bin_counts[l]}' for l in blabels]

    acc_data = {}
    for m in models:
        if m not in df.columns: continue
        df['correct'] = (df[m] == df['Answer']).astype(int)
        acc_data[m]   = df.groupby('bin', observed=True)['correct'].mean().values * 100

    prop_m = [m for m in prop_m if m in acc_data]
    open_m = [m for m in open_m if m in acc_data]

    col_p  = {m: LINE_COLORS_PROP[i % len(LINE_COLORS_PROP)] for i, m in enumerate(prop_m)}
    col_o  = {m: LINE_COLORS_OPEN[i % len(LINE_COLORS_OPEN)] for i, m in enumerate(open_m)}

    fig, (ax_p, ax_o) = plt.subplots(1, 2, figsize=(8.0, 3.6),
                                      sharey=True, gridspec_kw={'wspace': 0.08})

    def style_ax(ax, ylabel=True):
        ax.set_xlim(-0.5, len(blabels) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=7)
        ax.set_xlabel(r'$|\Delta\log S|$ (best vs. 2nd best solvent)', labelpad=6)
        if ylabel: ax.set_ylabel('Accuracy (%)', labelpad=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--')
        ax.set_axisbelow(True)

    for m in prop_m:
        ax_p.plot(x, acc_data[m], color=col_p[m], linewidth=0.9,
                  marker='o', markersize=3.0, label=NAME_MAP.get(m, m))
    style_ax(ax_p, ylabel=True)
    ax_p.set_title('Proprietary', fontsize=9, fontweight='bold', pad=4)
    ax_p.legend(loc='lower right', ncol=2, borderpad=0.4, handlelength=1.3,
                handletextpad=0.4, labelspacing=0.2, columnspacing=0.8,
                fontsize=7, frameon=True, framealpha=0.9, edgecolor='#cccccc')

    for m in open_m:
        ax_o.plot(x, acc_data[m], color=col_o[m], linewidth=0.9,
                  marker='s', markersize=3.0, linestyle='--', label=NAME_MAP.get(m, m))
    style_ax(ax_o, ylabel=False)
    ax_o.set_title('Open-source', fontsize=9, fontweight='bold', pad=4)
    ax_o.legend(loc='lower right', ncol=2, borderpad=0.4, handlelength=1.3,
                handletextpad=0.4, labelspacing=0.2, columnspacing=0.8,
                fontsize=7, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax_o.tick_params(axis='y', length=0)

    ax_p.set_ylim(20, 100)

    plt.tight_layout()
    _save(fig, out_dir / 'figure_task2_delta')

# ══════════════════════════════════════════════════════════════════════════════
# Task 3
# ══════════════════════════════════════════════════════════════════════════════

def plot_task3(df, out_dir):
    print('\nTask 3:')
    models = [m for m in model_cols(df, 'task3') if m not in TASK_EXCLUDE.get('task3', set())]

    # Overall bar
    results = overall_results(df, models, 'Answer', set(PROP_MODELS), set(OPEN_MODELS), None)
    overall_bar(results, (30, 96), 'Task 3', out_dir / 'figure_task3_overall')

    # MolWt line chart
    _molwt_line_chart(df, 'task3', 'Answer', out_dir, 'figure_task3_molwt')

    # Delta LogS line chart
    plt.rcParams.update(RCPARAMS_STANDARD)
    df = df.copy()
    df['abs_delta'] = df['Delta_LogS_new_minus_base'].abs()
    bins   = [0.1, 0.25, 0.5, 0.75, 1.0, 4.0]
    blabels = ['0.1–0.25', '0.25–0.5', '0.5–0.75', '0.75–1.0', '≥1.0']
    df['bin'] = pd.cut(df['abs_delta'], bins=bins, labels=blabels)
    bin_counts = df['bin'].value_counts().sort_index()
    xticklabels = [f'{l}\n$n$={bin_counts[l]}' for l in blabels]
    x = list(range(len(blabels)))

    acc_data = {}
    for m in models:
        if m not in df.columns: continue
        df['correct'] = (df[m] == df['Answer']).astype(int)
        acc_data[m] = df.groupby('bin', observed=True)['correct'].mean().values * 100

    prop_m = [m for m in models if m in PROP_MODELS and m in acc_data]
    open_m = [m for m in models if m in OPEN_MODELS and m in acc_data]
    mk_map = {m: 'o' for m in prop_m}
    mk_map.update({m: 's' for m in open_m})

    col_p = {m: LINE_COLORS_PROP[i % len(LINE_COLORS_PROP)] for i, m in enumerate(prop_m)}
    col_o = {m: LINE_COLORS_OPEN[i % len(LINE_COLORS_OPEN)] for i, m in enumerate(open_m)}

    fig, (ax_p, ax_o) = plt.subplots(1, 2, figsize=(7.0, 3.6),
                                      sharey=True, gridspec_kw={'wspace': 0.08})

    def style_ax(ax, ylabel=True):
        ax.set_xlim(-0.5, len(blabels) - 0.5)
        ax.set_ylim(40, 102)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=6)
        ax.set_yticks(np.arange(40, 105, 10))
        ax.set_yticklabels([str(int(v)) for v in np.arange(40, 105, 10)])
        ax.set_xlabel(r'$|\Delta\log S|$ (second solvent addition)', labelpad=6)
        if ylabel: ax.set_ylabel('Accuracy (%)', labelpad=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--')
        ax.set_axisbelow(True)

    for m in prop_m:
        ax_p.plot(x, acc_data[m], color=col_p[m], linewidth=0.9,
                  marker=mk_map[m], markersize=2.8, label=NAME_MAP.get(m, m))
    style_ax(ax_p, ylabel=True)
    ax_p.set_title('Proprietary', fontsize=7, fontweight='bold', pad=4)
    ax_p.legend(loc='lower left', ncol=2, borderpad=0.4, handlelength=1.3,
                handletextpad=0.4, labelspacing=0.2, columnspacing=0.8,
                fontsize=7, frameon=True, framealpha=0.9, edgecolor='#cccccc')

    for m in open_m:
        ax_o.plot(x, acc_data[m], color=col_o[m], linewidth=0.9,
                  marker=mk_map[m], markersize=2.8, linestyle='--', label=NAME_MAP.get(m, m))
    style_ax(ax_o, ylabel=False)
    ax_o.set_title('Open-source', fontsize=7, fontweight='bold', pad=4)
    ax_o.legend(loc='lower left', ncol=2, borderpad=0.4, handlelength=1.3,
                handletextpad=0.4, labelspacing=0.2, columnspacing=0.8,
                fontsize=7, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax_o.tick_params(axis='y', length=0)

    plt.tight_layout()
    _save(fig, out_dir / 'figure_task3_delta')


# ══════════════════════════════════════════════════════════════════════════════
# Task 4
# ══════════════════════════════════════════════════════════════════════════════

def plot_task4_delta(df, out_dir):
    """Line chart: accuracy by |delta logS| bins, all models on one panel -- Task 4."""
    plt.rcParams.update(RCPARAMS_STANDARD)
    models = [m for m in model_cols(df, 'task4') if m not in TASK_EXCLUDE.get('task4', set())]

    df = df.copy()
    bins    = [0.5, 1.0, 2.0, float('inf')]
    blabels = ['0.5–1.0', '1.0–2.0', '≥2.0']
    x = list(range(len(blabels)))

    df['abs_delta'] = df['delta_logS'].abs()
    df['bin'] = pd.cut(df['abs_delta'], bins=bins, labels=blabels)
    bin_counts  = df['bin'].value_counts().sort_index()
    xticklabels = [f'{l}\n$n$={bin_counts[l]}' for l in blabels]

    acc_data = {}
    for m in models:
        if m not in df.columns: continue
        df['correct'] = (df[m] == df['Answer']).astype(int)
        vals = df.groupby('bin', observed=True)['correct'].mean().values * 100
        acc_data[m] = vals

    models = [m for m in models if m in acc_data]

    # 9 distinct colors for all models
    COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#17becf', '#bcbd22',
    ]
    col = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(4.0, 3.6))

    for m in models:
        ax.plot(x, acc_data[m], color=col[m], linewidth=0.9,
                marker='o', markersize=3.0, linestyle='-',
                label=BAR_NAME_MAP_T4.get(m, NAME_MAP.get(m, m)))

    ax.set_xlim(-0.5, len(blabels) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=7)
    ax.set_xlabel(r'$|\Delta\log S|$', labelpad=6)
    ax.set_ylabel('Accuracy (%)', labelpad=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--')
    ax.set_axisbelow(True)

    all_vals = [v for m in models for v in acc_data[m] if not np.isnan(v)]
    if all_vals:
        ax.set_ylim(max(0, min(all_vals) - 5), 100)

    ax.legend(loc='upper left', ncol=1, borderpad=0.4, handlelength=1.3,
              handletextpad=0.4, labelspacing=0.2,
              fontsize=7, frameon=True, framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    _save(fig, out_dir / 'figure_task4_delta')

def plot_task4(df, out_dir):
    print('\nTask 4:')
    models = model_cols(df, 'task4')

    # Overall bar — classify reasoning models separately
    plt.rcParams.update(RCPARAMS_STANDARD)
    REAS = {'gemini-3.1-pro', 'gemini-3.1-pro(high)'}
    results_raw = []
    for m in models:
        if m not in df.columns: continue
        acc = accuracy(df, m, 'Answer')
        if acc is None: continue
        if m in REAS:           cat = 'reas'
        elif m in PROP_MODELS:  cat = 'prop'
        else:                   cat = 'open'
        results_raw.append((m, acc, cat))
    results_raw.append(('random_baseline', 50.0, 'rand'))
    results_raw.sort(key=lambda x: x[1])

    n = len(results_raw)
    C_REAS = '#1a4a6e'
    fig, ax = plt.subplots(figsize=(4.5, max(3.5, 0.35 * n + 1.0)))
    y = np.arange(n)
    values = [a for _, a, _ in results_raw]
    names  = [BAR_NAME_MAP_T4.get(m, NAME_MAP.get(m, m)) for m, _, _ in results_raw]
    colors = []
    for _, _, cat in results_raw:
        if cat == 'open': colors.append(C_OPEN)
        elif cat == 'rand': colors.append(C_RAND)
        elif cat == 'reas': colors.append(C_REAS)
        else: colors.append(C_PROP)

    ax.barh(y, values, height=0.7, color=colors, alpha=0.85,
            linewidth=0.4, edgecolor='white')
    for i, val in enumerate(values):
        ax.text(val + 0.5, i, f'{val:.1f}%', va='center', ha='left', fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(45, 88)
    ax.set_xlabel('Accuracy (%)', labelpad=4)
    ax.legend(handles=[
        Patch(facecolor=C_RAND,  label='Random baseline'),
        Patch(facecolor=C_OPEN,  label='Open-source'),
        Patch(facecolor=C_PROP,  label='Proprietary'),
        Patch(facecolor=C_REAS,  label='Proprietary (reasoning)'),
    ], loc='lower right', fontsize=7.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linewidth=0.3, color='#ccc', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.30, right=0.97, bottom=0.08, top=0.98)
    _save(fig, out_dir / 'figure_task4_overall')

    # Delta LogS line chart
    plot_task4_delta(df, out_dir)

    # Solvent heatmap
    plt.rcParams.update(RCPARAMS_STANDARD)
    solvents = df['Solvent'].unique()
    # Fixed order: no-reasoning first, then reasoning
    NO_REAS_ORDER = ['gemini-3-flash', 'gemini-3-flash-name', 'glm-5',
                     'gpt-5.2', 'gpt-5.2-name', 'glm-5-name']
    REAS_ORDER    = ['gemini-3.1-pro', 'gemini-3.1-pro(high)']
    hm_models = [m for m in NO_REAS_ORDER + REAS_ORDER if m in df.columns]
    sep = len([m for m in NO_REAS_ORDER if m in df.columns]) - 0.5
    hm_labels = [HEATMAP_NAME_MAP.get(m, NAME_MAP.get(m, m)) for m in hm_models]

    raw = {}
    for s in solvents:
        sub = df[df['Solvent'] == s]
        row = []
        for m in hm_models:
            filled = sub[m].dropna()
            filled = filled[filled.str.strip() != '']
            if len(filled) == 0:
                row.append(np.nan)
            else:
                row.append((filled == sub.loc[filled.index, 'Answer']).mean() * 100)
        raw[s] = row

    solvent_order = sorted(raw.keys(), key=lambda s: np.nanmean(raw[s]), reverse=True)
    data = np.array([raw[s] for s in solvent_order])



    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=30, vcenter=50, vmax=90)
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks(np.arange(len(hm_models)))
    ax.set_xticklabels(hm_labels, fontsize=6.0, ha='center', linespacing=1.25)
    ax.set_yticks(np.arange(len(solvent_order)))
    ax.set_yticklabels(solvent_order, fontsize=7.0)
    ax.tick_params(length=0)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center', fontsize=6.5, color='#999999')
            else:
                tc = 'white' if (val > 72 or val < 42) else '#1a1a1a'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6.5, color=tc)

    if sep is not None:
        ax.axvline(x=sep, color='#333333', linewidth=1.0, linestyle='--')

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015, aspect=25)
    cbar.ax.tick_params(labelsize=6, width=0.6, length=2)
    cbar.set_label('Accuracy (%)', fontsize=6.5, labelpad=4)
    cbar.outline.set_linewidth(0.6)

    if sep is not None:
        mid_no = sep / (len(hm_models) - 1) * 0.72
        mid_re = (sep + len(hm_models)) / (2 * len(hm_models)) * 0.72 + 0.15
        fig.text(0.30, 0.01, 'No reasoning', ha='center', fontsize=6, color='#444444')
        fig.text(0.72, 0.01, 'Reasoning',    ha='center', fontsize=6, color='#8b0000')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, out_dir / 'figure_task4_heatmap_solvent')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SoluBench figure generation')
    parser.add_argument('--results_dir', default='results',
                        help='Directory with task CSV files (default: results/)')
    parser.add_argument('--out_dir', default='figures',
                        help='Output directory for figures (default: figures/)')
    parser.add_argument('--tasks', nargs='+', type=int, default=[1, 2, 3, 4],
                        help='Tasks to plot (default: all)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Results dir : {Path(args.results_dir).resolve()}')
    print(f'Output dir  : {out_dir.resolve()}')

    if 1 in args.tasks:
        plot_task1(load(args.results_dir, 'task1'), out_dir)
    if 2 in args.tasks:
        plot_task2(load(args.results_dir, 'task2'), out_dir)
    if 3 in args.tasks:
        plot_task3(load(args.results_dir, 'task3'), out_dir)
    if 4 in args.tasks:
        plot_task4(load(args.results_dir, 'task4'), out_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()