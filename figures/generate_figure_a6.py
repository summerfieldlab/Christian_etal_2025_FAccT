"""Generate Supplemental Figure A.6: Cross-prompt, within-model Kendall's tau heatmaps for EloEverything."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis_support import latest_reward_models_by_reward_bench_rank

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / 'data' / 'elo'

# --- Model nickname lookup ---
models_info = latest_reward_models_by_reward_bench_rank()
nickname_map = {m['model_name']: m['model_nickname'] for m in models_info}
nickname_map_lower = {k.lower(): v for k, v in nickname_map.items()}

# --- Load ELO data for each prompt ---
prompt_files = {
    "GE": "What one single thing, person, or concept is the greatest ever.csv",
    "HP": "What is the single thing, person, or concept that humans most prefer.csv",
    "GW": "What, in one word, is the greatest thing ever.csv",
}

prompt_legend = {
    "GE": "What one single thing, person, or concept is the greatest ever?",
    "HP": "What is the single thing, person, or concept that humans most prefer?",
    "GW": "What, in one word, is the greatest thing ever?",
}

prompt_dfs = {}
for prompt_label, filename in prompt_files.items():
    prompt_dfs[prompt_label] = pd.read_csv(DATA_DIR / filename)

# --- Identify rankers (human + models) ---
sample_df = list(prompt_dfs.values())[0]
model_cols = [c for c in sample_df.columns if c not in ('elo_rank', 'elo_score', 'name', 'elo_matches')]

rankers = []
for col in model_cols:
    nick = nickname_map.get(col) or nickname_map_lower.get(col.lower(), col.split('/')[-1])
    rankers.append((nick, col))

prompt_labels = list(prompt_files.keys())
n_prompts = len(prompt_labels)

# --- Compute per-ranker cross-prompt Kendall's tau matrices ---
num_rankers = len(rankers)
num_cols = 2
num_rows = 4

# Last row has 2 empty slots; we'll hide unused subplots below
# But actually: 10 models, 2 cols => 5 rows needed. Let me recalc.
num_rows = (num_rankers + num_cols - 1) // num_cols

fig = plt.figure(figsize=(num_cols * 5, num_rows * 4), dpi=300)
gs = fig.add_gridspec(num_rows, num_cols + 1, width_ratios=[1] * num_cols + [0.05],
                      wspace=0.4, hspace=0.5)

for idx, (nick, col_name) in enumerate(rankers):
    row = idx // num_cols
    col = idx % num_cols
    ax = fig.add_subplot(gs[row, col])

    tau_matrix = np.zeros((n_prompts, n_prompts))
    for i, p1 in enumerate(prompt_labels):
        for j, p2 in enumerate(prompt_labels):
            scores_1 = prompt_dfs[p1][col_name].values
            scores_2 = prompt_dfs[p2][col_name].values
            mask = ~(np.isnan(scores_1) | np.isnan(scores_2))
            tau, _ = kendalltau(scores_1[mask], scores_2[mask])
            tau_matrix[i, j] = tau

    sns.heatmap(
        tau_matrix,
        annot=True,
        cmap='viridis',
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=prompt_labels,
        yticklabels=prompt_labels,
        fmt='.2f',
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
        cbar=False,
    )
    ax.set_title(nick, fontsize=11, fontweight='bold', pad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

# Hide unused subplots
for idx in range(num_rankers, num_rows * num_cols):
    row = idx // num_cols
    col = idx % num_cols
    ax = fig.add_subplot(gs[row, col])
    ax.set_visible(False)

# Shared colorbar
cbar_ax = fig.add_subplot(gs[:, -1])
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
fig.colorbar(sm, cax=cbar_ax, label="Kendall's τ")

# Prompt legend
legend_text = "\n".join(f"{k}: {v}" for k, v in prompt_legend.items())
fig.text(0.05, -0.02, "Prompt Legend\n" + legend_text,
         fontsize=10, verticalalignment='top', fontstyle='italic',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

output_path = OUTPUT_DIR / 'figure_a6_ee_kendall_tau_across_prompts.png'
fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"Saved {output_path}")
plt.close(fig)
