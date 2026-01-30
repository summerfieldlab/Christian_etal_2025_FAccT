"""Generate Supplemental Figure A.5: Kendall's tau heatmaps for EloEverything data."""
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
from analysis_support import latest_reward_models_by_reward_bench_rank, format_name

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / 'data' / 'elo'

# --- Model nickname lookup ---
models_info = latest_reward_models_by_reward_bench_rank()
nickname_map = {m['model_name']: m['model_nickname'] for m in models_info}
# Also map by case-insensitive lookup for CSV column mismatches
nickname_map_lower = {k.lower(): v for k, v in nickname_map.items()}

# --- ELO prompt files (keys are verbatim prompts used as titles) ---
prompt_files = {
    "What one single thing, person, or concept is the greatest ever?": "What one single thing, person, or concept is the greatest ever.csv",
    "What is the single thing, person, or concept that humans most prefer?": "What is the single thing, person, or concept that humans most prefer.csv",
    "What, in one word, is the greatest thing ever?": "What, in one word, is the greatest thing ever.csv",
}

# --- Compute Kendall's tau matrices ---
matrices = []
labels_list = []

for prompt_name, filename in prompt_files.items():
    df = pd.read_csv(DATA_DIR / filename)

    # Build ranker columns: human ELO + reward models
    model_cols = [c for c in df.columns if c not in ('elo_rank', 'elo_score', 'name', 'elo_matches')]

    # Create a DataFrame of rankings
    rankings = pd.DataFrame()
    rankings['Human (ELO)'] = df['elo_score']
    for col in model_cols:
        nick = nickname_map.get(col) or nickname_map_lower.get(col.lower(), col.split('/')[-1])
        rankings[nick] = df[col]

    # Drop rows with NaN
    rankings = rankings.dropna()

    labels = list(rankings.columns)
    n = len(labels)
    tau_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            tau, _ = kendalltau(rankings.iloc[:, i], rankings.iloc[:, j])
            tau_matrix[i, j] = tau

    matrices.append((prompt_name, tau_matrix, labels))

# --- Plot: 3 heatmaps stacked vertically with colorbar ---
fig, axs = plt.subplots(3, 1, figsize=(10, 22), dpi=300)

heatmaps = []
for idx, (prompt_name, matrix, labels) in enumerate(matrices):
    hm = sns.heatmap(
        matrix,
        annot=True,
        cmap='viridis',
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        fmt='.2f',
        ax=axs[idx],
        annot_kws={"size": 8, "weight": "bold"},
        cbar=False,
    )
    heatmaps.append(hm)

    axs[idx].set_title(f'"{prompt_name}"', fontsize=11, pad=10, style='italic')
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=45, ha='right', fontsize=10, fontweight='bold')
    axs[idx].set_yticklabels(axs[idx].get_yticklabels(), fontsize=10, fontweight='bold')

cbar_ax = fig.add_axes([0.92, 0.35, 0.05, 0.3])
cbar = fig.colorbar(heatmaps[0].collections[0], cax=cbar_ax, orientation="vertical")
cbar.set_label("Kendall's τ", fontsize=20)
cbar.ax.tick_params(labelsize=18)

plt.subplots_adjust(left=0.1, right=0.8, top=0.95, bottom=0.05, hspace=0.4)

output_path = OUTPUT_DIR / 'figure_a5_ee_kendall_tau_heatmap.png'
fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"Saved {output_path}")
plt.close(fig)
