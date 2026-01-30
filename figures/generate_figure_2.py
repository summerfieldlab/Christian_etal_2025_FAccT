"""Generate Figure 2: (A) Kendall's tau heatmap, (B) MDS plot, (C) RSA dissimilarity matrices."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis_support import (
    latest_reward_models_by_reward_bench_rank,
    load_models,
    get_shared_vocabulary,
    prune_all_model_responses_to_set,
    identify_duplicate_encodings,
    kendall_tau_matrix,
    heatmap_subplot,
    mds_plot_subplot_improved,
)

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / 'data'

# --- Load and prepare data ---
models = latest_reward_models_by_reward_bench_rank()
models = load_models(models, DATA_DIR)

shared_vocab = get_shared_vocabulary(models)
prune_all_model_responses_to_set(models, shared_vocab)

dupes = identify_duplicate_encodings(models)
if not dupes.empty:
    dupe_tokens = set(dupes['token_decoded'].unique())
    shared_vocab = shared_vocab - dupe_tokens
    prune_all_model_responses_to_set(models, shared_vocab)

# Build combined scores DataFrame per model
for model in models:
    for rt in model['response_types']:
        model[rt] = model[rt].sort_values('token_id').reset_index(drop=True)
    assert (model["greatest_responses"]["token_id"] == model["best_responses"]["token_id"]).all()
    assert (model["best_responses"]["token_id"] == model["worst_responses"]["token_id"]).all()
    model["scores"] = model["greatest_responses"][["token_id", "token_name", "token_decoded"]].copy()
    model["scores"]["greatest"] = model["greatest_responses"]["score"].values
    model["scores"]["best"] = model["best_responses"]["score"].values
    model["scores"]["worst"] = model["worst_responses"]["score"].values

# --- Compute Kendall's tau ---
score_column = "greatest"
kendall_corr, kendall_p = kendall_tau_matrix(models, score_column)

# --- Figure 2A: Heatmap ---
fig_a, gs_a = plt.subplots(1, 1, figsize=(8, 7), dpi=300)
cbar_ax = fig_a.add_axes([0.92, 0.15, 0.02, 0.7])
heatmap_subplot(models, kendall_corr, "Kendall's τ", fig_a.axes[0],
                cbar_ax=cbar_ax, first_plot=True, show_title=False)
fig_a.tight_layout(rect=[0, 0, 0.9, 1])
fig_a.savefig(OUTPUT_DIR / 'figure_2a_heatmap.png', format='png', dpi=300, bbox_inches='tight')
print(f"Saved {OUTPUT_DIR / 'figure_2a_heatmap.png'}")
plt.close(fig_a)

# --- Figure 2B: MDS ---
fig_b = plt.figure(figsize=(6, 6), dpi=300)
ax_b = fig_b.add_subplot(111)
mds_plot_subplot_improved(models, kendall_corr, "MDS (Kendall's τ)", ax_b, show_title=False)
fig_b.tight_layout()
fig_b.savefig(OUTPUT_DIR / 'figure_2b_mds.png', format='png', dpi=300, bbox_inches='tight')
print(f"Saved {OUTPUT_DIR / 'figure_2b_mds.png'}")
plt.close(fig_b)

# --- Figure 2C: RSA theoretical dissimilarity matrices ---
n = len(models)
creators = [m['creator'] for m in models]
base_models_list = [m['base_model'] for m in models]
parameters = np.array([int(m['size'].replace('B', '')) for m in models])
ranks = np.array([int(m['reward_bench_rank']) for m in models])

def build_indicator_matrix(values, compare_func):
    return np.array([[compare_func(values[i], values[j]) for j in range(n)] for i in range(n)])

model_base = build_indicator_matrix(base_models_list, lambda a, b: 1 if a == b else 0)
model_creator = build_indicator_matrix(creators, lambda a, b: 1 if a == b else 0)
model_params = build_indicator_matrix(parameters, lambda a, b: 1 if a == b else 1 / (1 + abs(a - b)))
model_ranks = build_indicator_matrix(ranks, lambda a, b: 1 if a == b else 1 / (1 + abs(a - b)))

fig_c, axs = plt.subplots(4, 1, figsize=(3, 10.5), dpi=300)
cmap = "viridis"

for ax, matrix in zip(axs, [model_base, model_creator, model_params, model_ranks]):
    sns.heatmap(matrix, ax=ax, cmap=cmap, cbar=False, xticklabels=[], yticklabels=[], vmin=0, vmax=1)

plt.tight_layout()
fig_c.savefig(OUTPUT_DIR / 'figure_2c_rsa.png', format='png', dpi=300, bbox_inches='tight')
print(f"Saved {OUTPUT_DIR / 'figure_2c_rsa.png'}")
plt.close(fig_c)
