"""Generate Supplemental Figure A.1: Spearman's rho and RBO correlation heatmaps for the 'greatest' prompt."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis_support import (
    latest_reward_models_by_reward_bench_rank,
    load_models,
    get_shared_vocabulary,
    prune_all_model_responses_to_set,
    identify_duplicate_encodings,
    spearman_matrix,
    rbo_matrix,
    heatmap_subplot,
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
    model["scores"] = model["greatest_responses"][["token_id", "token_name", "token_decoded"]].copy()
    model["scores"]["greatest"] = model["greatest_responses"]["score"].values

# --- Compute correlation matrices ---
score_column = "greatest"
spearman_corr, spearman_p = spearman_matrix(models, score_column)
rbo_corr = rbo_matrix(models, score_to_compare=score_column)

# --- Plot side-by-side heatmaps (matching Figure 2A cell sizing) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300,
                         gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.05})
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

heatmap_subplot(models, spearman_corr, "Spearman's ρ", axes[0],
                cbar_ax=cbar_ax, first_plot=True, show_title=True)
heatmap_subplot(models, rbo_corr, "Rank-Biased Overlap (RBO)", axes[1],
                cbar_ax=None, first_plot=False, show_title=True)

fig.tight_layout(rect=[0, 0, 0.9, 1])

output_path = OUTPUT_DIR / 'figure_a1_correlation_heatmaps_spearman_rbo_greatest.png'
fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"Saved {output_path}")
plt.close(fig)
