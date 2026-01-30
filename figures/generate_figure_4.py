"""Generate Figure 4: Best vs Worst scatter plots for all reward models."""
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
    plot_scores_x_y,
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

# --- Figure 4: Best vs Worst ---
fig = plot_scores_x_y(models, "best", "worst")
fig.savefig(OUTPUT_DIR / 'figure_4_best_worst.png', format='png', dpi=300, bbox_inches='tight')
print(f"Saved {OUTPUT_DIR / 'figure_4_best_worst.png'}")
plt.close(fig)
