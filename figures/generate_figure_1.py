"""Generate Figure 1: Violin plot of exhaustive score distributions to the 'greatest thing' prompt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis_support import (
    latest_reward_models_by_reward_bench_rank,
    load_models,
    get_shared_vocabulary,
    prune_all_model_responses_to_set,
    identify_duplicate_encodings,
    create_violin_plot,
)

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / 'data'

models = latest_reward_models_by_reward_bench_rank()
models = load_models(models, DATA_DIR)

shared_vocab = get_shared_vocabulary(models)
prune_all_model_responses_to_set(models, shared_vocab)

dupes = identify_duplicate_encodings(models)
if not dupes.empty:
    dupe_tokens = set(dupes['token_decoded'].unique())
    shared_vocab = shared_vocab - dupe_tokens
    prune_all_model_responses_to_set(models, shared_vocab)

fig = create_violin_plot(models, responses_to_compare='greatest_responses', show_title=False)
fig.savefig(OUTPUT_DIR / 'figure_1_violin_greatest.png', format='png', dpi=300, bbox_inches='tight')
print(f"Saved {OUTPUT_DIR / 'figure_1_violin_greatest.png'}")
