"""Generate Table 3: First three moments of reward distribution across shared tokens."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import skew

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis_support import (
    latest_reward_models_by_reward_bench_rank,
    load_models,
    get_shared_vocabulary,
    prune_all_model_responses_to_set,
    identify_duplicate_encodings,
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

rows = []
for model in models:
    scores = model["greatest_responses"]["score"].values
    rows.append({
        "Model": model["model_nickname"],
        "Mean": round(float(np.mean(scores)), 3),
        "Variance": round(float(np.var(scores)), 3),
        "Skewness": round(float(skew(scores)), 3),
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / 'table_3_distribution_moments.csv', index=False)
print(df.to_string(index=False))
print(f"\nSaved {OUTPUT_DIR / 'table_3_distribution_moments.csv'}")
