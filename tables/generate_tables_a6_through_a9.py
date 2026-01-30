"""Generate Appendix Tables A6–A9: Top and bottom tokens ranked by reward model scores.

A6: Raw vocabulary, greatest prompt
A7: Shared vocabulary, greatest prompt
A8: Shared vocabulary, best + worst sum
A9: Shared vocabulary, best − worst difference
"""

import sys
from pathlib import Path
import pandas as pd

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

NUM_ROWS = 20

models = latest_reward_models_by_reward_bench_rank()
models = load_models(models, DATA_DIR)


def top_bottom_table(models, response_type, num_rows=NUM_ROWS):
    """Build a top-N / ... / bottom-N table with one column per model."""
    top = pd.DataFrame()
    bottom = pd.DataFrame()
    for model in models:
        df = model[response_type].sort_values("score", ascending=False).reset_index(drop=True)
        cells = df["token_decoded"]
        top[model['model_nickname']] = cells.head(num_rows).values
        bottom[model['model_nickname']] = cells.tail(num_rows).values

    ellipsis_row = pd.DataFrame([['...'] * len(top.columns)], columns=top.columns)
    return pd.concat([top, ellipsis_row, bottom], ignore_index=True)


def top_bottom_derived_table(models, score_fn, num_rows=NUM_ROWS):
    """Build top/bottom table from a derived score (e.g. best+worst)."""
    top = pd.DataFrame()
    bottom = pd.DataFrame()
    for model in models:
        df = score_fn(model).sort_values("score", ascending=False).reset_index(drop=True)
        cells = df["token_decoded"]
        top[model['model_nickname']] = cells.head(num_rows).values
        bottom[model['model_nickname']] = cells.tail(num_rows).values

    ellipsis_row = pd.DataFrame([['...'] * len(top.columns)], columns=top.columns)
    return pd.concat([top, ellipsis_row, bottom], ignore_index=True)


def best_plus_worst(model):
    best = model["best_responses"][["token_decoded", "score"]].rename(columns={"score": "best"})
    worst = model["worst_responses"][["token_decoded", "score"]].rename(columns={"score": "worst"})
    merged = best.merge(worst, on="token_decoded")
    merged["score"] = merged["best"] + merged["worst"]
    return merged


def best_minus_worst(model):
    best = model["best_responses"][["token_decoded", "score"]].rename(columns={"score": "best"})
    worst = model["worst_responses"][["token_decoded", "score"]].rename(columns={"score": "worst"})
    merged = best.merge(worst, on="token_decoded")
    merged["score"] = merged["best"] - merged["worst"]
    return merged


def save(table, filename):
    table.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"Saved {OUTPUT_DIR / filename}")


# A6: raw vocabulary, greatest
save(top_bottom_table(models, "greatest_responses"),
     "table_a6_raw_vocab_greatest.csv")

# Prune to shared vocabulary for A7–A9
shared_vocab = get_shared_vocabulary(models)
prune_all_model_responses_to_set(models, shared_vocab)

dupes = identify_duplicate_encodings(models)
if not dupes.empty:
    dupe_tokens = set(dupes['token_decoded'].unique())
    shared_vocab = shared_vocab - dupe_tokens
    prune_all_model_responses_to_set(models, shared_vocab)

# A7: shared vocabulary, greatest
save(top_bottom_table(models, "greatest_responses"),
     "table_a7_shared_vocab_greatest.csv")

# A8: shared vocabulary, best + worst
save(top_bottom_derived_table(models, best_plus_worst),
     "table_a8_shared_vocab_best_plus_worst.csv")

# A9: shared vocabulary, best − worst
save(top_bottom_derived_table(models, best_minus_worst),
     "table_a9_shared_vocab_best_minus_worst.csv")
