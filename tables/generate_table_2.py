"""Generate Tables 2a and 2b: Top and bottom tokens for R-Gem-2B and R-Lla-3B."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis_support import latest_reward_models_by_reward_bench_rank, load_models

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / 'data'

models = latest_reward_models_by_reward_bench_rank()
models = load_models(models, DATA_DIR)

TABLE_CONFIG = {
    "R-Gem-2B": "table_2a_top_bottom_tokens_R-Gem-2B.csv",
    "R-Lla-3B": "table_2b_top_bottom_tokens_R-Lla-3B.csv",
}

N = 22

for nickname, output_file in TABLE_CONFIG.items():
    model = next(m for m in models if m["model_nickname"] == nickname)
    responses = model["greatest_responses"].sort_values("score", ascending=False).reset_index(drop=True)

    top = responses.head(N)
    bottom = responses.tail(N)

    rows = []
    for _, r in top.iterrows():
        rows.append({"Token ID": int(r["token_id"]), "Decoded": r["token_decoded"], "Score": r["score"]})
    rows.append({"Token ID": "...", "Decoded": "...", "Score": "..."})
    for _, r in bottom.iterrows():
        rows.append({"Token ID": int(r["token_id"]), "Decoded": r["token_decoded"], "Score": r["score"]})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / output_file, index=False)
    print(f"Saved {OUTPUT_DIR / output_file}")
