"""Generate Table 1: Open-source reward models studied."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis_support import latest_reward_models_by_reward_bench_rank

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

models = latest_reward_models_by_reward_bench_rank()

rows = []
for m in models:
    base_model = f"{m['base_model']} {m['base_version']}"
    params = int(m['size'].replace('B', ''))
    model_name = m['model_name'].split('/')[-1]
    rows.append({
        "RewardBench Rank": int(m['reward_bench_rank']),
        "Model ID": m['model_nickname'],
        "Developer": m['creator'],
        "Model Name": model_name,
        "Base Model": base_model,
        "Parameters (B)": params,
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / 'table_1_reward_models.csv', index=False)
print(f"Saved {OUTPUT_DIR / 'table_1_reward_models.csv'}")
