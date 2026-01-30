"""Generate Appendix Tables A1–A5: Top and bottom tokens ranked by base model (Gemma) log-probabilities.

A1: Greatest prompt
A2: Best prompt
A3: Worst prompt
A4: Best + worst sum
A5: Best − worst difference
"""

import yaml
import pandas as pd
from pathlib import Path

SCRIPT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = SCRIPT_ROOT / 'config'
DATA_DIR = SCRIPT_ROOT / 'data' / 'base_model_logits'
OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_ROWS = 20


def model_name_to_filename(model_name):
    return model_name.lower().replace("/", "--")


def top_bottom_table(models, score_col, num_rows=NUM_ROWS):
    top = pd.DataFrame()
    bottom = pd.DataFrame()
    for model in models:
        sorted_data = model["data"].sort_values(score_col, ascending=False).reset_index(drop=True)
        cells = sorted_data["token_decoded"]
        top[model['name']] = cells.head(num_rows)
        bottom[model["name"]] = cells.tail(num_rows)

    ellipsis_row = pd.DataFrame([['...'] * len(top.columns)], index=['...'], columns=top.columns)
    return pd.concat([top, ellipsis_row, bottom])


def save(table, filename):
    table.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"Saved {OUTPUT_DIR / filename}")


models = yaml.safe_load(open(CONFIG_DIR / "gemma_base_models.yaml", "r"))
for model in models:
    model['data'] = pd.read_csv(DATA_DIR / f"{model_name_to_filename(model['name'])}.csv")

assert all(len(m['data']) == len(models[0]['data']) for m in models)

for model in models:
    model['data']['best_plus_worst']  = model['data']['best'] + model['data']['worst']
    model['data']['best_minus_worst'] = model['data']['best'] - model['data']['worst']

save(top_bottom_table(models, 'greatest'),        'table_a1_logprobs_greatest.csv')
save(top_bottom_table(models, 'best'),            'table_a2_logprobs_best.csv')
save(top_bottom_table(models, 'worst'),           'table_a3_logprobs_worst.csv')
save(top_bottom_table(models, 'best_plus_worst'), 'table_a4_logprobs_best_plus_worst.csv')
save(top_bottom_table(models, 'best_minus_worst'),'table_a5_logprobs_best_minus_worst.csv')
