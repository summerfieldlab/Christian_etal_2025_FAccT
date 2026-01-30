"""
Generate reward model scores for all tokens in the vocabulary.

For each reward model listed in config/reward_models.yaml, scores every token
in the model's vocabulary against each prompt in config/prompts.yaml, and saves
results to a CSV file in data/.
"""

import yaml
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from reward_model_support import RewardModel
from reward_model_registry import *  # registers all models

SCRIPT_ROOT = Path(__file__).parent
CONFIG_DIR = SCRIPT_ROOT / 'config'
OUTPUT_DIR = SCRIPT_ROOT / 'data'

# Load configs
with open(CONFIG_DIR / 'prompts.yaml') as f:
    prompts = yaml.safe_load(f)

with open(CONFIG_DIR / 'reward_models.yaml') as f:
    models = yaml.safe_load(f)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for model_info in models:
    model_name = model_info['name']
    safe_name = model_name.replace('/', '--')
    output_path = OUTPUT_DIR / f"{safe_name}.csv"

    if output_path.exists():
        print(f"Skipping {model_name} — {output_path} already exists")
        continue

    print(f"Processing model: {model_name}")
    reward_model = RewardModel.create(model_name)
    tokenizer = reward_model.tokenizer

    # Build vocabulary (shared across prompts for this model)
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
    token_names, token_ids = zip(*vocab)
    token_decoded = tokenizer.batch_decode([[tid] for tid in token_ids],
                                           skip_special_tokens=False)

    df = pd.DataFrame({
        'token_id': token_ids,
        'token_name': token_names,
        'token_decoded': token_decoded,
    })

    # Score each prompt
    batch_size = reward_model.default_batch_size
    for prompt_name, prompt_text in prompts.items():
        print(f"  Prompt: {prompt_name}")
        all_scores = []

        for i in tqdm(range(0, len(token_ids), batch_size),
                       desc=f"  {prompt_name}"):
            batch_ids = list(token_ids[i:i + batch_size])
            scores = reward_model.get_reward_scores_from_response_token_ids(
                prompt_text, batch_ids, batch_size)
            all_scores.extend(scores)

        df[prompt_name] = all_scores

    df.to_csv(output_path, index=False)
    print(f"Saved {safe_name}.csv")

    # Memory cleanup
    del reward_model
    torch.cuda.empty_cache()

print("Done.")
