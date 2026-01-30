# Reward Model Interpretability via Optimal and Pessimal Tokens

Code for the paper: **"Reward Model Interpretability via Optimal and Pessimal Tokens"** by Brian Christian, Hannah Rose Kirk, Jessica A.F. Thompson, Christopher Summerfield, and Tsvetomira Dumbalska (ACM FAccT 2025). [Read the paper.](https://dl.acm.org/doi/full/10.1145/3715275.3732068)

This repository ranks tokens by their reward scores, revealing biases and interpretability insights.

## Setup

```bash
conda create -n reward-model-tokens python=3.10
conda activate reward-model-tokens
pip install -r requirements.txt
```

You will also need a [HuggingFace](https://huggingface.co/) account with access to the relevant model weights.

Figures 3, A3, and A4 are generated with R. Install [R](https://www.r-project.org/) and the required packages:

```r
install.packages(c("tidyverse", "ggpubr", "broom", "ggh4x", "ggbeeswarm"))
```

## Data

All data needed to reproduce the analysis is included in `data/`:

| File | Description |
|------|-------------|
| `data/reward_model_scores/` | Per-model reward score CSVs (one per model, columns: `token_id, token_name, token_decoded, greatest, best, worst`) |
| `data/base_model_logits/` | Base model logit scores (output of `generate_base_model_logprobs.py`) |
| `data/corpora/` | Reference corpora (`1_1_all_alpha.txt`, `AFINN-111.txt`) |
| `data/elo/` | Reward scores for all EloEverything entities across 10 models |

## Usage

### Generate reward model scores

Score every token in each model's vocabulary against each prompt:

```bash
python generate_reward_model_scores.py
```

Models and prompts are configured in `config/reward_models.yaml` and `config/prompts.yaml`. Output CSVs are saved to `data/`.

### Core modules

- `reward_model_support.py` — Base `RewardModel` class with factory pattern and device management
- `reward_model_registry.py` — Registry of 10 reward models (Llama 3 and Gemma 2 families)
- `analysis_support.py` — Correlation metrics, vocabulary operations, and visualization helpers

### Generate figures and tables

```bash
python figures/generate_figure_1.py
python figures/generate_figure_2.py
Rscript figures/generate_figure_3.R
python figures/generate_figure_4.py
python figures/generate_figure_5.py

python tables/generate_table_1.py
python tables/generate_table_2.py
python tables/generate_table_3.py
python tables/generate_tables_a1_through_a5.py
python tables/generate_tables_a6_through_a9.py
```

### Multi-token search using Greedy Coordinate Gradient (GCG)

For code relating to multi-token search using a custom implementation of nanoGCG, see https://github.com/thompsonj/nanoGCG.

### Base model log-probabilities

Generate base model log-probabilities (required by Tables A1–A5):

```bash
python generate_base_model_logprobs.py
```

Configuration: `config/gemma_base_models.yaml` and `config/prompts.yaml`.

## Citation

```bibtex
@inproceedings{christian2025reward,
  title={Reward Model Interpretability via Optimal and Pessimal Tokens},
  author={Christian, Brian and Kirk, Hannah Rose and Thompson, Jessica A.F. and Summerfield, Christopher and Dumbalska, Tsvetomira},
  booktitle={Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT)},
  year={2025},
  doi={10.1145/3715275.3732068}
}
```

## License

MIT License. See [LICENSE](LICENSE).
