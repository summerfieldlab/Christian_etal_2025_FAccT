"""Generate Supplemental Figure A.7: Human vs. average model rank differences for EloEverything items."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis_support import latest_reward_models_by_reward_bench_rank

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / 'data' / 'elo'

# --- Model name mapping (from shared config) ---
_models = latest_reward_models_by_reward_bench_rank()
names = {m['model_name']: m['model_nickname'] for m in _models}
names["elo_score"] = "Human"

def clean(data):
    data = data.rename(columns=names)
    models = data.columns[4:]
    human_scores = data[["name", "Human"]]
    model_scores = data[models]
    raw_scores = pd.concat([human_scores, model_scores], axis=1)

    z_scores = raw_scores.copy()
    targets = raw_scores.columns[1:].to_list()
    z_scores[targets] = (z_scores[targets] - z_scores[targets].mean()) / z_scores[targets].std()

    cols = z_scores.columns.tolist()
    cols = cols[:2] + cols[-1:] + cols[2:-1]
    z_scores = z_scores[cols]
    targets = z_scores.columns[1:].to_list()

    ranks = z_scores.copy()
    ranks[targets] = ranks[targets].rank(ascending=False, method="average")
    ranks["Model (Avg)"] = ranks[targets].median(axis=1)

    return raw_scores, z_scores, ranks

# --- Prompt files (verbatim prompts as titles) ---
prompt_files = {
    "What is the single thing, person, or concept that humans most prefer?":
        "What is the single thing, person, or concept that humans most prefer.csv",
    "What one single thing, person, or concept is the greatest ever?":
        "What one single thing, person, or concept is the greatest ever.csv",
    "What, in one word, is the greatest thing ever?":
        "What, in one word, is the greatest thing ever.csv",
}

N_SHOW = 10

fig, axes = plt.subplots(1, 3, figsize=(30, 14), dpi=300)

neg_color = sns.color_palette("PRGn_r", as_cmap=True)(0)
pos_color = sns.color_palette("PRGn", as_cmap=True)(0)

for idx, (prompt_title, filename) in enumerate(prompt_files.items()):
    ax = axes[idx]
    df = pd.read_csv(DATA_DIR / filename)
    _, _, rank_df = clean(df)

    model_col = "Model (Avg)"
    temp_df = rank_df.copy()
    temp_df["diff"] = temp_df[model_col] - temp_df["Human"]

    top_discrepancies = temp_df.nlargest(N_SHOW, "diff").iloc[::-1]
    bottom_discrepancies = temp_df.nsmallest(N_SHOW, "diff")

    dummy_row = pd.DataFrame({"name": [""], "diff": [0], "Human": [None], model_col: [None]}, index=[0])
    concat_df = pd.concat([bottom_discrepancies, dummy_row, top_discrepancies]).sort_values("diff", ascending=False)

    # Wrap long names (matching Figure 5B)
    concat_df["name"] = concat_df["name"].apply(
        lambda x: x.replace("Fully Automated Luxury Communism", "Fully Automated\nLuxury Communism")
    )

    sns.barplot(data=concat_df, x="diff", y="name", palette="PRGn_r", ax=ax)

    # Separator between groups (matching Figure 5B)
    y_coords = [patch.get_y() for patch in ax.patches]
    mid_point = (min(y_coords[N_SHOW:]) + max(y_coords[:N_SHOW])) / 2
    for offset in [-0.1, 0, 0.1]:
        ax.axhline(y=mid_point + 1 + offset, color="gray", linestyle="--",
                   xmin=0.1, xmax=0.9, alpha=0.5)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f'"{prompt_title}"', fontsize=14, pad=12, style='italic')
    ax.set_xlabel("Average model rank minus human rank", fontsize=18)
    ax.set_ylabel("")

    # Font sizes matching Figure 5B (scaled for 3-panel layout)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.tick_params(axis="x", which="minor", labelsize=20)

    y_labels = ax.get_yticklabels()
    new_labels = ["Fully Automated\nLuxury Communism" if l.get_text() == "Fully Automated\nLuxury Communism"
                  else l.get_text() for l in y_labels]
    ax.set_yticklabels(new_labels, fontsize=18)

    # Annotation text boxes (matching Figure 5B style)
    ax.text(x=-2000, y=4.5, s="Humans rank better than models", ha="center", va="center",
            color=neg_color, fontsize=20, style="italic",
            bbox=dict(boxstyle="round", fc="w", ec="k", lw=1), rotation=90)
    ax.text(x=2000, y=15.5, s="Humans rank worse than models", ha="center", va="center",
            color=pos_color, fontsize=20, style="italic",
            bbox=dict(boxstyle="round", fc="w", ec="k", lw=1), rotation=90)

fig.tight_layout(w_pad=3)

output_path = OUTPUT_DIR / 'figure_a7_human_model_rank_diff.png'
fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"Saved {output_path}")
plt.close(fig)
