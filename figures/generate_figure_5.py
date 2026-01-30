"""Generate Figure 5: EloEverything analysis.
Panel A is a UI screenshot (not generated here).
Panel B: Bar chart of max human-model rank differences.
Panel C: Rank trajectory plot across human and model rankings.
"""
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

import yaml

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / 'data' / 'elo'

# --- Model name mapping (from config) ---
with open(Path(__file__).parent.parent / 'config' / 'reward_models.yaml') as _f:
    _config = yaml.safe_load(_f)
names = {entry['name']: entry['nickname'] for entry in _config}
names["elo_score"] = "Human"

fixed_model_order = [
    "Human", "N-Gem-27B", "S-Gem-27B-v0.2", "S-Gem-27B",
    "S-Lla-8B-v0.2", "N-Lla-8B", "L-Lla-8B",
    "R-Lla-8B", "R-Lla-3B", "F-Lla-8B-v0.1", "R-Gem-2B",
]

RB_ranks = [np.nan, 2, 3, 5, 10, 11, 12, 17, 19, 20, 31]


def clean(data):
    data = data.rename(columns=names)
    models = data.columns[4:]
    human_scores = data[["name", "Human"]]
    model_scores = data[models]
    raw_scores = pd.concat([human_scores, model_scores], axis=1)

    z_scores = raw_scores.copy()
    targets = raw_scores.columns[1:].to_list()
    z_scores[targets] = (z_scores[targets] - z_scores[targets].mean()) / z_scores[targets].std()

    # Reorder columns
    cols = z_scores.columns.tolist()
    cols = cols[:2] + cols[-1:] + cols[2:-1]
    z_scores = z_scores[cols]
    targets = z_scores.columns[1:].to_list()

    ranks = z_scores.copy()
    ranks[targets] = ranks[targets].rank(ascending=False, method="average")
    ranks["Model (Avg)"] = ranks[targets].median(axis=1)

    return raw_scores, z_scores, ranks


# --- Load data ---
dfs = {}
for f in sorted(glob.glob(str(DATA_DIR / "*.csv"))):
    df = pd.read_csv(f)
    prompt = Path(f).stem
    dfs[prompt] = df
    print(f'Loaded "{prompt}" with {len(df)} rows')

clean_dfs = {}
for prompt, df in dfs.items():
    raw_scores, z_scores, ranks = clean(df)
    clean_dfs[prompt] = {"raw": raw_scores, "z": z_scores, "ranks": ranks}

# --- Prompt for the main figure ---
prompt_key = "What one single thing, person, or concept is the greatest ever"
rank_df = clean_dfs[prompt_key]["ranks"]

# Get top results per model (for panel C)
targets = rank_df.columns[1:-1]  # exclude Model (Avg)
top_results = {}
for target in targets:
    sorted_df = rank_df.sort_values(target, ascending=True)
    top_results[target] = sorted_df["name"].iloc[0]

# =====================================================================
# Panel B: Human-model rank difference bar chart
# =====================================================================
model_col = "Model (Avg)"
temp_df = rank_df.copy()
temp_df["diff"] = temp_df[model_col] - temp_df["Human"]

top_discrepancies = temp_df.nlargest(10, "diff").iloc[::-1]
bottom_discrepancies = temp_df.nsmallest(10, "diff")

dummy_row = pd.DataFrame({"name": [""], "diff": [0], "Human": [None], model_col: [None]}, index=[0])
concat_df = pd.concat([bottom_discrepancies, dummy_row, top_discrepancies]).sort_values("diff", ascending=False)

fig_b, ax = plt.subplots(figsize=(10, 12))

sns.barplot(data=concat_df, x="diff", y="name", palette="PRGn_r", ax=ax)

y_coords = [patch.get_y() for patch in ax.patches]
mid_point = (min(y_coords[10:]) + max(y_coords[:10])) / 2
for offset in [-0.1, 0, 0.1]:
    ax.axhline(y=mid_point + 1 + offset, color="gray", linestyle="--", xmin=0.1, xmax=0.9, alpha=0.5)

ax.axvline(0, color="black", linewidth=1)

neg_color = sns.color_palette("PRGn_r", as_cmap=True)(0)
pos_color = sns.color_palette("PRGn", as_cmap=True)(0)

ax.text(x=2000, y=19.75, s="Humans rank worse\nthan models", ha="center",
        color=pos_color, fontsize=30, style="italic",
        bbox=dict(boxstyle="round", fc="w", ec="k", lw=1), rotation=90)
ax.text(x=-2000, y=9, s="Humans rank better\nthan models", ha="center",
        color=neg_color, fontsize=30, style="italic",
        bbox=dict(boxstyle="round", fc="w", ec="k", lw=1), rotation=90)

ax.set_xlabel("Average model rank minus human rank", fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="x", which="minor", labelsize=30)
ax.set_ylabel("")

y_labels = ax.get_yticklabels()
new_y_labels = ["Fully Automated\nLuxury Communism" if l.get_text() == "Fully Automated Luxury Communism" else l.get_text() for l in y_labels]
ax.set_yticklabels(new_y_labels, fontsize=25)

plt.tight_layout()
fig_b.savefig(OUTPUT_DIR / 'figure_5b_rank_diff.png', format='png', dpi=300, bbox_inches='tight')
print(f"Saved figure_5b_rank_diff.png")
plt.close(fig_b)

# =====================================================================
# Panel C: Rank trajectory plot
# =====================================================================
correlations = []
for model in fixed_model_order[1:]:
    correlation, _ = spearmanr(rank_df["Human"], rank_df[model])
    correlations.append((model, correlation))

all_columns = fixed_model_order

top_100_df = rank_df.nsmallest(100, "Human")
bottom_100_df = rank_df.nlargest(100, "Human")

highlight_top_df = rank_df.nsmallest(5, "Human")
highlight_bottom_df = rank_df.nlargest(5, "Human").iloc[::-1]

top_df = rank_df[rank_df["name"].isin(top_results.values())]
top_df = top_df[~top_df["name"].isin(highlight_top_df["name"])]

columns_to_keep = ["name"] + fixed_model_order
top_100_df = top_100_df[columns_to_keep]
bottom_100_df = bottom_100_df[columns_to_keep]
highlight_top_df = highlight_top_df[columns_to_keep]
highlight_bottom_df = highlight_bottom_df[columns_to_keep]

fig_c = plt.figure(figsize=(27, 18))
gs = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.15)
ax_top = fig_c.add_subplot(gs[0])
ax_bottom = fig_c.add_subplot(gs[1])

x_positions = np.arange(len(all_columns))
ZOOM_RANGE = (0, 1000)

# Background lines
for _, row in top_100_df.iterrows():
    y_ranks = row[all_columns].values.astype(float)
    ax_bottom.plot(x_positions, y_ranks, color="blue", alpha=0.06, linewidth=2)
for _, row in bottom_100_df.iterrows():
    y_ranks = row[all_columns].values.astype(float)
    ax_bottom.plot(x_positions, y_ranks, color="red", alpha=0.06, linewidth=2)

# Color palettes
blue_palette = sns.color_palette("Reds_d", n_colors=7)[2:][::-1]
red_palette = sns.color_palette("Blues_d", n_colors=7)[2:][::-1]
colors = sns.color_palette("Set2")
bright_palette = [colors[0]] + colors[2:5] + ["goldenrod"]
markers = ["o", "s", "*", "d", "^"]
markersizes = [15, 15, 20, 15, 15]


def plot_dual_line(row, color, marker, base_alpha, linewidth, marker_size, label):
    y_ranks = row[all_columns].values.astype(float)
    line = ax_bottom.plot(x_positions, y_ranks, color=color, alpha=base_alpha,
                          linewidth=linewidth, marker=marker, markersize=marker_size, label=label)
    ls = "--" if np.max(y_ranks) > ZOOM_RANGE[1] else "-"
    ax_top.plot(x_positions, y_ranks, color=color, alpha=base_alpha,
                linewidth=linewidth, linestyle=ls, marker=marker, markersize=marker_size)
    return line[0]


handles_bottom, handles_top, handles_bright = [], [], []

for i, (idx, row) in enumerate(top_df.iterrows()):
    line = plot_dual_line(row, bright_palette[i], markers[i], 0.8, 5, markersizes[i], row["name"])
    handles_bright.append(line)

highlight_bottom_df = highlight_bottom_df.iloc[::-1]
for i, (idx, row) in enumerate(highlight_bottom_df.iterrows()):
    line = plot_dual_line(row, blue_palette[i], markers[i], 0.8, 5, markersizes[i], row["name"])
    handles_bottom.append(line)

for i, (idx, row) in enumerate(highlight_top_df.iterrows()):
    line = plot_dual_line(row, red_palette[i], markers[i], 0.8, 5, markersizes[i], row["name"])
    handles_top.append(line)

# Axes setup
for ax in [ax_top, ax_bottom]:
    ax.set_xticks(x_positions)
    ax.invert_yaxis()

ax_top.set_xticklabels([])
x_labels = [fixed_model_order[0]]
for i, (model, corr) in enumerate(correlations):
    rb_rank = RB_ranks[i + 1]
    x_labels.append(f"{model}\nRB #={int(rb_rank)}\nρ={corr:.2f}")
ax_bottom.set_xticklabels(x_labels, rotation=90, fontsize=25)

ax_top.set_yscale("log")
ax_top.set_ylim(ZOOM_RANGE[1], 0)
ax_bottom.set_ylim(len(rank_df) + 200, -200)
ax_top.margins(y=0.3)

@plt.FuncFormatter
def rank_formatter(x, p):
    return "1" if x == 0 else str(int(x))

ax_bottom.yaxis.set_major_formatter(rank_formatter)

zoom_rect = Rectangle((-0.25, 0), len(all_columns) - 0.5, 1000,
                       fill=False, color="black", linestyle="--", linewidth=2)
ax_bottom.add_patch(zoom_rect)

for spine in ax_top.spines.values():
    spine.set_linestyle("--")
    spine.set_linewidth(2)

ax_bottom.set_ylabel("All Ranks (#1 = best, #7530 = worst)", fontsize=32)
ax_top.set_ylabel("Ranks 1-1000 (log scale)", fontsize=32)
ax_bottom.tick_params(axis="y", labelsize=30)
ax_top.tick_params(axis="y", labelsize=30)
ax_bottom.tick_params(axis="x", labelsize=30)

ax_bottom.yaxis.set_major_locator(plt.MultipleLocator(1000))
ax_bottom.grid(which="major", color="gray", linestyle="--", linewidth=2, alpha=0.2)
ax_top.grid(which="major", color="gray", linestyle="--", linewidth=2, alpha=0.2)

# Legend
legend_props = dict(borderaxespad=0.0, ncol=1, frameon=False, fancybox=False, fontsize=20, loc="center")
legend_ax = fig_c.add_axes([0.1, 0.85, 0.8, 0.1])
legend_ax.axis("off")

bottom_labels = [f"{row['name']} (#{int(row['Human'])})" for _, row in highlight_bottom_df.iterrows()]
top_labels = [f"{row['name']} (#{int(row['Human'])})" for _, row in highlight_top_df.iterrows()]
bright_labels = []
for _, row in top_df.iterrows():
    if "Gödel, Escher, Bach" in row["name"]:
        bright_labels.append(f"$\\it{{{row['name']}}}$ (#{int(row['Human'])})")
    else:
        bright_labels.append(f"{row['name']} (#{int(row['Human'])})")

legend1 = legend_ax.legend(handles_bottom[::-1], bottom_labels[::-1],
                           bbox_to_anchor=(0.5, 0.1), title="Bottom 5 ranked items by humans", **legend_props)
legend2 = legend_ax.legend(handles_top, top_labels,
                           bbox_to_anchor=(0.2, 0.1), title="Top 5 ranked items by humans", **legend_props)
legend3 = legend_ax.legend(handles_bright, bright_labels,
                           bbox_to_anchor=(0.8, 0.1), title="Best ranked items by models", **legend_props)

legend_ax.add_artist(legend1)
legend_ax.add_artist(legend2)
legend_ax.add_artist(legend3)

for leg in [legend1, legend2, legend3]:
    leg.get_title().set_fontsize(22)

plt.subplots_adjust(top=0.75)
plt.tight_layout()

fig_c.savefig(OUTPUT_DIR / 'figure_5c_rank_trajectory.png', format='png', dpi=300, bbox_inches='tight')
print(f"Saved figure_5c_rank_trajectory.png")
plt.close(fig_c)
