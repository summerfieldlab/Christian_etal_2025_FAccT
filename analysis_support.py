import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.titleweight'] = 'bold'  # Bold all titles
plt.rcParams['axes.labelweight'] = 'bold'  # Bold all axis labels


greatest_prompt = "What, in one word, is the greatest thing ever?"
best_prompt = "What, in one word, is the best thing ever?"
worst_prompt = "What, in one word, is the worst thing ever?"

def _load_config():
    """Load the full reward model config from config/reward_models.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'reward_models.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def format_name(name):
    return name.replace("/", "--")

def latest_reward_models_by_reward_bench_rank():
    config = _load_config()
    models = []
    for entry in config:
        models.append({
            "model_name": entry["name"],
            "creator": entry["name"].split("/")[0],
            "base_model": entry["base_model"],
            "base_version": str(entry["base_version"]),
            "size": str(entry["size"]),
            "reward_bench_rank": str(entry["reward_bench_rank"]),
            "model_nickname": entry["nickname"],
        })

    # Sort by reward_bench_rank
    models.sort(key=lambda m: int(m["reward_bench_rank"]))

    # Assign a color to each model
    colors = sns.color_palette()[:len(models)]
    for i, model in enumerate(models):
        model["color"] = colors[i]

    return models

def load_models(models, data_dir):
    """Load model scores from per-model CSV files in data_dir/reward_model_scores/.

    Each CSV has columns: token_id, token_name, token_decoded, greatest, best, worst.
    """
    import os

    scores_dir = os.path.join(data_dir, "reward_model_scores")

    # Sort models ascending by RewardBench rank
    for model in models:
        model["reward_bench_rank"] = int(model["reward_bench_rank"])
    models = sorted(models, key=lambda x: x['reward_bench_rank'])

    for model in models:
        model_key = format_name(model["model_name"])
        csv_path = os.path.join(scores_dir, f"{model_key}.csv")
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)

        for prompt_short in ["greatest", "best", "worst"]:
            responses = df[["token_id", "token_name", "token_decoded"]].copy()
            responses["score"] = df[prompt_short].values
            responses = responses.dropna(subset=["score"])
            model[f"{prompt_short}_responses"] = responses

        model["response_types"] = ["greatest_responses", "best_responses", "worst_responses"]

    print("All models loaded.")
    return models

def get_shared_vocabulary(models, shared_column='token_decoded'):
    # Get the shared vocabulary across all models (and all responses within each model)
    shared_vocab = set(models[0][models[0]['response_types'][0]][shared_column].unique())
    for model in models:
        for responses in model['response_types']:
            shared_vocab.intersection_update(set(model[responses][shared_column].unique()))
    return shared_vocab

def prune_responses_to_set(responses, vocab_set, vocab_column='token_decoded'):
    # Prune responses to only include tokens in the shared vocabulary
    return responses[responses[vocab_column].isin(vocab_set)]

def prune_all_model_responses_to_set(models, vocab_set, vocab_column='token_decoded'):
    # Prune all model responses to only include tokens in the shared vocabulary
    for model in models:
        for responses in model['response_types']:
            model[responses] = prune_responses_to_set(model[responses], vocab_set, vocab_column)

def identify_duplicate_encodings(models):
    all_duplicates = []
    for model in models:
        for response_type in model['response_types']:
            responses = model[response_type]
            # Find rows with duplicate token_decoded values
            mask = responses['token_decoded'].duplicated(keep=False)
            if mask.any():
                # Create a copy and add model information
                duplicates = responses[mask].copy()  # Create explicit copy
                duplicates.loc[:, 'model'] = model['model_name']
                duplicates.loc[:, 'response_type'] = response_type
                all_duplicates.append(duplicates)
    
    # Combine all duplicate information into one DataFrame
    if all_duplicates:
        return pd.concat(all_duplicates, ignore_index=True).sort_values('token_decoded')
    return pd.DataFrame()

def analyze_encoding_issues(df):
    # Look at rows where token_decoded contains '��'
    problematic = df[df['token_decoded'] == '��'].copy()
    
    print("Analyzing tokens that decode to ��:")
    for _, row in problematic.iterrows():
        print(f"\nToken ID: {row['token_id']}")
        print(f"Token name: {row['token_name']}")
        print("Token name bytes:", [hex(b) for b in row['token_name'].encode('utf-8')])
        if row['token_decoded']:
            print("Decoded bytes:", [hex(b) for b in row['token_decoded'].encode('utf-8')])

def plot_scores_x_y(models_to_plot, x_col, y_col, share_axis_limits=True, plot_reference_line=True, save_path=None):
    # Calculate subplot dimensions before creating the figure
    numrows = 2
    numcols = 5
    
    # Create figure
    fig = plt.figure(figsize=(10, 4.25), dpi=300)
    
    # Add spacing between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, model in enumerate(models_to_plot):
        df = model["scores"]
        token_ids = df["token_id"]
        # get (x, y) pairs for each token_id
        x_scores = df.set_index("token_id").loc[token_ids][x_col]
        y_scores = df.set_index("token_id").loc[token_ids][y_col]

        # Create subplot
        plt.subplot(numrows, numcols, i+1)

        sns.scatterplot(x=x_scores, y=y_scores, alpha=0.5, s=1, color=model["color"], rasterized=True) # Raster is key because these plots have 100s of ks of points
        plt.title(f"{model['model_nickname']}")
        plt.xlabel(f"{x_col.title()} score")
        plt.ylabel(f"{y_col.title()} score")

        if share_axis_limits:
            # Set x and y axes to be equal
            max_score = max(x_scores.max(), y_scores.max())
            min_score = min(x_scores.min(), y_scores.min())
            plt.xlim(min_score, max_score)
            plt.ylim(min_score, max_score)

        if plot_reference_line:
            # Plot a dotted reference line with the same color
            plt.plot([min_score, max_score], [min_score, max_score], color=model["color"], linestyle='--', alpha=0.75, linewidth=1)

    plt.tight_layout()  # Automatically adjust subplot params for better fit

    # Save the figure if a path is provided
    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)

    plt.show()

    return fig
    

def validate_statistical_value(value, name, i, j):
    """Validate statistical values, raising exceptions for invalid cases."""
    if np.isnan(value):
        raise ValueError(f"NaN {name} found for models {i} and {j}")
    if isinstance(value, (float, np.floating)):
        if name == "p-value" and value < 0:
            raise ValueError(f"Negative {name} ({value}) found for models {i} and {j}")
        if name == "p-value" and value > 1:
            raise ValueError(f"P-value greater than 1 ({value}) found for models {i} and {j}")
    return value

from scipy.stats import kendalltau
# Compute the Kendall's Tau distance between two models
def kendall_tau_distance(response1, response2, score_to_compare):
    # Merge the two dataframes
    merged = response1.merge(response2, on='token_decoded', suffixes=('_1', '_2'))

    # Compute the Kendall's Tau distance
    tau, p_value = kendalltau(merged[f'{score_to_compare}_1'], merged[f'{score_to_compare}_2'])
    return tau, p_value

# Compute the Kendall's Tau distance matrix
def kendall_tau_matrix(models, score_to_compare):
    n_models = len(models)
    kendall_tau_matrix = np.zeros((n_models, n_models))
    p_value_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            tau, p_val = kendall_tau_distance(models[i]["scores"], models[j]["scores"], score_to_compare)
            
            # Validate values - will raise exception if invalid
            tau = validate_statistical_value(tau, "correlation", i, j)
            p_val = validate_statistical_value(p_val, "p-value", i, j)
            
            kendall_tau_matrix[i, j] = tau
            p_value_matrix[i, j] = p_val

    return kendall_tau_matrix, p_value_matrix

from scipy.stats import spearmanr
# Compute the Spearman's Rank Correlation Coefficient between two models
def spearman_rank_correlation(response1, response2, score_to_compare):
    # Merge the two dataframes
    merged = response1.merge(response2, on='token_decoded', suffixes=('_1', '_2'))

    # Compute the Spearman's Rank Correlation Coefficient
    rho, p_value = spearmanr(merged[f'{score_to_compare}_1'], merged[f'{score_to_compare}_2'])
    return rho, p_value

# Compute the Spearman's Rank Correlation Coefficient matrix
def spearman_matrix(models, score_to_compare):
    n_models = len(models)
    spearman_rho_matrix = np.zeros((n_models, n_models))
    p_value_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            rho, p_val = spearman_rank_correlation(models[i]["scores"], models[j]["scores"], score_to_compare)
            
            # Validate values - will raise exception if invalid
            rho = validate_statistical_value(rho, "correlation", i, j)
            p_val = validate_statistical_value(p_val, "p-value", i, j)
            
            spearman_rho_matrix[i, j] = rho
            p_value_matrix[i, j] = p_val

    return spearman_rho_matrix, p_value_matrix


# Compute Spearman's D (Spearman's Foot Rule) (NOTE: this needs to be double-checked for accuracy)
def foot_rule_correlation(response1, response2, score_to_compare):
    # Merge the two dataframes
    merged = response1.merge(response2, on='token_decoded', suffixes=('_1', '_2'))
    
    # Get ranks for each score column
    ranks1 = merged[f'{score_to_compare}_1'].rank()
    ranks2 = merged[f'{score_to_compare}_2'].rank()
    
    # Compute foot rule distance (sum of absolute differences between ranks)
    foot_rule_distance = np.sum(np.abs(ranks1 - ranks2))
    
    # Normalize by maximum possible distance (n²/2 where n is number of items)
    n = len(merged)
    max_distance = (n * n) / 2
    
    # Convert to similarity (1 - normalized distance)
    similarity = 1 - (foot_rule_distance / max_distance)
    
    return similarity

def foot_rule_matrix(models, score_to_compare):
    n_models = len(models)
    foot_rule_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            foot_rule_matrix[i, j] = foot_rule_correlation(
                models[i]["scores"], 
                models[j]["scores"], 
                score_to_compare
            )
    
    return foot_rule_matrix

# NOTE: This needs to be double-checked for accuracy
def ndcg(response1, response2, score_to_compare, k=None):
    """
    Normalized Discounted Cumulative Gain
    """
    # Merge dataframes
    merged = response1.merge(response2, on='token_decoded', suffixes=('_true', '_pred'))
    
    # Check if we lost any rows
    if len(merged) != len(response1) or len(merged) != len(response2):
        print(f"Warning: Merge reduced rows from {len(response1)}/{len(response2)} to {len(merged)}")
    
    # Sort by predicted scores to get ranking
    sorted_df = merged.sort_values(f'{score_to_compare}_pred', ascending=False)
    if k:
        sorted_df = sorted_df.head(k)
    
    # Calculate DCG
    relevance = sorted_df[f'{score_to_compare}_true'].values
    gains = 2 ** relevance - 1
    positions = np.arange(1, len(relevance) + 1)
    discounts = np.log2(positions + 1)
    dcg = np.sum(gains / discounts)
    
    # Calculate IDCG using original relevance scores
    original_relevance = response1[score_to_compare].values
    if k:
        original_relevance = original_relevance[:k]
    ideal_relevance = np.sort(original_relevance)[::-1]
    ideal_gains = 2 ** ideal_relevance - 1
    idcg = np.sum(ideal_gains / discounts[:len(ideal_relevance)])
    
    return dcg / idcg if idcg > 0 else 0.0

def ndcg_matrix(models, score_to_compare, k=None):
    n_models = len(models)
    ndcg_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            ndcg_matrix[i, j] = ndcg(
                models[i]["scores"], 
                models[j]["scores"], 
                score_to_compare,
                k=k
            )
    return ndcg_matrix

def rbo_matrix(models, score_to_compare='score', p=0.9, threshold=1e-6):
    # RBO: Rank-Biased Overlap
    
    # Note, this will be within an error of {threshold} of the true RBO value
    n_models = len(models)
    rbo_matrix = np.zeros((n_models, n_models))

    # Pre-compute all ranks
    ranks = [model["scores"].sort_values(score_to_compare, ascending=False)['token_decoded'].values 
            for model in models]
    
    # Calculate depth where p^(d-1) falls below threshold
    max_depth = min(
        max(len(rank) for rank in ranks),
        int(np.log(threshold) / np.log(p)) + 1
    )
    ranks = [rank[:max_depth] for rank in ranks]
    
    weights = np.power(p, np.arange(max_depth)) * (1 - p)
    
    # Pre-compute all prefix sets for all ranks
    prefix_sets = []
    for rank in ranks:
        model_prefixes = []
        current_set = set()
        for item in rank:
            current_set = current_set | {item}  # Using | instead of add() for immutability
            model_prefixes.append(current_set)
        prefix_sets.append(model_prefixes)

    # Only calculate upper triangle
    for i in range(n_models):
        rank1_prefixes = prefix_sets[i]
        for j in range(i, n_models):
            rank2_prefixes = prefix_sets[j]
            depth = min(len(rank1_prefixes), len(rank2_prefixes))
            
            # Vectorized intersection calculation
            intersections = np.array([
                len(rank1_prefixes[d].intersection(rank2_prefixes[d]))
                for d in range(depth)
            ])
            
            # Calculate overlaps using broadcasting
            depths = np.arange(1, depth + 1)
            overlaps = intersections / depths
            
            # Compute final score using dot product
            score = np.dot(overlaps, weights[:depth])
            
            rbo_matrix[i, j] = score
            if i != j:  # Don't double-assign diagonal
                rbo_matrix[j, i] = score

    return rbo_matrix


import string
def create_subplots(num_rows, num_cols, figsize=(14, 10), dpi=300):
    """Create a figure and subplot axes with the specified dimensions."""
    # Create figure with extra space on the right for the colorbar
    fig = plt.figure(figsize=figsize)
    
    # Create a gridspec that ensures square subplots
    gs = fig.add_gridspec(num_rows, num_cols + 1,  # Add 1 column for colorbar
                         width_ratios=[1] * num_cols + [0.05],  # Make colorbar thinner
                         height_ratios=[1] * num_rows)
    
    return fig, gs

def heatmap_subplot(models, matrix, title, ax, cbar_ax=None, first_plot=False, show_title=True):
    """Create a heatmap on the specified subplot axis."""
    labels = [model["model_nickname"] for model in models]
    sns.heatmap(
        matrix,
        annot=True,
        cmap='viridis',
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=labels,
        yticklabels=labels if first_plot else False,  # Only show y-axis labels on first plot
        fmt='.2f',
        ax=ax,
        cbar=first_plot,  # Only show colorbar on first plot
        cbar_ax=cbar_ax if first_plot else None,
        annot_kws={"size": 12, "weight": "bold"}
    )
    if show_title:
        ax.set_title(title, pad=20, fontsize=12)  # Match MDS plot
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14, fontweight="bold")
    if first_plot:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, fontweight="bold")

    if first_plot and cbar_ax is not None:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        # Make the colorbar tick labels bold
        # cbar.ax.tick_params(labelsize=10)  # Set font size if needed
        plt.setp(cbar.ax.get_yticklabels(), weight='bold')  # Make labels bold


    

def plot_multiple_heatmaps(models, matrices, titles, num_rows, num_cols, show_legend=True, show_title=True):
    """Plot multiple heatmaps in a grid layout."""
    fig, gs = create_subplots(num_rows, num_cols, figsize=(6*num_cols + 2, 6*num_rows), dpi=300)
    
    # Create a separate axis for the colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])  # Span all rows in the last column
    
    # Create and plot each subplot
    for idx, (matrix, title) in enumerate(zip(matrices, titles)):
        row = idx // num_cols
        col = idx % num_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Only create colorbar for first plot
        first_plot = idx == 0
        heatmap_subplot(models, matrix, title, ax, cbar_ax=cbar_ax, first_plot=first_plot, show_title=show_title)
    
    # Add model name legend
    if show_legend:
        labels = list(string.ascii_uppercase[:matrices[0].shape[0]])
        legend_elements = [plt.Line2D([0], [0], marker='', color='none',
                                    label=f'{letter}: {name}')
                          for letter, name in zip(labels, [model["model_name"] for model in models])]
        
        fig.legend(handles=legend_elements,
                  loc='center right',
                  bbox_to_anchor=(1.2, 0.5))

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Adjust spacing for colorbar
    plt.subplots_adjust(right=0.85)
    
    plt.show()

    return fig, gs

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def pvalue_subplot(matrix, title, ax, first_plot=False):
    """Create a heatmap specifically for p-values."""
    labels = list(string.ascii_uppercase[:matrix.shape[0]])
    
    def sig_level(p):
        """Return significance level with guard clauses, checking most significant first."""
        if p == 0 or p < 0.001:  # Most significant
            return 3   # ***
        if p < 0.01:
            return 2   # **
        if p < 0.05:
            return 1   # *
        return 0       # ns (p >= 0.05)
    
    def sig_symbol(p):
        """Return significance symbol with guard clauses, checking most significant first."""
        if p == 0 or p < 0.001:  # Most significant
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return 'ns'    # p >= 0.05
    
    # Convert p-values to significance categories
    plot_matrix = np.vectorize(sig_level)(matrix)
    annot_matrix = np.vectorize(sig_symbol)(matrix)
    
    # Create color palette for the 4 categories (ns, *, **, ***)
    colors = ['#91BFDB', '#4575B4', '#D73027', '#A50026']  # Light blue to dark red
    cmap = ListedColormap(colors)
    
    sns.heatmap(
        plot_matrix,
        annot=annot_matrix,
        cmap=cmap,
        vmin=0,
        vmax=3,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        fmt='',
        ax=ax,
        cbar=False  # Remove colorbar
    )
    ax.set_title(title)
    
    if first_plot:
        # Create colored boxes for legend
        legend_elements = [
            Patch(facecolor=colors[0], label='ns: p ≥ 0.05'),
            Patch(facecolor=colors[1], label='*: p < 0.05'),
            Patch(facecolor=colors[2], label='**: p < 0.01'),
            Patch(facecolor=colors[3], label='***: p < 0.001')
        ]
        ax.legend(handles=legend_elements,
                 bbox_to_anchor=(1.5, 1),
                 loc='upper right')

def plot_multiple_pvalues(models, matrices, titles, num_rows, num_cols, show_legend=True):
    """Plot multiple p-value heatmaps in a grid layout."""
    fig, gs = create_subplots(num_rows, num_cols, figsize=(6*num_cols, 6*num_rows), dpi=300)
    
    # Create and plot each subplot
    for idx, (matrix, title) in enumerate(zip(matrices, titles)):
        row = idx // num_cols
        col = idx % num_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Only create legend for first plot
        first_plot = idx == 0
        pvalue_subplot(matrix, title, ax, first_plot=first_plot)
    
    # Add model name legend
    if show_legend:
        labels = list(string.ascii_uppercase[:matrices[0].shape[0]])
        legend_elements = [plt.Line2D([0], [0], marker='', color='none',
                                    label=f'{letter}: {name}')
                          for letter, name in zip(labels, [model["model_name"] for model in models])]
        
        fig.legend(handles=legend_elements,
                  loc='center right',
                  bbox_to_anchor=(1.1, 0.5))

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    plt.show()

from sklearn.manifold import MDS
def plot_multiple_mds_improved(models, similarity_matrices, titles, num_rows, num_cols, show_title=True):
    """Plot multiple MDS plots with better space management."""
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(6*num_cols, 6*num_rows), dpi=300)
    gs = fig.add_gridspec(num_rows, num_cols)
    
    # Create axes using GridSpec
    axes = []
    for i in range(num_rows):
        for j in range(num_cols):
            if i * num_cols + j < len(similarity_matrices):
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)
    
    # Set figure background to white
    fig.patch.set_facecolor('white')
    
    # Plot each subplot
    for similarity_matrix, title, ax in zip(similarity_matrices, titles, axes):
        mds_plot_subplot_improved(models, similarity_matrix, title, ax, show_title=show_title)
    
    # Adjust spacing
    plt.tight_layout(pad=1.5)  # Increase padding slightly
    
    return fig, axes

# Enhanced subplot function with better space management
def mds_plot_subplot_improved(models, similarity_matrix, title, ax, show_title=True):
    """Create an MDS plot with better space management."""
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    distance_matrix = 1 - similarity_matrix
    positions = mds.fit_transform(distance_matrix)

    # Set up grid with lower zorder
    ax.grid(True, linestyle='--', alpha=0.3, color='gray', zorder=0)
    ax.set_facecolor('#f8f9fa')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5, zorder=0)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5, zorder=0)
    
    # Plot points with higher zorder
    ax.scatter(positions[:, 0], positions[:, 1], 
              s=200,
              c=[model["color"] for model in models],
              alpha=1,
            #   edgecolor='black',
              linewidth=0,
              zorder=2)
    
    # Add labels with adjusted spacing
    for i, (x, y) in enumerate(positions):
        bbox_props = dict(
            boxstyle="round,pad=0.4",
            fc="white",
            ec="gray",
            alpha=0.8,
            zorder=10
        )
        
        ax.annotate(
            models[i]["model_nickname"],
            (x, y),
            xytext=(10, 10),  # Reduced offset
            textcoords='offset points',
            fontsize=11,
            bbox=bbox_props,
            ha='left',
            va='bottom',
            color=models[i]["color"],
            fontweight='bold'
        )
        
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Optional: if you want to remove the ticks themselves too
    ax.tick_params(axis='both', which='both', length=0)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    if show_title:
        ax.set_title(title, pad=20, fontsize=12)
    
    # Set aspect ratio while controlling boundaries
    ax.set_aspect('equal', adjustable='box')
    
    # Calculate and set balanced limits
    bounds = ax.get_xbound() + ax.get_ybound()
    max_range = max(abs(min(bounds)), abs(max(bounds)))
    padding = max_range * 0.1  # Reduced padding
    ax.set_xlim(-max_range - padding, max_range + padding)
    ax.set_ylim(-max_range - padding, max_range + padding)

def plot_multiple_mds(models, similarity_matrices, titles, num_rows, num_cols):
    """Plot multiple MDS plots with larger figure size."""
    # Much larger figure size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6))
    axes = axes.ravel()
    
    fig.patch.set_facecolor('white')
    
    # More spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    
    for similarity_matrix, title, ax in zip(similarity_matrices, titles, axes):
        mds_plot_subplot(models, similarity_matrix, title, ax)
    
    plt.tight_layout()
    plt.show()

    return fig, axes


def create_legend(models):
    """
    Create a standalone legend plot for model names
    
    Parameters:
    models: list of dicts, where each dict has 'model_name' string
    """
    # Create single subplot
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    
    # Create labels dictionary
    labels = {string.ascii_uppercase[i]: f'({string.ascii_uppercase[i]}) {model["model_name"]}'
             for i, model in enumerate(models)}
    
    # Define color mapping
    colors = sns.color_palette()[:len(models)]
    color_dict = {letter: color for letter, color in zip(labels.keys(), colors)}

    # Configure axis
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create legend handles
    handles = [plt.Line2D([0], [0], color=color_dict[letter], lw=10, label=label)
              for letter, label in labels.items()]
    ax.legend(handles=handles, loc='center', fontsize=12, title='Model Legend')
    
    plt.tight_layout()
    plt.show()


from scipy import stats
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_distributions(models, responses_to_compare='responses'):
    """
    Analyze distributions of reward model scores
    
    Parameters:
    models: list of dicts, where each dict has a 'responses' key containing a DataFrame with a 'score' column
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Reward Model Score Distributions', fontsize=16)
    
    # Create combined DataFrame for easier plotting
    combined_df = pd.DataFrame()
    labels = {}  # Dictionary to store letter-to-full-name mapping
    colors = {}
    for i, model in enumerate(models):
        combined_df[model['model_nickname']] = model[responses_to_compare]['score']
        labels[model['model_nickname']] = model['model_nickname']
        colors[model['model_nickname']] = model['color']

    # Histograms with KDE
    legend_handles = []
    for column in combined_df.columns:
        # Plot histogram and KDE without adding to legend
        hist = sns.histplot(data=combined_df, x=column, kde=True, ax=ax1, 
                          alpha=0.3, color=colors[column], label=None)
        kde = sns.kdeplot(data=combined_df[column], ax=ax1, color=colors[column], 
                   label=None, linewidth=2)
        # Create custom legend handle with thicker line
        handle = plt.Line2D([], [], color=colors[column], linewidth=4, label=column)
        legend_handles.append(handle)
    
    ax1.set_title('Overlaid Distributions')
    ax1.legend(handles=legend_handles)
    ax1.set_xlabel('Score')
    
    # Box plots
    sns.boxplot(data=combined_df, ax=ax2, palette=colors)
    ax2.set_title('Box Plots')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylabel('Score')
    
    # QQ plots
    for i, column in enumerate(combined_df.columns):
        # Get the plotting results
        (osm, osr), (slope, intercept, r) = stats.probplot(combined_df[column].dropna(), dist="norm")
        # Plot with matching colors
        ax3.scatter(osm, osr, color=colors[column], alpha=0.7, label=column)
        # Add reference line
        line = ax3.plot(osm, slope * osm + intercept, color=colors[column], linestyle='--', alpha=0.8)

    ax3.set_title('Normal Q-Q Plots')
    ax3.set_xlabel('Theoretical Quantiles')
    ax3.set_ylabel('Sample Quantiles')
    ax3.legend()
    
    # Violin plots in the fourth subplot for another view of the distributions
    sns.violinplot(data=combined_df, ax=ax4)
    ax4.set_title('Violin Plots')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylabel('Score')
    
    # Summary statistics
    stats_df = pd.DataFrame({
        'mean': combined_df.mean(),
        'std': combined_df.std(),
        'skew': combined_df.skew(),
        'kurtosis': combined_df.kurtosis(),
        'min': combined_df.min(),
        'max': combined_df.max()
    })
    
    print("\nSummary Statistics:")
    print(stats_df)
    
    # Correlation matrix
    corr_matrix = combined_df.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    plt.tight_layout()
    plt.show()
    
    return fig, stats_df, corr_matrix

def adjust_color_lightness(color, amount): # Added 2025-05-10
    """
    Adjusts the lightness of a color by multiplying the lightness by the given amount.
    
    Parameters:
    color: str, a color string in a format matplotlib understands (hex, name, etc.)
    amount: float, amount to adjust lightness by. Values > 1 lighten, values < 1 darken.
    
    Returns:
    str: Adjusted color as hex string
    """
    import matplotlib.colors as mc
    import colorsys
    
    # Convert color to RGB
    try:
        c = mc.to_rgb(color)
    except:
        # If conversion fails, return original color
        return color
    
    # Convert RGB to HLS (Hue, Lightness, Saturation)
    h, l, s = colorsys.rgb_to_hls(*c)
    
    # Adjust lightness
    l = max(0, min(1, l * amount))
    
    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    
    # Return as hex
    return mc.to_hex((r, g, b))

# For standalone violin plot
def create_violin_plot(models, responses_to_compare='responses', show_title=True):
    """
    Create a violin plot of reward model scores
    
    Parameters:
    models: list of dicts, where each dict has a 'responses' key containing a DataFrame with a 'score' column
    
    Returns:
    matplotlib.figure.Figure: The figure containing the violin plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    plt.rcParams['font.family'] = 'Libertinus Sans' # Default sans-serif font used by ACM papers
    
    # Create combined DataFrame for plotting
    combined_df = pd.DataFrame()
    base_colors = {}
    fill_colors = {}
    edge_colors = {}
    for model in models:
        combined_df[model['model_nickname']] = model[responses_to_compare]['score']
        base_colors[model['model_nickname']] = model['color']
        # Make fill slightly lighter
        fill_colors[model['model_nickname']] = adjust_color_lightness(
            model['color'], 1.1)
        # Make edge slightly darker
        edge_colors[model['model_nickname']] = adjust_color_lightness(
            model['color'], 0.9)

    
    # Create violin plot
    sns.violinplot(data=combined_df, ax=ax, palette=fill_colors, linewidth=0.8)
    
    # for violin, model_name in zip(ax.collections, combined_df.columns):
        # violin.set_edgecolor(edge_colors[model_name])
 
    # Customize plot
    if show_title:
        ax.set_title('Score Distribution by Model')
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Score')
    
    plt.tight_layout()
    
    return fig


def skewness_plot(models, responses_to_compare='responses', figsize=(10, 6)):
    """
    Create a visualization of skewness with confidence intervals for multiple models.
    
    Parameters:
    models: list of dicts, where each dict has a 'responses' key containing a DataFrame with a 'score' column
    responses_to_compare: key to access responses in model dict
    figsize: tuple of (width, height) for the figure
    
    Returns:
    fig: matplotlib figure
    ax: matplotlib axis
    skew_df: pandas DataFrame containing skewness statistics
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate skewness and confidence intervals
    skewness_data = []
    for i, model in enumerate(models):
        data = model[responses_to_compare]['score']
        model_id = model['model_nickname']
        
        # Calculate bootstrap confidence intervals for skewness
        n_bootstrap = 1000
        bootstrap_skews = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_skews.append(stats.skew(sample))
        
        ci_size = 95 # 95% confidence interval
        ci_lower = np.percentile(bootstrap_skews, (100 - ci_size) / 2)
        ci_upper = np.percentile(bootstrap_skews, ci_size + (100 - ci_size) / 2)
        point_skew = stats.skew(data)
        
        skewness_data.append({
            'Model': model_id,
            'Color': model['color'],
            'Skewness': point_skew,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper
        })
    
    # Create DataFrame
    skew_df = pd.DataFrame(skewness_data)
    
    # Plot each point with error bars
    for i, row in skew_df.iterrows():
        ax.errorbar(
            x=[i],
            y=[row['Skewness']],
            yerr=[[row['Skewness'] - row['CI_lower']], 
                  [row['CI_upper'] - row['Skewness']]],
            fmt='o',
            capsize=5,
            capthick=2,
            elinewidth=2,
            markersize=10,
            color=row['Color'],
            label=row['Model']
        )
    
    # Add reference line at zero
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Customize plot
    ax.set_xticks(range(len(skew_df)))
    ax.set_xticklabels(skew_df['Model'], rotation=0)
    ax.set_title(f'Distribution Skewness Comparison with {ci_size}% Confidence Intervals')
    ax.set_ylabel('Skewness')
    ax.grid(True, alpha=0.3)
    # Set x-axis labels at 45 degrees
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax, skew_df
