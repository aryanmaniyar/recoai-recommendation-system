"""
============================================================
  VISUALIZATIONS MODULE - Fixed Version
  Saare graphs aur charts yahan hain
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ── Custom Dark Theme Colors ──────────────────────────────
PALETTE    = ['#6C63FF', '#FF6B6B', '#48CAE4', '#06D6A0', '#FFD166', '#EF476F']
BG_COLOR   = '#0F0F1A'
CARD_COLOR = '#1A1A2E'
TEXT_COLOR = '#E0E0FF'


def set_dark_style():
    """Saare charts ke liye dark theme set karo."""
    plt.rcParams.update({
        'figure.facecolor' : BG_COLOR,
        'axes.facecolor'   : CARD_COLOR,
        'axes.edgecolor'   : '#2D2D4E',
        'axes.labelcolor'  : TEXT_COLOR,
        'xtick.color'      : TEXT_COLOR,
        'ytick.color'      : TEXT_COLOR,
        'text.color'       : TEXT_COLOR,
        'grid.color'       : '#2D2D4E',
        'grid.alpha'       : 0.5,
        'font.family'      : 'DejaVu Sans',
        'axes.titlesize'   : 13,
        'axes.titleweight' : 'bold',
    })


def plot_rating_distribution(ratings, save_path=None):
    """Rating distribution ka histogram."""
    set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Rating Analysis Dashboard',
                 fontsize=16, fontweight='bold', color=TEXT_COLOR)

    # Plot 1: Rating Distribution
    ax1 = axes[0]
    rating_counts = ratings['rating'].value_counts().sort_index()
    colors = PALETTE[:len(rating_counts)]
    bars = ax1.bar(
        rating_counts.index, rating_counts.values,
        color=colors, edgecolor='white', linewidth=0.5, width=0.35
    )
    ax1.set_title('Distribution of Ratings')
    ax1.set_xlabel('Rating Value')
    ax1.set_ylabel('Number of Ratings')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, rating_counts.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f'{val:,}', ha='center', va='bottom',
            fontsize=9, color=TEXT_COLOR
        )

    # Plot 2: Ratings per User
    ax2 = axes[1]
    ratings_per_user = ratings.groupby('userId')['rating'].count()
    ax2.hist(ratings_per_user, bins=40,
             color=PALETTE[2], edgecolor='white',
             linewidth=0.3, alpha=0.85)
    ax2.set_title('Ratings per User')
    ax2.set_xlabel('Number of Ratings by User')
    ax2.set_ylabel('Number of Users')
    mean_val = ratings_per_user.mean()
    ax2.axvline(mean_val, color=PALETTE[1], linestyle='--',
                linewidth=2, label=f'Mean = {mean_val:.1f}')
    ax2.legend(facecolor=CARD_COLOR, edgecolor='gray')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor=BG_COLOR)
    return fig


def plot_genre_analysis(movies, save_path=None):
    """Genre popularity analysis."""
    set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Genre Analysis',
                 fontsize=16, fontweight='bold', color=TEXT_COLOR)

    # Parse genres
    all_genres = []
    for genres in movies['genres'].dropna():
        all_genres.extend(genres.split('|'))
    genre_counts = pd.Series(all_genres).value_counts()

    # Plot 1: Horizontal Bar Chart
    ax1 = axes[0]
    top_genres = genre_counts.head(12)
    colors     = plt.cm.plasma(np.linspace(0.2, 0.9, len(top_genres)))
    bars       = ax1.barh(
        range(len(top_genres)), top_genres.values, color=colors
    )
    ax1.set_yticks(range(len(top_genres)))
    ax1.set_yticklabels(top_genres.index, fontsize=10)
    ax1.set_title('Top Genres by Movie Count')
    ax1.set_xlabel('Number of Movies')
    ax1.invert_yaxis()
    for bar, val in zip(bars, top_genres.values):
        ax1.text(
            val + 1, bar.get_y() + bar.get_height() / 2,
            str(val), va='center', fontsize=9, color=TEXT_COLOR
        )

    # Plot 2: Pie Chart
    ax2 = axes[1]
    top8   = genre_counts.head(8)
    colors8 = PALETTE * 2
    wedges, texts, autotexts = ax2.pie(
        top8.values,
        labels=top8.index,
        autopct='%1.1f%%',
        colors=colors8[:len(top8)],
        startangle=140,
        textprops={'color': TEXT_COLOR, 'fontsize': 9}
    )
    for at in autotexts:
        at.set_color('black')
        at.set_fontweight('bold')
    ax2.set_title('Genre Distribution')
    ax2.set_facecolor(CARD_COLOR)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor=BG_COLOR)
    return fig


def plot_similarity_heatmap(sim_matrix,
                             title='Cosine Similarity Heatmap',
                             save_path=None):
    """Movie similarity matrix heatmap."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 11))

    n          = min(20, len(sim_matrix))
    sub_matrix = sim_matrix.iloc[:n, :n]
    labels     = [
        t[:25] + '...' if len(t) > 25 else t
        for t in sub_matrix.index
    ]

    sns.heatmap(
        sub_matrix, ax=ax,
        cmap='RdYlGn', vmin=0, vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        annot=False,
        linewidths=0.3,
        linecolor='#0F0F1A',
        cbar_kws={'label': 'Cosine Similarity Score', 'shrink': 0.8}
    )
    ax.set_title(
        f'{title}\n(Higher value = More Similar)',
        fontsize=14, pad=20, color=TEXT_COLOR
    )
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor=BG_COLOR)
    return fig


def plot_recommendation_comparison(cb_recs, cf_recs,
                                    movie_title, save_path=None):
    """Content-Based vs Collaborative comparison chart."""
    set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'Recommendations for: "{movie_title}"',
        fontsize=14, fontweight='bold', color=TEXT_COLOR
    )

    datasets = [
        (cb_recs, axes[0], PALETTE[0],
         'Content-Based',  'similarity_score'),
        (cf_recs, axes[1], PALETTE[1],
         'Collaborative',  'similarity_score'),
    ]

    for df, ax, color, label, score_col in datasets:
        if df is None or df.empty or score_col not in df.columns:
            ax.text(0.5, 0.5, 'No data available',
                    transform=ax.transAxes,
                    ha='center', color=TEXT_COLOR, fontsize=12)
            ax.set_title(label)
            continue

        top = df.head(8).copy()
        short_titles = [
            t[:28] + '...' if len(t) > 28 else t
            for t in top['title']
        ]
        scores = top[score_col].fillna(0).values

        if len(scores) == 0:
            ax.text(0.5, 0.5, 'No data available',
                    transform=ax.transAxes,
                    ha='center', color=TEXT_COLOR)
            ax.set_title(label)
            continue

        bars = ax.barh(
            range(len(short_titles)), scores,
            color=color,
            edgecolor='white', linewidth=0.4, alpha=0.9
        )
        ax.set_yticks(range(len(short_titles)))
        ax.set_yticklabels(short_titles, fontsize=9)
        max_score = max(scores) if max(scores) > 0 else 1
        ax.set_xlim(0, max_score * 1.25)
        ax.set_xlabel('Similarity Score')
        ax.set_title(f'{label} Filtering', fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        for bar, val in zip(bars, scores):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center',
                fontsize=8, color=TEXT_COLOR
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor=BG_COLOR)
    return fig


def plot_evaluation_metrics(metrics, save_path=None):
    """Evaluation metrics - donut chart style."""
    set_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle('Model Evaluation Metrics',
                 fontsize=16, fontweight='bold', color=TEXT_COLOR)

    metric_configs = [
        ('RMSE',        metrics.get('RMSE', 0.85),
         'Lower is Better',  PALETTE[1]),
        ('MAE',         metrics.get('MAE', 0.65),
         'Lower is Better',  PALETTE[2]),
        ('Precision@K', metrics.get('Precision@K', 0.72),
         'Higher is Better', PALETTE[0]),
    ]

    for i, (name, value, note, color) in enumerate(metric_configs):
        ax = axes[i]

        # Safe value - must be between 0.001 and 0.999
        safe_val = float(value)
        safe_val = max(0.001, min(safe_val, 0.999))

        # Fix for values > 1 (like RMSE can be > 1)
        if safe_val > 1:
            safe_val = 0.999

        sizes        = [safe_val, 1.0 - safe_val]
        donut_colors = [color, '#2D2D4E']

        ax.pie(
            sizes,
            colors=donut_colors,
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor=BG_COLOR)
        )

        ax.text(0,  0.10, f'{value:.3f}',
                ha='center', va='center',
                fontsize=22, fontweight='bold', color=color)
        ax.text(0, -0.20, name,
                ha='center', va='center',
                fontsize=14, fontweight='bold', color=TEXT_COLOR)
        ax.text(0, -0.45, note,
                ha='center', va='center',
                fontsize=9, color='#888888', style='italic')
        ax.set_facecolor(CARD_COLOR)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor=BG_COLOR)
    return fig


def plot_popularity_vs_ratings(movies, save_path=None):
    """Average rating vs popularity scatter plot."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    movies = movies.copy()
    all_primary = []
    for g in movies['genres'].dropna():
        all_primary.append(g.split('|')[0])
    movies['primary_genre'] = all_primary

    top_genres   = movies['primary_genre'].value_counts().head(6).index
    genre_colors = {g: PALETTE[i] for i, g in enumerate(top_genres)}

    for genre in top_genres:
        subset = movies[movies['primary_genre'] == genre]
        ax.scatter(
            subset['avg_rating'],
            subset['rating_count'],
            label=genre,
            color=genre_colors[genre],
            alpha=0.7, s=40,
            edgecolors='white', linewidth=0.3
        )

    others = movies[~movies['primary_genre'].isin(top_genres)]
    if len(others) > 0:
        ax.scatter(
            others['avg_rating'],
            others['rating_count'],
            label='Other',
            color='#555555',
            alpha=0.4, s=20
        )

    ax.set_xlabel('Average Rating', fontsize=12)
    ax.set_ylabel('Number of Ratings (Popularity)', fontsize=12)
    ax.set_title('Popularity vs Average Rating by Genre',
                 fontsize=14, pad=15)
    ax.legend(
        facecolor=CARD_COLOR, edgecolor='gray',
        framealpha=0.8, fontsize=9
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor=BG_COLOR)
    return fig