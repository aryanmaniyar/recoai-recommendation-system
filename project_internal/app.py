"""
============================================================
  STREAMLIT WEB APP - Product Recommendation System
  Run with: streamlit run app.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.recommender import (
    load_movielens_data, preprocess_data,
    ContentBasedRecommender, CollaborativeFilteringRecommender,
    HybridRecommender, evaluate_model
)
from utils.visualizations import (
    plot_rating_distribution, plot_genre_analysis,
    plot_similarity_heatmap, plot_recommendation_comparison,
    plot_evaluation_metrics, plot_popularity_vs_ratings
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title = "🎬 RecoAI – Movie Recommendation System",
    page_icon  = "🎬",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
  :root {
    --bg-primary  : #0D0D1F;
    --bg-card     : #141428;
    --bg-card2    : #1A1A35;
    --accent-blue : #6C63FF;
    --accent-pink : #FF6B9D;
    --accent-cyan : #48CAE4;
    --accent-green: #06D6A0;
    --text-primary: #E8E8FF;
    --text-muted  : #9090B0;
    --border      : #2A2A4A;
    --radius      : 12px;
  }

  .stApp { background: var(--bg-primary) !important; }
  html, body, [class*="css"] { color: var(--text-primary) !important; }

  .hero-header {
    background: linear-gradient(135deg, #1A0533 0%, #0D1B4D 50%, #0A2B2B 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #A78BFA, #60A5FA, #34D399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
  }
  .hero-subtitle { font-size: 1rem; color: var(--text-muted); margin: 0; }

  .metric-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 140px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1rem; text-align: center;
    transition: transform 0.2s, border-color 0.2s;
  }
  .metric-card:hover { transform: translateY(-3px); border-color: var(--accent-blue); }
  .metric-card .value {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1;
  }
  .metric-card .label {
    font-size: 0.75rem; color: var(--text-muted);
    margin-top: 4px; text-transform: uppercase; letter-spacing: 1px;
  }

  .rec-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex; align-items: center; gap: 1rem;
    transition: all 0.2s;
  }
  .rec-card:hover { border-color: var(--accent-blue); transform: translateX(4px); }
  .rec-rank { font-size: 1.5rem; font-weight: 800; color: var(--accent-blue); min-width: 36px; }
  .rec-title { font-size: 1rem; font-weight: 600; color: var(--text-primary); }
  .rec-meta  { font-size: 0.8rem; color: var(--text-muted); }
  .score-badge {
    margin-left: auto; padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 700;
    background: linear-gradient(135deg,
      rgba(108,99,255,0.2), rgba(72,202,228,0.2));
    border: 1px solid rgba(108,99,255,0.4);
    color: var(--accent-cyan); white-space: nowrap;
  }

  .section-header {
    font-size: 1.4rem; font-weight: 700;
    color: var(--text-primary);
    border-left: 4px solid var(--accent-blue);
    padding-left: 1rem; margin: 2rem 0 1rem 0;
  }

  .algo-pill {
    display: inline-block; padding: 4px 16px;
    border-radius: 20px; font-size: 0.78rem; font-weight: 600;
    margin-right: 6px;
  }
  .pill-cb { background: rgba(108,99,255,0.15);
             border: 1px solid rgba(108,99,255,0.4); color: #A78BFA; }
  .pill-cf { background: rgba(255,107,157,0.15);
             border: 1px solid rgba(255,107,157,0.4); color: #FF6B9D; }
  .pill-hybrid { background: rgba(6,214,160,0.15);
                 border: 1px solid rgba(6,214,160,0.4); color: #06D6A0; }

  .stSelectbox > div > div {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
  }
  .stTabs [data-baseweb="tab"]     { color: var(--text-muted) !important; }
  .stTabs [aria-selected="true"]   { color: var(--accent-blue) !important; }

  .stButton > button {
    background: linear-gradient(135deg,
      var(--accent-blue), #8B5CF6) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(108,99,255,0.4) !important;
  }

  div[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA + MODEL LOADING (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="🤖 Training recommendation models…")
def load_and_train():
    movies, ratings                  = load_movielens_data()
    movies_f, ratings_f, rm, sm      = preprocess_data(movies, ratings)
    cb      = ContentBasedRecommender().fit(movies_f)
    cf      = CollaborativeFilteringRecommender(method='item-item').fit(rm, movies_f)
    hybrid  = HybridRecommender(alpha=0.5).fit(movies_f, rm)
    metrics = evaluate_model(rm)
    return movies_f, ratings_f, rm, cb, cf, hybrid, metrics


movies, ratings, rating_matrix, cb_model, cf_model, hybrid_model, metrics = load_and_train()
all_titles = sorted(movies['title'].unique().tolist())


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    n_recommendations = st.slider(
        "📋 Number of Recommendations", 5, 20, 10
    )

    st.markdown("---")
    st.markdown("### 🧠 Algorithm Info")
    st.markdown("""
    <div style='font-size:0.85rem; color:#9090B0; line-height:1.8'>
    <b style='color:#A78BFA'>Content-Based</b><br>
    Genres se similar movies dhundta hai.<br><br>
    <b style='color:#FF6B9D'>Collaborative</b><br>
    User ratings se similar movies dhundta hai.<br><br>
    <b style='color:#06D6A0'>Hybrid</b><br>
    Dono ka combination — best results!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown(f"""
    <div style='font-size:0.85rem; color:#9090B0; line-height:1.8'>
    🎬 Movies : <b style='color:#E8E8FF'>{len(movies):,}</b><br>
    👥 Users  : <b style='color:#E8E8FF'>{len(rating_matrix):,}</b><br>
    ⭐ Ratings: <b style='color:#E8E8FF'>{len(ratings):,}</b><br>
    📐 Matrix : <b style='color:#E8E8FF'>
      {rating_matrix.shape[0]}×{rating_matrix.shape[1]}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#555; text-align:center'>
    MSc Data Science Project<br>
    Product Recommendation System<br>
    <b>Python • Scikit-Learn • Streamlit</b>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────

st.markdown("""
<div class='hero-header'>
  <div class='hero-title'>🎬 RecoAI</div>
  <p style='font-size:1.2rem; color:#C0C0E0; margin:0.5rem 0'>
    Intelligent Product Recommendation System
  </p>
  <p class='hero-subtitle'>
    Content-Based &nbsp;•&nbsp; Collaborative Filtering
    &nbsp;•&nbsp; Hybrid Model &nbsp;•&nbsp; Real-time Recommendations
  </p>
</div>
""", unsafe_allow_html=True)

# ── Quick Stats ───────────────────────────────
sparsity = (rating_matrix == 0).sum().sum() / rating_matrix.size * 100
avg_rat  = ratings['rating'].mean()

st.markdown(f"""
<div class='metric-row'>
  <div class='metric-card'>
    <div class='value'>{len(movies):,}</div>
    <div class='label'>🎬 Movies</div>
  </div>
  <div class='metric-card'>
    <div class='value'>{len(rating_matrix):,}</div>
    <div class='label'>👥 Users</div>
  </div>
  <div class='metric-card'>
    <div class='value'>{len(ratings):,}</div>
    <div class='label'>⭐ Ratings</div>
  </div>
  <div class='metric-card'>
    <div class='value'>{avg_rat:.2f}</div>
    <div class='label'>📈 Avg Rating</div>
  </div>
  <div class='metric-card'>
    <div class='value'>{sparsity:.0f}%</div>
    <div class='label'>🕳️ Sparsity</div>
  </div>
  <div class='metric-card'>
    <div class='value'>{metrics.get('RMSE', 0):.3f}</div>
    <div class='label'>🎯 RMSE</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Get Recommendations",
    "📊 Data Analysis",
    "📈 Model Evaluation",
    "💡 How It Works"
])


# ╔══════════════════════════╗
# ║  TAB 1 : RECOMMENDATIONS ║
# ╚══════════════════════════╝
with tab1:
    st.markdown(
        "<div class='section-header'>🎯 Find Similar Movies</div>",
        unsafe_allow_html=True
    )

    col_select, col_info = st.columns([2, 1])

    with col_select:
        selected_movie = st.selectbox(
            "🎬 Select a Movie / Product:",
            all_titles, index=0
        )

    with col_info:
        movie_info = movies[movies['title'] == selected_movie].iloc[0]
        st.markdown(f"""
        <div style='background:var(--bg-card2);
             border:1px solid var(--border);
             border-radius:10px; padding:1rem; margin-top:1.5rem'>
          <b>📌 {movie_info['title'][:35]}</b><br>
          <span style='color:var(--text-muted); font-size:0.85rem'>
            🏷️ {movie_info['genres']}<br>
            📅 Year: {movie_info.get('year', 'N/A')}<br>
            ⭐ Avg Rating: {movie_info.get('avg_rating', 'N/A')}
          </span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚀 Generate Recommendations", use_container_width=True):

        with st.spinner("🔮 Finding best matches…"):
            cb_recs     = cb_model.recommend(selected_movie, n=n_recommendations)
            cf_recs     = cf_model.recommend(selected_movie, n=n_recommendations)
            hybrid_recs = hybrid_model.recommend(selected_movie, n=n_recommendations)

        rec_tab1, rec_tab2, rec_tab3 = st.tabs([
            "🟣 Content-Based",
            "🔴 Collaborative",
            "🟢 Hybrid (Best)"
        ])

        def render_rec_cards(df, score_col='similarity_score'):
            if df is None or df.empty or 'error' in df.columns:
                st.warning("No recommendations found.")
                return
            for i, row in df.iterrows():
                genres_str = str(row.get('genres', 'N/A'))[:50]
                score      = row.get(score_col,
                             row.get('hybrid_score', 0))
                score_val  = f"{score:.4f}" if pd.notna(score) else "N/A"
                st.markdown(f"""
                <div class='rec-card'>
                  <div class='rec-rank'>#{i+1}</div>
                  <div>
                    <div class='rec-title'>{row['title']}</div>
                    <div class='rec-meta'>
                      🏷️ {genres_str} &nbsp;|&nbsp;
                      📅 {row.get('year','N/A')} &nbsp;|&nbsp;
                      ⭐ {row.get('avg_rating','N/A')}
                    </div>
                  </div>
                  <div class='score-badge'>Score: {score_val}</div>
                </div>
                """, unsafe_allow_html=True)

        with rec_tab1:
            st.markdown(
                "<span class='algo-pill pill-cb'>TF-IDF + Cosine Similarity</span>"
                " Based on genres & content features",
                unsafe_allow_html=True
            )
            render_rec_cards(cb_recs)

        with rec_tab2:
            st.markdown(
                "<span class='algo-pill pill-cf'>User-Item Interaction Matrix</span>"
                " Based on rating patterns",
                unsafe_allow_html=True
            )
            render_rec_cards(cf_recs)

        with rec_tab3:
            st.markdown(
                "<span class='algo-pill pill-hybrid'>α=0.5 Weighted Blend</span>"
                " Best of both worlds!",
                unsafe_allow_html=True
            )
            score_col = ('hybrid_score'
                         if 'hybrid_score' in hybrid_recs.columns
                         else 'similarity_score')
            render_rec_cards(hybrid_recs, score_col=score_col)

        st.markdown(
            "<div class='section-header'>📊 Visual Comparison</div>",
            unsafe_allow_html=True
        )
        fig = plot_recommendation_comparison(cb_recs, cf_recs, selected_movie)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ╔══════════════════════════╗
# ║  TAB 2 : DATA ANALYSIS   ║
# ╚══════════════════════════╝
with tab2:
    st.markdown(
        "<div class='section-header'>📊 Exploratory Data Analysis</div>",
        unsafe_allow_html=True
    )

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("**⭐ Rating Distribution**")
        fig_r = plot_rating_distribution(ratings)
        st.pyplot(fig_r, use_container_width=True)
        plt.close()

    with viz_col2:
        st.markdown("**🎬 Genre Analysis**")
        fig_g = plot_genre_analysis(movies)
        st.pyplot(fig_g, use_container_width=True)
        plt.close()

    st.markdown("**🌟 Popularity vs Average Rating**")
    fig_p = plot_popularity_vs_ratings(movies)
    st.pyplot(fig_p, use_container_width=True)
    plt.close()

    st.markdown("**🔥 Content Similarity Heatmap (Top 20 Movies)**")
    sim_matrix = cb_model.get_similarity_matrix_subset(n=20)
    fig_h = plot_similarity_heatmap(sim_matrix, "Content-Based Cosine Similarity")
    st.pyplot(fig_h, use_container_width=True)
    plt.close()

    st.markdown(
        "<div class='section-header'>📋 Dataset Preview</div>",
        unsafe_allow_html=True
    )
    tab_m, tab_r = st.tabs(["Movies Dataset", "Ratings Dataset"])
    with tab_m:
        st.dataframe(movies.head(50), use_container_width=True)
    with tab_r:
        st.dataframe(ratings.head(100), use_container_width=True)


# ╔══════════════════════════╗
# ║  TAB 3 : EVALUATION      ║
# ╚══════════════════════════╝
with tab3:
    st.markdown(
        "<div class='section-header'>📈 Model Performance Metrics</div>",
        unsafe_allow_html=True
    )

    fig_ev = plot_evaluation_metrics(metrics)
    st.pyplot(fig_ev, use_container_width=True)
    plt.close()

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, label, val, help_text in [
        (mc1, "RMSE",        metrics.get('RMSE', 0),
               "Root Mean Square Error — Lower is Better"),
        (mc2, "MAE",         metrics.get('MAE', 0),
               "Mean Absolute Error — Lower is Better"),
        (mc3, "Precision@K", metrics.get('Precision@K', 0),
               "Precision at K — Higher is Better"),
        (mc4, "Train Users", metrics.get('Train_Users', 0),
               "Users used for training"),
    ]:
        col.metric(
            label = label,
            value = f"{val:.4f}" if isinstance(val, float) else val,
            help  = help_text
        )

    st.markdown("---")
    st.markdown("""
    <div style='background:var(--bg-card2);
         border:1px solid var(--border);
         border-radius:12px; padding:1.5rem'>
      <h4>📖 Metrics Samjho (Simple)</h4>
      <table style='width:100%; border-collapse:collapse; font-size:0.9rem'>
        <tr style='border-bottom:1px solid var(--border)'>
          <td style='padding:8px; color:#48CAE4; font-weight:600'>RMSE</td>
          <td style='padding:8px'>Predicted vs Actual rating ka average error.
            0.85 means ~0.85 stars off on average.</td>
        </tr>
        <tr style='border-bottom:1px solid var(--border)'>
          <td style='padding:8px; color:#FF6B9D; font-weight:600'>MAE</td>
          <td style='padding:8px'>Mean Absolute Error —
            RMSE se thoda simple metric hai.</td>
        </tr>
        <tr>
          <td style='padding:8px; color:#06D6A0; font-weight:600'>Precision@K</td>
          <td style='padding:8px'>Top-K recommendations mein kitne relevant the.
            0.72 = 72% relevant results.</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════╗
# ║  TAB 4 : HOW IT WORKS    ║
# ╚══════════════════════════╝
with tab4:
    st.markdown(
        "<div class='section-header'>💡 System Architecture</div>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style='background:var(--bg-card2);
         border:1px solid var(--border);
         border-radius:12px; padding:2rem; line-height:2'>

      <h3>🇮🇳 Simple Hinglish Explanation</h3>
      <p>Recommendation system ek aisa <b>AI system</b> hai jo batata hai:
         <em>"Tumhe ye bhi pasand aayega!"</em></p>

      <hr style='border-color:var(--border)'>

      <h4>🟣 1. Content-Based Filtering</h4>
      <p><b>Kya karta hai:</b> Movie ke features (genres) dekho aur
         similar features wali movies dhundo.<br>
         <b>Example:</b> "Inception" (Sci-Fi + Thriller) →
         System "Interstellar", "The Matrix" suggest karega.<br>
         <b>Algorithm:</b> TF-IDF → Cosine Similarity</p>

      <hr style='border-color:var(--border)'>

      <h4>🔴 2. Collaborative Filtering (Item-Item)</h4>
      <p><b>Kya karta hai:</b>
         "Jo logon ne ye movies ek saath rate ki, wo similar hain"<br>
         <b>Example:</b> Amazon ka "Customers who bought X also bought Y"<br>
         <b>Algorithm:</b> Rating Matrix → Item-Item Cosine Similarity</p>

      <hr style='border-color:var(--border)'>

      <h4>🟢 3. Hybrid System</h4>
      <p><b>Formula:</b>
         <code>score = 0.5 × content_score + 0.5 × collab_score</code><br>
         <b>Used by:</b> Netflix, Spotify, Amazon — sabhi hybrid use karte hain!</p>

      <hr style='border-color:var(--border)'>

      <h4>📐 Cosine Similarity Formula</h4>
      <p><code>similarity = (A · B) / (||A|| × ||B||)</code><br>
         <b>Range:</b> 0 (different) se 1 (same)</p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:1.5rem; background:var(--bg-card2);
         border:1px solid #06D6A050;
         border-radius:12px; padding:1.5rem'>
      <h4>🗂️ Project Structure</h4>
      <pre style='color:#06D6A0; font-size:0.85rem; line-height:1.8'>
recommendation_system/
├── app.py                ← Streamlit web app
├── requirements.txt      ← Python packages
├── README.md             ← Documentation
├── data/                 ← Dataset folder
├── models/
│   └── recommender.py    ← Core ML algorithms
├── utils/
│   └── visualizations.py ← Charts & graphs
├── notebooks/            ← Jupyter notebooks
└── tests/
    └── test_recommender.py</pre>
    </div>
    """, unsafe_allow_html=True)