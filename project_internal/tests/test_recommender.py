"""
============================================================
  UNIT TESTS - Recommendation System
  Run with: python -m pytest tests/ -v
============================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import numpy as np
from models.recommender import (
    load_movielens_data, preprocess_data,
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender
)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture(scope='module')
def sample_data():
    movies, ratings          = load_movielens_data()
    movies_f, ratings_f, rm, sm = preprocess_data(movies, ratings)
    return movies_f, ratings_f, rm


# ── Tests: Data Loading ───────────────────────────────────

class TestDataLoading:

    def test_movies_loaded(self, sample_data):
        movies, _, _ = sample_data
        assert len(movies) > 0

    def test_movies_has_required_columns(self, sample_data):
        movies, _, _ = sample_data
        for col in ['movieId', 'title', 'genres']:
            assert col in movies.columns

    def test_ratings_loaded(self, sample_data):
        _, ratings, _ = sample_data
        assert len(ratings) > 0

    def test_rating_matrix_not_empty(self, sample_data):
        _, _, rm = sample_data
        assert rm.shape[0] > 0 and rm.shape[1] > 0

    def test_ratings_valid_range(self, sample_data):
        _, ratings, _ = sample_data
        assert ratings['rating'].between(0.5, 5.0).all()


# ── Tests: Content-Based ──────────────────────────────────

class TestContentBasedRecommender:

    def test_fit_returns_self(self, sample_data):
        movies, _, _ = sample_data
        cb     = ContentBasedRecommender()
        result = cb.fit(movies)
        assert result is cb

    def test_recommendations_returns_dataframe(self, sample_data):
        movies, _, _ = sample_data
        cb    = ContentBasedRecommender().fit(movies)
        title = movies['title'].iloc[0]
        recs  = cb.recommend(title, n=5)
        assert isinstance(recs, pd.DataFrame)

    def test_recommendations_correct_count(self, sample_data):
        movies, _, _ = sample_data
        cb    = ContentBasedRecommender().fit(movies)
        title = movies['title'].iloc[0]
        recs  = cb.recommend(title, n=5)
        assert len(recs) <= 5

    def test_recommended_movie_not_same_as_input(self, sample_data):
        movies, _, _ = sample_data
        cb    = ContentBasedRecommender().fit(movies)
        title = movies['title'].iloc[0]
        recs  = cb.recommend(title, n=5)
        if 'title' in recs.columns:
            assert title not in recs['title'].values

    def test_similarity_score_between_0_and_1(self, sample_data):
        movies, _, _ = sample_data
        cb    = ContentBasedRecommender().fit(movies)
        title = movies['title'].iloc[0]
        recs  = cb.recommend(title, n=5)
        if 'similarity_score' in recs.columns:
            assert recs['similarity_score'].between(0, 1).all()


# ── Tests: Collaborative Filtering ───────────────────────

class TestCollaborativeRecommender:

    def test_fit_succeeds(self, sample_data):
        movies, _, rm = sample_data
        cf     = CollaborativeFilteringRecommender(method='item-item')
        result = cf.fit(rm, movies)
        assert result is cf

    def test_recommendations_returns_dataframe(self, sample_data):
        movies, _, rm = sample_data
        cf    = CollaborativeFilteringRecommender(
                    method='item-item').fit(rm, movies)
        title = movies['title'].iloc[0]
        recs  = cf.recommend(title, n=5)
        assert isinstance(recs, pd.DataFrame)

    def test_item_sim_matrix_created(self, sample_data):
        movies, _, rm = sample_data
        cf = CollaborativeFilteringRecommender(
                 method='item-item').fit(rm, movies)
        assert cf.item_sim_df is not None


# ── Tests: Hybrid ─────────────────────────────────────────

class TestHybridRecommender:

    def test_hybrid_fit_and_recommend(self, sample_data):
        movies, _, rm = sample_data
        hybrid = HybridRecommender(alpha=0.5).fit(movies, rm)
        title  = movies['title'].iloc[0]
        recs   = hybrid.recommend(title, n=5)
        assert isinstance(recs, pd.DataFrame)
        assert len(recs) <= 5

    def test_hybrid_alpha_range(self, sample_data):
        movies, _, rm = sample_data
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            h    = HybridRecommender(alpha=alpha).fit(movies, rm)
            recs = h.recommend(movies['title'].iloc[0], n=3)
            assert isinstance(recs, pd.DataFrame)


# ── Sanity Checks ─────────────────────────────────────────

def test_cosine_similarity_identical_vectors():
    from sklearn.metrics.pairwise import cosine_similarity
    v1 = np.array([[1, 0, 1]])
    v2 = np.array([[1, 0, 1]])
    result = cosine_similarity(v1, v2)[0][0]
    assert result == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal_vectors():
    from sklearn.metrics.pairwise import cosine_similarity
    v1 = np.array([[1, 0]])
    v2 = np.array([[0, 1]])
    result = cosine_similarity(v1, v2)[0][0]
    assert result == pytest.approx(0.0, abs=1e-5)


def test_tfidf_vectorizer_works():
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs   = ["Action Comedy", "Drama Romance", "Action Drama"]
    tfidf  = TfidfVectorizer()
    matrix = tfidf.fit_transform(docs)
    assert matrix.shape[0] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
    