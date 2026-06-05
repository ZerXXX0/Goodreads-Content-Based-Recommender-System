from __future__ import annotations

from functools import lru_cache

import streamlit as st

from .evaluation import evaluate_recommender
from .preprocessing import load_ratings
from .recommender import get_recommender


@st.cache_resource(show_spinner=True)
def load_model(force_rebuild: bool = False):
    return get_recommender(force_rebuild=force_rebuild)


@st.cache_data(show_spinner=True)
def load_evaluation_metrics(force_rebuild: bool = False):
    recommender = load_model(force_rebuild=force_rebuild)
    ratings = load_ratings()
    return evaluate_recommender(recommender, ratings, top_n=10)
