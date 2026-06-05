from __future__ import annotations

import json

import streamlit as st

from src.dashboard_utils import load_model
from src.preprocessing import load_ratings


st.set_page_config(page_title="Goodreads Book Recommendation System", page_icon="📚", layout="wide")


st.title("Goodreads Book Recommendation System")
st.caption("Content-based recommendations using Sentence-BERT embeddings and cosine similarity.")

with st.sidebar:
    st.header("Model Information")
    st.write("Embedding model: all-MiniLM-L6-v2")
    st.write("Positive feedback threshold: rating >= 4")
    st.write("Ranking: cosine similarity over unseen books")

    st.header("Dataset Information")
    ratings = load_ratings()
    st.write(f"Ratings rows: {len(ratings):,}")
    st.write(f"Unique users: {ratings['user_id'].nunique():,}")
    st.write(f"Unique books: {ratings['book_id'].nunique():,}")


user_id = st.number_input("User ID", min_value=1, step=1, value=1)
force_rebuild = st.checkbox("Rebuild embeddings from scratch", value=False)

if st.button("Generate Recommendations"):
    with st.spinner("Loading recommender and generating results..."):
        recommender = load_model(force_rebuild=force_rebuild)
        result = recommender.recommend(int(user_id), top_n=10)

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Top-10 Recommendations")
        if result.recommendations.empty:
            st.info("No recommendations available for this user.")
        else:
            table = result.recommendations.rename(
                columns={
                    "rank": "Rank",
                    "title": "Book Title",
                    "authors": "Author",
                    "similarity": "Similarity Score",
                }
            )[["Rank", "Book Title", "Author", "Similarity Score"]]
            st.dataframe(table, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Recommendation Summary")
        st.caption("Evaluation metrics are available in the separate metrics dashboard.")

    st.subheader("User Reading History")
    if result.history.empty:
        st.info("No positively rated books were found for this user.")
    else:
        st.dataframe(result.history.rename(columns={"title": "Book Title", "authors": "Author"}), use_container_width=True, hide_index=True)

    st.subheader("Run Summary")
    summary = {
        "user_id": int(user_id),
        "recommended_books": len(result.recommendations),
    }
    st.code(json.dumps(summary, indent=2), language="json")
