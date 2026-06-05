# Goodreads Content-Based Recommender System

A content-based recommender system for the Goodreads dataset using Sentence-BERT (SBERT) embeddings and cosine similarity. It ranks unseen books for users based on their positive reading history and tags.

---

## 📁 Repository Structure

The project has been restructured to separate the python application code from the raw dataset:

```
Goodreads-Content-Based-Recommender-System/
├── .gitignore               # Excludes virtualenvs, cache, and IDE configs
├── .gitattributes           # Git attributes configuration
├── README.md                # This project documentation
├── app.py                   # Streamlit web interface
├── metrics_dashboard.py     # Evaluation and metrics CLI
├── requirements.txt         # Project dependencies
├── results.json             # Cached latest evaluation results
├── analytics.md             # In-depth analysis of recommendations
├── models/                  # Precomputed embeddings and book IDs
│   ├── book_ids.npy
│   └── embeddings.npy
├── src/                     # Source code module
│   ├── __init__.py
│   ├── dashboard_utils.py   # Shared Streamlit caching helpers
│   ├── embeddings.py        # Generates/caches embeddings (SBERT / TF-IDF)
│   ├── evaluation.py        # Recommender evaluation metrics (Precision@10, NDCG, etc.)
│   ├── preprocessing.py     # Data loaders and text enrichment
│   ├── recommender.py       # Profile building and ranking logic
│   └── utils.py             # Project-wide path and file utilities
└── goodbooks-10k/           # Raw Dataset Directory
    ├── books.csv            # Book metadata
    ├── book_tags.csv        # Tags assigned by users to books (IDs)
    ├── tags.csv             # Map tag IDs to names
    ├── ratings.csv          # User ratings
    ├── to_read.csv          # Books marked "to read" by users
    ├── quick_look.ipynb     # Jupyter notebook for data exploration
    ├── README.md            # Original dataset documentation
    ├── THANKS.md            # Dataset credits
    ├── LICENSE              # Dataset license
    ├── books_xml/           # Raw XML source files
    ├── samples/             # Small CSV snippets for quick testing
    └── contrib/             # Community notebooks/scripts
```

---

## 🚀 Getting Started

### 1. Installation

Install all required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Web Application

The interactive web dashboard lets you enter a User ID to view their reading history and top-10 recommended books:

```bash
streamlit run app.py
```

### 3. Run the Evaluation Dashboard

To evaluate recommender performance (Precision@10, Recall@10, and NDCG@10) across active users:

```bash
python metrics_dashboard.py --top-n 10 --threshold 4 --min-positive-ratings 20
```

To rebuild the precomputed Sentence-BERT embeddings from scratch, add the `--rebuild` flag:

```bash
python metrics_dashboard.py --rebuild --top-n 10 --threshold 4 --min-positive-ratings 20
```

---

## 📊 Latest Evaluation Results

Latest evaluation metrics saved in `results.json`:

* **Precision@10:** 0.042
* **Recall@10:** 0.029
* **NDCG@10:** 0.049
* **Evaluated Users:** 52,783

For a deep-dive analysis, check out [analytics.md](analytics.md).
