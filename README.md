# Goodreads Content-Based Recommender System

A content-based recommender system for the Goodreads dataset using Sentence-BERT (SBERT) embeddings and cosine similarity. It ranks unseen books for users based on their positive reading history and tags.

---

## 📁 Repository Structure

```text
Goodreads-Content-Based-Recommender-System/
├── .gitignore
├── .gitattributes
├── README.md
├── app.py
├── metrics_dashboard.py
├── requirements.txt
├── results.json
├── analytics.md
├── models/
│   ├── book_ids.npy
│   └── embeddings.npy
├── src/
│   ├── __init__.py
│   ├── dashboard_utils.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── preprocessing.py
│   ├── recommender.py
│   └── utils.py
└── goodbooks-10k/
```

---

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Web Application

```bash
streamlit run app.py
```

The application allows users to:

* Enter a Goodreads User ID.
* View books positively rated by the user.
* Receive Top-10 personalized recommendations.
* Inspect similarity scores for each recommendation.

### 3. Run the Evaluation Dashboard

```bash
python metrics_dashboard.py --top-n 10 --threshold 4 --min-positive-ratings 20
```

Rebuild embeddings:

```bash
python metrics_dashboard.py --rebuild --top-n 10 --threshold 4 --min-positive-ratings 20
```

---

## 🧠 Methodology

### Content Enrichment

Each book is represented by:

* Title
* Author(s)
* Top user-generated tags

These attributes are merged into a single textual representation.

### Embedding Generation

The system uses:

* Sentence-BERT (`all-MiniLM-L6-v2`)
* TF-IDF fallback when SBERT is unavailable

Embeddings are normalized and cached.

### User Profiling

User preferences are modeled using a weighted average of positively rated books:

* Rating 5 contributes more than Rating 4.
* Profile vectors are L2-normalized.

### Recommendation Generation

Recommendations are produced by:

1. Building a user profile vector.
2. Computing cosine similarity against all unseen books.
3. Ranking books by similarity score.
4. Returning the Top-N results.

---

## 📊 Latest Evaluation Results

Latest results from `results.json`:

| Metric          |  Score |
| --------------- | -----: |
| Precision@10    | 0.0765 |
| Recall@10       | 0.0518 |
| NDCG@10         | 0.0825 |
| Evaluated Users | 52,783 |

### Interpretation

* Approximately 7.65% of recommended books are relevant.
* The system retrieves 5.18% of all relevant future interactions.
* Relevant books tend to appear near the top of the recommendation list.

### Improvement over Previous Version

| Metric       | Previous | Current | Improvement |
| ------------ | -------: | ------: | ----------: |
| Precision@10 |   0.0420 |  0.0765 |      +82.1% |
| Recall@10    |   0.0290 |  0.0518 |      +78.6% |
| NDCG@10      |   0.0490 |  0.0825 |      +68.4% |

---

## 📈 Performance Analysis

Given a catalog of approximately 10,000 books, recommending only 10 books per user is a highly challenging retrieval task.

Random recommendation would achieve approximately:

```text
0.1%
```

while the model achieves:

```text
7.65%
```

making it roughly:

```text
76× better than random selection
```

under the same evaluation conditions.

---

## 🔮 Future Improvements

* Hybrid Content-Based + Collaborative Filtering
* Book Synopsis Integration
* Advanced Embedding Models (MPNet, E5, BGE)
* Learning-to-Rank Re-ranking
* Implicit Feedback Integration
* User Preference Adaptation

---

## 📄 Documentation

For detailed implementation details and evaluation analysis:

* `analytics.md`
* `src/evaluation.py`
* `src/recommender.py`

---

## 📜 License

This project uses the Goodreads Goodbooks-10k dataset and follows the original dataset licensing terms.
