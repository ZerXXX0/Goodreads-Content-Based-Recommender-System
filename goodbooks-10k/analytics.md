# Goodreads Content-Based Recommender Analytics

## Evaluation summary

This analysis is based on the latest evaluation run with:

`python3 metrics_dashboard.py --rebuild --top-n 10 --threshold 4 --min-positive-ratings 20 --output results.json`

### Key metrics

* Precision@10: 0.042
* Recall@10: 0.029
* NDCG@10: 0.049
* Evaluated users: 52,783

## Dataset and evaluation context

* Ratings rows: 5,976,479
* Unique users: 53,424
* Unique books: 10,000
* Positive threshold: 4 (ratings 4 and 5 are treated as positive)
* Minimum positive ratings per user for evaluation: 20

## Coverage and diagnostics

* Active users included in evaluation: 52,783
* Book ID overlap with the recommender corpus: 10,000 / 10,000 (100%)
* Book coverage in embeddings: 100%
* Average positive ratings per user: 77.18
* Median positive ratings per user: 76.00
* Min positive ratings per user: 1
* Max positive ratings per user: 194

## Notes

The evaluation uses a content-based recommender to rank unseen books for each user, holding out a random split of positive ratings for evaluation. The low precision and recall values reflect the difficulty of recommending from a very large candidate set of 10,000 books using only implicit positive feedback.
