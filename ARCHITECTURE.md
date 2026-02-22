# ARCHITECTURE

## Goals

1. **Part 1:** Build a correct supervised text classification baseline using **BoW/TF-IDF** + classical ML models.
2. **Part 2:** Build a second supervised pipeline using **SentenceTransformer embeddings**, then train the same set of models for a fair comparison.
3. **Part 3:** Cluster documents into meaningful topics and produce an interpretable **2-level topic tree**.

This repo is structured to be reproducible and easy to explain.

---

## Data flow

### Part 1 (Supervised: BoW/TF-IDF)
Raw text
→ Vectorizer (BoW or TF-IDF)
→ Classifier (NB / Logistic Regression / Linear SVM / Random Forest)
→ Predictions on test set
→ Metrics + plots saved to `outputs/`

### Part 2 (Supervised: Embeddings)
Raw text
→ SentenceTransformer encoder (`all-MiniLM-L6-v2`)
→ Dense embeddings for train/test
→ Classifier (NB / Logistic Regression / Linear SVM / Random Forest)
→ Predictions on test set
→ Metrics saved to `outputs/`

### Part 3 (Unsupervised: Clustering + topic tree)
Raw text
→ SentenceTransformer embeddings
→ KMeans top-level clustering (K < 10 chosen by elbow)
→ For each cluster: pick centroid-nearest docs → generate label
→ Find 2 biggest clusters
→ Re-cluster each into exactly 3 subclusters → label each
→ Save a partial topic tree

---

## Module responsibilities

### `src/config.py`
Central configuration:
- random seed
- vectorizer parameters
- model hyperparameters
- embedding model name
- clustering parameters (K range, # representative docs)
- output file paths

### `src/data.py`
Loads 20 Newsgroups dataset:
- canonical train/test split
- remove headers/footers/quotes
Returns: `X_train, y_train, X_test, y_test, class_names`.

### `src/metrics.py`
Reusable metric functions:
- accuracy
- macro-F1
Also builds a results DataFrame.

### `src/part1_supervised.py`
- Builds 8 pipelines (2 vectorizers × 4 classifiers)
- Trains each on train set
- Evaluates on test set
- Saves CSV + comparison bar chart

### `src/part2_embeddings.py`
- Computes SentenceTransformer embeddings for train/test
- Trains/evaluates the same set of classifiers on embeddings
- Saves CSV

### `src/labeling.py`
Topic labeling interface:
- Builds a prompt/text bundle from representative docs
- Default offline labeler: top TF-IDF keywords
- A stub you can connect to an LLM API if required

### `src/part3_clustering.py`
- Elbow method for K (<10)
- Top-level KMeans clustering
- Representative docs selection
- Labeling top-level clusters
- Re-clustering 2 biggest clusters into 3 subclusters each
- Saves elbow plot, CSV summaries, and a text tree

### `src/utils.py`
Small helpers:
- ensure output directory exists
- safe saving utilities

### `scripts/run_part*.py`
Command-line entrypoints that call the corresponding `src/` functions.

---


### TF-IDF
TF-IDF converts text into a large sparse vector:
- words that appear often in a document matter
- words that appear in *every* document matter less


### Embeddings
Embeddings convert each document into a dense vector representing meaning.
They enable:
- semantic clustering (topics group by meaning, not just keyword overlap)
- a fair “dense-feature” supervised baseline in Part 2
