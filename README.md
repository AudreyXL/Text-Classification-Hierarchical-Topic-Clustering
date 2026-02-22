# Text Topic Modeling -- 20 Newsgroups

## Project Overview

This project shows three different approaches to working with text data
using the 20 Newsgroups dataset:

1.  TF-IDF + Machine Learning (Supervised Classification)
2.  Sentence Embeddings + Machine Learning (Supervised Classification)
3.  Clustering + Topic Tree (Unsupervised Topic Discovery)

Parts 1 and 2 classify documents using labels.
Part 3 ignores labels and discovers topics automatically.

------------------------------------------------------------------------

## How to Run

From the project root:

python -m scripts.run_part1 

python -m scripts.run_part2 

python -m scripts.run_part3

Each part can run independently.

Outputs are saved in the `outputs/` folder.

------------------------------------------------------------------------

## Part 1 -- TF-IDF Classification

What it does: - Converts text into TF-IDF vectors - Trains models like
Logistic Regression or SVM - Evaluates performance

Outputs: - part1_results.csv - part1_model_comparison.png

Metrics: - Accuracy - Macro F1 - Training and prediction time

------------------------------------------------------------------------

## Part 2 -- Embedding Classification

What it does: - Converts text into SentenceTransformer embeddings -
Trains the same types of classifiers - Compares results to TF-IDF

Outputs:  part2_results.csv

Metrics: Accuracy, Macro F1, Embedding time, Training and
prediction time

------------------------------------------------------------------------

## Part 3 -- Topic Clustering (Unsupervised)

What it does: Converts text into vectors 

Groups documents using K-Means  

Builds a 2-level topic tree: Labels clusters using top
keywords

Outputs: elbow_plot.png 

topic_tree.txt 

top_level_clusters.csv 

subclusters.csv

Metrics: Inertia 

Silhouette score 

------------------------------------------------------------------------

## Expected Results

1. Similar documents group together
2. Embeddings capture deeper meaning than TF-IDF
3. Clear topic clusters form in the unsupervised setting

------------------------------------------------------------------------

## Summary

This project demonstrates shows text vectorization, supervised classification, unsupervised clustering, model comparison and basic evaluation metrics. It shows how different text representations affect classification and
topic discovery.
