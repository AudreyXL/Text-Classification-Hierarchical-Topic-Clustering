# Unsupervised Topic Modeling --- 20 Newsgroups

## Overview

This project explores how to automatically discover topics in text data
**without using any labels**. Instead of training a classifier, the goal
is to uncover natural structure directly from the text.

## Problem

Can meaningful topics be discovered in a large document collection
without labeled data?\
This is important because real-world datasets often lack annotations.

## Approach

1.  Convert text to numeric vectors using:
    -   TF-IDF
    -   SentenceTransformer embeddings\
2.  Apply K-Means clustering to group similar documents.
3.  Build a simple 2-level topic hierarchy (main topics + subtopics).
4.  Automatically label clusters using top TF-IDF keywords.

## Technology

-   Python\
-   Scikit-learn\
-   SentenceTransformers\
-   Pandas & Matplotlib

## Run

From the project root:

python -m scripts.run_part3

Outputs are saved in the `outputs/` folder.

------------------------------------------------------------------------

This project shows unsupervised topic discovery, clustering, and
hierarchical structure learning without using any labels.
