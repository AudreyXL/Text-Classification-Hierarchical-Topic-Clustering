# Text Topic Modeling (20 Newsgroups)

This repo contains three parts:

- **Part 1 — TF-IDF / BoW + Classical Classifiers (Supervised)**
- **Part 2 — SentenceTransformer Embeddings + Classical Classifiers (Supervised)**
- **Part 3 — Topic Clustering + 2-Level Topic Tree (Unsupervised)**

Dataset: scikit-learn **20 Newsgroups** using the **canonical train/test split**, with `("headers","footers","quotes")` removed to reduce metadata leakage.

---

## Repo layout

```
text-topic-modeling/
  README.md
  ARCHITECTURE.md
  requirements.txt
  .gitignore
  notebooks/
    text_classification_hw.ipynb
  src/
    config.py
    data.py
    metrics.py
    part1_supervised.py
    part2_embeddings.py
    part3_clustering.py
    labeling.py
    utils.py
  scripts/
    run_part1.py
    run_part2.py
    run_part3.py
  outputs/
    (generated files)
```

- `src/` = reusable “library” code (functions you import)
- `scripts/` = runnable entrypoints for each part
- `outputs/` = generated CSVs/plots/tree text

---

## Setup

### Option A — Conda (Anaconda/Miniconda)

```bash
conda create -n textproj python=3.10 -y
conda activate textproj
pip install -r requirements.txt
```

### Option B — venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to run

Run these from the **repo root** (the folder that contains `README.md`).

### Part 1 — TF-IDF/BoW classification

```bash
python -m scripts.run_part1
```

Outputs:
- `outputs/part1_results.csv`
- `outputs/part1_model_comparison.png`

### Part 2 — SentenceTransformer embeddings classification

```bash
python -m scripts.run_part2
```

Outputs:
- `outputs/part2_results.csv`

### Part 3 — Clustering + 2-level topic tree

```bash
python -m scripts.run_part3
```

Outputs:
- `outputs/elbow_plot.png`
- `outputs/topic_tree.txt`
- `outputs/top_level_clusters.csv` (cluster sizes + labels)
- `outputs/subclusters.csv` (subcluster sizes + labels)

---

## Notes on cluster labeling (LLM)

Part 3 includes an LLM-labeling interface in `src/labeling.py`.

- Default behavior (works offline): creates **simple labels from top TF-IDF keywords**.
- If you want to use an LLM, implement `label_with_llm()` in `src/labeling.py`

---

## Expected demo flow

1. Run Part 1 → show results 
2. Run Part 2 → show results compare to Part 1
3. Run Part 3 → show tree output 
