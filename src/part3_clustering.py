"""
Part 3 — Topic Clustering + 2-Level Topic Tree

Steps:
A) Top-level clustering with KMeans (K < 10), K chosen with elbow method
B) Find the 2 largest clusters; re-cluster each into exactly 3 subclusters
C) Label clusters using representative docs closest to centroid
D) Save a partial topic tree to outputs/

Outputs:
- outputs/elbow_plot.png
- outputs/topic_tree.txt
- outputs/top_level_clusters.csv
- outputs/subclusters.csv
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

from .config import (
    RANDOM_STATE,
    OUTPUT_DIR,
    EMBEDDING_MODEL_NAME,
    EMBED_BATCH_SIZE,
    NORMALIZE_EMBEDDINGS,
    TOP_LEVEL_K_RANGE,
    REPRESENTATIVES_PER_CLUSTER,
    SUBCLUSTERS_PER_BIG_CLUSTER,
)
from .labeling import make_cluster_label
from .utils import ensure_dir, write_text


def embed_docs(docs: List[str]) -> np.ndarray:
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedder.encode(
        docs,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    )


def elbow_choose_k(emb: np.ndarray, k_range=TOP_LEVEL_K_RANGE) -> Tuple[int, List[int], List[float]]:
    """
    Compute KMeans inertia for each K and pick an elbow automatically.

    Returns:
        k_elbow, ks, inertias
    """
    ks = list(k_range)
    inertias: List[float] = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        km.fit(emb)
        inertias.append(float(km.inertia_))

    # Simple elbow heuristic: max second difference in inertia
    d1 = np.diff(inertias)
    d2 = np.diff(d1)
    elbow_idx = 1 + int(np.argmax(np.abs(d2)))  # +1 because of diffs
    k_elbow = ks[elbow_idx]
    return k_elbow, ks, inertias


def save_elbow_plot(ks: List[int], inertias: List[float], out_path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Inertia (lower is better)")
    plt.title("Elbow Method (K < 10)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def top_representatives(
    emb: np.ndarray,
    cluster_ids: np.ndarray,
    centroids: np.ndarray,
    cluster_idx: int,
    n: int,
) -> np.ndarray:
    """Indices of n docs closest to the centroid within a cluster (most representative)."""
    dist = cosine_distances(emb, centroids)
    dist_to_own = dist[np.arange(len(emb)), cluster_ids]
    idx = np.where(cluster_ids == cluster_idx)[0]
    idx_sorted = idx[np.argsort(dist_to_own[idx])]
    return idx_sorted[:n]


def short_label(label_text: str) -> str:
    """Extract the label portion from 'LABEL: ... | KEYWORDS: ...'."""
    if "LABEL:" in label_text:
        return label_text.split("LABEL:", 1)[1].split("|", 1)[0].strip()
    return label_text.strip()


def run_part3(docs: List[str], use_llm_labels: bool = False) -> Dict[str, Any]:
    """
    Build the top-level clusters and a 2-level tree.

    Args:
        docs: list of documents to cluster (recommend: X_train or X_train+X_test)
        use_llm_labels: if True, will call the LLM stub in labeling.py
    """
    ensure_dir(OUTPUT_DIR)

    print("=" * 72)
    print("PART 3 — Topic clustering + 2-level tree")
    print("=" * 72)

    emb = embed_docs(docs)

    # Step A: elbow + top-level KMeans
    k_elbow, ks, inertias = elbow_choose_k(emb)
    save_elbow_plot(ks, inertias, OUTPUT_DIR / "elbow_plot.png")
    print(f"Chosen K (elbow): {k_elbow}")

    kmeans = KMeans(n_clusters=k_elbow, random_state=RANDOM_STATE, n_init="auto")
    cluster_ids = kmeans.fit_predict(emb)

    # Label each top-level cluster
    cluster_sizes = {c: int(np.sum(cluster_ids == c)) for c in range(k_elbow)}
    cluster_labels: Dict[int, str] = {}

    for c in range(k_elbow):
        reps = top_representatives(
            emb, cluster_ids, kmeans.cluster_centers_, c, REPRESENTATIVES_PER_CLUSTER
        )
        rep_texts = [docs[i] for i in reps]
        cluster_labels[c] = make_cluster_label(rep_texts, use_llm=use_llm_labels)

    top_level_df = pd.DataFrame([
        {"Cluster": c, "Size": cluster_sizes[c], "Label": cluster_labels[c]}
        for c in range(k_elbow)
    ]).sort_values("Size", ascending=False)
    top_level_df.to_csv(OUTPUT_DIR / "top_level_clusters.csv", index=False)

    # Step B: pick the 2 largest clusters and create 3 subclusters each
    top2 = list(top_level_df["Cluster"].head(2))
    print(f"Two largest clusters: {top2}")

    subcluster_rows = []
    subtree: Dict[int, Any] = {}

    for parent_c in top2:
        member_idx = np.where(cluster_ids == parent_c)[0]
        emb_sub = emb[member_idx]

        km_sub = KMeans(
            n_clusters=SUBCLUSTERS_PER_BIG_CLUSTER,
            random_state=RANDOM_STATE,
            n_init="auto",
        )
        sub_ids = km_sub.fit_predict(emb_sub)

        # distances for representatives within subcluster
        dist_sub = cosine_distances(emb_sub, km_sub.cluster_centers_)
        dist_to_own = dist_sub[np.arange(len(member_idx)), sub_ids]

        sub_labels = []
        sub_sizes = []

        for sc in range(SUBCLUSTERS_PER_BIG_CLUSTER):
            local = np.where(sub_ids == sc)[0]
            local_sorted = local[np.argsort(dist_to_own[local])]
            reps_local = local_sorted[:REPRESENTATIVES_PER_CLUSTER]
            reps_global = member_idx[reps_local]

            rep_texts = [docs[i] for i in reps_global]
            label = make_cluster_label(rep_texts, use_llm=use_llm_labels)

            size = int(np.sum(sub_ids == sc))
            sub_labels.append(label)
            sub_sizes.append(size)

            subcluster_rows.append({
                "ParentCluster": parent_c,
                "Subcluster": sc,
                "Size": size,
                "Label": label,
            })

        subtree[parent_c] = {"sub_labels": sub_labels, "sub_sizes": sub_sizes}

    pd.DataFrame(subcluster_rows).sort_values(["ParentCluster", "Size"], ascending=[True, False])        .to_csv(OUTPUT_DIR / "subclusters.csv", index=False)

    # Step C: write a simple text tree
    lines = []
    lines.append("PARTIAL TOPIC TREE")
    lines.append("=" * 60)

    for c in range(k_elbow):
        lines.append(f"• {short_label(cluster_labels[c])} (n={cluster_sizes[c]})")
        if c in subtree:
            for j in range(SUBCLUSTERS_PER_BIG_CLUSTER):
                lines.append(
                    f"   ├─ {short_label(subtree[c]['sub_labels'][j])} (n={subtree[c]['sub_sizes'][j]})"
                )

    tree_text = "\n".join(lines)
    write_text(OUTPUT_DIR / "topic_tree.txt", tree_text)

    print()
    print(tree_text)
    print()
    print("Saved: outputs/elbow_plot.png, outputs/topic_tree.txt, outputs/top_level_clusters.csv, outputs/subclusters.csv")

    return {
        "k": k_elbow,
        "cluster_sizes": cluster_sizes,
        "cluster_labels": cluster_labels,
        "subtree": subtree,
    }
