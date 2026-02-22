from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_tfidf(texts, top_k=6):
    """
    Extract simple keywords using TF-IDF.
    Robust to empty/stopword-only documents.
    """
    # 1) Clean + keep only non-empty texts
    cleaned = []
    for t in texts:
        if t is None:
            continue
        t = " ".join(str(t).split())
        if len(t) >= 20:          # avoid tiny fragments
            cleaned.append(t)

    # 2) If nothing usable, return empty list (caller can handle)
    if len(cleaned) == 0:
        return []

    try:
        # For small representative sets, be gentle:
        # - stop_words=None avoids wiping everything out
        # - min_df=1 because we might have only a few docs
        vec = TfidfVectorizer(stop_words=None, min_df=1, max_df=1.0)
        X = vec.fit_transform(cleaned)
        vocab = vec.get_feature_names_out()

        # mean TF-IDF score across the representative docs
        scores = X.mean(axis=0).A1
        top_idx = scores.argsort()[::-1][:top_k]
        return [vocab[i] for i in top_idx if scores[i] > 0]

    except ValueError:
        # This is the "empty vocabulary" case
        return []
def make_cluster_label(texts, use_llm: bool = False) -> str:
    """
    Create a simple, interpretable label for a cluster.

    - Default behavior (use_llm=False): build a label from TF-IDF keywords
    - If use_llm=True: you can plug in an LLM call (stubbed for now)
    """
    # Clean representative texts (avoid empty docs)
    cleaned = []
    for t in texts:
        if t is None:
            continue
        t = " ".join(str(t).split())
        if len(t) >= 20:
            cleaned.append(t)

    if len(cleaned) == 0:
        return "Misc Topic"

    if use_llm:
        # Stub: replace with your real LLM call if required
        # prompt = build_label_prompt(cleaned)  # if you implement it
        # return call_llm(prompt)
        return "LABEL: (LLM label stub) | KEYWORDS: (stub)"

    keywords = extract_keywords_tfidf(cleaned, top_k=6)
    if len(keywords) == 0:
        return "Misc Topic"

    # Simple, readable label: top 3 keywords
    return f"LABEL: {' / '.join(keywords[:3])} | KEYWORDS: {', '.join(keywords[:6])}"