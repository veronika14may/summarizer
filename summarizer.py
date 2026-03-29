import math
from collections import Counter
from preprocessor import split_sentences, preprocess_sentence

MAX_CHARS = 300

def compute_tf(tokens):
    if not tokens:
        return {}
    counter = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counter.items()}

def compute_idf(sentences_tokens):
    n = len(sentences_tokens)
    df = {}
    for tokens in sentences_tokens:
        for word in set(tokens):
            df[word] = df.get(word, 0) + 1
    return {word: math.log((n + 1) / (freq + 1)) + 1 for word, freq in df.items()}

def sentence_tfidf_score(tf,idf):
    if not tf:
        return 0.0
    scores = [tf[w] * idf.get(w, 1.0) for w in tf]
    return sum(scores) / len(scores)

def _position_bonus(idx, total):
    if total <= 1:
        return 1.0
    if idx == 0:
        return 1.5
    if idx == 1:
        return 1.2
    if idx == total - 1:
        return 1.1
    return 1.0

def cosine_similarity(v1, v2):
    """Косинусное сходство двух TF-векторов"""
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[w] * v2[w] for w in common)
    norm1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    norm2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def connectivity_scores(tf_list):
    n = len(tf_list)
    scores = [0.0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                scores[i] += cosine_similarity(tf_list[i], tf_list[j])
    max_score = max(scores) if scores else 1.0
    if max_score == 0:
        return [0.0] * n
    return [s / max_score for s in scores]

def summarize(text, max_chars):
    sentences = split_sentences(text)
    if not sentences:
        return ""

    # Если исходный текст уже короче порога - вернуть как есть
    if len(text) <= max_chars:
        return text[:max_chars]

    tokens_per_sentence = [preprocess_sentence(s) for s in sentences]

    # TF-IDF
    idf = compute_idf(tokens_per_sentence)
    tf_list = [compute_tf(toks) for toks in tokens_per_sentence]
    tfidf_scores = [sentence_tfidf_score(tf, idf) for tf in tf_list]

    # Связность
    conn_scores = connectivity_scores(tf_list)

    # Итоговая оценка предложения
    total = len(sentences)
    final_scores = [
        (tfidf_scores[i] * 0.6 + conn_scores[i] * 0.4)
        * _position_bonus(i, total)
        for i in range(total)
    ]

    # Жадный выбор предложений
    ranked = sorted(range(total), key=lambda i: final_scores[i], reverse=True)
    selected: list[int] = []
    current_len = 0

    for idx in ranked:
        sentence = sentences[idx]
        addition = len(sentence) + (1 if selected else 0)
        if current_len + addition <= max_chars:
            selected.append(idx)
            current_len += addition
        if current_len >= max_chars:
            break

    # Если ни одно предложение не вошло (слишком длинные) — обрезать лучшее
    if not selected:
        best_idx = ranked[0]
        return sentences[best_idx][:max_chars]

    # Восстановить порядок и собрать текст
    selected.sort()
    summary = " ".join(sentences[i] for i in selected)

    return summary[:max_chars]

def summarize_batch(texts, max_chars):
    """Суммаризировать массив текстов"""
    return [summarize(t, max_chars) for t in texts]
