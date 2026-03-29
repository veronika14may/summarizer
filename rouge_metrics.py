"""
rouge_metrics.py
Вычисление метрик ROUGE-N: Precision, Recall, F-measure.

ROUGE-N считает перекрытие N-грамм между сгенерированным рефератом
(hypothesis) и эталонным рефератом (reference).

Формулы:
    Precision = |совпадающие N-граммы| / |N-граммы гипотезы|
    Recall    = |совпадающие N-граммы| / |N-граммы эталона|
    F-measure = 2 * Precision * Recall / (Precision + Recall)
"""

from collections import Counter
from preprocessor import tokenize_words, lemmatize, STOP_WORDS


# ── Нормализация текста для метрик ────────────────────────────────────────────

def _normalize_for_rouge(text: str) -> list[str]:
    """
    Токенизировать и лемматизировать текст без фильтрации стоп-слов
    (ROUGE обычно считается по всем словам, но мы предлагаем оба варианта).
    """
    tokens = tokenize_words(text)
    return [lemmatize(t) for t in tokens]


def _normalize_no_stopwords(text: str) -> list[str]:
    """Токенизировать, лемматизировать и убрать стоп-слова."""
    tokens = tokenize_words(text)
    return [lemmatize(t) for t in tokens if lemmatize(t) not in STOP_WORDS]


# ── N-граммы ─────────────────────────────────────────────────────────────────

def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Построить Counter N-грамм из списка токенов."""
    ngrams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


# ── Ядро метрики ──────────────────────────────────────────────────────────────

def rouge_n(
    hypothesis: str,
    reference: str,
    n: int = 1,
    remove_stopwords: bool = False,
) -> dict[str, float]:
    """
    Вычислить ROUGE-N между одним рефератом и одним эталоном.

    :param hypothesis:       сгенерированный реферат
    :param reference:        эталонный реферат
    :param n:                порядок N-граммы (1 или 2)
    :param remove_stopwords: убирать стоп-слова перед подсчётом
    :return: словарь {"precision": float, "recall": float, "f1": float}
    """
    norm = _normalize_no_stopwords if remove_stopwords else _normalize_for_rouge

    hyp_tokens = norm(hypothesis)
    ref_tokens = norm(reference)

    hyp_ngrams = _get_ngrams(hyp_tokens, n)
    ref_ngrams = _get_ngrams(ref_tokens, n)

    # Пересечение: минимум из двух счётчиков
    overlap = sum((hyp_ngrams & ref_ngrams).values())
    hyp_total = sum(hyp_ngrams.values())
    ref_total = sum(ref_ngrams.values())

    precision = overlap / hyp_total if hyp_total > 0 else 0.0
    recall = overlap / ref_total if ref_total > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_n_batch(
    hypotheses: list[str],
    references: list[str],
    n: int = 1,
    remove_stopwords: bool = False,
) -> dict[str, float]:
    """
    Macro-average ROUGE-N по всему батчу.

    :param hypotheses: список сгенерированных рефератов
    :param references: список эталонных рефератов
    :param n:          порядок N-граммы
    :return: усреднённые precision, recall, f1
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Количество гипотез ({len(hypotheses)}) "
            f"не совпадает с количеством эталонов ({len(references)})"
        )

    scores = [
        rouge_n(h, r, n=n, remove_stopwords=remove_stopwords)
        for h, r in zip(hypotheses, references)
    ]

    avg_precision = sum(s["precision"] for s in scores) / len(scores)
    avg_recall = sum(s["recall"] for s in scores) / len(scores)
    avg_f1 = sum(s["f1"] for s in scores) / len(scores)

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "per_doc": scores,  # отдельные значения для каждого документа
    }


def evaluate(
    hypotheses: list[str],
    references: list[str],
    remove_stopwords: bool = False,
) -> dict:
    """
    Полный отчёт: ROUGE-1 и ROUGE-2.

    :return: вложенный словарь с метриками по ROUGE-1 и ROUGE-2
    """
    r1 = rouge_n_batch(hypotheses, references, n=1, remove_stopwords=remove_stopwords)
    r2 = rouge_n_batch(hypotheses, references, n=2, remove_stopwords=remove_stopwords)
    return {"ROUGE-1": r1, "ROUGE-2": r2}
