import re
import pymorphy3
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

morph = pymorphy3.MorphAnalyzer()

STOP_WORDS = set(stopwords.words("russian"))

def tokenize_words(text):
    return re.findall(r"[а-яёА-ЯЁa-zA-Z]+", text)

def lemmatize(word):
    """Привести слово к нормальной форме (лемме)"""
    parsed = morph.parse(word.lower())
    if parsed:
        return parsed[0].normal_form
    return word.lower()

def normalize_tokens(tokens):
    """Лемматизировать список токенов и убрать стоп-слова"""
    result = []
    for tok in tokens:
        lemma = lemmatize(tok)
        if lemma not in STOP_WORDS and len(lemma) > 1:
            result.append(lemma)
    return result

def split_sentences(text):
    """Разбить текст на предложения"""
    raw = re.split(r"(?<=[.!?…])\s+|(?<=\n)\s*", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def preprocess_sentence(sentence):
    """Токенизировать и нормализовать одно предложение"""
    tokens = tokenize_words(sentence)
    return normalize_tokens(tokens)