"""Proyecto 3: clasificador SPAM/HAM con Bayes.

Este modulo implementa un filtro de SMS usando probabilidades por palabra
segun las formulas del anexo del proyecto. El modelo usa presencia de palabras
(document frequency), suavizado de Laplace y una regla de combinacion bayesiana.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FALLBACK_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by can did do does doing down during each few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just me more most my myself
no nor not now of off on once only or other our ours ourselves out over own same she should so some such
than that the their theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with you your yours yourself yourselves
""".split())


def load_dataset(path: str = "spam_ham.csv") -> pd.DataFrame:
    """Carga y normaliza el dataset del laboratorio."""
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df = df.rename(columns={"Label": "label", "SMS_TEXT": "message"})
    df["label"] = (
        df["label"].astype(str).str.replace('"', "", regex=False).str.strip().str.lower()
    )
    df["message"] = df["message"].fillna("").astype(str)
    df = df[df["label"].isin(["spam", "ham"])].copy()
    return df.reset_index(drop=True)


def get_stopwords() -> set[str]:
    """Usa stopwords de NLTK si estan disponibles; si no, usa respaldo local."""
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return FALLBACK_STOPWORDS.copy()


@dataclass
class SpamHamBayesClassifier:
    """Clasificador Bayesiano basado en presencia de palabras."""

    min_word_probability: float = 0.01
    max_word_probability: float = 0.99
    stopwords: set[str] = field(default_factory=get_stopwords)
    prior_spam: float = 0.0
    prior_ham: float = 0.0
    word_spam_probability: dict[str, float] = field(default_factory=dict)
    word_likelihoods: dict[str, tuple[float, float]] = field(default_factory=dict)
    vocabulary: set[str] = field(default_factory=set)

    def preprocess(self, text: str) -> list[str]:
        """Tokeniza, convierte a minusculas y elimina ruido/stopwords."""
        tokens = wordpunct_tokenize(str(text).lower())
        tokens = [token for token in tokens if re.fullmatch(r"[a-z]+", token)]
        tokens = [token for token in tokens if token not in self.stopwords]
        return tokens

    def fit(self, messages: Iterable[str], labels: Iterable[str]) -> "SpamHamBayesClassifier":
        train_df = pd.DataFrame({"message": list(messages), "label": list(labels)})
        train_df["tokens"] = train_df["message"].apply(self.preprocess)

        spam_docs = train_df[train_df["label"] == "spam"]
        ham_docs = train_df[train_df["label"] == "ham"]
        if spam_docs.empty or ham_docs.empty:
            raise ValueError("El entrenamiento necesita ejemplos de spam y ham.")

        self.prior_spam = len(spam_docs) / len(train_df)
        self.prior_ham = len(ham_docs) / len(train_df)
        self.vocabulary = set(token for tokens in train_df["tokens"] for token in tokens)

        spam_document_frequency = Counter()
        ham_document_frequency = Counter()
        for tokens in spam_docs["tokens"]:
            spam_document_frequency.update(set(tokens))
        for tokens in ham_docs["tokens"]:
            ham_document_frequency.update(set(tokens))

        self.word_spam_probability = {}
        self.word_likelihoods = {}
        for word in sorted(self.vocabulary):
            p_word_given_spam = (spam_document_frequency[word] + 1) / (len(spam_docs) + 2)
            p_word_given_ham = (ham_document_frequency[word] + 1) / (len(ham_docs) + 2)
            denominator = p_word_given_spam * self.prior_spam + p_word_given_ham * self.prior_ham
            p_spam_given_word = (p_word_given_spam * self.prior_spam) / denominator
            p_spam_given_word = min(
                max(p_spam_given_word, self.min_word_probability), self.max_word_probability
            )
            self.word_spam_probability[word] = p_spam_given_word
            self.word_likelihoods[word] = (p_word_given_spam, p_word_given_ham)
        return self

    def predict_proba(self, text: str) -> float:
        """Retorna P(spam | texto) combinando las probabilidades de sus palabras."""
        tokens = sorted(set(self.preprocess(text)))
        probabilities = [self.word_spam_probability[token] for token in tokens if token in self.vocabulary]
        if not probabilities:
            return self.prior_spam

        log_spam_product = sum(math.log(probability) for probability in probabilities)
        log_ham_product = sum(math.log(1 - probability) for probability in probabilities)
        max_log = max(log_spam_product, log_ham_product)
        spam_product = math.exp(log_spam_product - max_log)
        ham_product = math.exp(log_ham_product - max_log)
        return spam_product / (spam_product + ham_product)

    def predict(self, text: str, threshold: float = 0.5) -> str:
        return "spam" if self.predict_proba(text) >= threshold else "ham"

    def top_predictive_words(self, text: str, n: int = 3) -> list[tuple[str, float]]:
        """Devuelve las n palabras del texto con mayor P(spam | palabra)."""
        unique_tokens = sorted(set(self.preprocess(text)))
        scored_tokens = [
            (token, self.word_spam_probability[token])
            for token in unique_tokens
            if token in self.word_spam_probability
        ]
        return sorted(scored_tokens, key=lambda item: item[1], reverse=True)[:n]

    def classify_prompt(self, text: str, threshold: float = 0.5) -> dict[str, object]:
        probability = self.predict_proba(text)
        return {
            "text": text,
            "spam_probability": probability,
            "spam_probability_percent": round(probability * 100, 2),
            "prediction": "spam" if probability >= threshold else "ham",
            "threshold": threshold,
            "top_predictive_words": self.top_predictive_words(text, 3),
        }


def train_project_model(path: str = "spam_ham.csv", random_state: int = 42):
    """Carga datos, divide 80/20 y entrena el modelo del proyecto."""
    df = load_dataset(path)
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=random_state, stratify=df["label"]
    )
    model = SpamHamBayesClassifier().fit(train_df["message"], train_df["label"])
    return model, train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def evaluate_thresholds(model: SpamHamBayesClassifier, test_df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    """Evalua matriz de confusion y metricas para varios thresholds."""
    if thresholds is None:
        thresholds = [round(value / 10, 1) for value in range(1, 10)]

    y_true = (test_df["label"] == "spam").astype(int).to_numpy()
    probabilities = test_df["message"].apply(model.predict_proba).to_numpy()
    rows = []
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "TP": int(matrix[0, 0]),
                "FN": int(matrix[0, 1]),
                "FP": int(matrix[1, 0]),
                "TN": int(matrix[1, 1]),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    model, train_df, test_df = train_project_model()
    metrics = evaluate_thresholds(model, test_df)
    print(metrics.round(4).to_string(index=False))
    best_threshold = float(metrics.loc[metrics["f1_score"].idxmax(), "threshold"])
    print(model.classify_prompt("Free entry to win a cash prize. Claim now by calling this number.", best_threshold))
