"""A small image classifier (scikit-learn digits, 8x8 pixels).

The point of the project is the streaming and monitoring layer, not the model, so the
model is deliberately simple, fast to train and fully reproducible."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_data(seed: int = 0):
    digits = load_digits()
    x = digits.data / 16.0  # pixel values in [0, 1]
    y = digits.target
    return train_test_split(x, y, test_size=0.4, random_state=seed, stratify=y)


def train(seed: int = 0):
    x_train, x_test, y_train, y_test = load_data(seed)
    clf = LogisticRegression(max_iter=3000)
    clf.fit(x_train, y_train)
    return clf, (x_test, y_test)


def save(clf, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)


def load(path: str):
    return joblib.load(path)


def predict(clf, features: np.ndarray) -> tuple[int, float]:
    """Predicted label and confidence (highest class probability) for one sample."""
    probs = clf.predict_proba(features.reshape(1, -1))[0]
    best = int(np.argmax(probs))
    return int(clf.classes_[best]), float(probs[best])
