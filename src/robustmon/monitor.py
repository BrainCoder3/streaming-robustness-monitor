"""Label-free drift monitoring on a stream of predictions.

In production the true label usually arrives late or never, so the main signal is the model's
own confidence. When labels are available (here: in the simulation) the windowed accuracy is
used as a second, stronger signal.

A window is "bad" when its mean confidence falls below `baseline - k * std / sqrt(window)` or its
accuracy falls `accuracy_drop` below the baseline. The monitor raises DEGRADED after
`consecutive` bad windows in a row and RECOVERED after `consecutive` good ones, which avoids
alarms caused by a single unlucky window."""

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .model import predict


@dataclass(frozen=True)
class Baseline:
    conf_mean: float
    conf_std: float
    accuracy: float


@dataclass(frozen=True)
class Alert:
    kind: str  # "DEGRADED" or "RECOVERED"
    window_index: int
    last_event_id: int
    conf_mean: float
    accuracy: float | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_baseline(clf, x_val: np.ndarray, y_val: np.ndarray) -> Baseline:
    """Reference behaviour of the model on clean validation data."""
    confs, correct = [], []
    for features, label in zip(x_val, y_val):
        pred, conf = predict(clf, features)
        confs.append(conf)
        correct.append(pred == int(label))
    return Baseline(float(np.mean(confs)), float(np.std(confs)), float(np.mean(correct)))


@dataclass
class DriftMonitor:
    baseline: Baseline
    window: int = 50
    k_sigma: float = 3.0
    accuracy_drop: float = 0.10
    consecutive: int = 2
    state: str = "OK"
    history: list = field(default_factory=list)
    _buffer: list = field(default_factory=list)
    _bad_streak: int = 0
    _good_streak: int = 0
    _window_index: int = 0

    @property
    def confidence_threshold(self) -> float:
        return self.baseline.conf_mean - self.k_sigma * self.baseline.conf_std / math.sqrt(self.window)

    def update(self, event: dict) -> list[Alert]:
        """Feed one prediction event, return the alerts it triggers (usually none)."""
        self._buffer.append(event)
        if len(self._buffer) < self.window:
            return []
        batch, self._buffer = self._buffer, []
        return self._evaluate(batch)

    def _evaluate(self, batch: list[dict]) -> list[Alert]:
        conf_mean = float(np.mean([e["confidence"] for e in batch]))
        labelled = [e for e in batch if e.get("label") is not None]
        accuracy = float(np.mean([e["pred"] == e["label"] for e in labelled])) if labelled else None

        reasons = []
        if conf_mean < self.confidence_threshold:
            reasons.append(f"confidence {conf_mean:.3f} < {self.confidence_threshold:.3f}")
        if accuracy is not None and accuracy < self.baseline.accuracy - self.accuracy_drop:
            reasons.append(f"accuracy {accuracy:.3f} < {self.baseline.accuracy - self.accuracy_drop:.3f}")
        bad = bool(reasons)

        last_id = int(batch[-1]["id"])
        self.history.append({
            "window_index": self._window_index,
            "last_event_id": last_id,
            "conf_mean": conf_mean,
            "accuracy": accuracy,
            "bad": bad,
            "state": self.state,
        })

        alerts: list[Alert] = []
        if bad:
            self._bad_streak += 1
            self._good_streak = 0
            if self.state == "OK" and self._bad_streak >= self.consecutive:
                self.state = "DEGRADED"
                alerts.append(Alert("DEGRADED", self._window_index, last_id, conf_mean, accuracy, "; ".join(reasons)))
        else:
            self._good_streak += 1
            self._bad_streak = 0
            if self.state == "DEGRADED" and self._good_streak >= self.consecutive:
                self.state = "OK"
                alerts.append(Alert("RECOVERED", self._window_index, last_id, conf_mean, accuracy, "back within baseline"))
        self.history[-1]["state_after"] = self.state
        self._window_index += 1
        return alerts
