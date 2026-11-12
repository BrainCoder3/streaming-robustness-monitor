"""The three stages of the pipeline. Each one only talks to the bus, never to another stage.

    producer  --inputs-->  inference  --predictions-->  monitor  --alerts-->  (anyone)
"""

import numpy as np

from .config import TOPIC_ALERTS, TOPIC_INPUTS, TOPIC_PREDICTIONS
from .model import predict
from .monitor import DriftMonitor
from .noise import DEFAULT_SCHEDULE, Phase, add_noise, sigma_at


class Producer:
    """Replays test images, degraded according to a noise schedule."""

    def __init__(self, bus, x_pool: np.ndarray, y_pool: np.ndarray, schedule=DEFAULT_SCHEDULE, seed: int = 0) -> None:
        self.bus = bus
        self.x_pool, self.y_pool = x_pool, y_pool
        self.schedule = schedule
        self.rng = np.random.default_rng(seed)
        self.sent = 0

    @property
    def total(self) -> int:
        return sum(p.events for p in self.schedule)

    def emit(self, n: int) -> int:
        count = 0
        while count < n and self.sent < self.total:
            i = int(self.rng.integers(len(self.x_pool)))
            sigma = sigma_at(self.sent, self.schedule)
            features = add_noise(self.x_pool[i], sigma, self.rng)
            self.bus.publish(TOPIC_INPUTS, str(self.sent), {
                "id": self.sent,
                "features": [round(float(v), 4) for v in features],
                "label": int(self.y_pool[i]),
                "sigma": sigma,  # kept for the evaluation only, the monitor never reads it
            })
            self.sent += 1
            count += 1
        return count


class InferenceService:
    def __init__(self, bus, clf, group: str = "inference") -> None:
        self.bus, self.clf, self.group = bus, clf, group

    def step(self, max_messages: int = 100, timeout_s: float = 0.0) -> int:
        messages = self.bus.poll([TOPIC_INPUTS], self.group, max_messages, timeout_s)
        for msg in messages:
            event = msg.value
            pred, conf = predict(self.clf, np.asarray(event["features"], dtype=float))
            self.bus.publish(TOPIC_PREDICTIONS, msg.key, {
                "id": event["id"],
                "pred": pred,
                "confidence": conf,
                "label": event.get("label"),
                "sigma": event.get("sigma"),
            })
        return len(messages)


class MonitorService:
    def __init__(self, bus, monitor: DriftMonitor, group: str = "monitor") -> None:
        self.bus, self.monitor, self.group = bus, monitor, group
        self.alerts: list = []

    def step(self, max_messages: int = 100, timeout_s: float = 0.0) -> int:
        messages = self.bus.poll([TOPIC_PREDICTIONS], self.group, max_messages, timeout_s)
        for msg in messages:
            for alert in self.monitor.update(msg.value):
                self.alerts.append(alert)
                self.bus.publish(TOPIC_ALERTS, f"{alert.kind}-{alert.window_index}", alert.to_dict())
        return len(messages)


def run_pipeline(bus, clf, monitor: DriftMonitor, x_pool, y_pool, schedule=DEFAULT_SCHEDULE, seed: int = 0, batch: int = 25):
    """Run the three stages in lockstep inside one process (tests and demo)."""
    producer = Producer(bus, x_pool, y_pool, schedule, seed)
    inference = InferenceService(bus, clf)
    service = MonitorService(bus, monitor)
    while producer.sent < producer.total:
        producer.emit(batch)
        inference.step(max_messages=batch * 2)
        service.step(max_messages=batch * 2)
    while inference.step(500) or service.step(500):
        pass
    return service.alerts


def evaluate_detection(alerts, schedule=DEFAULT_SCHEDULE) -> dict:
    """How well did the monitor follow the known noise schedule?"""
    first_noisy: int | None = None
    start = 0
    for phase in schedule:
        if phase.sigma > 0:
            first_noisy = start
            break
        start += phase.events
    last_noisy_end, start = 0, 0
    for phase in schedule:
        start += phase.events
        if phase.sigma > 0:
            last_noisy_end = start
    degraded = [a for a in alerts if a.kind == "DEGRADED"]
    recovered = [a for a in alerts if a.kind == "RECOVERED"]
    false_alarms = [a for a in degraded if first_noisy is not None and a.last_event_id < first_noisy]
    first_detect = next((a for a in degraded if first_noisy is not None and a.last_event_id >= first_noisy), None)
    first_recover = next((a for a in recovered if a.last_event_id >= last_noisy_end), None)
    return {
        "first_noisy_event": first_noisy,
        "false_alarms_before_noise": len(false_alarms),
        "detection_delay_events": (first_detect.last_event_id - first_noisy + 1) if first_detect else None,
        "recovery_delay_events": (first_recover.last_event_id - last_noisy_end + 1) if first_recover else None,
        "alerts": [a.to_dict() for a in alerts],
    }
