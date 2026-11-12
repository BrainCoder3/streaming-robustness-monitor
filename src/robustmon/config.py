"""Shared settings. Everything can be overridden with environment variables so the
same code runs locally, in the tests and inside Docker Compose."""

import os
from dataclasses import dataclass

TOPIC_INPUTS = "inputs"
TOPIC_PREDICTIONS = "predictions"
TOPIC_ALERTS = "alerts"


@dataclass(frozen=True)
class Settings:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    model_path: str = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    seed: int = int(os.getenv("SEED", "0"))
    # monitor
    window: int = int(os.getenv("MONITOR_WINDOW", "50"))
    k_sigma: float = float(os.getenv("MONITOR_K_SIGMA", "3.0"))
    accuracy_drop: float = float(os.getenv("MONITOR_ACCURACY_DROP", "0.10"))
    consecutive: int = int(os.getenv("MONITOR_CONSECUTIVE", "2"))
    # producer
    events_per_second: float = float(os.getenv("EVENTS_PER_SECOND", "50"))
    total_events: int = int(os.getenv("TOTAL_EVENTS", "1600"))
