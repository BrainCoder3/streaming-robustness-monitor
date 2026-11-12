"""Synthetic input degradation with a known schedule.

The stream goes through phases (clean, mild noise, strong noise, clean again) so that we
know exactly when the world changes and can measure how fast the monitor notices."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Phase:
    events: int
    sigma: float


DEFAULT_SCHEDULE = (
    Phase(400, 0.0),   # clean
    Phase(400, 0.30),  # mild noise
    Phase(400, 0.70),  # strong noise
    Phase(400, 0.0),   # back to clean
)


def sigma_at(index: int, schedule: Sequence[Phase] = DEFAULT_SCHEDULE) -> float:
    """Noise level applied to the event number `index` (0-based)."""
    start = 0
    for phase in schedule:
        if index < start + phase.events:
            return phase.sigma
        start += phase.events
    return schedule[-1].sigma


def phase_boundaries(schedule: Sequence[Phase] = DEFAULT_SCHEDULE) -> list[int]:
    """Index of the first event of each phase."""
    bounds, start = [], 0
    for phase in schedule:
        bounds.append(start)
        start += phase.events
    return bounds


def add_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian pixel noise, clipped to the valid [0, 1] range."""
    if sigma <= 0:
        return x.copy()
    return np.clip(x + rng.normal(0.0, sigma, size=x.shape), 0.0, 1.0)
