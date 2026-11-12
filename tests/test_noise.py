import numpy as np

from robustmon.noise import DEFAULT_SCHEDULE, Phase, add_noise, phase_boundaries, sigma_at


def test_sigma_follows_the_schedule():
    schedule = (Phase(3, 0.0), Phase(2, 0.5))
    assert [sigma_at(i, schedule) for i in range(5)] == [0.0, 0.0, 0.0, 0.5, 0.5]


def test_sigma_after_the_end_keeps_the_last_phase():
    schedule = (Phase(2, 0.0), Phase(2, 0.7))
    assert sigma_at(100, schedule) == 0.7


def test_phase_boundaries():
    assert phase_boundaries(DEFAULT_SCHEDULE) == [0, 400, 800, 1200]


def test_no_noise_returns_an_unchanged_copy():
    x = np.linspace(0, 1, 64)
    out = add_noise(x, 0.0, np.random.default_rng(0))
    assert np.array_equal(out, x) and out is not x


def test_noise_is_clipped_and_reproducible():
    x = np.linspace(0, 1, 64)
    a = add_noise(x, 0.8, np.random.default_rng(1))
    b = add_noise(x, 0.8, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert not np.array_equal(a, x)
