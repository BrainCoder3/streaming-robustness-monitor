import pytest

from robustmon.monitor import Baseline, DriftMonitor

BASELINE = Baseline(conf_mean=0.9, conf_std=0.1, accuracy=0.95)


def feed(monitor, start, n, confidence, correct=True):
    alerts = []
    for i in range(start, start + n):
        alerts += monitor.update({"id": i, "confidence": confidence, "pred": 1, "label": 1 if correct else 0})
    return alerts


@pytest.fixture
def monitor():
    return DriftMonitor(BASELINE, window=10, k_sigma=3.0, accuracy_drop=0.10, consecutive=2)


def test_threshold_uses_the_standard_error_of_the_window(monitor):
    assert monitor.confidence_threshold == pytest.approx(0.9 - 3 * 0.1 / 10 ** 0.5)


def test_no_alert_on_baseline_like_traffic(monitor):
    assert feed(monitor, 0, 100, 0.9) == []
    assert monitor.state == "OK"


def test_a_single_bad_window_does_not_raise_an_alert(monitor):
    feed(monitor, 0, 10, 0.9)
    assert feed(monitor, 10, 10, 0.5) == []
    assert feed(monitor, 20, 10, 0.9) == []


def test_two_bad_windows_raise_degraded_then_recovery(monitor):
    feed(monitor, 0, 10, 0.9)
    alerts = feed(monitor, 10, 20, 0.5)
    assert [a.kind for a in alerts] == ["DEGRADED"]
    assert alerts[0].last_event_id == 29
    recovered = feed(monitor, 30, 20, 0.9)
    assert [a.kind for a in recovered] == ["RECOVERED"]
    assert monitor.state == "OK"


def test_accuracy_alone_can_trigger_the_alert(monitor):
    # confidence stays high but the answers are wrong
    alerts = feed(monitor, 0, 20, 0.9, correct=False)
    assert [a.kind for a in alerts] == ["DEGRADED"]
    assert "accuracy" in alerts[0].reason


def test_alert_is_not_repeated_while_degraded(monitor):
    alerts = feed(monitor, 0, 100, 0.5)
    assert len(alerts) == 1


def test_works_without_labels(monitor):
    alerts = []
    for i in range(20):
        alerts += monitor.update({"id": i, "confidence": 0.5, "pred": 1, "label": None})
    assert [a.kind for a in alerts] == ["DEGRADED"]
    assert alerts[0].accuracy is None
