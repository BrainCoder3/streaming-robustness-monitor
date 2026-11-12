from robustmon.bus import InMemoryBus
from robustmon.monitor import DriftMonitor, compute_baseline
from robustmon.noise import DEFAULT_SCHEDULE
from robustmon.services import evaluate_detection, run_pipeline


def run(trained, seed=0):
    clf, x_cal, y_cal, x_stream, y_stream = trained
    monitor = DriftMonitor(compute_baseline(clf, x_cal, y_cal), window=50)
    bus = InMemoryBus()
    alerts = run_pipeline(bus, clf, monitor, x_stream, y_stream, DEFAULT_SCHEDULE, seed)
    return bus, monitor, alerts


def test_every_event_goes_through_the_three_stages(trained):
    bus, _, _ = run(trained)
    total = sum(p.events for p in DEFAULT_SCHEDULE)
    assert bus.count("inputs") == bus.count("predictions") == total


def test_no_false_alarm_on_the_clean_phase(trained):
    _, _, alerts = run(trained)
    assert evaluate_detection(alerts)["false_alarms_before_noise"] == 0


def test_degradation_is_detected_quickly_and_recovery_follows(trained):
    _, _, alerts = run(trained)
    result = evaluate_detection(alerts)
    assert result["detection_delay_events"] is not None and result["detection_delay_events"] <= 200
    assert result["recovery_delay_events"] is not None and result["recovery_delay_events"] <= 200


def test_the_monitor_never_reads_the_noise_level(trained):
    _, monitor, _ = run(trained)
    assert all("sigma" not in w for w in monitor.history)


def test_the_result_does_not_depend_on_one_lucky_seed(trained):
    delays = [evaluate_detection(run(trained, seed=s)[2])["detection_delay_events"] for s in range(5)]
    assert all(d is not None and d <= 250 for d in delays), delays
