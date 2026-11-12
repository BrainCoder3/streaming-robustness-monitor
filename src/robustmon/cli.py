"""Command line entry points.

    python -m robustmon train      train the model and write artifacts/model.joblib
    python -m robustmon demo       run the whole pipeline in one process (no Kafka needed)
    python -m robustmon produce    Kafka: stream degraded images to the `inputs` topic
    python -m robustmon infer      Kafka: read `inputs`, write `predictions`
    python -m robustmon monitor    Kafka: read `predictions`, write `alerts`
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import model
from .bus import InMemoryBus, connect_with_retry
from .config import Settings
from .monitor import DriftMonitor, compute_baseline
from .noise import DEFAULT_SCHEDULE, phase_boundaries
from .services import InferenceService, MonitorService, Producer, evaluate_detection, run_pipeline


def _load_or_train(settings: Settings):
    if Path(settings.model_path).exists():
        clf = model.load(settings.model_path)
        _, (x_test, y_test) = model.train(settings.seed)  # same split, only needed for the data
        return clf, x_test, y_test
    clf, (x_test, y_test) = model.train(settings.seed)
    model.save(clf, settings.model_path)
    return clf, x_test, y_test


def _make_monitor(settings: Settings, clf, x_val, y_val) -> DriftMonitor:
    baseline = compute_baseline(clf, x_val, y_val)
    return DriftMonitor(baseline, settings.window, settings.k_sigma, settings.accuracy_drop, settings.consecutive)


def cmd_train(settings: Settings) -> None:
    clf, (x_test, y_test) = model.train(settings.seed)
    model.save(clf, settings.model_path)
    print(f"model saved to {settings.model_path} (clean test accuracy {clf.score(x_test, y_test):.3f})")


def cmd_demo(settings: Settings, out_dir: str) -> None:
    clf, x_test, y_test = _load_or_train(settings)
    # half of the clean test split calibrates the monitor, the other half feeds the stream
    half = len(x_test) // 2
    monitor = _make_monitor(settings, clf, x_test[:half], y_test[:half])
    bus = InMemoryBus()
    alerts = run_pipeline(bus, clf, monitor, x_test[half:], y_test[half:], DEFAULT_SCHEDULE, settings.seed)
    summary = evaluate_detection(alerts)
    summary["baseline"] = {"conf_mean": monitor.baseline.conf_mean, "conf_std": monitor.baseline.conf_std, "accuracy": monitor.baseline.accuracy}
    summary["confidence_threshold"] = monitor.confidence_threshold
    summary["events"] = bus.count("predictions")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_report(summary)
    _plot(monitor, summary, out / "demo_timeline.png")
    print(f"written: {out / 'demo_summary.json'} and {out / 'demo_timeline.png'}")


def _print_report(summary: dict) -> None:
    print(f"events processed          : {summary['events']}")
    print(f"baseline confidence / acc : {summary['baseline']['conf_mean']:.3f} / {summary['baseline']['accuracy']:.3f}")
    print(f"false alarms before noise : {summary['false_alarms_before_noise']}")
    print(f"detection delay (events)  : {summary['detection_delay_events']}")
    print(f"recovery delay (events)   : {summary['recovery_delay_events']}")
    for a in summary["alerts"]:
        print(f"  [{a['kind']:9}] after event {a['last_event_id']:>4}  {a['reason']}")


def _plot(monitor: DriftMonitor, summary: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping the figure")
        return
    h = monitor.history
    x = [w["last_event_id"] for w in h]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, [w["conf_mean"] for w in h], label="mean confidence (label-free)", color="#7A1F2E")
    ax.plot(x, [w["accuracy"] for w in h], label="accuracy (labels)", color="#444444", linestyle="--")
    ax.axhline(summary["confidence_threshold"], color="#7A1F2E", linestyle=":", linewidth=1, label="confidence threshold")
    bounds = phase_boundaries() + [sum(p.events for p in DEFAULT_SCHEDULE)]
    for i, phase in enumerate(DEFAULT_SCHEDULE):
        if phase.sigma > 0:
            ax.axvspan(bounds[i], bounds[i + 1], color="#7A1F2E", alpha=0.07 + 0.10 * phase.sigma)
            ax.text((bounds[i] + bounds[i + 1]) / 2, 0.02, f"noise σ={phase.sigma}", ha="center", fontsize=8)
    for a in summary["alerts"]:
        ax.axvline(a["last_event_id"], color="black", linewidth=1)
        ax.text(a["last_event_id"], 1.02, a["kind"].lower(), rotation=90, va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("event number")
    ax.set_ylim(0, 1.3)
    ax.set_title("Model behaviour as the input noise changes", pad=10)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def cmd_produce(settings: Settings) -> None:
    clf, x_test, y_test = _load_or_train(settings)
    half = len(x_test) // 2
    bus = connect_with_retry(settings.bootstrap_servers, client_id="producer")
    producer = Producer(bus, x_test[half:], y_test[half:], DEFAULT_SCHEDULE, settings.seed)
    batch = max(1, int(settings.events_per_second // 10))
    print(f"producing {producer.total} events at ~{settings.events_per_second}/s")
    while producer.sent < producer.total:
        producer.emit(batch)
        bus.flush()
        time.sleep(batch / settings.events_per_second)
    print("producer done")


def cmd_infer(settings: Settings) -> None:
    clf = model.load(settings.model_path)
    bus = connect_with_retry(settings.bootstrap_servers, client_id="inference")
    service = InferenceService(bus, clf)
    print("inference service running")
    while True:
        service.step(max_messages=200, timeout_s=0.5)
        bus.flush()


def cmd_monitor(settings: Settings) -> None:
    clf, x_test, y_test = _load_or_train(settings)
    half = len(x_test) // 2
    monitor = _make_monitor(settings, clf, x_test[:half], y_test[:half])
    bus = connect_with_retry(settings.bootstrap_servers, client_id="monitor")
    service = MonitorService(bus, monitor)
    print(f"monitor running (confidence threshold {monitor.confidence_threshold:.3f})")
    shown = 0
    while True:
        service.step(max_messages=200, timeout_s=0.5)
        bus.flush()
        for alert in service.alerts[shown:]:
            print(f"ALERT {alert.kind} after event {alert.last_event_id}: {alert.reason}", flush=True)
        shown = len(service.alerts)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="robustmon", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "demo", "produce", "infer", "monitor"])
    parser.add_argument("--out", default="docs", help="output folder for the demo (default: docs)")
    args = parser.parse_args(argv)
    settings = Settings()
    {
        "train": lambda: cmd_train(settings),
        "demo": lambda: cmd_demo(settings, args.out),
        "produce": lambda: cmd_produce(settings),
        "infer": lambda: cmd_infer(settings),
        "monitor": lambda: cmd_monitor(settings),
    }[args.command]()


if __name__ == "__main__":
    main()
