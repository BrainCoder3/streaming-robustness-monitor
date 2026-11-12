"""Runs only against a real broker: KAFKA_BOOTSTRAP=localhost:9092 pytest tests/test_kafka_integration.py
(start it with `docker compose up -d kafka`). Skipped everywhere else, including the CI."""

import os
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.getenv("KAFKA_BOOTSTRAP"), reason="needs a running Kafka broker")


def test_message_round_trip_through_a_real_broker():
    from robustmon.bus import connect_with_retry

    topic = f"test-{uuid.uuid4().hex[:8]}"
    bus = connect_with_retry(os.environ["KAFKA_BOOTSTRAP"], attempts=5, delay_s=1.0, client_id="it")
    bus.publish(topic, "k", {"hello": "kafka"})
    bus.flush()
    received = []
    deadline = time.time() + 20
    while not received and time.time() < deadline:
        received = bus.poll([topic], f"group-{topic}", timeout_s=1.0)
    assert received and received[0].value == {"hello": "kafka"}
