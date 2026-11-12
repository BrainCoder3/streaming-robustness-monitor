"""Message bus abstraction.

`InMemoryBus` is used by the tests and by the one-process demo. `KafkaBus` talks to a real
Apache Kafka broker (Docker Compose). Both expose the same two methods, so the services do
not know which one they run on."""

import json
import time
from collections import defaultdict
from typing import NamedTuple, Protocol, Sequence


class Message(NamedTuple):
    topic: str
    key: str
    value: dict
    offset: int


class Bus(Protocol):
    def publish(self, topic: str, key: str, value: dict) -> None: ...

    def poll(self, topics: Sequence[str], group: str, max_messages: int = 100, timeout_s: float = 0.0) -> list[Message]: ...


class InMemoryBus:
    """A tiny Kafka-like log: append-only topics and one committed offset per (group, topic)."""

    def __init__(self) -> None:
        self._log: dict[str, list[Message]] = defaultdict(list)
        self._offsets: dict[tuple[str, str], int] = defaultdict(int)

    def publish(self, topic: str, key: str, value: dict) -> None:
        self._log[topic].append(Message(topic, key, value, len(self._log[topic])))

    def poll(self, topics, group, max_messages=100, timeout_s=0.0):
        out: list[Message] = []
        for topic in topics:
            pos = self._offsets[(group, topic)]
            batch = self._log[topic][pos:pos + max(0, max_messages - len(out))]
            self._offsets[(group, topic)] = pos + len(batch)
            out.extend(batch)
        return out

    def count(self, topic: str) -> int:
        return len(self._log[topic])


class KafkaBus:
    """Thin wrapper over kafka-python. Imported lazily so the tests do not need the client."""

    def __init__(self, bootstrap_servers: str, client_id: str = "robustmon") -> None:
        from kafka import KafkaProducer  # noqa: WPS433

        self._bootstrap = bootstrap_servers
        self._client_id = client_id
        self._consumers: dict[tuple[str, tuple[str, ...]], object] = {}
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id,
            key_serializer=lambda k: k.encode("utf-8"),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            linger_ms=5,
        )

    def publish(self, topic: str, key: str, value: dict) -> None:
        self._producer.send(topic, key=key, value=value)

    def flush(self) -> None:
        self._producer.flush()

    def poll(self, topics, group, max_messages=100, timeout_s=0.5):
        from kafka import KafkaConsumer  # noqa: WPS433

        cache_key = (group, tuple(topics))
        if cache_key not in self._consumers:
            self._consumers[cache_key] = KafkaConsumer(
                *topics,
                bootstrap_servers=self._bootstrap,
                group_id=group,
                client_id=f"{self._client_id}-{group}",
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )
        consumer = self._consumers[cache_key]
        records = consumer.poll(timeout_ms=int(timeout_s * 1000), max_records=max_messages)
        out: list[Message] = []
        for batch in records.values():
            for rec in batch:
                key = rec.key.decode("utf-8") if rec.key else ""
                out.append(Message(rec.topic, key, rec.value, rec.offset))
        return out


def connect_with_retry(bootstrap_servers: str, attempts: int = 30, delay_s: float = 2.0, client_id: str = "robustmon") -> KafkaBus:
    """Kafka needs a few seconds to be ready in Docker Compose: retry instead of crashing."""
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return KafkaBus(bootstrap_servers, client_id=client_id)
        except Exception as exc:  # noqa: BLE001 - the client raises several exception types
            last = exc
            time.sleep(delay_s)
    raise RuntimeError(f"Kafka not reachable at {bootstrap_servers}") from last
