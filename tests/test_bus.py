from robustmon.bus import InMemoryBus


def test_each_group_has_its_own_offset():
    bus = InMemoryBus()
    for i in range(3):
        bus.publish("t", str(i), {"n": i})
    a = bus.poll(["t"], "group-a")
    b = bus.poll(["t"], "group-b")
    assert [m.value["n"] for m in a] == [0, 1, 2]
    assert [m.value["n"] for m in b] == [0, 1, 2]
    assert bus.poll(["t"], "group-a") == []  # already consumed by this group


def test_poll_respects_max_messages_and_keeps_order():
    bus = InMemoryBus()
    for i in range(5):
        bus.publish("t", str(i), {"n": i})
    first = bus.poll(["t"], "g", max_messages=2)
    rest = bus.poll(["t"], "g", max_messages=10)
    assert [m.offset for m in first] == [0, 1]
    assert [m.offset for m in rest] == [2, 3, 4]


def test_topics_are_independent():
    bus = InMemoryBus()
    bus.publish("a", "k", {"x": 1})
    assert bus.poll(["b"], "g") == []
    assert bus.count("a") == 1 and bus.count("b") == 0
