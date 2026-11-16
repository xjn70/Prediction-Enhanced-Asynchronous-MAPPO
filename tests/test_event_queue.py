"""Heap ordering and clock invariants for the event queue."""

import pytest

from uwsn.environment import EventQueue, EventType, SimulationEvent


def test_time_then_priority_then_sequence_orders_events():
    queue = EventQueue()
    mission_end = queue.push(5, EventType.MISSION_END)
    wait = queue.push(5, EventType.WAIT_COMPLETE, 1, {"a": 1})
    ready = queue.push(5, EventType.AUV_DECISION_READY, 0, {"b": 2})
    packet = queue.push(5, EventType.CH_DATA_GENERATION, 0)
    initial = queue.push(0, EventType.AUV_DECISION_READY, 0)

    assert [queue.pop() for _ in range(5)] == [initial, packet, wait, ready, mission_end]
    assert queue.current_time_s == 5
    assert queue.pop() is None
    assert queue.is_empty()


def test_repeated_peek_leaves_clock_and_contents_unchanged():
    queue = EventQueue()
    event = queue.push(7, EventType.WAIT_COMPLETE, 0)
    assert queue.peek() is queue.peek() is event
    assert queue.current_time_s == 0
    assert len(queue) == 1
    assert queue.pop() is event
    assert len(queue) == 0
    assert queue.peek() is None


@pytest.mark.parametrize("time_s", [-1, float("inf"), float("-inf"), float("nan"), True])
def test_invalid_times_are_rejected(time_s):
    with pytest.raises(ValueError, match="time_s"):
        SimulationEvent(time_s, EventType.WAIT_COMPLETE, 0, 0)
    queue = EventQueue()
    with pytest.raises(ValueError, match="time_s"):
        queue.push(time_s, EventType.WAIT_COMPLETE, 0)
    assert queue.is_empty()


def test_clock_cannot_skip_events_or_schedule_in_the_past():
    queue = EventQueue()
    event = queue.push(8, EventType.WAIT_COMPLETE, 0)
    queue.advance_to(5)
    with pytest.raises(ValueError, match="past"):
        queue.push(4, EventType.WAIT_COMPLETE, 0)
    with pytest.raises(ValueError, match="backwards"):
        queue.advance_to(4)
    with pytest.raises(ValueError, match="pending event"):
        queue.advance_to(9)
    queue.advance_to(8)
    assert queue.pop() is event

    with pytest.raises(ValueError, match="earlier priority"):
        queue.push(8, EventType.CH_DATA_GENERATION, 0)
    same_time = queue.push(8, EventType.WAIT_COMPLETE, 0)
    assert queue.pop() is same_time
    assert same_time.sequence_id > event.sequence_id


def test_clear_discards_pending_entries_and_retains_clock_and_sequence():
    queue = EventQueue()
    queue.advance_to(2)
    discarded = queue.push(3, EventType.MISSION_END)
    queue.clear()
    assert queue.is_empty()
    assert queue.current_time_s == 2
    assert queue.push(2, EventType.WAIT_COMPLETE, 0).sequence_id > discarded.sequence_id
