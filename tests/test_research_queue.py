from atsf.research_queue import ResearchQueue, ResearchReason, ResearchRequest


def test_queue_orders_by_priority_then_id():
    queue = ResearchQueue()
    queue.enqueue(ResearchRequest("b", "s1", ResearchReason.DEGRADED, 2))
    queue.enqueue(ResearchRequest("a", "s2", ResearchReason.DIVERSIFICATION, 1))
    assert [item.request_id for item in queue.pending()] == ["a", "b"]


def test_duplicate_request_keeps_higher_priority():
    queue = ResearchQueue()
    queue.enqueue(ResearchRequest("x", "s1", ResearchReason.DEGRADED, 5))
    queue.enqueue(ResearchRequest("x", "s1", ResearchReason.HALTED, 1))
    assert queue.pending()[0].priority == 1


def test_complete_removes_request():
    queue = ResearchQueue()
    queue.enqueue(ResearchRequest("x", None, ResearchReason.CAPACITY, 3))
    completed = queue.complete("x")
    assert completed.request_id == "x"
    assert len(queue) == 0
