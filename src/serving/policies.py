from __future__ import annotations
from typing import List
from .request import Request


# def build_batch_fifo(queue: List[Request], max_batch: int) -> List[Request]:
#     """Pop up to max_batch requests from the head of queue (FIFO)."""
#     batch: List[Request] = []
#     while queue and len(batch) < max_batch:
#         batch.append(queue.pop(0))
#     return batch
#
#
# def fill_batch_fifo(active: List[Request], queue: List[Request], max_batch: int) -> None:
#     """Fill empty slots in active from queue head (FIFO)."""
#     while queue and len(active) < max_batch:
#         active.append(queue.pop(0))


def build_batch_priority(queue_hi: List[Request], queue_lo: List[Request], max_batch: int) -> List[Request]:
    """Build a batch preferring priority queue first (FIFO within each queue)."""
    batch: List[Request] = []
    while queue_hi and len(batch) < max_batch:
        batch.append(queue_hi.pop(0))
    while queue_lo and len(batch) < max_batch:
        batch.append(queue_lo.pop(0))
    return batch


def fill_batch_priority(active: List[Request], queue_hi: List[Request], queue_lo: List[Request], max_batch: int) -> None:
    """Fill empty slots in active, preferring priority queue first."""
    while len(active) < max_batch and (queue_hi or queue_lo):
        if queue_hi:
            active.append(queue_hi.pop(0))
        else:
            active.append(queue_lo.pop(0))
