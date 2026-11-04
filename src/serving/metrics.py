from __future__ import annotations
from dataclasses import dataclass, field
from serving.request import Request


@dataclass
class SimulationResult:
    # Time accounting
    t_end: float
    busy_time: float

    # Finished and running requests
    finished: list[Request]
    running: list[Request]

    # Per-device raw timing statistics accumulated across the whole simulation.
    # Unit: milliseconds.
    device_compute_ms: dict[str, float] = field(default_factory=dict)
    device_comm_ms: dict[str, float] = field(default_factory=dict)
    device_busy_wo_overlap_ms: dict[str, float] = field(default_factory=dict)
    device_busy_wi_overlap_ms: dict[str, float] = field(default_factory=dict)
