from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

# ******************** Model Part ********************
@dataclass
class ModelConfig:
    model_name: str
    hidden_size: int
    query_head_num: int
    kv_head_num: int
    intermediate_size: int
    layer_num: int
    head_dim: int = None
    kv_hidden_size: int = None

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.query_head_num
        self.kv_hidden_size = self.head_dim * self.kv_head_num

ModelConfigs = {
    0: ModelConfig("llama3-8B", 4096, 32, 8, 14336, 32),
    1: ModelConfig("qwen-14B", 5120, 40, 8, 13824, 48),
    2: ModelConfig("qwen-32B", 5120, 40, 8, 27848, 64),
    3: ModelConfig("llama3-70B", 8192, 64, 8, 28672, 80)
}

# ******************** System Part ********************
Mode = Literal["preempt", "reserve"]

@dataclass
class SystemConfig:
    hcase_index: int
    pcase_index: int

    req_type_num: int
    lam: float                 # req/s
    t_end: float               # seconds
    req_dist: list[float]
    # Pipline parallelism and sub batch num
    use_pp_sub_batch: bool = False
    sub_batch_num: int = 1

    # Module Parallelism
    use_mp_sub_batch: bool = False

    # QoS / priority
    priority_ratio: float = 0.05
    mode: Mode = "preempt"     # "preempt" or "reserve"

    # Capacity controls
    max_batch_hi: int = 16     # preempt mode cap_total when priority_present
    max_batch_lo: int = 256    # normal-only cap_total; also total cap in reserve mode
    reserve_hi: int = 16       # reserved slots for priority in reserve mode (hard reserve)

    # Exploration-aligned memory feasibility knobs
    peak_seq_len: int = 2048
    runtime_reserve_ratio: float = 0.0

    # Idle batching wait (seconds)
    max_wait_s: float = 0.0       # normal-only idle wait
    max_wait_hi_s: float = 0.0    # priority-present idle wait (default 0 protects latency)

    seed: int = 0
    verbose: bool = False