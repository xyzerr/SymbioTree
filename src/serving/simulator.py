from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple
import numpy as np
from math import ceil, floor
import random

from system.config import ModelConfig

from serving.request import Request
from serving.metrics import SimulationResult
from serving.config import SimulatorConfig

from parallelism.pnode import BasicParallelismNode
from parallelism.ptree import ParallelismTree

@dataclass(frozen=True)
class Caps:
    """Admission caps for the current moment."""
    cap_total: int          # max len(active)
    cap_normal: int         # max number of normal in active (reserve mode), large in preempt
    priority_present: bool  # queue_hi non-empty OR priority already in active

class Simulator:
    def __init__(self,
                 sim_cfg: SimulatorConfig,
                 model_cfg: ModelConfig,
                 req_prob: List[float],
                 ptree:ParallelismTree):
        self.sim_cfg = sim_cfg
        random.seed(self.sim_cfg.seed)
        self.model_cfg = model_cfg
        self.req_prob = req_prob
        self.ptree = ptree
        self._validate_sim_cfg(sim_cfg)
        self.rng = np.random.default_rng(sim_cfg.seed)

        self.t: float = 0.0
        self.device_compute_ms = defaultdict(float)
        self.device_comm_ms = defaultdict(float)
        self.device_busy_wo_overlap_ms = defaultdict(float)
        self.device_busy_wi_overlap_ms = defaultdict(float)
        self.perf_profile_history = None
        self.next_arrival: float = self._exp_interarrival()

        # Two-level FIFO queues
        self.queue_hi: List[Request] = []
        self.queue_lo: List[Request] = []

        self.active: List[Request] = []
        self.finished: List[Request] = []

        self.req_counter: int = 0
        self.busy_time: float = 0.0

        self.perf_num_counter: int = 0
        self.perf_result_history: float = 0.0

    # -------------------------
    # Basic utils / validation
    # -------------------------
    @staticmethod
    def _validate_sim_cfg(sim_cfg: SimulatorConfig) -> None:
        if sim_cfg.lam <= 0:
            raise ValueError("lam must be > 0")
        if sim_cfg.t_end <= 0:
            raise ValueError("t_end must be > 0")
        if sim_cfg.max_batch_hi <= 0 or sim_cfg.max_batch_lo <= 0:
            raise ValueError("max_batch_hi/max_batch_lo must be > 0")
        if sim_cfg.max_wait_s < 0 or sim_cfg.max_wait_hi_s < 0:
            raise ValueError("max_wait_s/max_wait_hi_s must be >= 0")
        if not (0.0 <= sim_cfg.priority_ratio <= 1.0):
            raise ValueError("priority_ratio must be in [0, 1]")
        if sim_cfg.mode not in ("preempt", "reserve"):
            raise ValueError("mode must be 'preempt' or 'reserve'")
        if sim_cfg.reserve_hi < 0:
            raise ValueError("reserve_hi must be >= 0")
        if sim_cfg.mode == "reserve" and sim_cfg.reserve_hi > sim_cfg.max_batch_lo:
            raise ValueError("reserve_hi must be <= max_batch_lo in reserve mode")

    def _log(self, msg: str) -> None:
        if self.sim_cfg.verbose:
            print(f"[t={self.t:.6f}] {msg}")

    def _exp_interarrival(self) -> float:
        return float(self.rng.exponential(1.0 / self.sim_cfg.lam))

    # -------------------------
    # Workload sampling
    # -------------------------
    def _sample_nonneg_int_normal(self, mean: float, std: float, min_value: int = 0) -> int:
        x = float(self.rng.normal(mean, std))
        x = int(np.round(x))
        return max(x, min_value)

    def _sample_req_tokens(self) -> Tuple[int, int, int]:
        """Mixture of 4 request classes (A/B/C/D) exactly as original code."""

        # cls = int(self.rng.choice(self.sim_cfg.req_type_num, p=[0.5, 0.2, 0.2, 0.1]))
        # if cls == 0:
        #     pm, gm, ps, gs = 128, 128, 32.0, 32.0 # prompt_mean, gen_mean, prompt_std, gen_std
        # elif cls == 1:
        #     pm, gm, ps, gs = 128, 2048, 32.0, 1024.0 # prompt_mean, gen_mean, prompt_std, gen_std
        # elif cls == 2:
        #     pm, gm, ps, gs = 2048, 128, 1024.0, 32.0 # prompt_mean, gen_mean, prompt_std, gen_std
        # else:
        #     pm, gm, ps, gs = 2048, 2048, 1024.0, 1024.0 # prompt_mean, gen_mean, prompt_std, gen_std

        cls = int(self.rng.choice(self.sim_cfg.req_type_num, p=self.req_prob))
        if cls == 0:
            pm, gm, ps, gs = 256, 256, 128.0, 128.0 # prompt_mean, gen_mean, prompt_std, gen_std
        elif cls == 1:
            # pm, gm, ps, gs = 1024, 1024, 256.0, 256.0 # prompt_mean, gen_mean, prompt_std, gen_std
            # pm, gm, ps, gs = 4096, 4096, 2048.0, 2048.0 # prompt_mean, gen_mean, prompt_std, gen_std
            pm, gm, ps, gs = 1024, 1024, 512.0, 512.0 # prompt_mean, gen_mean, prompt_std, gen_std
        else:
            # pm, gm, ps, gs = 2048, 2048, 1024.0, 1024.0 # prompt_mean, gen_mean, prompt_std, gen_std
            # pm, gm, ps, gs = 10240, 10240, 2048.0, 2048.0 # prompt_mean, gen_mean, prompt_std, gen_std
            # pm, gm, ps, gs = 8192, 8192, 2048.0, 2048.0 # prompt_mean, gen_mean, prompt_std, gen_std
            pm, gm, ps, gs = 4096, 4096, 2048.0, 2048.0 # prompt_mean, gen_mean, prompt_std, gen_std

        prompt_tokens = self._sample_nonneg_int_normal(pm, ps, min_value=1)
        target_gen_tokens = self._sample_nonneg_int_normal(gm, gs, min_value=1)
        assert 0 <= cls <= self.sim_cfg.req_type_num
        return cls, prompt_tokens, target_gen_tokens

    def _new_request(self, arrival_time: float) -> Request:
        cls, prompt_tokens, target_gen_tokens = self._sample_req_tokens()
        is_prio = bool(self.rng.random() < self.sim_cfg.priority_ratio)
        r = Request(
            req_id=self.req_counter,
            req_type=cls,
            arrival_time=arrival_time,
            prompt_tokens=prompt_tokens,
            target_gen_tokens=target_gen_tokens,
            is_priority=is_prio
        )
        self.req_counter += 1
        return r

    def _enqueue_arrivals_up_to(self, t_limit: float) -> None:
        """Enqueue all arrivals with arrival_time <= t_limit."""
        while self.next_arrival <= t_limit:
            r = self._new_request(self.next_arrival)
            if r.is_priority:
                self.queue_hi.append(r)
            else:
                self.queue_lo.append(r)
            self.next_arrival += self._exp_interarrival()

    # -------------------------
    # Priority presence & caps
    # -------------------------
    def _active_counts(self) -> Tuple[int, int]:
        pr = sum(1 for r in self.active if r.is_priority)
        nr = len(self.active) - pr
        return pr, nr

    def _priority_present(self) -> bool:
        if self.queue_hi:
            return True
        return any(r.is_priority for r in self.active)

    def _compute_caps(self) -> Caps:
        """
        根据当前情况，决定 active 队列的容量。返回 Caps 变量。
        对于 preempt 模式，容量为 cap_total，cap_normal 无意义。
        """
        prio_present = self._priority_present()
        prio_active, _ = self._active_counts()

        if self.sim_cfg.mode == "preempt":
            if prio_present:
                cap_total = self.sim_cfg.max_batch_hi
            else:
                cap_total = self.sim_cfg.max_batch_lo
            cap_normal = 10**9  # effectively unlimited; total cap governs
            return Caps(cap_total=cap_total, cap_normal=cap_normal, priority_present=prio_present)

        # reserve mode (hard reserve)
        cap_total = self.sim_cfg.max_batch_lo
        cap_normal = max(self.sim_cfg.max_batch_lo - self.sim_cfg.reserve_hi, 0)
        return Caps(cap_total=cap_total, cap_normal=cap_normal, priority_present=prio_present)

    # -------------------------
    # Rebalance = preempt + admit
    # -------------------------
    def _preempt_if_needed(self, caps: Caps) -> None:
        if self.sim_cfg.mode != "preempt":
            return
        if not self.queue_hi:
            return

        max_evict = 1   # 每次最多 evict 1 个，当 active 全是 lo 时可能会导致 hi 的排队延迟增加
        # max_evict = max(1, len(self.active) - caps.cap_total) # 全是 lo 时来一个 hi，则evict大量lo，否则evict 1个。

        evicted = 0

        i = len(self.active) - 1
        while i >= 0:
            if evicted >= max_evict:
                break
            if len(self.active) < caps.cap_total:
                break

            if not self.active[i].is_priority:
                r = self.active.pop(i)
                self.queue_lo.insert(0, r)
                evicted += 1
            i -= 1

    def _admit_priority(self, caps: Caps) -> None:
        while len(self.active) < caps.cap_total and self.queue_hi:
            r = self.queue_hi.pop(0)
            if r.start_time is None:
                r.start_time = self.t
            self.active.append(r)

    def _admit_normal(self, caps: Caps) -> None:
        # In reserve mode, enforce normal cap in active.
        _, nr = self._active_counts()
        while len(self.active) < caps.cap_total and self.queue_lo and nr < caps.cap_normal:
            r = self.queue_lo.pop(0)
            if r.start_time is None:
                r.start_time = self.t
            self.active.append(r)
            nr += 1

    def _rebalance(self, caps: Caps) -> None:
        """Unified admission control."""
        self._preempt_if_needed(caps)
        self._admit_priority(caps)
        self._admit_normal(caps)

    # -------------------------
    # Idle dispatch policy
    # -------------------------
    def _idle_should_dispatch_now(self, caps: Caps) -> bool:
        """When active is empty, decide if we should dispatch immediately (vs wait for arrivals/deadline)."""
        if not (self.queue_hi or self.queue_lo):
            return False

        target = caps.cap_total
        if (len(self.queue_hi) + len(self.queue_lo)) >= target:
            return True

        max_wait = self.sim_cfg.max_wait_hi_s if self.queue_hi else self.sim_cfg.max_wait_s
        if max_wait == 0.0:
            return True

        oldest = (self.queue_hi[0] if self.queue_hi else self.queue_lo[0]).arrival_time
        deadline = oldest + max_wait
        return self.t >= deadline

    def _idle_advance_time(self) -> None:
        """Advance time to next arrival or batching deadline when GPU is idle, and we didn't dispatch."""
        if not (self.queue_hi or self.queue_lo):
            # Jump to next arrival, create exactly one request at that time.
            self.t = self.next_arrival
            r = self._new_request(self.t)
            (self.queue_hi if r.is_priority else self.queue_lo).append(r)
            self.next_arrival += self._exp_interarrival()
            return

        max_wait = self.sim_cfg.max_wait_hi_s if self.queue_hi else self.sim_cfg.max_wait_s
        oldest = (self.queue_hi[0] if self.queue_hi else self.queue_lo[0]).arrival_time
        deadline = oldest + max_wait

        self.t = min(self.next_arrival, deadline)
        self._enqueue_arrivals_up_to(self.t)

    # -------------------------
    # Step simulation
    # -------------------------
    def _decode_one_step(self, begin_node: BasicParallelismNode) -> None:
        """One decode step
        """
        assert self.active

        # (0) count and print
        self.perf_num_counter += 1

        # (1) time advance
        evaluation_cycle_stride = 1000
        sub_batch_num = self.sim_cfg.sub_batch_num
        if self.perf_num_counter % evaluation_cycle_stride == 1:
            if not self.sim_cfg.use_pp_sub_batch or sub_batch_num == 1:
                step_profile = self.ptree.run_from_begin_node(begin_node,
                                                              self.active,
                                                              False,
                                                              self.sim_cfg.use_mp_sub_batch,
                                                              return_profile=True)
                step_time_s = step_profile.total_time_ms / 1000.0
                # print("batch_size:", len(self.active), "step_time_s:", step_time_s)
            else:
                sub_batch_size = max(1, ceil(len(self.active) / sub_batch_num))
                random_indexes = random.sample(range(len(self.active)), sub_batch_size)
                sub_batch_active = [self.active[i] for i in random_indexes]
                sub_batch_profile = self.ptree.run_from_begin_node(begin_node,
                                                                   sub_batch_active,
                                                                   True,
                                                                   self.sim_cfg.use_mp_sub_batch,
                                                                   return_profile=True)
                sub_batch_step_time_s = sub_batch_profile.total_time_ms / 1000.0
                step_time_s = sub_batch_num * sub_batch_step_time_s
                # step_time_s = sub_batch_step_time_s
                step_profile = sub_batch_profile.scaled(sub_batch_num)
                # step_profile = sub_batch_profile.scaled(1)
                # update delay time for requests
                for idx, req in enumerate(self.active):
                    if req in sub_batch_active:
                        continue
                    offset_cycle_num = floor(idx / sub_batch_size)
                    acc_times =  min(evaluation_cycle_stride, (req.target_gen_tokens - req.gen_tokens))
                    req.accum_delay_time += sub_batch_step_time_s * offset_cycle_num * acc_times
            #
            self.perf_result_history = step_time_s
            self.perf_profile_history = step_profile
        else:
            step_time_s = self.perf_result_history
            step_profile = self.perf_profile_history
        # print(step_time_s)

        if step_profile is not None:
            for device_name, value in step_profile.per_device_compute_ms.items():
                self.device_compute_ms[device_name] += value
            for device_name, value in step_profile.per_device_comm_ms.items():
                self.device_comm_ms[device_name] += value
            for device_name, value in step_profile.per_device_busy_wo_overlap_ms.items():
                self.device_busy_wo_overlap_ms[device_name] += value
            for device_name, value in step_profile.per_device_busy_wi_overlap_ms.items():
                self.device_busy_wi_overlap_ms[device_name] += value

        self.t += step_time_s
        self.busy_time += step_time_s

        # (2) token generation
        for r in self.active:
            r.gen_tokens += 1

        # (3) retire finished
        still: List[Request] = []
        for r in self.active:
            if r.done():
                r.finish_time = self.t
                self.finished.append(r)
            else:
                still.append(r)
        self.active = still

        # (4) arrivals
        self._enqueue_arrivals_up_to(self.t)

        # (5) fill/admit after retire
        caps = self._compute_caps()
        self._rebalance(caps)

    def run(self, begin_node:BasicParallelismNode) -> SimulationResult:
        while self.t < self.sim_cfg.t_end:
            if not self.active:
                caps = self._compute_caps()
                if self._idle_should_dispatch_now(caps):
                    self._rebalance(caps)  # dispatch by admitting under unified rules
                else:
                    self._idle_advance_time()
                    continue

            self._decode_one_step(begin_node)
            # return

            if self.t >= self.sim_cfg.t_end:
                break

        return SimulationResult(t_end=self.sim_cfg.t_end,
                                busy_time=self.busy_time,
                                finished=self.finished,
                                running=self.active,
                                device_compute_ms=dict(self.device_compute_ms),
                                device_comm_ms=dict(self.device_comm_ms),
                                device_busy_wo_overlap_ms=dict(self.device_busy_wo_overlap_ms),
                                device_busy_wi_overlap_ms=dict(self.device_busy_wi_overlap_ms))