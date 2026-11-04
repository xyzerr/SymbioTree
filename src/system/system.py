from copy import deepcopy
import numpy as np

from system.config import SystemConfig, ModelConfig
from serving.simulator import Simulator
from parallelism.ptree import ParallelismTree
from hardware.htree import HardwareTree
from system.metrics import summarize_metrics, summarize_metrics_data
from exploration.decoder import RootInit
from exploration.feasibility import (
    FeasibilityConfig,
    compute_feasible_batch_caps_from_root,
    compute_subgraph_memory_stats_from_root,
)


class System:
    def __init__(self, sys_cfg: SystemConfig, model_cfg: ModelConfig):
        self.sys_cfg = sys_cfg
        self.model_cfg = model_cfg

        hcase_idx = sys_cfg.hcase_index
        self.htree = HardwareTree(hcase_idx)

        pcase_idx = sys_cfg.pcase_index
        self.ptree = ParallelismTree(sys_cfg, model_cfg, self.htree, case_idx=pcase_idx)

        # self.req_prob = [0.0, 1.0, 0.0]
        self.req_prob = sys_cfg.req_dist
        self.last_subgraph_batch_info = []

    # ---------------------------------------------------------------------
    # Memory-feasibility helpers
    # ---------------------------------------------------------------------
    def _build_mem_cap_by_device_gb(self):
        return {
            int(d.idx): float(d.meta["mem_cap"])
            for d in self.htree.devices
            if d.meta.get("mem_cap") is not None
        }

    def _build_bytes_by_device(self):
        return {
            int(d.idx): int(d.meta.get("byte", 2))
            for d in self.htree.devices
        }

    def _build_root_init_for_feasibility(self):
        return RootInit(
            dp_attr=[[0.0, 1.0] for _ in range(self.sys_cfg.req_type_num)],
            pp_attr=[0, self.model_cfg.layer_num - 1],
            tp_attr=[0.0, 1.0],
        )

    def _build_feasibility_cfg(self):
        mem_cap_by_device_gb = self._build_mem_cap_by_device_gb()
        if not mem_cap_by_device_gb:
            return None

        return FeasibilityConfig(
            model_cfg=self.model_cfg,
            root_init=self._build_root_init_for_feasibility(),
            mem_cap_by_device_gb=mem_cap_by_device_gb,
            bytes_by_device=self._build_bytes_by_device(),
            peak_seq_len=int(getattr(self.sys_cfg, "peak_seq_len", 2048)),
            runtime_reserve_ratio=float(getattr(self.sys_cfg, "runtime_reserve_ratio", 0.0)),
            attach_hardware_leaves=False,
        )

    def _compute_subgraph_batch_constraints(self):
        feasibility_cfg = self._build_feasibility_cfg()
        default_upper = int(self.sys_cfg.max_batch_lo)

        if feasibility_cfg is None:
            return None, {}, {}

        subgraph_stats = compute_subgraph_memory_stats_from_root(
            self.ptree.root_node,
            feasibility_cfg,
            default_upper=default_upper,
        )
        feasible_caps = compute_feasible_batch_caps_from_root(
            self.ptree.root_node,
            feasibility_cfg,
            default_upper=default_upper,
        )
        return feasibility_cfg, subgraph_stats, feasible_caps

    def _resolve_subgraph_id(self, begin_node, begin_node_idx: int) -> int:
        return int(getattr(begin_node, "_topo_node_id", begin_node_idx))

    def _clamp_max_batch_lo(self, original_max_batch_lo: int, topo_node_id: int, feasible_caps: dict) -> int:
        feasible_cap = int(feasible_caps.get(topo_node_id, original_max_batch_lo))
        return max(0, min(int(original_max_batch_lo), feasible_cap))

    def _append_subgraph_batch_log(
        self,
        begin_node,
        begin_node_idx: int,
        topo_node_id: int,
        original_max_batch_lo: int,
        clamped_max_batch_lo: int,
        this_lambda: float,
        feasibility_cfg,
        subgraph_stats: dict,
    ) -> None:
        stat = subgraph_stats.get(topo_node_id)
        self.last_subgraph_batch_info.append({
            "begin_node_index": int(begin_node_idx),
            "topo_node_id": int(topo_node_id),
            "begin_node_name": str(begin_node.name),
            "original_max_batch_lo": int(original_max_batch_lo),
            "clamped_max_batch_lo": int(clamped_max_batch_lo),
            "feasible_batch_enabled": feasibility_cfg is not None,
            "mem_cap_total_gb": None if stat is None or stat.mem_cap_total_gb is None else round(float(stat.mem_cap_total_gb), 6),
            "available_mem_gb": None if stat is None or stat.available_mem_gb is None else round(float(stat.available_mem_gb), 6),
            "param_mem_gb": None if stat is None else round(float(stat.param_mem_gb), 6),
            "kv_mem_per_batch_gb": None if stat is None else round(float(stat.kv_mem_per_batch_gb), 6),
            "max_feasible_batch": None if stat is None or stat.max_feasible_batch is None else int(stat.max_feasible_batch),
            "lambda": float(this_lambda),
        })

    def _apply_subgraph_batch_constraint(
        self,
        begin_node,
        begin_node_idx: int,
        this_sys_cfg: SystemConfig,
        this_lambda: float,
        feasibility_cfg,
        subgraph_stats: dict,
        feasible_caps: dict,
    ) -> None:
        topo_node_id = self._resolve_subgraph_id(begin_node, begin_node_idx)
        original_max_batch_lo = int(this_sys_cfg.max_batch_lo)
        clamped_max_batch_lo = self._clamp_max_batch_lo(
            original_max_batch_lo,
            topo_node_id,
            feasible_caps,
        )
        this_sys_cfg.max_batch_lo = clamped_max_batch_lo
        self._append_subgraph_batch_log(
            begin_node=begin_node,
            begin_node_idx=begin_node_idx,
            topo_node_id=topo_node_id,
            original_max_batch_lo=original_max_batch_lo,
            clamped_max_batch_lo=clamped_max_batch_lo,
            this_lambda=this_lambda,
            feasibility_cfg=feasibility_cfg,
            subgraph_stats=subgraph_stats,
        )

    # ---------------------------------------------------------------------
    # Original run path with a small inserted hook
    # ---------------------------------------------------------------------
    def run_system(self):
        # 每个 begin_nodes 都对应一个 simulator
        original_lambda = self.sys_cfg.lam
        simulation_result = []

        # Memory-feasibility precomputation (optional)
        self.last_subgraph_batch_info = []
        feasibility_cfg, subgraph_stats, feasible_caps = self._compute_subgraph_batch_constraints()

        # 原始逻辑
        for begin_node_idx, begin_node in enumerate(self.ptree.begin_nodes):
            print("begin_node_name:", begin_node.name)
            print("begin_node_dp_attr:", begin_node.dp_attr)
            this_sys_cfg = deepcopy(self.sys_cfg)

            # 指定 PP sub batch num
            sub_batch_num = self.ptree.summarise_layer_info(begin_node)
            this_sys_cfg.sub_batch_num = sub_batch_num
            # this_sys_cfg.use_pp_sub_batch = False
            this_sys_cfg.use_pp_sub_batch = True

            # 指定是否采用 MP sub batch
            this_sys_cfg.use_mp_sub_batch = False
            # this_sys_cfg.use_mp_sub_batch = True

            # 分配 lambda (req rate)
            this_lambda = sum([
                (dp_attr[1] - dp_attr[0]) * prob * original_lambda
                for dp_attr, prob in zip(begin_node.dp_attr, self.req_prob)
            ])
            if this_lambda <= 0:
                continue
            this_sys_cfg.lam = this_lambda
            print("this_lambda:", this_lambda)

            # MEMORY CONSTRAINT: small insertion point
            self._apply_subgraph_batch_constraint(
                begin_node=begin_node,
                begin_node_idx=begin_node_idx,
                this_sys_cfg=this_sys_cfg,
                this_lambda=this_lambda,
                feasibility_cfg=feasibility_cfg,
                subgraph_stats=subgraph_stats,
                feasible_caps=feasible_caps,
            )
            print("this_max_batch=", this_sys_cfg.max_batch_lo)
            if this_sys_cfg.max_batch_lo <= 0:
                continue

            # 启动 Simulator
            sample_prob = np.array([
                (dp_attr[1] - dp_attr[0]) * prob
                for dp_attr, prob in zip(begin_node.dp_attr, self.req_prob)
            ])
            sample_prob = sample_prob / np.sum(sample_prob)
            simulator = Simulator(this_sys_cfg, self.model_cfg, sample_prob, self.ptree)
            single_thread_result = simulator.run(begin_node)
            simulation_result.append(single_thread_result)
            print(summarize_metrics([single_thread_result], self.sys_cfg.t_end))
            print("\n\n\n")

        return simulation_result
