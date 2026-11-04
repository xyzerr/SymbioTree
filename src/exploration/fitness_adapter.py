# fitness_adapter_v3.py
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Union
import numpy as np

from system.config import SystemConfig, ModelConfig
from system.metrics import summarize_metrics_data, summarize_metrics

from serving.simulator import Simulator
from parallelism.ptree import ParallelismTree
from parallelism.ptraversal import derive_from_node, detect_begin_nodes
from hardware.htree import HardwareTree
from parallelism.pnode import BasicHardwareNode

Number = Union[int, float]

def default_result_to_fitness(sim_results: List[Any]) -> float:
    # smaller is better
    if not sim_results:
        return float("-inf")
    if all(isinstance(x, (int, float)) for x in sim_results):
        return -float(max(sim_results))
    return float("-inf")

@dataclass
class SystemEvaluatorV3:
    sys_cfg: SystemConfig
    model_cfg: ModelConfig
    req_prob: Sequence[float]
    hcase_idx: int = 0
    pcase_idx_for_init: int = 3
    result_to_fitness: Callable[[List[Any]], float] = default_result_to_fitness
    pareto_mode: bool = False

    def __post_init__(self) -> None:
        self.htree = HardwareTree(self.hcase_idx)
        self.ptree = ParallelismTree(self.sys_cfg, self.model_cfg, self.htree, case_idx=self.pcase_idx_for_init)
        self._dev_by_idx = {getattr(d, "idx", i): d for i, d in enumerate(self.htree.devices)}
        # print(self._dev_by_idx)

    def _override_ptree_with_root(self, root_node: Any) -> None:
        # IMPORTANT: derive_from_node() will call derive_child_info on *all* parallel nodes,
        # including leaf-parallel -> hardware edges.
        # In v3, leaf parallel nodes have parallel_attr sized to their device-group, so this is safe.
        self.ptree.root_node = root_node
        self.ptree.begin_nodes = list(detect_begin_nodes(root_node))
        leaf_nodes = derive_from_node(root_node)
        self.ptree.leaf_nodes = leaf_nodes

        mapper = {}
        for leaf in leaf_nodes:
            # leaf.print_info()
            if isinstance(leaf, BasicHardwareNode):
                dev = self._dev_by_idx.get(leaf.idx)
                if dev is None:
                    raise ValueError(f"Hardware idx {leaf.idx} not found.")
                mapper[leaf.name] = dev
        # print(mapper)
        self.ptree.mapper = mapper

    def _resolve_sub_graph_batch_size(self, root_node: Any, begin_node: Any, default_batch_size: int) -> int:
        batch_dict = getattr(root_node, "sub_graph_batch_sizes", None)
        if not isinstance(batch_dict, dict):
            return int(default_batch_size)

        topo_node_id = getattr(begin_node, "_topo_node_id", None)
        if topo_node_id in batch_dict:
            return max(1, int(batch_dict[topo_node_id]))

        name = getattr(begin_node, "name", None)
        if name in batch_dict:
            return max(1, int(batch_dict[name]))

        return int(default_batch_size)

    def run_system_on_root(self, root_node: Any, batch_size: int = 1):
        self._override_ptree_with_root(root_node)

        original_lambda = self.sys_cfg.lam
        results: List[Any] = []

        for begin_node in self.ptree.begin_nodes:
            this_sys_cfg = deepcopy(self.sys_cfg)
            sub_batch_num = self.ptree.summarise_layer_info(begin_node)
            this_sys_cfg.sub_batch_num = sub_batch_num
            this_sys_cfg.use_pp_sub_batch = True
            # this_sys_cfg.use_mp_sub_batch = True
            this_sys_cfg.use_mp_sub_batch = False
            this_sys_cfg.max_batch_lo = self._resolve_sub_graph_batch_size(root_node, begin_node, batch_size)

            this_lambda = sum(
                [(dp_attr[1] - dp_attr[0]) * prob * original_lambda
                 for dp_attr, prob in zip(begin_node.dp_attr, self.req_prob)]
            )
            this_sys_cfg.lam = this_lambda
            sample_prob = np.array([(dp_attr[1]-dp_attr[0])*prob for dp_attr, prob in zip(begin_node.dp_attr, self.req_prob)])
            sample_prob = sample_prob/np.sum(sample_prob)
            # simulator = Simulator(this_sys_cfg, self.model_cfg, self.req_prob, self.ptree)
            simulator = Simulator(this_sys_cfg, self.model_cfg, sample_prob, self.ptree)
            single_thread_result = simulator.run(begin_node)
            # print(summarize_metrics([single_thread_result], self.sys_cfg.t_end))
            results.append(single_thread_result)

        # return results
        print(summarize_metrics(results, self.sys_cfg.t_end))
        return summarize_metrics_data(results, self.sys_cfg.t_end)

    def fitness(self, root_node: Any, batch_size: int = 1) -> Any:
        # return float(self.result_to_fitness(self.run_system_on_root(root_node)))

        if self.pareto_mode:
            # 返回两个指标，Throughput 和 Latency（TPOT）
            return self.run_system_on_root(root_node, batch_size)
        else:
            # 只返回 Throughput，目前已废弃
            return self.run_system_on_root(root_node, batch_size)[0]


def make_fitness_fn(
    sys_cfg: SystemConfig,
    model_cfg: ModelConfig,
    *,
    pareto_mode: bool = True,
    req_prob: Sequence[float],
    hcase_idx: int = 0,
    pcase_idx_for_init: int = 3,
    result_to_fitness: Callable[[List[Any]], float] = default_result_to_fitness,
) -> Callable[[Any, int], Any]:
    return SystemEvaluatorV3(
        sys_cfg=sys_cfg,
        model_cfg=model_cfg,
        req_prob=req_prob,
        hcase_idx=hcase_idx,
        pcase_idx_for_init=pcase_idx_for_init,
        result_to_fitness=result_to_fitness,
        pareto_mode=pareto_mode
    ).fitness
