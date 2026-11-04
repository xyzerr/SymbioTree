from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import copy
import math

from exploration.decoder import RootInit, try_decode_to_root
from exploration.individual import Individual
from parallelism.pnode import BasicHardwareNode, Parallelism, XpTag
from parallelism.ptraversal import detect_begin_nodes, derive_from_node
from system.config import ModelConfig

_GB = float(1024 ** 3)


@dataclass(frozen=True)
class FeasibilityConfig:
    model_cfg: ModelConfig
    root_init: RootInit
    mem_cap_by_device_gb: Dict[int, float]
    bytes_by_device: Dict[int, int]
    peak_seq_len: int = 2048
    runtime_reserve_ratio: float = 0.10
    attach_hardware_leaves: bool = True

    def __post_init__(self) -> None:
        if int(self.peak_seq_len) <= 0:
            raise ValueError(f"peak_seq_len must be > 0, got {self.peak_seq_len}")
        ratio = float(self.runtime_reserve_ratio)
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError(
                f"runtime_reserve_ratio must be in [0, 1), got {self.runtime_reserve_ratio}"
            )


@dataclass(frozen=True)
class SubGraphMemoryStats:
    topo_node_id: int
    mem_cap_total_gb: Optional[float]
    available_mem_gb: Optional[float]
    param_mem_gb: float
    kv_mem_per_batch_gb: float
    max_feasible_batch: Optional[int]



def _local_layer_count(leaf: BasicHardwareNode) -> int:
    if len(getattr(leaf, "pp_attr", [])) != 2:
        return 0
    return max(0, int(leaf.pp_attr[1]) - int(leaf.pp_attr[0]) + 1)



def _local_tp_ratio(leaf: BasicHardwareNode) -> float:
    if len(getattr(leaf, "tp_attr", [])) != 2:
        return 0.0
    return max(0.0, float(leaf.tp_attr[1]) - float(leaf.tp_attr[0]))



def _bytes_per_elem(device_id: int, cfg: FeasibilityConfig) -> int:
    return max(1, int(cfg.bytes_by_device.get(int(device_id), 2)))



def _non_attention_param_elems_per_layer(model_cfg: ModelConfig) -> int:
    hidden = int(model_cfg.hidden_size)
    kv_hidden = int(model_cfg.kv_hidden_size)
    inter = int(model_cfg.intermediate_size)
    # IMPORTANT: keep this aligned with parallelism/ptraversal.py.
    # In the current XP semantics, XpTag.ATTENTION covers only the attention core
    # (module_idx == 1). QKV / PROJ / FFN all belong to the non-attention path,
    # i.e. XpTag.LINEAR or XpTag.BOTH. Therefore ATTENTION contributes no parameters
    # here; only the non-attention path owns parameter memory.
    qkv = hidden * (hidden + 2 * kv_hidden)
    proj = hidden * hidden
    ffn = 2 * hidden * inter
    return int(qkv + proj + ffn)



def _kv_elems_per_token_per_layer(model_cfg: ModelConfig) -> int:
    # GQA-aware: KV cache size depends on kv heads, not query heads.
    return int(2 * model_cfg.kv_hidden_size)



def _leaf_param_mem_gb(leaf: BasicHardwareNode, cfg: FeasibilityConfig) -> float:
    # ATTENTION-only leaves have no parameters under the current XP definition.
    # BOTH means the leaf executes the full layer, but only the non-attention modules
    # (QKV / PROJ / FFN) contribute parameter memory here.
    if leaf.xp_attr not in (XpTag.BOTH, XpTag.LINEAR):
        return 0.0

    layer_count = _local_layer_count(leaf)
    tp_ratio = _local_tp_ratio(leaf)
    if layer_count <= 0 or tp_ratio <= 0.0:
        return 0.0

    elems = layer_count * tp_ratio * _non_attention_param_elems_per_layer(cfg.model_cfg)
    return float(elems * _bytes_per_elem(leaf.idx, cfg)) / _GB



def _leaf_kv_mem_per_batch_gb(leaf: BasicHardwareNode, cfg: FeasibilityConfig) -> float:
    # KV cache exists only for the attention path. LINEAR-only leaves do not reserve KV.
    if leaf.xp_attr not in (XpTag.BOTH, XpTag.ATTENTION):
        return 0.0

    layer_count = _local_layer_count(leaf)
    tp_ratio = _local_tp_ratio(leaf)
    if layer_count <= 0 or tp_ratio <= 0.0:
        return 0.0

    elems = (
        layer_count
        * tp_ratio
        * int(cfg.peak_seq_len)
        * _kv_elems_per_token_per_layer(cfg.model_cfg)
    )
    return float(elems * _bytes_per_elem(leaf.idx, cfg)) / _GB



def compute_subgraph_memory_stats(
    ind: Individual,
    cfg: FeasibilityConfig,
    *,
    default_upper: int,
) -> Dict[int, SubGraphMemoryStats]:
    root = try_decode_to_root(
        ind,
        copy.deepcopy(cfg.root_init),
        attach_hardware_leaves=cfg.attach_hardware_leaves,
    )
    if root is None:
        return {}

    out: Dict[int, SubGraphMemoryStats] = {}
    reserve_ratio = float(cfg.runtime_reserve_ratio)
    default_upper = max(1, int(default_upper))

    for begin_node in detect_begin_nodes(root):
        topo_node_id = int(getattr(begin_node, "_topo_node_id"))
        leaf_nodes = [x for x in derive_from_node(begin_node) if isinstance(x, BasicHardwareNode)]
        if not leaf_nodes:
            out[topo_node_id] = SubGraphMemoryStats(
                topo_node_id=topo_node_id,
                mem_cap_total_gb=None,
                available_mem_gb=None,
                param_mem_gb=0.0,
                kv_mem_per_batch_gb=0.0,
                max_feasible_batch=default_upper,
            )
            continue

        mem_caps = [cfg.mem_cap_by_device_gb.get(int(leaf.idx)) for leaf in leaf_nodes]
        if any(v is None for v in mem_caps):
            out[topo_node_id] = SubGraphMemoryStats(
                topo_node_id=topo_node_id,
                mem_cap_total_gb=None,
                available_mem_gb=None,
                param_mem_gb=0.0,
                kv_mem_per_batch_gb=0.0,
                max_feasible_batch=default_upper,
            )
            continue

        mem_cap_total_gb = float(sum(float(v) for v in mem_caps if v is not None))
        available_mem_gb = mem_cap_total_gb * (1.0 - reserve_ratio)

        param_mem_gb = 0.0
        kv_mem_per_batch_gb = 0.0
        for leaf in leaf_nodes:
            param_mem_gb += _leaf_param_mem_gb(leaf, cfg)
            kv_mem_per_batch_gb += _leaf_kv_mem_per_batch_gb(leaf, cfg)

        if available_mem_gb <= 0.0 or available_mem_gb < param_mem_gb:
            max_batch = 0
        elif kv_mem_per_batch_gb <= 0.0:
            max_batch = default_upper
        else:
            max_batch = int(math.floor((available_mem_gb - param_mem_gb) / kv_mem_per_batch_gb))
            max_batch = max(0, min(default_upper, max_batch))

        out[topo_node_id] = SubGraphMemoryStats(
            topo_node_id=topo_node_id,
            mem_cap_total_gb=mem_cap_total_gb,
            available_mem_gb=available_mem_gb,
            param_mem_gb=param_mem_gb,
            kv_mem_per_batch_gb=kv_mem_per_batch_gb,
            max_feasible_batch=max_batch,
        )

    return out




def compute_subgraph_memory_stats_from_root(
    root: object,
    cfg: FeasibilityConfig,
    *,
    default_upper: int,
) -> Dict[int, SubGraphMemoryStats]:
    out: Dict[int, SubGraphMemoryStats] = {}
    reserve_ratio = float(cfg.runtime_reserve_ratio)
    default_upper = max(1, int(default_upper))

    for begin_idx, begin_node in enumerate(detect_begin_nodes(root)):
        topo_node_id = int(getattr(begin_node, "_topo_node_id", begin_idx))
        leaf_nodes = [x for x in derive_from_node(begin_node) if isinstance(x, BasicHardwareNode)]
        if not leaf_nodes:
            out[topo_node_id] = SubGraphMemoryStats(
                topo_node_id=topo_node_id,
                mem_cap_total_gb=None,
                available_mem_gb=None,
                param_mem_gb=0.0,
                kv_mem_per_batch_gb=0.0,
                max_feasible_batch=default_upper,
            )
            continue

        mem_caps = [cfg.mem_cap_by_device_gb.get(int(leaf.idx)) for leaf in leaf_nodes]
        if any(v is None for v in mem_caps):
            out[topo_node_id] = SubGraphMemoryStats(
                topo_node_id=topo_node_id,
                mem_cap_total_gb=None,
                available_mem_gb=None,
                param_mem_gb=0.0,
                kv_mem_per_batch_gb=0.0,
                max_feasible_batch=default_upper,
            )
            continue

        mem_cap_total_gb = float(sum(float(v) for v in mem_caps if v is not None))
        available_mem_gb = mem_cap_total_gb * (1.0 - reserve_ratio)

        param_mem_gb = 0.0
        kv_mem_per_batch_gb = 0.0
        for leaf in leaf_nodes:
            param_mem_gb += _leaf_param_mem_gb(leaf, cfg)
            kv_mem_per_batch_gb += _leaf_kv_mem_per_batch_gb(leaf, cfg)

        if available_mem_gb <= 0.0 or available_mem_gb < param_mem_gb:
            max_batch = 0
        elif kv_mem_per_batch_gb <= 0.0:
            max_batch = default_upper
        else:
            max_batch = int(math.floor((available_mem_gb - param_mem_gb) / kv_mem_per_batch_gb))
            max_batch = max(0, min(default_upper, max_batch))

        out[topo_node_id] = SubGraphMemoryStats(
            topo_node_id=topo_node_id,
            mem_cap_total_gb=mem_cap_total_gb,
            available_mem_gb=available_mem_gb,
            param_mem_gb=param_mem_gb,
            kv_mem_per_batch_gb=kv_mem_per_batch_gb,
            max_feasible_batch=max_batch,
        )

    return out


def compute_feasible_batch_caps_from_root(
    root: object,
    cfg: Optional[FeasibilityConfig],
    *,
    default_upper: int,
) -> Dict[int, int]:
    default_upper = max(1, int(default_upper))
    begin_nodes = list(detect_begin_nodes(root))
    begin_ids = [int(getattr(begin_node, "_topo_node_id", idx)) for idx, begin_node in enumerate(begin_nodes)]

    if cfg is None:
        return {int(nid): default_upper for nid in begin_ids}

    stats = compute_subgraph_memory_stats_from_root(root, cfg, default_upper=default_upper)
    if not stats:
        return {int(nid): default_upper for nid in begin_ids}

    out = {int(nid): default_upper for nid in begin_ids}
    for nid, stat in stats.items():
        cap = stat.max_feasible_batch
        if cap is None:
            out[int(nid)] = default_upper
        else:
            out[int(nid)] = max(0, min(default_upper, int(cap)))
    return out


def _detect_begin_node_ids_from_topology(ind: Individual) -> list[int]:
    out: list[int] = []

    def dfs(nid: int) -> None:
        gene = ind.topology.gene(int(nid))
        if gene.ptype == Parallelism.DP:
            for cid in ind.topology.children_of(int(nid)):
                dfs(int(cid))
        else:
            out.append(int(nid))

    dfs(int(ind.topology.root_id))
    return out


def _build_memory_feasibility_log(
    begin_ids: list[int],
    cfg: Optional[FeasibilityConfig],
    stats: Dict[int, SubGraphMemoryStats],
    *,
    default_upper: int,
) -> dict:
    subgraphs: Dict[str, dict] = {}
    for nid in begin_ids:
        stat = stats.get(int(nid))
        if stat is None:
            subgraphs[str(int(nid))] = {
                "mem_cap_total_gb": None,
                "available_mem_gb": None,
                "param_mem_gb": None,
                "kv_mem_per_batch_gb": None,
                "max_feasible_batch": int(default_upper),
            }
            continue

        subgraphs[str(int(nid))] = {
            "mem_cap_total_gb": None if stat.mem_cap_total_gb is None else round(float(stat.mem_cap_total_gb), 6),
            "available_mem_gb": None if stat.available_mem_gb is None else round(float(stat.available_mem_gb), 6),
            "param_mem_gb": round(float(stat.param_mem_gb), 6),
            "kv_mem_per_batch_gb": round(float(stat.kv_mem_per_batch_gb), 6),
            "max_feasible_batch": None if stat.max_feasible_batch is None else int(stat.max_feasible_batch),
        }

    return {
        "enabled": cfg is not None,
        "peak_seq_len": None if cfg is None else int(cfg.peak_seq_len),
        "runtime_reserve_ratio": None if cfg is None else float(cfg.runtime_reserve_ratio),
        "subgraphs": subgraphs,
    }



def compute_feasible_batch_caps(
    ind: Individual,
    cfg: Optional[FeasibilityConfig],
    *,
    default_upper: int,
) -> Dict[int, int]:
    default_upper = max(1, int(default_upper))
    begin_ids = _detect_begin_node_ids_from_topology(ind)

    if cfg is None:
        setattr(
            ind,
            "memory_feasibility_log",
            _build_memory_feasibility_log(begin_ids, None, {}, default_upper=default_upper),
        )
        return {int(nid): default_upper for nid in begin_ids}

    stats = compute_subgraph_memory_stats(ind, cfg, default_upper=default_upper)
    setattr(
        ind,
        "memory_feasibility_log",
        _build_memory_feasibility_log(begin_ids, cfg, stats, default_upper=default_upper),
    )
    if not stats:
        return {int(nid): default_upper for nid in begin_ids}

    out = {int(nid): default_upper for nid in begin_ids}
    for nid, stat in stats.items():
        cap = stat.max_feasible_batch
        if cap is None:
            out[int(nid)] = default_upper
        else:
            out[int(nid)] = max(0, min(default_upper, int(cap)))
    return out
