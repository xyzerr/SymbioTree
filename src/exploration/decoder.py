# decoder_v3.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from parallelism.pnode import (
    BasicNode,
    BasicHardwareNode,
    DataParallelismNode,
    PipelineParallelismNode,
    TensorParallelismNode,
    ModuleParallelismNode,
    Parallelism,
    XpTag,
)
from exploration.individual import Individual, Topology


@dataclass(frozen=True)
class RootInit:
    dp_attr: List[List[float]]
    pp_attr: List[int]
    tp_attr: List[float]
    xp_attr: XpTag = XpTag.BOTH


class DecodeError(Exception):
    pass


def decode_to_root(ind: Individual, root_init: RootInit, *, attach_hardware_leaves: bool = True) -> BasicNode:
    topo: Topology = ind.topology

    try:
        ind.check_legality()  # includes topology/attrs/device consistency
    except Exception as e:
        raise DecodeError(f"Individual illegal: {e}") from e

    id2node: Dict[int, BasicNode] = {}

    def make_node(node_id: int) -> BasicNode:
        g = topo.gene(node_id)
        ptype = g.ptype

        if ptype == Parallelism.DP:
            pa = ind.attrs.dp_attr.get(node_id)
            if pa is None:
                raise DecodeError(f"Missing DP parallel_attr for node {node_id}")
            return DataParallelismNode(name=f"DP_{node_id}", parallel_attr=pa)

        if ptype == Parallelism.PP:
            pa = ind.attrs.pp_attr.get(node_id)
            if pa is None:
                raise DecodeError(f"Missing PP parallel_attr for node {node_id}")
            return PipelineParallelismNode(name=f"PP_{node_id}", parallel_attr=pa)

        if ptype == Parallelism.TP:
            pa = ind.attrs.tp_attr.get(node_id)
            if pa is None:
                raise DecodeError(f"Missing TP parallel_attr for node {node_id}")
            return TensorParallelismNode(name=f"TP_{node_id}", parallel_attr=pa)

        if ptype == Parallelism.XP:
            pa = ind.attrs.xp_attr.get(node_id)
            if pa is None:
                raise DecodeError(f"Missing XP parallel_attr for node {node_id}")
            return ModuleParallelismNode(name=f"XP_{node_id}", parallel_attr=pa)

        raise DecodeError(f"Unsupported ptype: {ptype}")

    # 1) instantiate parallel nodes
    for nid in topo.iter_dfs():
        node = make_node(nid)
        setattr(node, "_topo_node_id", int(nid))
        id2node[nid] = node

    # 2) connect parallel topology edges
    for pid in topo.iter_dfs():
        p = id2node[pid]
        for cid in topo.children_of(pid):
            p.add_child(id2node[cid])

    root = id2node[topo.root_id]

    # 3) seed root attrs and derive along parallel edges
    root.dp_attr = root_init.dp_attr
    root.pp_attr = root_init.pp_attr
    root.tp_attr = root_init.tp_attr
    root.xp_attr = root_init.xp_attr

    try:
        for pid in topo.iter_dfs():
            parent = id2node[pid]
            for child_idx, cid in enumerate(topo.children_of(pid)):
                parent.derive_child_info(id2node[cid], child_idx)
    except Exception as e:
        raise DecodeError(f"Derive parallel attrs failed: {e}") from e

    # 4) attach hardware children for leaf parallel nodes (arity is device_group size)
    if attach_hardware_leaves:
        try:
            _attach_hardware(ind, topo, id2node)
        except Exception as e:
            raise DecodeError(f"Attach hardware failed: {e}") from e

    setattr(root, "sub_graph_batch_sizes", dict(getattr(ind, "sub_graph_batch_sizes", {})))
    return root


def _attach_hardware(ind: Individual, topo: Topology, id2node: Dict[int, BasicNode]) -> None:
    leaf_ids = topo.leaf_parallel_nodes()
    for leaf_id in leaf_ids:
        dev_group = ind.device_assign.leaf_to_devices[leaf_id]
        if not isinstance(dev_group, list) or not dev_group:
            raise ValueError(f"Leaf {leaf_id} must map to non-empty list[int], got {dev_group}")

        leaf = id2node[leaf_id]
        # Attach one HW child per device (order matters for splitting)
        for d in dev_group:
            leaf.add_child(BasicHardwareNode(name=f"HW_{d}", idx=int(d)))


def try_decode_to_root(ind: Individual, root_init: RootInit, *, attach_hardware_leaves: bool = True) -> Optional[BasicNode]:
    try:
        return decode_to_root(ind, root_init, attach_hardware_leaves=attach_hardware_leaves)
    except DecodeError:
        return None
