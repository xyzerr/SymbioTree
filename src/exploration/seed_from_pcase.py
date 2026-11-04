# seed_from_pcase_v3.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional, Any

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

from exploration.individual import (
    Topology,
    TopologyNodeGene,
    DeviceAssign,
    Attrs,
    Individual,
)


def _ptype_of(node: BasicNode) -> Parallelism:
    if isinstance(node, DataParallelismNode):
        return Parallelism.DP
    if isinstance(node, PipelineParallelismNode):
        return Parallelism.PP
    if isinstance(node, TensorParallelismNode):
        return Parallelism.TP
    if isinstance(node, ModuleParallelismNode):
        return Parallelism.XP
    raise TypeError(f"Unsupported node type for topology: {type(node)}")


def _is_parallel(node: BasicNode) -> bool:
    return isinstance(
        node,
        (DataParallelismNode, PipelineParallelismNode, TensorParallelismNode, ModuleParallelismNode),
    )


def _parallel_children(node: BasicNode) -> List[BasicNode]:
    return [c for c in getattr(node, "children", []) if _is_parallel(c)]


def _hardware_children(node: BasicNode) -> List[BasicHardwareNode]:
    return [c for c in getattr(node, "children", []) if isinstance(c, BasicHardwareNode)]


def individual_from_pcase_root(
    root: BasicNode,
    *,
    devices: Sequence[int],
    req_type_num: int,
    batch_size: int = 1,
    strict_device_partition: bool = True,
) -> Individual:
    """
    Convert a pcase-built *phenotype* tree (parallel nodes + BasicHardwareNode leaves)
    into an Individual_v3 (Topology + DeviceAssign + Attrs).

    Notes
    -----
    - Topology includes ONLY parallel nodes.
    - For each topology leaf parallel node, DeviceAssign maps it to the list of hardware idxs
      from its BasicHardwareNode children (order preserved).
    - Attrs are copied from each parallel node's `parallel_attr` (and must match effective arity):
        * internal parallel node: arity = number of parallel children
        * leaf parallel node: arity = number of hardware children

    strict_device_partition:
      - True: require that assigned devices == set(devices) and no duplicates
      - False: allow subset; (still no duplicates). Useful if your pcase uses fewer devices
        than the target hardware; such seeds won't crash, but may be inconsistent with your
        System/Simulator assumptions unless you handle unused devices elsewhere.
    """
    dev_list = list(devices)
    dev_set = set(dev_list)

    # 1) Traverse parallel nodes; assign canonical node_id by BFS order
    node2id: Dict[int, int] = {}
    id2node: Dict[int, BasicNode] = {}

    q: List[BasicNode] = [root]
    next_id = 0

    while q:
        n = q.pop(0)
        if not _is_parallel(n):
            continue
        key = id(n)
        if key in node2id:
            continue
        node2id[key] = next_id
        id2node[next_id] = n
        next_id += 1
        q.extend(_parallel_children(n))

    # 2) Build topology genes with child_slot among PARALLEL children
    genes: List[TopologyNodeGene] = []
    root_id = node2id[id(root)]
    genes.append(TopologyNodeGene(node_id=root_id, parent_id=-1, ptype=_ptype_of(root), child_slot=0))

    for pid, pnode in id2node.items():
        pch = _parallel_children(pnode)
        for slot, child in enumerate(pch):
            cid = node2id[id(child)]
            genes.append(
                TopologyNodeGene(
                    node_id=cid,
                    parent_id=pid,
                    ptype=_ptype_of(child),
                    child_slot=slot,
                )
            )

    topo = Topology(nodes=genes)

    # 3) DeviceAssign: topology leaves -> hardware idx list (order preserved)
    da = DeviceAssign()
    used: List[int] = []
    for leaf_id in topo.leaf_parallel_nodes():
        leaf_node = id2node[leaf_id]
        hw = _hardware_children(leaf_node)
        if not hw:
            raise ValueError(f"Topology leaf parallel node {leaf_id} has no BasicHardwareNode children in pcase tree.")
        grp = [int(h.idx) for h in hw]
        da.leaf_to_devices[leaf_id] = grp
        used.extend(grp)

    # duplicates check
    if len(set(used)) != len(used):
        raise ValueError(f"pcase seed uses duplicated devices: {used}")

    if strict_device_partition:
        if set(used) != dev_set:
            raise ValueError(f"pcase seed devices mismatch. want={sorted(dev_set)} got={sorted(set(used))}")
    else:
        # subset allowed
        if not set(used).issubset(dev_set):
            raise ValueError(f"pcase seed uses devices outside target set. want={sorted(dev_set)} got={sorted(set(used))}")

    # 4) Attrs: copy parallel_attr into appropriate dict by node_id
    attrs = Attrs()
    for nid, node in id2node.items():
        t = _ptype_of(node)
        pa = getattr(node, "parallel_attr", None)
        if pa is None:
            raise ValueError(f"pcase node {nid} missing parallel_attr")

        if t == Parallelism.DP:
            attrs.dp_attr[nid] = [row[:] for row in pa]
        elif t == Parallelism.PP:
            attrs.pp_attr[nid] = list(pa)
        elif t == Parallelism.TP:
            attrs.tp_attr[nid] = list(pa)
        elif t == Parallelism.XP:
            attrs.xp_attr[nid] = list(pa)
        else:
            raise ValueError(f"Unsupported type: {t}")

    ind = Individual(
        topology=topo,
        device_assign=da,
        attrs=attrs,
        devices=dev_list,
        req_type_num=req_type_num,
        batch_size=int(batch_size),
    )
    # validate shapes/constraints (will also enforce strict partition if enabled at Individual level)
    ind.check_legality()
    return ind
