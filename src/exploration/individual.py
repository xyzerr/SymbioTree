# individual_v3.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from parallelism.pnode import Parallelism, XpTag


@dataclass(frozen=True)
class TopologyNodeGene:
    node_id: int
    parent_id: int           # -1 means root
    ptype: Parallelism       # DP/PP/TP/XP
    child_slot: int = 0      # order under parent

    def __post_init__(self):
        if self.ptype == Parallelism.NONE:
            raise ValueError("ptype cannot be NONE")
        if self.parent_id == -1 and self.child_slot != 0:
            raise ValueError("Root must have child_slot=0")


@dataclass
class Topology:
    nodes: List[TopologyNodeGene]

    _id2gene: Dict[int, TopologyNodeGene] = field(init=False, default_factory=dict)
    _children: Dict[int, List[int]] = field(init=False, default_factory=dict)
    _root_id: int = field(init=False, default=-1)

    def __post_init__(self):
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.nodes:
            raise ValueError("Topology must contain at least one node.")

        id2gene: Dict[int, TopologyNodeGene] = {}
        for g in self.nodes:
            if g.node_id in id2gene:
                raise ValueError(f"Duplicate node_id: {g.node_id}")
            id2gene[g.node_id] = g

        roots = [g.node_id for g in self.nodes if g.parent_id == -1]
        if len(roots) != 1:
            raise ValueError(f"Topology must have exactly one root, got {roots}")
        root_id = roots[0]

        tmp: Dict[int, List[Tuple[int, int]]] = {g.node_id: [] for g in self.nodes}
        for g in self.nodes:
            if g.parent_id == -1:
                continue
            if g.parent_id not in id2gene:
                raise ValueError(f"Parent {g.parent_id} not found for node {g.node_id}")
            tmp[g.parent_id].append((g.child_slot, g.node_id))

        children: Dict[int, List[int]] = {}
        for pid, items in tmp.items():
            items.sort(key=lambda x: x[0])
            slots = [s for s, _ in items]
            if slots:
                exp = list(range(len(slots)))
                if slots != exp:
                    raise ValueError(f"Invalid child_slot under parent {pid}: expected {exp}, got {slots}")
            children[pid] = [nid for _, nid in items]

        self._check_acyclic_and_connected(root_id, children)

        self._id2gene = id2gene
        self._children = children
        self._root_id = root_id

    @staticmethod
    def _check_acyclic_and_connected(root_id: int, children: Dict[int, List[int]]) -> None:
        visiting: Set[int] = set()
        visited: Set[int] = set()

        def dfs(u: int) -> None:
            if u in visiting:
                raise ValueError("Cycle detected.")
            if u in visited:
                return
            visiting.add(u)
            for v in children.get(u, []):
                dfs(v)
            visiting.remove(u)
            visited.add(u)

        dfs(root_id)
        all_nodes = set(children.keys())
        if visited != all_nodes:
            miss = sorted(list(all_nodes - visited))
            raise ValueError(f"Disconnected nodes: {miss}")

    @property
    def root_id(self) -> int:
        return self._root_id

    def gene(self, node_id: int) -> TopologyNodeGene:
        return self._id2gene[node_id]

    def children_of(self, node_id: int) -> List[int]:
        return self._children.get(node_id, [])

    def parent_of(self, node_id: int) -> int:
        return self._id2gene[node_id].parent_id

    def iter_dfs(self) -> Iterable[int]:
        stack = [self.root_id]
        while stack:
            u = stack.pop()
            yield u
            for v in reversed(self.children_of(u)):
                stack.append(v)

    def leaf_parallel_nodes(self) -> List[int]:
        return [nid for nid in self._id2gene.keys() if len(self.children_of(nid)) == 0]

    def check_legality(self) -> None:
        # XP legality:
        # - internal XP: exactly 2 parallel children
        # - leaf XP: allowed, but its two branches will be realized as 2 devices
        for nid, g in self._id2gene.items():
            if g.ptype != Parallelism.XP:
                continue
            child_k = len(self.children_of(nid))
            if child_k not in (0, 2):
                raise ValueError(
                    f"XP node {nid} must either be a leaf or have exactly 2 parallel children."
                )

        # order constraints along root->leaf path
        def dfs(u: int, seen_pp: bool, seen_tp: bool) -> None:
            t = self.gene(u).ptype
            if t == Parallelism.DP and (seen_pp or seen_tp):
                raise ValueError(f"DP appears after PP/TP at node {u}")
            if t == Parallelism.PP and seen_tp:
                raise ValueError(f"PP appears after TP at node {u}")
            npp = seen_pp or (t == Parallelism.PP)
            ntp = seen_tp or (t == Parallelism.TP)
            for v in self.children_of(u):
                dfs(v, npp, ntp)

        dfs(self.root_id, False, False)


@dataclass
class Attrs:
    """
    parallel_attr for each parallel node.

    IMPORTANT (v3)
    --------------
    For *leaf parallel nodes* in topology, the effective arity is NOT 0.
    It equals the size of device group assigned to that leaf.
    We must generate parallel_attr with that arity so that ptraversal.derive_from_node()
    can derive attrs for BasicHardwareNode children without IndexError.
    """
    dp_attr: Dict[int, List[List[float]]] = field(default_factory=dict)
    pp_attr: Dict[int, List[float]] = field(default_factory=dict)
    tp_attr: Dict[int, List[float]] = field(default_factory=dict)
    xp_attr: Dict[int, List[XpTag]] = field(default_factory=dict)

    def check_shapes(self, topo: Topology, device_assign: "DeviceAssign", req_type_num: int) -> None:
        required_xp_tags = {XpTag.ATTENTION, XpTag.LINEAR}

        for nid in topo.iter_dfs():
            t = topo.gene(nid).ptype

            # effective arity:
            # - internal node: number of parallel children
            # - leaf node: size of its device group
            parallel_k = len(topo.children_of(nid))
            if parallel_k == 0:
                dev_group = device_assign.leaf_to_devices.get(nid)
                if not isinstance(dev_group, list) or len(dev_group) == 0:
                    raise ValueError(f"Leaf {nid} must have non-empty device group.")
                k = len(dev_group)
            else:
                k = parallel_k

            if t == Parallelism.DP:
                v = self.dp_attr.get(nid)
                if v is None or len(v) != req_type_num:
                    raise ValueError(f"DP node {nid}: missing or wrong req_type_num")
                for ri, row in enumerate(v):
                    if len(row) != k:
                        raise ValueError(f"DP node {nid}: req_idx={ri} expected len={k}, got {len(row)}")
            elif t == Parallelism.PP:
                v = self.pp_attr.get(nid)
                if v is None or len(v) != k:
                    raise ValueError(f"PP node {nid}: expected len={k}")
            elif t == Parallelism.TP:
                v = self.tp_attr.get(nid)
                if v is None or len(v) != k:
                    raise ValueError(f"TP node {nid}: expected len={k}")
            elif t == Parallelism.XP:
                # XP supports two realizations:
                # - internal XP: two parallel children
                # - leaf XP: exactly two devices, each device is one XP branch
                if parallel_k == 0:
                    dev_group = device_assign.leaf_to_devices.get(nid)
                    if not isinstance(dev_group, list) or len(dev_group) != 2:
                        raise ValueError(
                            f"XP leaf node {nid}: expected exactly 2 devices, got {dev_group}"
                        )
                elif parallel_k != 2:
                    raise ValueError(
                        f"XP node {nid}: must either be a leaf or have 2 parallel children."
                    )

                v = self.xp_attr.get(nid)
                if v is None or len(v) != 2:
                    raise ValueError(f"XP node {nid}: expected 2 tags.")
                if set(v) != required_xp_tags:
                    raise ValueError(
                        f"XP node {nid}: tags must be exactly {{ATTENTION, LINEAR}}, got {v}"
                    )
            else:
                raise ValueError(f"Unexpected type: {t}")


@dataclass
class DeviceAssign:
    """
    语义：leaf parallel node 可以挂多个 device
    leaf parallel node -> list[int] (device ids).
    All device ids across leaves must be a partition of devices (no duplicates).
    """

    # 键: leaf parallel node 的 node_id
    # 值: list[int]，表示这一个 leaf parallel node 下面要挂接的 device id（硬件节点的 idx）
    leaf_to_devices: Dict[int, List[int]] = field(default_factory=dict)

    def check_complete_for_leaves(self, topo: Topology) -> None:
        # 确保每个 leaf parallel node 都有分配
        leaves = topo.leaf_parallel_nodes()
        missing = [nid for nid in leaves if nid not in self.leaf_to_devices]
        if missing:
            raise ValueError(f"Missing device assignment for leaves: {missing}")

    def check_total_devices(self, devices: List[int]) -> None:
        # 每个 device 必须恰好出现一次（既不允许漏，也不允许重复）
        want = set(devices)
        got: List[int] = []
        for nid, grp in self.leaf_to_devices.items():
            if not isinstance(grp, list) or not grp:
                raise ValueError(f"Leaf {nid} has invalid device group: {grp}")
            got.extend(int(x) for x in grp)
        if set(got) != want:
            raise ValueError(f"Assigned devices mismatch. want={sorted(want)} got_unique={sorted(set(got))}")
        if len(got) != len(want):
            raise ValueError(f"Assigned devices must be a partition (no duplicates). got={got}")


@dataclass
class Individual:
    topology: Topology
    device_assign: DeviceAssign
    attrs: Attrs

    devices: List[int]
    req_type_num: int

    batch_size: int = 1
    # begin_node/sub-graph specific max_batch_lo, keyed by topology node_id
    sub_graph_batch_sizes: Dict[int, int] = field(default_factory=dict)

    # Multi-objective metrics (v4: Pareto throughput-latency)
    # objectives: (throughput, latency) where throughput is MAXIMIZED and latency is MINIMIZED.
    throughput: Optional[float] = None
    throughput_req: Optional[float] = None
    latency: Optional[float] = None
    objectives: Optional[Tuple[float, float]] = None

    f_dist: Optional[List[Any]] = None
    p_dist: Optional[List[Any]] = None

    # NSGA-II helpers
    pareto_rank: Optional[int] = None  # 0 is best (non-dominated front)
    crowding: float = 0.0

    uid: Optional[str] = None

    def check_legality(self) -> None:
        self.topology.check_legality()
        self.device_assign.check_complete_for_leaves(self.topology)
        self.device_assign.check_total_devices(self.devices)
        self.attrs.check_shapes(self.topology, self.device_assign, self.req_type_num)

        if getattr(self, "batch_size", 1) < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        for nid, bs in getattr(self, "sub_graph_batch_sizes", {}).items():
            if int(bs) < 1:
                raise ValueError(f"sub_graph batch size for node {nid} must be >= 1, got {bs}")
