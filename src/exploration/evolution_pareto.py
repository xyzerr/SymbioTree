# evolution_pareto.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import random
import math
import copy
import uuid
import hashlib
from collections import deque
import json

from parallelism.pnode import Parallelism, XpTag

from exploration.individual import (
    Attrs,
    DeviceAssign,
    Individual,
    Topology,
    TopologyNodeGene,
)

from exploration.decoder import RootInit, try_decode_to_root
from exploration.feasibility import FeasibilityConfig, compute_feasible_batch_caps
from exploration.seed_from_pcase import individual_from_pcase_root
from exploration.parallelism_filter import (
    allowed_parallelism_types,
    filter_init_patterns_by_parallelism,
    filter_rewrite_patterns_by_parallelism,
    individual_contains_disabled_parallelisms,
    normalize_disabled_parallelisms,
    symbolic_contains_disabled_parallelisms,
)
from exploration.ind_io import (
    log_individual_json,
    print_individual,
    format_topology,
    save_individual_json
)
from exploration.rewrite_mechanism import (
    InitPatternSpec,
    RewriteEngine,
    RewriteFamily,
    default_init_patterns,
    default_numeric_patterns,
    default_parameterized_patterns,
    default_patterns,
    has_open_nodes,
    individual_to_symbolic,
    is_materializable,
    symbolic_to_individual,
)


@dataclass
class InitConfig:
    population_size: int = 50
    max_depth: int = 4
    max_children: int = 4
    p_stop_expand: float = 0.35

    # XP MUST be exactly one ATTENTION and one LINEAR
    xp_swap_prob: float = 0.5

    shuffle_devices_per_individual: bool = True

    # Mixed initialization ratios
    p_pattern_seed_init: float = 0.50
    p_stratified_init: float = 0.45
    p_random_init: float = 0.05

    # Per-individual batch size search space (resampled on every mutation)
    batch_size_choices: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    disabled_parallelisms: Sequence[Any] = ()

@dataclass
class EvoConfig:

    rewrite_mode: str = "legacy"  # options: legacy, family_driven

    generations: int = 50
    elite_size: int = 5
    offspring_size: Optional[int] = None

    # New mutation taxonomy
    p_rewrite_mut: float = 0.60
    p_numeric_mut: float = 0.35
    p_mapping_refine_mut: float = 0.05

    # Backward-compatible aliases.
    p_topology_mut: Optional[float] = None
    p_device_mut: Optional[float] = None

    # Rewrite family weights
    p_skeleton_expand: float = 0.25
    p_local_refine: float = 0.30
    p_relabel: float = 0.15
    p_repartition: float = 0.20
    p_rollback: float = 0.10
    rewrite_max_steps: int = 4

    # Rewrite pattern source: "static" preserves the original hand-written patterns.
    # "parameterized" enables the new candidate-generating rewrite families.
    # "hybrid" concatenates both libraries.
    rewrite_pattern_mode: str = "static"
    param_rewrite_max_children: int = 4
    param_rewrite_partition_budget: int = 12
    param_rewrite_candidate_budget: int = 12

    weight_noise_sigma: float = 0.25
    max_attempt_factor: int = 20

    tournament_k: int = 3
    enable_cache: bool = True
    enable_subgraph_batch_mut: bool = False
    subgraph_batch_max_mutated: int = 1
    numeric_mutation_max_targets: int = 1

    def __post_init__(self) -> None:
        if self.p_topology_mut is not None or self.p_device_mut is not None:
            top = float(self.p_topology_mut or 0.0)
            dev = float(self.p_device_mut or 0.0)
            num = float(self.p_numeric_mut)
            total = top + dev + num
            if total > 0:
                self.p_rewrite_mut = (top + dev) / total
                self.p_numeric_mut = num / total
                self.p_mapping_refine_mut = 0.0

        total = self.p_rewrite_mut + self.p_numeric_mut + self.p_mapping_refine_mut
        if total <= 0:
            self.p_rewrite_mut, self.p_numeric_mut, self.p_mapping_refine_mut = 0.6, 0.35, 0.05
        else:
            self.p_rewrite_mut /= total
            self.p_numeric_mut /= total
            self.p_mapping_refine_mut /= total


def canonical_key(ind: Individual, *, round_digits: int = 3) -> str:
    topo = ind.topology
    topo_items = sorted(
        [(g.node_id, g.parent_id, int(g.ptype.value), g.child_slot) for g in topo.nodes],
        key=lambda x: (x[1], x[3], x[0]),
    )

    def round_list(xs):
        return [round(float(x), round_digits) for x in xs]

    attrs_items = []
    for nid in topo.iter_dfs():
        t = topo.gene(nid).ptype
        if t == Parallelism.DP:
            v = [[round(float(x), round_digits) for x in row] for row in ind.attrs.dp_attr.get(nid, [])]
            attrs_items.append(("DP", nid, v))
        elif t == Parallelism.PP:
            attrs_items.append(("PP", nid, round_list(ind.attrs.pp_attr.get(nid, []))))
        elif t == Parallelism.TP:
            attrs_items.append(("TP", nid, round_list(ind.attrs.tp_attr.get(nid, []))))
        elif t == Parallelism.XP:
            attrs_items.append(("XP", nid, [int(x.value) for x in ind.attrs.xp_attr.get(nid, [])]))

    leaves = sorted(topo.leaf_parallel_nodes())
    dev_items = [(nid, ind.device_assign.leaf_to_devices.get(nid, [])) for nid in leaves]

    sub_graph_batch_items = sorted((int(k), int(v)) for k, v in getattr(ind, "sub_graph_batch_sizes", {}).items())
    s = repr((topo_items, attrs_items, dev_items, int(getattr(ind, "batch_size", 1)), sub_graph_batch_items)).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


def _allowed_types(
    seen_pp: bool,
    seen_tp: bool,
    disabled_parallelisms: Sequence[Any] = (),
) -> List[Parallelism]:
    return allowed_parallelism_types(seen_pp, seen_tp, disabled_parallelisms)


def _get_disabled_parallelisms(init_cfg: InitConfig) -> List[Parallelism]:
    return list(normalize_disabled_parallelisms(getattr(init_cfg, "disabled_parallelisms", ())))


def _individual_allowed_by_parallelism_filter(ind: Individual, init_cfg: InitConfig) -> bool:
    return not individual_contains_disabled_parallelisms(ind, _get_disabled_parallelisms(init_cfg))


def _symbolic_allowed_by_parallelism_filter(sym_root: Any, init_cfg: InitConfig) -> bool:
    return not symbolic_contains_disabled_parallelisms(sym_root, _get_disabled_parallelisms(init_cfg))


def random_topology(cfg: InitConfig) -> Topology:
    disabled_parallelisms = normalize_disabled_parallelisms(getattr(cfg, "disabled_parallelisms", ()))
    nodes: List[TopologyNodeGene] = []
    root_allowed = _allowed_types(False, False, disabled_parallelisms)
    if not root_allowed:
        raise ValueError("No root parallelism types remain after applying disabled_parallelisms.")
    root_type = random.choice(root_allowed)
    nodes.append(TopologyNodeGene(node_id=0, parent_id=-1, ptype=root_type, child_slot=0))
    next_id = 1

    q: deque[Tuple[int, int, bool, bool]] = deque()
    q.append((0, 0, False, False))
    id2ptype: Dict[int, Parallelism] = {0: root_type}

    while q:
        nid, depth, seen_pp, seen_tp = q.popleft()
        t = id2ptype[nid]
        npp = seen_pp or (t == Parallelism.PP)
        ntp = seen_tp or (t == Parallelism.TP)

        if depth >= cfg.max_depth or random.random() < cfg.p_stop_expand:
            continue

        k = 2 if t == Parallelism.XP else random.randint(2, cfg.max_children)
        for slot in range(k):
            child_allowed = _allowed_types(npp, ntp, disabled_parallelisms)
            if not child_allowed:
                continue
            ctype = random.choice(child_allowed)
            nodes.append(TopologyNodeGene(node_id=next_id, parent_id=nid, ptype=ctype, child_slot=slot))
            id2ptype[next_id] = ctype
            q.append((next_id, depth + 1, npp, ntp))
            next_id += 1

    return Topology(nodes=nodes)


def _rand_positive_weights(k: int) -> List[float]:
    return [random.lognormvariate(0.0, 0.8) for _ in range(k)]


def _random_partition(n: int, k: int) -> List[int]:
    if k > n:
        raise ValueError("leaf_count > device_count")
    cuts = sorted(random.sample(range(1, n), k - 1)) if k > 1 else []
    parts = []
    prev = 0
    for c in cuts + [n]:
        parts.append(c - prev)
        prev = c
    return parts


def sample_device_assign(topo: Topology, devices: Sequence[int], *, shuffle: bool = True) -> DeviceAssign:
    # 最终目的：把 devices 随机切分到各个 leaf 上，每个 leaf 至少一个
    # 取 leaf 列表
    leaves = sorted(topo.leaf_parallel_nodes())
    if not leaves:
        raise ValueError("Topology has no leaves.")
    dev_list = list(devices)
    if shuffle:
        random.shuffle(dev_list)
    L = len(leaves)
    N = len(dev_list)
    # 生成一个随机 partition：把 N 分成 L 个正整数（每个 ≥1）
    sizes = _random_partition(N, L)  # each leaf >=1 device

    da = DeviceAssign()
    idx = 0
    for leaf_id, sz in zip(leaves, sizes):
        da.leaf_to_devices[leaf_id] = dev_list[idx: idx + sz]
        idx += sz
    return da

def _detect_begin_node_ids(topo: Topology) -> List[int]:
    """Mirror ptraversal.detect_begin_nodes() on topology node ids."""
    out: List[int] = []

    def dfs(nid: int) -> None:
        if topo.gene(nid).ptype == Parallelism.DP:
            for cid in topo.children_of(nid):
                dfs(cid)
        else:
            out.append(nid)

    dfs(topo.root_id)
    return out


def _subtree_device_ids(ind: Individual, node_id: int) -> List[int]:
    topo = ind.topology
    out: List[int] = []

    def dfs(nid: int) -> None:
        children = topo.children_of(nid)
        if not children:
            out.extend(int(d) for d in ind.device_assign.leaf_to_devices.get(nid, []))
            return
        for cid in children:
            dfs(cid)

    dfs(int(node_id))
    return out


def _subgraph_contains_pim(
    ind: Individual,
    node_id: int,
    device_type_by_id: Optional[Dict[int, str]],
) -> bool:
    dtype_map = device_type_by_id or {int(d): str(int(d)) for d in ind.devices}
    for did in _subtree_device_ids(ind, int(node_id)):
        dtype = str(dtype_map.get(int(did), "")).upper()
        if "PIM" in dtype:
            return True
    return False


def _eligible_pim_begin_ids(
    ind: Individual,
    begin_ids: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]],
) -> List[int]:
    return [
        int(nid)
        for nid in begin_ids
        if _subgraph_contains_pim(ind, int(nid), device_type_by_id)
    ]


def _compute_feasible_batch_caps(
    ind: Individual,
    upper: int,
    feasibility_cfg: Optional[FeasibilityConfig],
) -> Dict[int, int]:
    upper = max(1, int(upper))
    begin_ids = _detect_begin_node_ids(ind.topology)
    if feasibility_cfg is None:
        return {int(nid): int(upper) for nid in begin_ids}

    caps = compute_feasible_batch_caps(ind, feasibility_cfg, default_upper=upper)
    return {int(nid): max(0, int(caps.get(int(nid), upper))) for nid in begin_ids}


def _repair_sub_graph_batch_sizes_by_feasibility(
    ind: Individual,
    feasibility_cfg: Optional[FeasibilityConfig],
) -> bool:
    upper = max(1, int(getattr(ind, "batch_size", 1)))
    begin_ids = _detect_begin_node_ids(ind.topology)
    caps = _compute_feasible_batch_caps(ind, upper, feasibility_cfg)
    cur_map = {int(k): int(v) for k, v in getattr(ind, "sub_graph_batch_sizes", {}).items()}

    feasible = True
    repaired: Dict[int, int] = {}
    for nid in begin_ids:
        nid = int(nid)
        cap = max(0, int(caps.get(nid, upper)))
        if cap < 1:
            feasible = False
            cap = 1
        cur = int(cur_map.get(nid, upper))
        repaired[nid] = max(1, min(upper, cap, cur))

    ind.sub_graph_batch_sizes = repaired
    return feasible


def _sample_sub_graph_batch_sizes_for_topology(
    ind: Individual,
    max_batch_lo: int,
    enable_subgraph_batch_mut: bool,
    *,
    device_type_by_id: Optional[Dict[int, str]] = None,
    max_mutated_subgraphs: int = 1,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> Dict[int, int]:
    """
    Resample sub-graph batch sizes with PIM-aware constraints.

    Rule:
      - begin-node sub-graphs containing PIM devices may use a smaller local batch size
      - non-PIM sub-graphs stay at the global batch size
      - at most `max_mutated_subgraphs` begin nodes are allowed to deviate from the global batch size
    """
    begin_ids = _detect_begin_node_ids(ind.topology)
    upper = max(1, int(max_batch_lo))
    feasible_caps = _compute_feasible_batch_caps(ind, upper, feasibility_cfg)
    child_map = {int(nid): max(1, int(feasible_caps.get(int(nid), upper))) for nid in begin_ids}

    if (not enable_subgraph_batch_mut) or upper <= 1:
        return child_map

    pim_begin_ids = _eligible_pim_begin_ids(ind, begin_ids, device_type_by_id)
    if not pim_begin_ids:
        return child_map

    max_touch = max(0, int(max_mutated_subgraphs))
    if max_touch <= 0:
        return child_map

    if random.random() < 0.50:
        return child_map

    touch_k = random.randint(1, min(max_touch, len(pim_begin_ids)))
    for nid in random.sample(pim_begin_ids, k=touch_k):
        nid = int(nid)
        local_upper = max(1, int(feasible_caps.get(nid, upper)))
        if local_upper <= 1:
            child_map[nid] = 1
        else:
            child_map[nid] = int(random.randint(1, local_upper - 1 if local_upper > 1 else 1))
    return child_map


def _tweak_sub_graph_batch_sizes(
    parent_map: Dict[int, int],
    child: Individual,
    child_begin_ids: Sequence[int],
    max_batch_lo: int,
    *,
    device_type_by_id: Optional[Dict[int, str]] = None,
    max_mutated_subgraphs: int = 1,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> Dict[int, int]:
    """Micro-tune sub-graph batch sizes only around the parent's assignments."""
    upper = max(1, int(max_batch_lo))
    if not child_begin_ids:
        return {}

    feasible_caps = _compute_feasible_batch_caps(child, upper, feasibility_cfg)
    pim_begin_ids = set(_eligible_pim_begin_ids(child, child_begin_ids, device_type_by_id))
    child_map: Dict[int, int] = {}
    for nid in child_begin_ids:
        nid = int(nid)
        if nid in pim_begin_ids:
            base = int(parent_map.get(nid, upper))
            child_map[nid] = max(1, min(int(feasible_caps.get(nid, upper)), upper, base))
        else:
            child_map[nid] = max(1, int(feasible_caps.get(nid, upper)))

    if upper <= 1:
        return child_map

    mutable_ids = [nid for nid in child_begin_ids if int(nid) in pim_begin_ids]
    max_touch = max(0, int(max_mutated_subgraphs))
    if (not mutable_ids) or max_touch <= 0:
        return child_map

    touch_k = random.randint(1, min(max_touch, len(mutable_ids)))
    touched = False
    for nid in random.sample([int(x) for x in mutable_ids], k=touch_k):
        base = int(child_map[nid])
        op = random.choices(
            ["decrease", "increase", "resample"],
            weights=[0.50, 0.20, 0.30],
            k=1,
        )[0]
        if op == "decrease":
            child_map[nid] = int(random.randint(1, base))
        elif op == "increase":
            child_map[nid] = int(random.randint(base, max(base, int(feasible_caps.get(nid, upper)))))
        else:
            child_map[nid] = int(random.randint(1, max(1, int(feasible_caps.get(nid, upper)))))
        touched = touched or (child_map[nid] != base)

    if (not touched) and mutable_ids:
        nid = int(random.choice([int(x) for x in mutable_ids]))
        base = int(child_map[nid])
        if base > 1:
            child_map[nid] = base - 1
        elif upper > base:
            child_map[nid] = base + 1
    return child_map


def _mutate_sub_graph_batch_sizes(
    parent: Individual,
    child: Individual,
    mutation_kind: str,
    enable_subgraph_batch_mut: bool,
    *,
    device_type_by_id: Optional[Dict[int, str]] = None,
    max_mutated_subgraphs: int = 1,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> Dict[int, int]:
    child_begin_ids = _detect_begin_node_ids(child.topology)
    upper = max(1, int(getattr(child, "batch_size", 1)))

    # 不做 subgraph 级搜索时，仍需要 obey 内存可行性上限。
    if not enable_subgraph_batch_mut:
        caps = _compute_feasible_batch_caps(child, upper, feasibility_cfg)
        return {int(nid): max(1, int(caps.get(int(nid), upper))) for nid in child_begin_ids}

    if mutation_kind == "topology":
        return _sample_sub_graph_batch_sizes_for_topology(
            child,
            upper,
            enable_subgraph_batch_mut,
            device_type_by_id=device_type_by_id,
            max_mutated_subgraphs=max_mutated_subgraphs,
            feasibility_cfg=feasibility_cfg,
        )

    parent_map = {int(k): int(v) for k, v in getattr(parent, "sub_graph_batch_sizes", {}).items()}
    if (not parent_map) or (set(parent_map.keys()) != set(int(x) for x in child_begin_ids)):
        return _sample_sub_graph_batch_sizes_for_topology(
            child,
            upper,
            enable_subgraph_batch_mut,
            device_type_by_id=device_type_by_id,
            max_mutated_subgraphs=max_mutated_subgraphs,
            feasibility_cfg=feasibility_cfg,
        )

    r = random.random()
    if r < 0.60:
        out: Dict[int, int] = {}
        pim_begin_ids = set(_eligible_pim_begin_ids(child, child_begin_ids, device_type_by_id))
        feasible_caps = _compute_feasible_batch_caps(child, upper, feasibility_cfg)
        for nid in child_begin_ids:
            nid = int(nid)
            local_upper = max(1, int(feasible_caps.get(nid, upper)))
            if nid in pim_begin_ids:
                out[nid] = max(1, min(local_upper, int(parent_map.get(nid, local_upper))))
            else:
                out[nid] = local_upper
        return out
    if r < 0.80:
        return _tweak_sub_graph_batch_sizes(
            parent_map,
            child,
            child_begin_ids,
            upper,
            device_type_by_id=device_type_by_id,
            max_mutated_subgraphs=max_mutated_subgraphs,
            feasibility_cfg=feasibility_cfg,
        )
    return _sample_sub_graph_batch_sizes_for_topology(
        child,
        upper,
        enable_subgraph_batch_mut,
        device_type_by_id=device_type_by_id,
        max_mutated_subgraphs=max_mutated_subgraphs,
        feasibility_cfg=feasibility_cfg,
    )


def sample_attrs(topo: Topology, device_assign: DeviceAssign, req_type_num: int, init_cfg: InitConfig) -> Attrs:
    """
    v3: leaf arity is device_group_size; XP tags are exactly [ATTENTION, LINEAR] (maybe swapped).
    """
    attrs = Attrs()
    for nid in topo.iter_dfs():
        t = topo.gene(nid).ptype

        parallel_k = len(topo.children_of(nid))
        if parallel_k == 0:
            k = len(device_assign.leaf_to_devices[nid])
        else:
            k = parallel_k

        if t == Parallelism.DP:
            attrs.dp_attr[nid] = [_rand_positive_weights(k) for _ in range(req_type_num)]
        elif t == Parallelism.PP:
            attrs.pp_attr[nid] = _rand_positive_weights(k)
        elif t == Parallelism.TP:
            attrs.tp_attr[nid] = _rand_positive_weights(k)
        elif t == Parallelism.XP:
            tags = [XpTag.ATTENTION, XpTag.LINEAR]
            if random.random() < init_cfg.xp_swap_prob:
                tags.reverse()
            attrs.xp_attr[nid] = tags
        else:
            raise ValueError(f"Unexpected type: {t}")
    return attrs



def _normalized_init_mix(init_cfg: InitConfig) -> Tuple[float, float, float]:
    mix = [
        max(0.0, float(getattr(init_cfg, "p_pattern_seed_init", 0.50))),
        max(0.0, float(getattr(init_cfg, "p_stratified_init", 0.45))),
        max(0.0, float(getattr(init_cfg, "p_random_init", 0.05))),
    ]
    total = sum(mix)
    if total <= 0:
        return 0.50, 0.45, 0.05
    return mix[0] / total, mix[1] / total, mix[2] / total


def _plan_init_counts(total: int, init_cfg: InitConfig) -> Tuple[int, int, int]:
    p_pat, p_strat, p_rand = _normalized_init_mix(init_cfg)
    raw = [total * p_pat, total * p_strat, total * p_rand]
    counts = [int(math.floor(x)) for x in raw]
    remain = int(total - sum(counts))
    frac_order = sorted(
        range(3),
        key=lambda i: (raw[i] - counts[i], i),
        reverse=True,
    )
    for i in frac_order:
        if remain <= 0:
            break
        counts[i] += 1
        remain -= 1
    return max(0, counts[0]), max(0, counts[1]), max(0, counts[2])


def _build_device_type_to_ids(
    devices: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]],
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for did in devices:
        dtype = str((device_type_by_id or {}).get(int(did), str(int(did))))
        out.setdefault(dtype, []).append(int(did))
    return out


def _shuffle_device_type_to_ids(device_type_to_ids: Dict[str, List[int]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for dtype, ids in device_type_to_ids.items():
        vals = list(ids)
        random.shuffle(vals)
        out[str(dtype)] = vals
    return out


def _evaluate_individual(
    ind: Individual,
    *,
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> bool:
    try:
        ind.check_legality()
    except Exception:
        return False

    if not _repair_sub_graph_batch_sizes_by_feasibility(ind, feasibility_cfg):
        return False

    local_root_init = copy.deepcopy(root_init)
    root = try_decode_to_root(ind, local_root_init, attach_hardware_leaves=attach_hardware_leaves)
    if root is None:
        return False

    try:
        res = fitness_fn(root, ind.batch_size)
        thr, lat, f_dist, p_dist, thr_req = _parse_objectives(res)
        if (not math.isfinite(thr)) or (not math.isfinite(lat)):
            return False
        ind.throughput = thr
        ind.latency = lat
        ind.objectives = (thr, lat)
        ind.f_dist = f_dist
        ind.p_dist = p_dist
        ind.throughput_req = thr_req
    except Exception:
        return False
    return True


def _try_register_individual(
    pop: List[Individual],
    seen: set[str],
    ind: Individual,
    *,
    init_cfg: InitConfig,
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> bool:

    if not _individual_allowed_by_parallelism_filter(ind, init_cfg):
        return False

    if not _repair_sub_graph_batch_sizes_by_feasibility(ind, feasibility_cfg):
        return False

    uid = canonical_key(ind)
    ind.uid = uid
    if uid in seen:
        return False

    if not _evaluate_individual(
        ind,
        root_init=root_init,
        fitness_fn=fitness_fn,
        attach_hardware_leaves=attach_hardware_leaves,
        feasibility_cfg=feasibility_cfg,
    ):
        return False

    seen.add(uid)
    pop.append(ind)
    return True


def _make_individual_from_symbolic_seed(
    sym_root,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    req_type_num: int,
    devices: Sequence[int],
    device_type_to_ids: Dict[str, List[int]],
    device_type_by_id: Optional[Dict[int, str]] = None,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> Optional[Individual]:
    if not _symbolic_allowed_by_parallelism_filter(sym_root, init_cfg):
        return None

    batch_size = int(random.choice(init_cfg.batch_size_choices))
    try:
        ind = symbolic_to_individual(
            sym_root,
            device_type_to_ids=_shuffle_device_type_to_ids(device_type_to_ids),
            req_type_num=req_type_num,
            batch_size=batch_size,
            devices=list(devices),
            sub_graph_batch_sizes={},
        )
    except Exception:
        return None

    ind.sub_graph_batch_sizes = _sample_sub_graph_batch_sizes_for_topology(
        ind,
        batch_size,
        evo_cfg.enable_subgraph_batch_mut,
        device_type_by_id=device_type_by_id,
        max_mutated_subgraphs=evo_cfg.subgraph_batch_max_mutated,
        feasibility_cfg=feasibility_cfg,
    )
    return ind


def _fill_pattern_seeded_population(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    req_type_num: int,
    devices: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> None:
    if target_count <= 0:
        return

    start_len = len(pop)
    device_type_to_ids = _build_device_type_to_ids(devices, device_type_by_id)
    patterns = filter_init_patterns_by_parallelism(
        default_init_patterns(device_type_to_ids),
        _get_disabled_parallelisms(init_cfg),
    )
    if not patterns:
        return

    weights = [max(1e-9, float(p.weight)) for p in patterns]
    attempts = 0
    max_attempts = max(20, target_count * 40)

    while len(pop) - start_len < target_count and attempts < max_attempts:
        attempts += 1
        pat = random.choices(patterns, weights=weights, k=1)[0]
        ind = _make_individual_from_symbolic_seed(
            pat.instantiate(),
            init_cfg=init_cfg,
            evo_cfg=evo_cfg,
            req_type_num=req_type_num,
            devices=devices,
            device_type_to_ids=device_type_to_ids,
            device_type_by_id=device_type_by_id,
            feasibility_cfg=feasibility_cfg,
        )
        if ind is None:
            continue

        # print(ind.batch_size, format_topology(ind, True, True),"\n")

        _try_register_individual(
            pop,
            seen,
            ind,
            init_cfg=init_cfg,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
            feasibility_cfg=feasibility_cfg,
        )


def _fill_stratified_population(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    req_type_num: int,
    devices: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> None:
    if target_count <= 0:
        return

    start_len = len(pop)
    device_type_to_ids = _build_device_type_to_ids(devices, device_type_by_id)
    patterns = filter_init_patterns_by_parallelism(
        default_init_patterns(device_type_to_ids),
        _get_disabled_parallelisms(init_cfg),
    )
    if not patterns:
        return

    by_stratum: Dict[str, List[InitPatternSpec]] = {}
    for pat in patterns:
        by_stratum.setdefault(str(pat.stratum), []).append(pat)
    strata = list(sorted(by_stratum.keys()))
    if not strata:
        return

    attempts = 0
    max_attempts = max(30, target_count * 50)
    idx = 0

    while len(pop) - start_len < target_count and attempts < max_attempts:
        attempts += 1
        stratum = strata[idx % len(strata)]
        idx += 1
        plist = by_stratum[stratum]
        weights = [max(1e-9, float(p.weight)) for p in plist]
        pat = random.choices(plist, weights=weights, k=1)[0]
        ind = _make_individual_from_symbolic_seed(
            pat.instantiate(),
            init_cfg=init_cfg,
            evo_cfg=evo_cfg,
            req_type_num=req_type_num,
            devices=devices,
            device_type_to_ids=device_type_to_ids,
            device_type_by_id=device_type_by_id,
            feasibility_cfg=feasibility_cfg,
        )
        if ind is None:
            continue
        _try_register_individual(
            pop,
            seen,
            ind,
            init_cfg=init_cfg,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
            feasibility_cfg=feasibility_cfg,
        )


def _fill_random_population(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    req_type_num: int,
    devices: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> None:
    if target_count <= 0:
        return

    start_len = len(pop)
    attempts = 0
    max_attempts = max(40, target_count * 80)
    devices_list = list(devices)

    while len(pop) - start_len < target_count and attempts < max_attempts:
        attempts += 1
        topo = random_topology(init_cfg)

        try:
            da = sample_device_assign(
                topo,
                devices_list,
                shuffle=init_cfg.shuffle_devices_per_individual,
            )
        except Exception:
            continue

        attrs = sample_attrs(topo, da, req_type_num=req_type_num, init_cfg=init_cfg)
        batch_size = int(random.choice(init_cfg.batch_size_choices))
        ind = Individual(
            topology=topo,
            device_assign=da,
            attrs=attrs,
            devices=devices_list,
            req_type_num=req_type_num,
            batch_size=batch_size,
            sub_graph_batch_sizes={},
        )
        ind.sub_graph_batch_sizes = _sample_sub_graph_batch_sizes_for_topology(
            ind,
            batch_size,
            evo_cfg.enable_subgraph_batch_mut,
            device_type_by_id=device_type_by_id,
            max_mutated_subgraphs=evo_cfg.subgraph_batch_max_mutated,
            feasibility_cfg=feasibility_cfg,
        )
        _try_register_individual(
            pop,
            seen,
            ind,
            init_cfg=init_cfg,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
            feasibility_cfg=feasibility_cfg,
        )


def _fill_batch_variant_population(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
    device_type_by_id: Optional[Dict[int, str]] = None,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> None:
    """Final fallback: keep the structure/device assignment, resample batch knobs.

    This is intentionally aligned with the user's goal of broad batch exploration.
    It should only be used after structural seed pools and pure-random structure
    generation have already been tried.
    """
    if target_count <= 0 or not pop:
        return

    start_len = len(pop)
    base_pool = list(pop)
    attempts = 0
    max_attempts = max(60, target_count * 120)

    while len(pop) - start_len < target_count and attempts < max_attempts:
        attempts += 1
        donor = copy.deepcopy(random.choice(base_pool))

        batch_size = int(random.choice(init_cfg.batch_size_choices))
        donor.batch_size = batch_size
        donor.sub_graph_batch_sizes = _sample_sub_graph_batch_sizes_for_topology(
            donor,
            batch_size,
            evo_cfg.enable_subgraph_batch_mut,
            device_type_by_id=device_type_by_id,
            max_mutated_subgraphs=evo_cfg.subgraph_batch_max_mutated,
            feasibility_cfg=feasibility_cfg,
        )

        _try_register_individual(
            pop,
            seen,
            donor,
            init_cfg=init_cfg,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
            feasibility_cfg=feasibility_cfg,
        )


def _fill_numeric_variant_population(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
    device_type_by_id: Optional[Dict[int, str]] = None,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> None:
    """Final fallback after batch-only variants: keep validated structures/devices,
    but resample attrs and batch knobs to unlock additional unique seeds.
    """
    if target_count <= 0 or not pop:
        return

    start_len = len(pop)
    base_pool = list(pop)
    attempts = 0
    max_attempts = max(80, target_count * 160)

    while len(pop) - start_len < target_count and attempts < max_attempts:
        attempts += 1
        donor = copy.deepcopy(random.choice(base_pool))
        donor.attrs = sample_attrs(donor.topology, donor.device_assign, req_type_num=donor.req_type_num, init_cfg=init_cfg)
        batch_size = int(random.choice(init_cfg.batch_size_choices))
        donor.batch_size = batch_size
        donor.sub_graph_batch_sizes = _sample_sub_graph_batch_sizes_for_topology(
            donor,
            batch_size,
            evo_cfg.enable_subgraph_batch_mut,
            device_type_by_id=device_type_by_id,
            max_mutated_subgraphs=evo_cfg.subgraph_batch_max_mutated,
            feasibility_cfg=feasibility_cfg,
        )

        _try_register_individual(
            pop,
            seen,
            donor,
            init_cfg=init_cfg,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
            feasibility_cfg=feasibility_cfg,
        )


def _fill_population_mixed(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    req_type_num: int,
    devices: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> None:
    if target_count <= 0:
        return

    start_len = len(pop)

    # Phase 1: honour the configured mixture once.
    c_pat, c_strat, c_rand = _plan_init_counts(target_count, init_cfg)
    print(c_pat, c_strat, c_rand)

    _fill_pattern_seeded_population(
        pop, seen, c_pat,
        init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
        devices=devices, device_type_by_id=device_type_by_id,
        root_init=root_init, fitness_fn=fitness_fn,
        attach_hardware_leaves=attach_hardware_leaves,
        feasibility_cfg=feasibility_cfg,
    )
    _fill_stratified_population(
        pop, seen, c_strat,
        init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
        devices=devices, device_type_by_id=device_type_by_id,
        root_init=root_init, fitness_fn=fitness_fn,
        attach_hardware_leaves=attach_hardware_leaves,
        feasibility_cfg=feasibility_cfg,
    )
    _fill_random_population(
        pop, seen, c_rand,
        init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
        devices=devices, device_type_by_id=device_type_by_id,
        root_init=root_init, fitness_fn=fitness_fn,
        attach_hardware_leaves=attach_hardware_leaves,
        feasibility_cfg=feasibility_cfg,
    )

    # Phase 2: adaptive backfill. If high-quality seed pools are exhausted,
    # progressively lean harder on random legal sampling instead of failing
    # immediately after one no-progress round.
    stagnation_rounds = 0
    while len(pop) - start_len < target_count and stagnation_rounds < 4:
        before = len(pop)
        remaining = target_count - (len(pop) - start_len)

        # keep some seed pressure, but do not let a tiny seed library starve init
        seeded_quota = max(0, int(math.ceil(remaining * 0.40)))
        strat_quota = max(0, int(math.ceil(remaining * 0.30)))
        rand_quota = max(remaining, int(math.ceil(remaining * 1.50)))

        _fill_pattern_seeded_population(
            pop, seen, seeded_quota,
            init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
            devices=devices, device_type_by_id=device_type_by_id,
            root_init=root_init, fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
            feasibility_cfg=feasibility_cfg,
        )
        if len(pop) - start_len < target_count:
            _fill_stratified_population(
                pop, seen, strat_quota,
                init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
                devices=devices, device_type_by_id=device_type_by_id,
                root_init=root_init, fitness_fn=fitness_fn,
                attach_hardware_leaves=attach_hardware_leaves,
                feasibility_cfg=feasibility_cfg,
            )
        if len(pop) - start_len < target_count:
            _fill_random_population(
                pop, seen, rand_quota,
                init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
                devices=devices, device_type_by_id=device_type_by_id,
                root_init=root_init, fitness_fn=fitness_fn,
                attach_hardware_leaves=attach_hardware_leaves,
                feasibility_cfg=feasibility_cfg,
            )

        if len(pop) == before:
            # Last-resort top-up: exploit the already-valid structures by widening
            # batch/sub-graph batch exploration.
            remaining = target_count - (len(pop) - start_len)
            _fill_batch_variant_population(
                pop, seen, max(remaining * 2, remaining),
                init_cfg=init_cfg, evo_cfg=evo_cfg,
                root_init=root_init, fitness_fn=fitness_fn,
                attach_hardware_leaves=attach_hardware_leaves,
                device_type_by_id=device_type_by_id,
                feasibility_cfg=feasibility_cfg,
            )
            if len(pop) == before:
                _fill_numeric_variant_population(
                    pop, seen, max(remaining * 3, remaining),
                    init_cfg=init_cfg, evo_cfg=evo_cfg,
                    root_init=root_init, fitness_fn=fitness_fn,
                    attach_hardware_leaves=attach_hardware_leaves,
                    device_type_by_id=device_type_by_id,
                    feasibility_cfg=feasibility_cfg,
                )
            if len(pop) == before:
                stagnation_rounds += 1
            else:
                stagnation_rounds = 0
        else:
            stagnation_rounds = 0


def initialize_population(
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    *,
    req_type_num: int,
    devices: Sequence[int],
    root_init: RootInit,
    device_type_by_id: Optional[Dict[int, str]] = None,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool = True,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> List[Individual]:
    """Pattern/device-abstraction aware mixed initialization with global dedup."""
    pop: List[Individual] = []
    seen: set[str] = set()

    _fill_population_mixed(
        pop,
        seen,
        init_cfg.population_size,
        init_cfg=init_cfg,
        evo_cfg=evo_cfg,
        req_type_num=req_type_num,
        devices=list(devices),
        device_type_by_id=device_type_by_id,
        root_init=root_init,
        fitness_fn=fitness_fn,
        attach_hardware_leaves=attach_hardware_leaves,
        feasibility_cfg=feasibility_cfg,
    )

    if len(pop) < init_cfg.population_size:
        raise RuntimeError(
            f"Init failed: {len(pop)}/{init_cfg.population_size}. "
            f"pattern/stratified/random mix could not produce enough unique valid individuals."
        )

    fronts = fast_nondominated_sort(pop)
    for f in fronts:
        crowding_distance(f)
    pop.sort(key=lambda x: (10**9 if x.pareto_rank is None else x.pareto_rank, -x.crowding))
    return pop


def initialize_population_with_seeds(
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    *,
    req_type_num: int,
    devices: Sequence[int],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    seed_roots: Optional[Sequence[Any]] = None,
    seed_individuals: Optional[Sequence[Individual]] = None,
    attach_hardware_leaves: bool = True,
    strict_device_partition: bool = True,
    device_type_by_id: Optional[Dict[int, str]] = None,
    feasibility_cfg: Optional[FeasibilityConfig] = None,
) -> List[Individual]:
    """Seed-aware initialization: external seeds first, then mixed-fill the remainder with global dedup."""
    pop: List[Individual] = []
    seen: set[str] = set()
    devices_list = list(devices)

    def try_add(ind: Individual) -> None:
        if not getattr(ind, "sub_graph_batch_sizes", None):
            ind.sub_graph_batch_sizes = _sample_sub_graph_batch_sizes_for_topology(
                ind,
                int(getattr(ind, "batch_size", 1)),
                evo_cfg.enable_subgraph_batch_mut,
                device_type_by_id=device_type_by_id,
                max_mutated_subgraphs=evo_cfg.subgraph_batch_max_mutated,
                feasibility_cfg=feasibility_cfg,
            )
        _try_register_individual(
            pop,
            seen,
            ind,
            init_cfg=init_cfg,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
            feasibility_cfg=feasibility_cfg,
        )

    if seed_individuals:
        for ind in seed_individuals:
            try_add(copy.deepcopy(ind))

    if seed_roots:
        for root in seed_roots:
            try:
                batch_size = int(random.choice(init_cfg.batch_size_choices))
                local_root = copy.deepcopy(root)
                ind = individual_from_pcase_root(
                    local_root,
                    devices=devices_list,
                    req_type_num=req_type_num,
                    batch_size=batch_size,
                    strict_device_partition=strict_device_partition,
                )
                ind.sub_graph_batch_sizes = _sample_sub_graph_batch_sizes_for_topology(
                    ind,
                    batch_size,
                    evo_cfg.enable_subgraph_batch_mut,
                    device_type_by_id=device_type_by_id,
                    max_mutated_subgraphs=evo_cfg.subgraph_batch_max_mutated,
                    feasibility_cfg=feasibility_cfg,
                )
            except Exception:
                continue
            try_add(ind)

    if len(pop) > init_cfg.population_size:
        return nsga2_environmental_select(pop, init_cfg.population_size)

    remaining = init_cfg.population_size - len(pop)
    if remaining > 0:
        _fill_population_mixed(
            pop,
            seen,
            remaining,
            init_cfg=init_cfg,
            evo_cfg=evo_cfg,
            req_type_num=req_type_num,
            devices=devices_list,
            device_type_by_id=device_type_by_id,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
            feasibility_cfg=feasibility_cfg,
        )

    if len(pop) < init_cfg.population_size:
        raise RuntimeError(
            f"Init with seeds failed: {len(pop)}/{init_cfg.population_size}. "
            f"Not enough unique valid individuals after seed ingestion and mixed initialization."
        )

    fronts = fast_nondominated_sort(pop)
    for f in fronts:
        crowding_distance(f)
    pop.sort(key=lambda x: (10**9 if x.pareto_rank is None else x.pareto_rank, -x.crowding))
    return pop

def _parse_objectives(fitness_result: Any) -> Tuple[float, float, List[Any], List[Any], float]:
    """Parse evaluator output into (throughput, latency).

    In this codebase, pareto-mode simulator outputs are often shaped like:
      [throughput, p99_tpot_or_summary, finished_dist, processed_dist]
    where the second item may be a numeric latency *or* a status string such as
    ``"[Finished] finished=0\n"`` when no request has completed yet.

    We therefore parse more defensively:
      - first numeric field -> throughput
      - first later numeric scalar -> latency
      - if no later numeric scalar exists, assign a large finite latency penalty
        instead of rejecting the individual outright
    """
    if fitness_result is None:
        raise ValueError("fitness_result is None")

    latency_penalty = 1e5

    # tuple/list
    if isinstance(fitness_result, (tuple, list)) and len(fitness_result) >= 1:
        thr = float(fitness_result[0])
        lat = None
        for item in list(fitness_result)[1:]:
            if isinstance(item, (int, float)):
                lat = float(item)
                break
        if lat is None:
            lat = latency_penalty
        f_dist = fitness_result[2]
        p_dist = fitness_result[3]
        thr_req = fitness_result[4]

        return thr, lat, f_dist, p_dist, thr_req

    # # dict
    # if isinstance(fitness_result, dict):
    #     lower = {str(k).lower(): v for k, v in fitness_result.items()}
    #
    #     def pick(*names):
    #         for n in names:
    #             if n in lower:
    #                 return lower[n]
    #         return None
    #
    #     thr = pick("T", "throughput", "tps", "qps", "req_s", "req_per_s", "requests_per_s")
    #     lat = pick("L","latency", "p95_latency", "p99_latency", "p90_latency", "tail_latency", "p95", "p99", "lat_ms", "lat_s")
    #
    #     if thr is None:
    #         raise ValueError(f"Cannot parse throughput from dict keys={sorted(lower.keys())}")
    #     return float(thr), float(latency_penalty if lat is None else lat)
    #
    # # single number: old behavior (maximize this)
    # if isinstance(fitness_result, (int, float)):
    #     return float(fitness_result), latency_penalty

    raise ValueError(f"Unsupported fitness_result type: {type(fitness_result)}")


def dominates(a: Individual, b: Individual) -> bool:
    """Return True if a Pareto-dominates b (maximize throughput, minimize latency)."""
    if a.objectives is None or b.objectives is None:
        return False
    a_thr, a_lat = a.objectives
    b_thr, b_lat = b.objectives
    return (a_thr >= b_thr and a_lat <= b_lat) and (a_thr > b_thr or a_lat < b_lat)


def fast_nondominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    """NSGA-II fast non-dominated sorting. Assigns pareto_rank."""
    S: Dict[str, List[Individual]] = {}
    n: Dict[str, int] = {}
    fronts: List[List[Individual]] = [[]]

    for p in pop:
        S[p.uid] = []
        n[p.uid] = 0
        for q in pop:
            if p is q:
                continue
            if dominates(p, q):
                S[p.uid].append(q)
            elif dominates(q, p):
                n[p.uid] += 1
        if n[p.uid] == 0:
            p.pareto_rank = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[Individual] = []
        for p in fronts[i]:
            for q in S[p.uid]:
                n[q.uid] -= 1
                if n[q.uid] == 0:
                    q.pareto_rank = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def crowding_distance(front: List[Individual]) -> None:
    """Compute NSGA-II crowding distance in-place for a front."""
    if not front:
        return
    for p in front:
        p.crowding = 0.0

    # Objective 0: throughput (max)
    # Objective 1: latency (min)  -> treat as negative for sorting so "larger is better" is consistent? 
    # We'll implement separately with correct normalization.
    thr_vals = [p.objectives[0] for p in front]  # type: ignore[index]
    lat_vals = [p.objectives[1] for p in front]  # type: ignore[index]

    # Throughput (ascending for distance calc)
    order = sorted(range(len(front)), key=lambda i: thr_vals[i])
    front[order[0]].crowding = float("inf")
    front[order[-1]].crowding = float("inf")
    thr_min, thr_max = thr_vals[order[0]], thr_vals[order[-1]]
    denom = (thr_max - thr_min) if thr_max != thr_min else 1.0
    for j in range(1, len(order) - 1):
        i_prev, i_next, i_cur = order[j - 1], order[j + 1], order[j]
        front[i_cur].crowding += (thr_vals[i_next] - thr_vals[i_prev]) / denom

    # Latency (ascending: smaller is better)
    order = sorted(range(len(front)), key=lambda i: lat_vals[i])
    front[order[0]].crowding = float("inf")
    front[order[-1]].crowding = float("inf")
    lat_min, lat_max = lat_vals[order[0]], lat_vals[order[-1]]
    denom = (lat_max - lat_min) if lat_max != lat_min else 1.0
    for j in range(1, len(order) - 1):
        i_prev, i_next, i_cur = order[j - 1], order[j + 1], order[j]
        # For latency, larger normalized gap means more isolated => bigger crowding (still good)
        front[i_cur].crowding += (lat_vals[i_next] - lat_vals[i_prev]) / denom


def nsga2_environmental_select(pop: List[Individual], N: int) -> List[Individual]:
    """Select next generation of size N from pop (already evaluated)."""
    fronts = fast_nondominated_sort(pop)
    next_pop: List[Individual] = []
    for f in fronts:
        crowding_distance(f)
        if len(next_pop) + len(f) <= N:
            next_pop.extend(f)
        else:
            # sort by crowding descending
            f_sorted = sorted(f, key=lambda x: x.crowding, reverse=True)
            next_pop.extend(f_sorted[: max(0, N - len(next_pop))])
            break
    return next_pop


def tournament_select(pop: List[Individual], k: int) -> Individual:
    """Binary tournament on (pareto_rank asc, crowding desc)."""
    cand = random.sample(pop, k=min(k, len(pop)))
    # Ensure ranks/crowding exist
    return min(
        cand,
        key=lambda x: (
            10**9 if x.pareto_rank is None else x.pareto_rank,
            -float("inf") if x.crowding is None else -x.crowding,
        ),
    )


def _build_rewrite_pattern_library(cfg: EvoConfig) -> List[Any]:
    mode = str(getattr(cfg, "rewrite_pattern_mode", "static")).strip().lower()
    if mode in ("parameterized", "parametric"):
        return list(
            default_parameterized_patterns(
                max_split_children=int(getattr(cfg, "param_rewrite_max_children", 4)),
                partition_budget=int(getattr(cfg, "param_rewrite_partition_budget", 12)),
                candidate_budget=int(getattr(cfg, "param_rewrite_candidate_budget", 12)),
            )
        )
    if mode == "hybrid":
        return list(default_patterns()) + list(
            default_parameterized_patterns(
                max_split_children=int(getattr(cfg, "param_rewrite_max_children", 4)),
                partition_budget=int(getattr(cfg, "param_rewrite_partition_budget", 12)),
                candidate_budget=int(getattr(cfg, "param_rewrite_candidate_budget", 12)),
            )
        )
    return list(default_patterns())


def _choose_rewrite_family(cfg: EvoConfig) -> RewriteFamily:
    families = [
        RewriteFamily.SKELETON_EXPANSION,
        RewriteFamily.LOCAL_REFINEMENT,
        RewriteFamily.RELABEL,
        RewriteFamily.REPARTITION,
        RewriteFamily.ROLLBACK,
    ]
    weights = [
        max(1e-9, float(cfg.p_skeleton_expand)),
        max(1e-9, float(cfg.p_local_refine)),
        max(1e-9, float(cfg.p_relabel)),
        max(1e-9, float(cfg.p_repartition)),
        max(1e-9, float(cfg.p_rollback)),
    ]
    return random.choices(families, weights=weights, k=1)[0]


def rewrite_mutation(
    ind: Individual,
    cfg: EvoConfig,
    init_cfg: InitConfig,
    *,
    devices: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]] = None,
) -> Individual:
    """Bounded multi-step closure rewrite.

    In one mutation call, perform up to `cfg.rewrite_max_steps` rewrite steps.
    If the symbolic tree contains unclosed nodes, force LOCAL_REFINEMENT first.
    Only materialize back to Individual when the symbolic tree becomes fully
    materializable within the step budget; otherwise fall back to the parent.
    """
    disabled_parallelisms = _get_disabled_parallelisms(init_cfg)
    sym_root = individual_to_symbolic(
        ind,
        device_type_by_id=device_type_by_id or {int(d): str(int(d)) for d in devices},
    )
    if symbolic_contains_disabled_parallelisms(sym_root, disabled_parallelisms):
        return copy.deepcopy(ind)

    patterns = filter_rewrite_patterns_by_parallelism(_build_rewrite_pattern_library(cfg), disabled_parallelisms)
    if not patterns:
        return copy.deepcopy(ind)
    engine = RewriteEngine(patterns, rng=random)

    max_steps = max(1, int(getattr(cfg, "rewrite_max_steps", 1)))
    changed_any = False

    for _ in range(max_steps):
        if has_open_nodes(sym_root):
            changed = engine.rewrite_random(sym_root, family=RewriteFamily.LOCAL_REFINEMENT)
            if not changed:
                break
        else:
            family = _choose_rewrite_family(cfg)
            changed = engine.rewrite_random(sym_root, family=family)
            if not changed:
                changed = engine.rewrite_random(sym_root, family=None)
            if not changed:
                break

        changed_any = True

        if is_materializable(sym_root):
            break

    if (not changed_any) or (not is_materializable(sym_root)):
        return copy.deepcopy(ind)

    if symbolic_contains_disabled_parallelisms(sym_root, disabled_parallelisms):
        return copy.deepcopy(ind)

    try:
        child = symbolic_to_individual(
            sym_root,
            device_type_to_ids=_build_device_type_to_ids(devices, device_type_by_id),
            req_type_num=ind.req_type_num,
            batch_size=int(getattr(ind, "batch_size", 1)),
            devices=list(devices),
            sub_graph_batch_sizes={},
        )
        if not _individual_allowed_by_parallelism_filter(child, init_cfg):
            return copy.deepcopy(ind)
        return child
    except Exception:
        return copy.deepcopy(ind)


def mapping_refinement(ind: Individual, *, device_type_by_id: Optional[Dict[int, str]] = None) -> Individual:
    """Keep symbolic semantics fixed and only adjust concrete same-type mapping."""
    topo = ind.topology
    leaves = sorted(topo.leaf_parallel_nodes())
    if len(leaves) <= 1:
        return ind

    new_da = DeviceAssign(leaf_to_devices={lid: grp[:] for lid, grp in ind.device_assign.leaf_to_devices.items()})
    dtype_map = device_type_by_id or {int(d): str(int(d)) for d in ind.devices}

    by_type: Dict[str, List[Tuple[int, int]]] = {}
    for lid in leaves:
        grp = new_da.leaf_to_devices.get(lid, [])
        for idx, did in enumerate(grp):
            by_type.setdefault(str(dtype_map[int(did)]), []).append((lid, idx))

    candidate_types = [t for t, pos in by_type.items() if len(pos) >= 2]
    if not candidate_types:
        return ind

    dtype = random.choice(candidate_types)
    (l1, i1), (l2, i2) = random.sample(by_type[dtype], 2)
    if l1 == l2:
        grp = new_da.leaf_to_devices[l1]
        random.shuffle(grp)
    else:
        g1 = new_da.leaf_to_devices[l1]
        g2 = new_da.leaf_to_devices[l2]
        g1[i1], g2[i2] = g2[i2], g1[i1]

    return Individual(
        topology=copy.deepcopy(ind.topology),
        device_assign=copy.deepcopy(new_da),
        attrs=copy.deepcopy(ind.attrs),
        devices=list(ind.devices),
        req_type_num=ind.req_type_num,
        batch_size=int(getattr(ind, "batch_size", 1)),
        sub_graph_batch_sizes=dict(getattr(ind, "sub_graph_batch_sizes", {})),
    )



def _clone_attrs(attrs: Attrs) -> Attrs:
    return Attrs(
        dp_attr={nid: [row[:] for row in mat] for nid, mat in attrs.dp_attr.items()},
        pp_attr={nid: vec[:] for nid, vec in attrs.pp_attr.items()},
        tp_attr={nid: vec[:] for nid, vec in attrs.tp_attr.items()},
        xp_attr={nid: tags[:] for nid, tags in attrs.xp_attr.items()},
    )


def _build_symbolic_node_map(
    ind: Individual,
    device_type_by_id: Optional[Dict[int, str]],
) -> Dict[int, Any]:
    dtype_map = device_type_by_id or {int(d): str(int(d)) for d in ind.devices}
    sym_root = individual_to_symbolic(ind, device_type_by_id=dtype_map)
    return {
        int(nid): sym_node
        for nid, sym_node in zip(ind.topology.iter_dfs(), sym_root.walk())
    }


def _pattern_candidate_shape_ok(
    hint: Dict[str, Any],
    *,
    ptype: Parallelism,
    arity: int,
    req_type_num: int,
) -> bool:
    if ptype == Parallelism.DP:
        val = hint.get("dp_attr")
        return (
            isinstance(val, list)
            and len(val) == req_type_num
            and all(isinstance(row, list) and len(row) == arity for row in val)
        )
    if ptype == Parallelism.PP:
        val = hint.get("pp_attr")
        return isinstance(val, list) and len(val) == arity
    if ptype == Parallelism.TP:
        val = hint.get("tp_attr")
        return isinstance(val, list) and len(val) == arity
    if ptype == Parallelism.XP:
        val = hint.get("xp_attr")
        return isinstance(val, list) and len(val) == 2 and set(val) == {XpTag.ATTENTION, XpTag.LINEAR}
    return False


def _apply_numeric_pattern(
    new_attrs: Attrs,
    *,
    nid: int,
    ptype: Parallelism,
    arity: int,
    req_type_num: int,
    sym_node: Any,
    numeric_patterns: Sequence[Any],
) -> bool:
    matches = [pat for pat in numeric_patterns if pat.matches(sym_node)]
    if not matches:
        return False

    pat = random.choices(
        matches,
        weights=[max(1e-9, float(pat.weight)) for pat in matches],
        k=1,
    )[0]
    hint = pat.choose_candidate(rng=random)
    if not hint or (not _pattern_candidate_shape_ok(hint, ptype=ptype, arity=arity, req_type_num=req_type_num)):
        return False

    if ptype == Parallelism.DP:
        new_attrs.dp_attr[nid] = copy.deepcopy(hint["dp_attr"])
    elif ptype == Parallelism.PP:
        new_attrs.pp_attr[nid] = copy.deepcopy(hint["pp_attr"])
    elif ptype == Parallelism.TP:
        new_attrs.tp_attr[nid] = copy.deepcopy(hint["tp_attr"])
    elif ptype == Parallelism.XP:
        new_attrs.xp_attr[nid] = copy.deepcopy(hint["xp_attr"])
    else:
        return False
    return True


def _numeric_log_noise(
    new_attrs: Attrs,
    *,
    nid: int,
    ptype: Parallelism,
    sigma: float,
) -> bool:
    def perturb(x: float) -> float:
        return max(1e-9, float(x) * math.exp(random.gauss(0.0, sigma)))

    if ptype == Parallelism.DP:
        mat = new_attrs.dp_attr[nid]
        if not mat:
            return False
        row_ids = random.sample(range(len(mat)), k=random.randint(1, len(mat)))
        changed = False
        for r in row_ids:
            cols = random.sample(range(len(mat[r])), k=random.randint(1, len(mat[r])))
            for c in cols:
                before = mat[r][c]
                mat[r][c] = perturb(before)
                changed = changed or (mat[r][c] != before)
        return changed
    if ptype == Parallelism.PP:
        vec = new_attrs.pp_attr[nid]
        idxs = random.sample(range(len(vec)), k=random.randint(1, len(vec)))
        for i in idxs:
            vec[i] = perturb(vec[i])
        return True
    if ptype == Parallelism.TP:
        vec = new_attrs.tp_attr[nid]
        idxs = random.sample(range(len(vec)), k=random.randint(1, len(vec)))
        for i in idxs:
            vec[i] = perturb(vec[i])
        return True
    if ptype == Parallelism.XP:
        tags = new_attrs.xp_attr[nid]
        new_attrs.xp_attr[nid] = [tags[1], tags[0]]
        return True
    return False


def _numeric_pair_rebalance(
    new_attrs: Attrs,
    *,
    nid: int,
    ptype: Parallelism,
    sigma: float,
) -> bool:
    factor = math.exp(random.gauss(0.0, sigma))

    if ptype == Parallelism.DP:
        mat = new_attrs.dp_attr[nid]
        row_ids = [r for r, row in enumerate(mat) if len(row) >= 2]
        if not row_ids:
            return False
        r = random.choice(row_ids)
        i, j = random.sample(range(len(mat[r])), 2)
        mat[r][i] = max(1e-9, float(mat[r][i]) * factor)
        mat[r][j] = max(1e-9, float(mat[r][j]) / factor)
        return True

    if ptype == Parallelism.PP:
        vec = new_attrs.pp_attr[nid]
        if len(vec) < 2:
            return False
        i, j = random.sample(range(len(vec)), 2)
        vec[i] = max(1e-9, float(vec[i]) * factor)
        vec[j] = max(1e-9, float(vec[j]) / factor)
        return True

    if ptype == Parallelism.TP:
        vec = new_attrs.tp_attr[nid]
        if len(vec) < 2:
            return False
        i, j = random.sample(range(len(vec)), 2)
        vec[i] = max(1e-9, float(vec[i]) * factor)
        vec[j] = max(1e-9, float(vec[j]) / factor)
        return True

    if ptype == Parallelism.XP:
        tags = new_attrs.xp_attr[nid]
        new_attrs.xp_attr[nid] = [tags[1], tags[0]]
        return True

    return False


def _numeric_partial_reset(
    new_attrs: Attrs,
    *,
    nid: int,
    ptype: Parallelism,
) -> bool:
    if ptype == Parallelism.DP:
        mat = new_attrs.dp_attr[nid]
        if not mat:
            return False
        r = random.randrange(len(mat))
        cols = random.sample(range(len(mat[r])), k=random.randint(1, len(mat[r])))
        for c in cols:
            mat[r][c] = max(1e-9, random.lognormvariate(0.0, 0.8))
        return True

    if ptype == Parallelism.PP:
        vec = new_attrs.pp_attr[nid]
        idxs = random.sample(range(len(vec)), k=random.randint(1, len(vec)))
        for i in idxs:
            vec[i] = max(1e-9, random.lognormvariate(0.0, 0.8))
        return True

    if ptype == Parallelism.TP:
        vec = new_attrs.tp_attr[nid]
        idxs = random.sample(range(len(vec)), k=random.randint(1, len(vec)))
        for i in idxs:
            vec[i] = max(1e-9, random.lognormvariate(0.0, 0.8))
        return True

    if ptype == Parallelism.XP:
        tags = new_attrs.xp_attr[nid]
        new_attrs.xp_attr[nid] = [tags[1], tags[0]]
        return True

    return False


def numeric_mutation(
    ind: Individual,
    cfg: EvoConfig,
    init_cfg: InitConfig,
    *,
    device_type_by_id: Optional[Dict[int, str]] = None,
) -> Individual:
    del init_cfg
    topo = ind.topology
    new_attrs = _clone_attrs(ind.attrs)
    sigma = max(1e-9, float(cfg.weight_noise_sigma))
    max_targets = max(1, int(getattr(cfg, "numeric_mutation_max_targets", 1)))
    numeric_patterns = default_numeric_patterns()
    node_to_symbolic = _build_symbolic_node_map(ind, device_type_by_id)

    candidates = [int(nid) for nid in topo.iter_dfs()]
    if not candidates:
        return copy.deepcopy(ind)

    target_k = random.randint(1, min(max_targets, len(candidates)))
    target_nodes = random.sample(candidates, k=target_k)

    changed_any = False
    for nid in target_nodes:
        ptype = topo.gene(nid).ptype
        arity = len(topo.children_of(nid))
        if arity == 0:
            arity = len(ind.device_assign.leaf_to_devices.get(nid, []))
        sym_node = node_to_symbolic.get(int(nid))

        ops: List[str] = []
        if ptype == Parallelism.XP:
            ops.extend(["pattern", "swap"])
        else:
            ops.extend(["noise", "rebalance", "reset"])
            if sym_node is not None:
                matched = [pat for pat in numeric_patterns if pat.matches(sym_node)]
                if matched:
                    ops.append("pattern")

        if not ops:
            continue

        random.shuffle(ops)
        changed_this = False
        for op in ops:
            if op == "pattern" and sym_node is not None:
                changed_this = _apply_numeric_pattern(
                    new_attrs,
                    nid=int(nid),
                    ptype=ptype,
                    arity=arity,
                    req_type_num=ind.req_type_num,
                    sym_node=sym_node,
                    numeric_patterns=numeric_patterns,
                )
            elif op == "noise":
                changed_this = _numeric_log_noise(new_attrs, nid=int(nid), ptype=ptype, sigma=sigma)
            elif op == "rebalance":
                changed_this = _numeric_pair_rebalance(new_attrs, nid=int(nid), ptype=ptype, sigma=sigma)
            elif op == "reset":
                changed_this = _numeric_partial_reset(new_attrs, nid=int(nid), ptype=ptype)
            elif op == "swap":
                changed_this = _numeric_log_noise(new_attrs, nid=int(nid), ptype=ptype, sigma=sigma)

            if changed_this:
                changed_any = True
                break

    if not changed_any:
        return copy.deepcopy(ind)

    return Individual(
        copy.deepcopy(ind.topology),
        copy.deepcopy(ind.device_assign),
        new_attrs,
        devices=list(ind.devices),
        req_type_num=ind.req_type_num,
        batch_size=int(getattr(ind, "batch_size", 1)),
        sub_graph_batch_sizes=dict(getattr(ind, "sub_graph_batch_sizes", {})),
    )


def topology_mutation(ind: Individual, cfg: EvoConfig, init_cfg: InitConfig, *, devices: Sequence[int], device_type_by_id: Optional[Dict[int, str]] = None) -> Individual:
    """Backward-compatible alias for the new rewrite-based mutation."""
    return rewrite_mutation(ind, cfg, init_cfg, devices=devices, device_type_by_id=device_type_by_id)

def device_mutation(ind: Individual, *, p_swap=0.6, p_move=0.3, p_shuffle=0.1, device_type_by_id: Optional[Dict[int, str]] = None) -> Individual:
    """Backward-compatible alias. Device mutation is absorbed into rewrite mutation; this now only refines concrete mapping."""
    del p_swap, p_move, p_shuffle
    return mapping_refinement(ind, device_type_by_id=device_type_by_id)

def evolve(
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    *,
    req_type_num: int,
    devices: Sequence[int],
    root_init: RootInit,
    device_type_by_id: Optional[Dict[int, str]] = None,
    fitness_fn: Callable[[Any, int], float],
    feasibility_cfg: Optional[FeasibilityConfig] = None,
    # 使用经验进行种群初始化
    with_pop_seeds: bool = False,
    pop_seed_roots: Optional[Sequence[Any]] = None,
    pop_seed_individuals: Optional[Sequence[Individual]] = None,
    # decoder 是否把 HW 节点挂在 leaf 下（一般 True）
    attach_hardware_leaves: bool = True,
    # 随机数种子
    random_seed: Optional[int] = None,
    dse_out: Optional[str] = None,

) -> Tuple[Individual, List[Individual]]:
    '''
    # 该函数的关键行为/隐含假设
    # 1. 丢弃策略：任何非法/解码失败/仿真失败就扔掉（不会修复）
    # 2. v3的变异：rewrite-based mutation，直接在 symbolic 上改，允许多步，允许不完全 materialize（但必须 fully decode 成 topology），只要最后能 materialize 就行。相比 v2 的“单步、完全 materialize、失败回退”更灵活，能探索更大空间。
    # 3. 缓存：相同 uid 的个体不会重复跑仿真，能省很多时间（仿真很贵，这点很关键）
    # 4. attempt_limit：防止“丢弃太多导致卡死”
    '''

    if random_seed is not None:
        random.seed(random_seed)

    devices_list = list(devices)

    if not with_pop_seeds:
        pop = initialize_population(
            init_cfg,
            evo_cfg,
            req_type_num=req_type_num,
            devices=devices_list,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
            device_type_by_id=device_type_by_id,
            feasibility_cfg=feasibility_cfg,
        )
    else:
        pop = initialize_population_with_seeds(
            init_cfg,
            evo_cfg,
            req_type_num=req_type_num,
            devices=devices_list,
            root_init=root_init,
            fitness_fn=fitness_fn,
            seed_roots=pop_seed_roots,
            seed_individuals=pop_seed_individuals,
            attach_hardware_leaves=attach_hardware_leaves,
            strict_device_partition=True,
            device_type_by_id=device_type_by_id,
            feasibility_cfg=feasibility_cfg,
        )

    for init_ind in pop:
        # dse scatter point
        if dse_out:
            log_individual_json(init_ind, dse_out)
            print("\n [init ind] ", init_ind.uid, init_ind.batch_size, init_ind.throughput, init_ind.latency)
            print(format_topology(init_ind, True, True))
            print("\n")
            save_individual_json(init_ind, f"./debug/individuals/init_{init_ind.uid}.json")

    cache: Dict[str, Tuple[float, float]] = {}

    def eval_ind(ind: Individual) -> Optional[Individual]:
        try:
            # 合法性检查共检查3部分：
            # 1. topology 合法（DP/PP/TP 顺序约束、XP 子数=2）
            # 2. device partition 完整、无重复、覆盖全部 devices
            # 3. attrs shape 匹配“有效 arity”（含 leaf=设备组大小）
            ind.check_legality()
        except Exception:
            return None
        if not _individual_allowed_by_parallelism_filter(ind, init_cfg):
            return None
        if not _repair_sub_graph_batch_sizes_by_feasibility(ind, feasibility_cfg):
            return None
        key = canonical_key(ind)
        ind.uid = key
        if evo_cfg.enable_cache and key in cache:
            thr, lat = cache[key]
            ind.throughput = thr
            ind.latency = lat
            ind.objectives = (thr, lat)
            return ind
        local_root_init = copy.deepcopy(root_init)
        root = try_decode_to_root(ind, local_root_init, attach_hardware_leaves=attach_hardware_leaves)
        if root is None:
            return None
        try:
            res = fitness_fn(root, ind.batch_size)
            thr, lat, f_dist, p_dist, thr_req = _parse_objectives(res)
            if (not math.isfinite(thr)) or (not math.isfinite(lat)):
                return None
            ind.throughput = thr
            ind.latency = lat
            ind.objectives = (thr, lat)
            ind.f_dist = f_dist
            ind.p_dist = p_dist
            ind.throughput_req = thr_req

            # dse scatter point
            if dse_out:
                log_individual_json(ind, dse_out)
                # print("\n [eval ind] ", ind.uid, ind.batch_size, ind.throughput, ind.latency)
                # print(format_topology(ind, True, True))
                # save_individual_json(ind, f"./debug/individuals/eval_{ind.uid}.json")
                # print("\n")


        except Exception as e:
            print("evaluate/write failed:", repr(e))
            # fitness 阶段报错（含 ValueError），视为 invalid 个体丢弃
            return None

        if evo_cfg.enable_cache:
            cache[key] = (float(ind.throughput), float(ind.latency))
        return ind

    # 默认每代新产生 population_size - elite_size 个 offspring
    offspring_target = evo_cfg.offspring_size or init_cfg.population_size
    attempt_limit = evo_cfg.max_attempt_factor * max(1, offspring_target)

    # 每一代主要做四件事：排序 → 精英保留 → 生成 offspring → 更新种群
    for _gen in range(evo_cfg.generations):
        print("gen:", _gen)

        # 1) Assign Pareto rank + crowding distance for current population
        fronts = fast_nondominated_sort(pop)
        for f in fronts:
            crowding_distance(f)

        if fronts and fronts[0]:
            # Print a quick snapshot of the current Pareto front (top few by throughput)
            snap = sorted(fronts[0], key=lambda x: float('-inf') if x.throughput is None else x.throughput, reverse=True)[:5]
            print("  front0 size:", len(fronts[0]), " snapshot:", [(x.throughput, x.latency) for x in snap])

        # 2) Create offspring
        offspring: List[Individual] = []
        seen: set[str] = set()
        attempts = 0

        # 循环生成 offspring
        while len(offspring) < offspring_target and attempts < attempt_limit:
            attempts += 1
            # 选择父代，锦标赛选择：从 pop 随机抽 k 个，取 fitness 最大的那个。k 越大选择压力越强。
            parent = tournament_select(pop, evo_cfg.tournament_k)

            r = random.random()
            if r < evo_cfg.p_rewrite_mut:
                child = rewrite_mutation(
                    parent,
                    evo_cfg,
                    init_cfg,
                    devices=devices_list,
                    device_type_by_id=device_type_by_id,
                )
                mutation_kind = "topology"
            elif r < evo_cfg.p_rewrite_mut + evo_cfg.p_numeric_mut:
                child = numeric_mutation(parent, evo_cfg, init_cfg, device_type_by_id=device_type_by_id)
                mutation_kind = "numeric"
            else:
                child = mapping_refinement(parent, device_type_by_id=device_type_by_id)
                mutation_kind = "mapping_refine"

            # Always resample batch_size on every mutation, then mutate sub-graph batch sizes.
            # NOTE: some mutation ops may return the parent object; avoid in-place edits.
            child_batch_size = int(random.choice(init_cfg.batch_size_choices))
            child = copy.deepcopy(child)
            child.batch_size = child_batch_size
            child.sub_graph_batch_sizes = _mutate_sub_graph_batch_sizes(
                parent,
                Individual(
                    topology=copy.deepcopy(child.topology),
                    device_assign=copy.deepcopy(child.device_assign),
                    attrs=copy.deepcopy(child.attrs),
                    devices=list(child.devices),
                    req_type_num=child.req_type_num,
                    batch_size=child_batch_size,
                    sub_graph_batch_sizes=dict(getattr(child, "sub_graph_batch_sizes", {})),
                ),
                mutation_kind,
                evo_cfg.enable_subgraph_batch_mut,
                device_type_by_id=device_type_by_id,
                max_mutated_subgraphs=evo_cfg.subgraph_batch_max_mutated,
                feasibility_cfg=feasibility_cfg,
            )

            child2 = eval_ind(child)
            if child2 is None or child2.uid is None or child2.objectives is None:
                continue
            # 去重（按 uid）
            if child2.uid in seen:
                continue
            if child2.throughput is None or child2.latency is None:
                continue
            if math.isnan(child2.throughput) or math.isnan(child2.latency):
                continue

            seen.add(child2.uid)
            offspring.append(child2)

        # If offspring not enough, pad with random individuals from current pop
        # offspring 不够时，用旧种群补齐。即，如果因为丢弃太多导致 offspring 产量不足，就从上一代非精英里随机抽一些填充，避免种群规模缩水。
        if len(offspring) < offspring_target:
            pool = pop
            if pool:
                offspring.extend(random.sample(pool, k=min(offspring_target - len(offspring), len(pool))))

        # 3) Environmental selection (NSGA-II): parents + offspring -> next generation
        pop = nsga2_environmental_select(pop + offspring, init_cfg.population_size)
    # Final: compute Pareto info and return a representative "best" + the final population
    fronts = fast_nondominated_sort(pop)
    for f in fronts:
        crowding_distance(f)

    # Choose representative best: highest-throughput solution on the first Pareto front.
    if fronts and fronts[0]:
        best = max(fronts[0], key=lambda x: float('-inf') if x.throughput is None else x.throughput)
    else:
        # fallback
        best = max(pop, key=lambda x: float('-inf') if x.throughput is None else x.throughput)

    # Sort population for readability (rank asc, then throughput desc)
    pop.sort(key=lambda x: (
        10**9 if x.pareto_rank is None else x.pareto_rank,
        float('-inf') if x.throughput is None else -x.throughput,
    ))
    return best, pop

    if getattr(cfg, "rewrite_mode", "legacy") == "family_driven":
        from exploration.rewrite_mechanism import default_family_driven_patterns
        rewrite_patterns = default_family_driven_patterns()
