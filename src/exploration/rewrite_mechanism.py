from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import random

from parallelism.pnode import Parallelism, XpTag
from exploration.individual import Attrs, DeviceAssign, Individual, Topology, TopologyNodeGene


class RewriteFamily(str, Enum):
    SKELETON_EXPANSION = "skeleton_expansion"
    LOCAL_REFINEMENT = "local_refinement"
    RELABEL = "relabel"
    REPARTITION = "repartition"
    ROLLBACK = "rollback"


@dataclass
class SymbolicNode:
    op: Parallelism
    device_counts: Dict[str, int] = field(default_factory=dict)
    children: List["SymbolicNode"] = field(default_factory=list)
    hint: Dict[str, Any] = field(default_factory=dict)
    closed: bool = True

    def clone(self) -> "SymbolicNode":
        return copy.deepcopy(self)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def total_devices(self) -> int:
        return sum(int(v) for v in self.device_counts.values())

    def normalized_device_counts(self) -> Dict[str, int]:
        return {str(k): int(v) for k, v in sorted(self.device_counts.items()) if int(v) > 0}

    def recompute_counts(self) -> Dict[str, int]:
        if self.children:
            merged: Dict[str, int] = {}
            for c in self.children:
                c.recompute_counts()
                for t, cnt in c.device_counts.items():
                    merged[t] = merged.get(t, 0) + int(cnt)
            self.device_counts = merged
        else:
            self.device_counts = self.normalized_device_counts()
        return self.device_counts

    def walk(self) -> Iterable["SymbolicNode"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def pretty(self) -> str:
        if self.children:
            body = ", ".join(c.pretty() for c in self.children)
            return f"{self.op.name}({body})"
        if self.device_counts:
            dc = ",".join(f"{k}*{v}" for k, v in sorted(self.device_counts.items()))
            return f"{self.op.name}[{dc}]"
        return f"{self.op.name}[]"


@dataclass
class PatternSpec:
    name: str
    family: RewriteFamily
    match: Dict[str, Any]
    rewrite: Dict[str, Any]
    weight: float = 1.0

    def matches(self, node: SymbolicNode) -> bool:
        return _match_symbolic_node(node, self.match)

    def apply(self, node: SymbolicNode) -> Optional[SymbolicNode]:
        if not self.matches(node):
            return None
        spec = copy.deepcopy(self.rewrite)
        out = _build_from_spec(spec)
        out.recompute_counts()
        return out


@dataclass
class RewriteCandidate:
    replacement: SymbolicNode
    weight: float = 1.0


@dataclass
class ParametricPatternSpec:
    name: str
    family: RewriteFamily
    match: Dict[str, Any]
    generator: Callable[[SymbolicNode, Optional[random.Random], Dict[str, Any]], List[RewriteCandidate]]
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)
    declared_parallelisms: Tuple[Parallelism, ...] = field(default_factory=tuple)

    def matches(self, node: SymbolicNode) -> bool:
        return _match_symbolic_node(node, self.match)

    def generate(self, node: SymbolicNode, *, rng: Optional[random.Random] = None) -> List[RewriteCandidate]:
        if not self.matches(node):
            return []
        out = self.generator(node.clone(), rng, dict(self.params or {}))
        cleaned: List[RewriteCandidate] = []
        for cand in out:
            if cand is None or cand.replacement is None:
                continue
            cand.replacement.recompute_counts()
            cleaned.append(cand)
        return cleaned


@dataclass
class NumericAttrCandidate:
    attrs: Dict[str, Any]
    weight: float = 1.0


@dataclass
class NumericPatternSpec:
    name: str
    match: Dict[str, Any]
    candidates: List[NumericAttrCandidate]
    weight: float = 1.0

    def matches(self, node: SymbolicNode) -> bool:
        return _match_symbolic_node(node, self.match)

    def choose_candidate(self, *, rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
        if not self.candidates:
            return None
        local_rng = rng or random.Random()
        weights = [max(1e-9, float(c.weight)) for c in self.candidates]
        cand = local_rng.choices(self.candidates, weights=weights, k=1)[0]
        return copy.deepcopy(cand.attrs)


@dataclass
class InitPatternSpec:
    name: str
    stratum: str
    root: Dict[str, Any]
    weight: float = 1.0

    def instantiate(self) -> "SymbolicNode":
        spec = copy.deepcopy(self.root)
        out = _build_from_spec(spec)
        out.recompute_counts()
        _ensure_materializable(out)
        return out


class RewriteEngine:
    def __init__(self, patterns: Sequence[Any], *, rng: Optional[random.Random] = None):
        self.patterns = list(patterns)
        self.rng = rng or random.Random()

    def rewrite_random(self, root: SymbolicNode, *, family: Optional[RewriteFamily] = None) -> bool:
        matches: List[Tuple[List[int], RewriteCandidate, float]] = []

        def dfs(node: SymbolicNode, path: List[int]) -> None:
            for p in self.patterns:
                if family is not None and p.family != family:
                    continue
                if not p.matches(node):
                    continue

                if isinstance(p, ParametricPatternSpec):
                    for cand in p.generate(node, rng=self.rng):
                        eff_weight = max(1e-9, float(p.weight) * float(cand.weight))
                        matches.append((path[:], cand, eff_weight))
                else:
                    replacement = p.apply(node.clone())
                    if replacement is not None:
                        matches.append((path[:], RewriteCandidate(replacement=replacement, weight=1.0), max(1e-9, float(p.weight))))
            for i, c in enumerate(node.children):
                path.append(i)
                dfs(c, path)
                path.pop()

        dfs(root, [])
        if not matches:
            return False

        weights = [w for _, _, w in matches]
        path, cand, _ = self.rng.choices(matches, weights=weights, k=1)[0]
        _replace_node_by_path(root, path, cand.replacement)
        root.recompute_counts()
        return True


# -----------------------------
# Manual, declarative rewrite DSL
# -----------------------------

def _normalize_counts(counts: Dict[str, int]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counts.items()) if int(v) > 0}


def _parallelism_from_any(v: Any) -> Parallelism:
    if isinstance(v, Parallelism):
        return v
    return Parallelism[str(v)]


def _effective_arity(node: SymbolicNode) -> int:
    return len(node.children) if node.children else node.total_devices()


def _match_symbolic_node(node: SymbolicNode, match: Dict[str, Any]) -> bool:
    op = match.get("op")
    if op is not None:
        want = op if isinstance(op, Parallelism) else Parallelism[str(op)]
        if node.op != want:
            return False

    leaf = match.get("leaf")
    if leaf is not None and bool(leaf) != node.is_leaf():
        return False

    closed = match.get("closed")
    if closed is not None and bool(closed) != bool(node.closed):
        return False

    child_count = match.get("child_count")
    if child_count is not None and len(node.children) != int(child_count):
        return False

    arity = match.get("arity")
    if arity is not None and _effective_arity(node) != int(arity):
        return False

    min_arity = match.get("min_arity")
    if min_arity is not None and _effective_arity(node) < int(min_arity):
        return False

    max_arity = match.get("max_arity")
    if max_arity is not None and _effective_arity(node) > int(max_arity):
        return False

    min_devices = match.get("min_total_devices")
    if min_devices is not None and node.total_devices() < int(min_devices):
        return False

    max_devices = match.get("max_total_devices")
    if max_devices is not None and node.total_devices() > int(max_devices):
        return False

    require_types = match.get("require_types")
    if require_types is not None:
        types = set(node.device_counts.keys())
        if not set(str(x) for x in require_types).issubset(types):
            return False

    exact_counts = match.get("device_counts")
    if exact_counts is not None and _normalize_counts(exact_counts) != _normalize_counts(node.device_counts):
        return False

    return True



def _build_from_spec(spec: Dict[str, Any]) -> SymbolicNode:
    children = [_build_from_spec(c) for c in spec.get("children", [])]
    op = _parallelism_from_any(spec["op"])
    node = SymbolicNode(
        op=op,
        device_counts=_normalize_counts(spec.get("device_counts", {})),
        children=children,
        hint=copy.deepcopy(spec.get("hint", {})),
        closed=bool(spec.get("closed", len(children) > 0 or op != Parallelism.XP)),
    )
    node.recompute_counts()
    return node


def _get_node_by_path(root: SymbolicNode, path: Sequence[int]) -> SymbolicNode:
    cur = root
    for idx in path:
        cur = cur.children[idx]
    return cur


def _replace_node_by_path(root: SymbolicNode, path: Sequence[int], node: SymbolicNode) -> None:
    if not path:
        root.op = node.op
        root.device_counts = node.device_counts
        root.children = node.children
        root.hint = node.hint
        root.closed = node.closed
        return
    parent = _get_node_by_path(root, path[:-1])
    parent.children[path[-1]] = node


# -----------------------------
# Individual <-> Symbolic conversion
# -----------------------------

def individual_to_symbolic(ind: Individual, *, device_type_by_id: Dict[int, str]) -> SymbolicNode:
    topo = ind.topology

    def build(nid: int) -> SymbolicNode:
        ptype = topo.gene(nid).ptype
        children = [build(cid) for cid in topo.children_of(nid)]
        hint: Dict[str, Any] = {}
        if ptype == Parallelism.DP and nid in ind.attrs.dp_attr:
            hint["dp_attr"] = copy.deepcopy(ind.attrs.dp_attr[nid])
        elif ptype == Parallelism.PP and nid in ind.attrs.pp_attr:
            hint["pp_attr"] = copy.deepcopy(ind.attrs.pp_attr[nid])
        elif ptype == Parallelism.TP and nid in ind.attrs.tp_attr:
            hint["tp_attr"] = copy.deepcopy(ind.attrs.tp_attr[nid])
        elif ptype == Parallelism.XP and nid in ind.attrs.xp_attr:
            hint["xp_attr"] = copy.deepcopy(ind.attrs.xp_attr[nid])

        if children:
            node = SymbolicNode(op=ptype, children=children, hint=hint, closed=True)
            node.recompute_counts()
            return node

        counts: Dict[str, int] = {}
        for did in ind.device_assign.leaf_to_devices.get(nid, []):
            dtype = str(device_type_by_id[int(did)])
            counts[dtype] = counts.get(dtype, 0) + 1
        leaf_closed = True
        if ptype == Parallelism.XP:
            leaf_closed = sum(int(v) for v in counts.values()) == 2
        return SymbolicNode(op=ptype, device_counts=counts, hint=hint, closed=leaf_closed)

    root = build(topo.root_id)
    root.recompute_counts()
    return root


def symbolic_to_individual(
    root: SymbolicNode,
    *,
    device_type_to_ids: Dict[str, List[int]],
    req_type_num: int,
    batch_size: int,
    devices: Sequence[int],
    sub_graph_batch_sizes: Optional[Dict[int, int]] = None,
) -> Individual:
    _ensure_materializable(root)
    root = root.clone()
    root.recompute_counts()

    pools: Dict[str, List[int]] = {str(k): list(v) for k, v in device_type_to_ids.items()}
    nodes: List[TopologyNodeGene] = []
    leaf_to_devices: Dict[int, List[int]] = {}
    attrs = Attrs()
    next_id = 0

    def build(node: SymbolicNode, parent_id: int, child_slot: int) -> int:
        nonlocal next_id
        nid = next_id
        next_id += 1
        nodes.append(TopologyNodeGene(node_id=nid, parent_id=parent_id, ptype=node.op, child_slot=child_slot))

        if node.children:
            for idx, ch in enumerate(node.children):
                build(ch, nid, idx)
            _materialize_attrs(node, nid, attrs, req_type_num=req_type_num, arity=len(node.children))
            return nid

        assigned: List[int] = []
        for dtype, cnt in sorted(node.device_counts.items()):
            bucket = pools.get(str(dtype), [])
            if len(bucket) < int(cnt):
                raise ValueError(f"Not enough devices for type {dtype}: need {cnt}, have {len(bucket)}")
            take = bucket[: int(cnt)]
            del bucket[: int(cnt)]
            assigned.extend(int(x) for x in take)
        if not assigned:
            raise ValueError(f"Leaf {node.op.name} must receive at least one device.")
        leaf_to_devices[nid] = assigned
        _materialize_attrs(node, nid, attrs, req_type_num=req_type_num, arity=len(assigned))
        return nid

    build(root, -1, 0)
    leftovers = [x for bucket in pools.values() for x in bucket]
    if leftovers:
        raise ValueError(f"Unused devices after materialization: {sorted(leftovers)}")

    topo = Topology(nodes=nodes)
    ind = Individual(
        topology=topo,
        device_assign=DeviceAssign(leaf_to_devices=leaf_to_devices),
        attrs=attrs,
        devices=list(devices),
        req_type_num=req_type_num,
        batch_size=int(batch_size),
        sub_graph_batch_sizes=dict(sub_graph_batch_sizes or {}),
    )
    ind.check_legality()
    return ind


# -----------------------------
# Materialization checks / attr filling
# -----------------------------

def _uniform_dp_attr(req_type_num: int, arity: int) -> List[List[float]]:
    return [[1.0] * arity for _ in range(req_type_num)]


def _materialize_attrs(node: SymbolicNode, node_id: int, attrs: Attrs, *, req_type_num: int, arity: int) -> None:
    hint = node.hint or {}
    if node.op == Parallelism.DP:
        val = copy.deepcopy(hint.get("dp_attr"))
        if val is None or len(val) != req_type_num or any(len(row) != arity for row in val):
            val = _uniform_dp_attr(req_type_num, arity)
        attrs.dp_attr[node_id] = val
    elif node.op == Parallelism.PP:
        val = copy.deepcopy(hint.get("pp_attr"))
        if val is None or len(val) != arity:
            val = [1.0] * arity
        attrs.pp_attr[node_id] = val
    elif node.op == Parallelism.TP:
        val = copy.deepcopy(hint.get("tp_attr"))
        if val is None or len(val) != arity:
            val = [1.0] * arity
        attrs.tp_attr[node_id] = val
    elif node.op == Parallelism.XP:
        val = copy.deepcopy(hint.get("xp_attr"))
        if val is None or len(val) != 2 or set(val) != {XpTag.ATTENTION, XpTag.LINEAR}:
            val = [XpTag.ATTENTION, XpTag.LINEAR]
        attrs.xp_attr[node_id] = val
    else:
        raise ValueError(f"Unexpected op: {node.op}")


def _ensure_materializable(root: SymbolicNode) -> None:
    for node in root.walk():
        if node.op == Parallelism.XP:
            if node.children:
                if len(node.children) != 2:
                    raise ValueError("Internal XP must be materialized as a binary parallel node.")
            else:
                if node.total_devices() != 2:
                    raise ValueError("XP leaf must own exactly 2 devices.")
        if node.children:
            if not node.closed:
                raise ValueError(f"Node {node.pretty()} is not closed.")
        elif node.total_devices() <= 0:
            raise ValueError(f"Leaf {node.pretty()} must own at least one device.")



def is_materializable(root: SymbolicNode) -> bool:
    try:
        _ensure_materializable(root)
        return True
    except Exception:
        return False


def has_open_nodes(root: SymbolicNode) -> bool:
    for node in root.walk():
        if not bool(node.closed):
            return True
        if node.op == Parallelism.XP:
            if node.children:
                if len(node.children) != 2:
                    return True
            elif node.total_devices() != 2:
                return True
    return False


# -----------------------------
# Initialization pattern library
# Fixed, hand-written seed templates driven by device abstraction.
# Every pattern returned here is already CLOSED and directly materializable. XP may be internal-binary or a leaf with exactly 2 devices.
# -----------------------------

def _all_device_counts(device_type_to_ids: Dict[str, List[int]]) -> Dict[str, int]:
    return {
        str(dtype): int(len(ids))
        for dtype, ids in sorted(device_type_to_ids.items())
        if len(ids) > 0
    }


def _split_counts_two_buckets(counts: Dict[str, int]) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
    counts = _normalize_counts(counts)
    if sum(counts.values()) < 2:
        return None

    items = [(str(k), int(v)) for k, v in sorted(counts.items()) if int(v) > 0]
    if not items:
        return None

    if len(items) == 1:
        dtype, total = items[0]
        if total < 2:
            return None
        left_n = total // 2
        right_n = total - left_n
        if left_n <= 0 or right_n <= 0:
            return None
        return ({dtype: left_n}, {dtype: right_n})

    total = sum(v for _, v in items)
    target_left = max(1, total // 2)

    left: Dict[str, int] = {}
    right: Dict[str, int] = {k: v for k, v in items}
    left_total = 0

    for dtype, avail in items:
        if left_total >= target_left:
            break
        take = min(avail, target_left - left_total)
        if take > 0:
            left[dtype] = take
            right[dtype] -= take
            left_total += take

    right = _normalize_counts(right)
    if sum(left.values()) == 0 or sum(right.values()) == 0:
        # deterministic fallback: move exactly one full type to the left
        dtype0, avail0 = items[0]
        if len(items) >= 2:
            left = {dtype0: avail0}
            right = _normalize_counts({k: v for k, v in items[1:]})
        else:
            return None

    if sum(left.values()) == 0 or sum(right.values()) == 0:
        return None
    return _normalize_counts(left), _normalize_counts(right)


def _split_counts_for_dp_xp_tail(
    counts: Dict[str, int],
) -> Optional[Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]]:
    counts = _normalize_counts(counts)
    items = [(str(k), int(v)) for k, v in sorted(counts.items()) if int(v) > 0]
    if len(items) < 2:
        return None

    pivot = [(k, v) for k, v in items if v >= 2]
    if len(pivot) < 2:
        return None

    (t0, c0), (t1, c1) = pivot[:2]
    xp_left = {t0: max(1, c0 // 2)}
    xp_right = {t1: max(1, c1 // 2)}

    tail: Dict[str, int] = {k: v for k, v in items}
    tail[t0] -= xp_left[t0]
    tail[t1] -= xp_right[t1]
    tail = _normalize_counts(tail)

    if not tail:
        return None
    return _normalize_counts(xp_left), _normalize_counts(xp_right), tail


def _enumerate_two_way_seed_splits(counts: Dict[str, int]) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    counts = _normalize_counts(counts)
    items = [(str(k), int(v)) for k, v in sorted(counts.items()) if int(v) > 0]
    out: List[Tuple[Dict[str, int], Dict[str, int]]] = []
    seen: set[Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, int], ...]]]] = set()

    def add(left: Dict[str, int], right: Dict[str, int]) -> None:
        l = _normalize_counts(left)
        r = _normalize_counts(right)
        if not l or not r:
            return
        key_lr = (tuple(sorted(l.items())), tuple(sorted(r.items())))
        key_rl = (key_lr[1], key_lr[0])
        key = min(key_lr, key_rl)
        if key in seen:
            return
        seen.add(key)
        out.append((l, r))

    bal = _split_counts_two_buckets(counts)
    if bal is not None:
        add(*bal)

    n = len(items)
    if n >= 2:
        # full-type subset splits
        for mask in range(1, (1 << n) - 1):
            left = {items[i][0]: items[i][1] for i in range(n) if (mask >> i) & 1}
            right = {items[i][0]: items[i][1] for i in range(n) if not ((mask >> i) & 1)}
            add(left, right)
            if len(out) >= 12:
                break

    # single-type partial splits against the remainder
    for dtype, cnt in items:
        if cnt < 2:
            continue
        probe_points = sorted(set([1, cnt // 2, cnt - 1]))
        for take in probe_points:
            if take <= 0 or take >= cnt:
                continue
            left = {dtype: take}
            right = dict(counts)
            right[dtype] -= take
            add(left, right)
            if len(out) >= 18:
                break
        if len(out) >= 18:
            break

    return out


def _clone_with_counts(node: SymbolicNode, counts: Dict[str, int]) -> SymbolicNode:
    out = node.clone()
    out.children = []
    out.device_counts = _normalize_counts(counts)
    out.closed = True
    out.recompute_counts()
    return out


def _counts_to_tokens(counts: Dict[str, int]) -> List[str]:
    tokens: List[str] = []
    for dtype, cnt in sorted(_normalize_counts(counts).items()):
        tokens.extend([str(dtype)] * int(cnt))
    return tokens


def _counts_from_tokens(tokens: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for tok in tokens:
        out[str(tok)] = out.get(str(tok), 0) + 1
    return _normalize_counts(out)


def _balanced_sizes(total: int, k: int) -> List[int]:
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def _random_composition(total: int, k: int, *, rng: random.Random) -> List[int]:
    if k == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), k - 1))
    prev = 0
    out: List[int] = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _add_partition_candidate(
    buckets: Sequence[Sequence[str]],
    out: List[Tuple[Dict[str, int], ...]],
    seen: set[Tuple[Tuple[Tuple[str, int], ...], ...]],
) -> None:
    parts = tuple(_counts_from_tokens(bucket) for bucket in buckets)
    if any(not part for part in parts):
        return
    key = tuple(tuple(sorted(part.items())) for part in parts)
    if key in seen:
        return
    seen.add(key)
    out.append(parts)


def _enumerate_k_way_partitions(
    counts: Dict[str, int],
    k: int,
    *,
    max_out: int = 12,
    rng: Optional[random.Random] = None,
) -> List[Tuple[Dict[str, int], ...]]:
    counts = _normalize_counts(counts)
    total = sum(counts.values())
    if k <= 0 or total < k:
        return []

    local_rng = rng or random.Random()
    tokens = _counts_to_tokens(counts)
    out: List[Tuple[Dict[str, int], ...]] = []
    seen: set[Tuple[Tuple[Tuple[str, int], ...], ...]] = set()

    sizes_bal = _balanced_sizes(total, k)

    ordered = list(tokens)
    buckets: List[List[str]] = []
    cursor = 0
    for size in sizes_bal:
        buckets.append(ordered[cursor: cursor + size])
        cursor += size
    _add_partition_candidate(buckets, out, seen)

    rev = list(reversed(tokens))
    buckets = []
    cursor = 0
    for size in sizes_bal:
        buckets.append(rev[cursor: cursor + size])
        cursor += size
    _add_partition_candidate(buckets, out, seen)

    rr_buckets = [[] for _ in range(k)]
    for idx, tok in enumerate(tokens):
        rr_buckets[idx % k].append(tok)
    _add_partition_candidate(rr_buckets, out, seen)

    rr_buckets_rev = [[] for _ in range(k)]
    for idx, tok in enumerate(reversed(tokens)):
        rr_buckets_rev[idx % k].append(tok)
    _add_partition_candidate(rr_buckets_rev, out, seen)

    trials = max(4, max_out * 3)
    for _ in range(trials):
        shuffled = list(tokens)
        local_rng.shuffle(shuffled)
        sizes = sizes_bal if local_rng.random() < 0.55 else _random_composition(total, k, rng=local_rng)
        buckets = []
        cursor = 0
        for size in sizes:
            buckets.append(shuffled[cursor: cursor + size])
            cursor += size
        _add_partition_candidate(buckets, out, seen)
        if len(out) >= max_out:
            break

    return out[:max_out]


def _limit_candidates(cands: List[RewriteCandidate], *, max_out: int, rng: Optional[random.Random]) -> List[RewriteCandidate]:
    if len(cands) <= max_out:
        return cands
    local_rng = rng or random.Random()
    idxs = list(range(len(cands)))
    local_rng.shuffle(idxs)
    keep = set(idxs[:max_out])
    return [cand for idx, cand in enumerate(cands) if idx in keep]


def _generate_leaf_expansion(node: SymbolicNode, rng: Optional[random.Random], params: Dict[str, Any]) -> List[RewriteCandidate]:
    if not node.is_leaf():
        return []

    requested_k = int(params.get("child_count", 2))
    if requested_k < 2:
        return []
    if node.op == Parallelism.XP and requested_k != 2:
        return []

    max_partitions = int(params.get("max_partitions", 10))
    partitions = _enumerate_k_way_partitions(node.device_counts, requested_k, max_out=max_partitions, rng=rng)
    if not partitions:
        return []

    child_op = Parallelism.TP if node.op != Parallelism.XP else Parallelism.TP
    out: List[RewriteCandidate] = []
    for idx, parts in enumerate(partitions):
        children = [
            SymbolicNode(op=child_op, device_counts=part, closed=True)
            for part in parts
        ]
        cand = SymbolicNode(
            op=node.op,
            children=children,
            hint=copy.deepcopy(node.hint),
            closed=True,
        )
        cand.recompute_counts()
        out.append(RewriteCandidate(replacement=cand, weight=1.0 if idx == 0 else 0.75))

    return _limit_candidates(out, max_out=int(params.get("max_candidates", max_partitions)), rng=rng)


def _generate_internal_to_leaf(node: SymbolicNode, rng: Optional[random.Random], params: Dict[str, Any]) -> List[RewriteCandidate]:
    if node.is_leaf() or not node.children:
        return []
    if node.op == Parallelism.XP and node.total_devices() != 2:
        return []
    out = SymbolicNode(
        op=node.op,
        device_counts=_normalize_counts(node.device_counts),
        hint=copy.deepcopy(node.hint),
        closed=True,
    )
    out.recompute_counts()
    return [RewriteCandidate(replacement=out, weight=1.0)]


def _build_binary_tree(op: Parallelism, children: Sequence[SymbolicNode], *, preserve_root_hint: Dict[str, Any]) -> SymbolicNode:
    if len(children) < 2:
        raise ValueError("Need at least 2 children to binaryize")
    cur = SymbolicNode(op=op, children=[children[0].clone(), children[1].clone()], closed=True)
    cur.recompute_counts()
    for child in children[2:]:
        cur = SymbolicNode(op=op, children=[cur, child.clone()], closed=True)
        cur.recompute_counts()
    cur.hint = copy.deepcopy(preserve_root_hint)
    cur.recompute_counts()
    return cur


def _generate_binaryized(node: SymbolicNode, rng: Optional[random.Random], params: Dict[str, Any]) -> List[RewriteCandidate]:
    if node.is_leaf() or len(node.children) < 3 or node.op == Parallelism.XP:
        return []

    out: List[RewriteCandidate] = []
    children = [child.clone() for child in node.children]
    left_deep = _build_binary_tree(node.op, children, preserve_root_hint=node.hint)
    out.append(RewriteCandidate(replacement=left_deep, weight=1.0))

    right_deep = _build_binary_tree(node.op, list(reversed(children)), preserve_root_hint=node.hint)
    right_deep.children = list(reversed(right_deep.children))
    right_deep.recompute_counts()
    out.append(RewriteCandidate(replacement=right_deep, weight=0.85))
    return _limit_candidates(out, max_out=int(params.get("max_candidates", 2)), rng=rng)


def _generate_flattened(node: SymbolicNode, rng: Optional[random.Random], params: Dict[str, Any]) -> List[RewriteCandidate]:
    if node.is_leaf() or len(node.children) != 2:
        return []

    out: List[RewriteCandidate] = []
    for idx, child in enumerate(node.children):
        if child.op != node.op or not child.children:
            continue
        new_children: List[SymbolicNode] = []
        if idx == 0:
            new_children.extend(grand.clone() for grand in child.children)
            new_children.append(node.children[1].clone())
        else:
            new_children.append(node.children[0].clone())
            new_children.extend(grand.clone() for grand in child.children)
        flat = SymbolicNode(op=node.op, children=new_children, hint=copy.deepcopy(node.hint), closed=True)
        flat.recompute_counts()
        out.append(RewriteCandidate(replacement=flat, weight=1.0))
    return _limit_candidates(out, max_out=int(params.get("max_candidates", 2)), rng=rng)


def _generate_repartition(node: SymbolicNode, rng: Optional[random.Random], params: Dict[str, Any]) -> List[RewriteCandidate]:
    if node.is_leaf() or len(node.children) < 2:
        return []
    if not all(child.is_leaf() for child in node.children):
        return []
    if node.op == Parallelism.XP and len(node.children) != 2:
        return []

    max_partitions = int(params.get("max_partitions", 12))
    partitions = _enumerate_k_way_partitions(node.device_counts, len(node.children), max_out=max_partitions, rng=rng)
    current = tuple(_normalize_counts(child.device_counts) for child in node.children)
    out: List[RewriteCandidate] = []
    for idx, parts in enumerate(partitions):
        if tuple(parts) == current:
            continue
        children = []
        for child, part in zip(node.children, parts):
            new_child = _clone_with_counts(child, part)
            children.append(new_child)
        cand = SymbolicNode(op=node.op, children=children, hint=copy.deepcopy(node.hint), closed=True)
        cand.recompute_counts()
        out.append(RewriteCandidate(replacement=cand, weight=1.0 if idx == 0 else 0.8))
    return _limit_candidates(out, max_out=int(params.get("max_candidates", max_partitions)), rng=rng)


def _generate_child_permutations(node: SymbolicNode, rng: Optional[random.Random], params: Dict[str, Any]) -> List[RewriteCandidate]:
    if node.is_leaf() or len(node.children) < 2:
        return []

    out: List[RewriteCandidate] = []
    if len(node.children) == 2:
        swapped = SymbolicNode(
            op=node.op,
            children=[node.children[1].clone(), node.children[0].clone()],
            hint=copy.deepcopy(node.hint),
            closed=True,
        )
        swapped.recompute_counts()
        out.append(RewriteCandidate(replacement=swapped, weight=1.0))
    else:
        rotated_children = [child.clone() for child in node.children[1:]] + [node.children[0].clone()]
        rotated = SymbolicNode(op=node.op, children=rotated_children, hint=copy.deepcopy(node.hint), closed=True)
        rotated.recompute_counts()
        out.append(RewriteCandidate(replacement=rotated, weight=1.0))

        reversed_children = [child.clone() for child in reversed(node.children)]
        reversed_node = SymbolicNode(op=node.op, children=reversed_children, hint=copy.deepcopy(node.hint), closed=True)
        reversed_node.recompute_counts()
        out.append(RewriteCandidate(replacement=reversed_node, weight=0.75))

    return _limit_candidates(out, max_out=int(params.get("max_candidates", 2)), rng=rng)


def _generate_same_op_rotation(node: SymbolicNode, rng: Optional[random.Random], params: Dict[str, Any]) -> List[RewriteCandidate]:
    if node.is_leaf() or len(node.children) != 2:
        return []

    out: List[RewriteCandidate] = []
    left, right = node.children
    if left.op == node.op and len(left.children) == 2:
        a, b = left.children
        rotated = SymbolicNode(
            op=node.op,
            children=[
                a.clone(),
                SymbolicNode(op=node.op, children=[b.clone(), right.clone()], closed=True),
            ],
            hint=copy.deepcopy(node.hint),
            closed=True,
        )
        rotated.recompute_counts()
        out.append(RewriteCandidate(replacement=rotated, weight=1.0))

    if right.op == node.op and len(right.children) == 2:
        b, c = right.children
        rotated = SymbolicNode(
            op=node.op,
            children=[
                SymbolicNode(op=node.op, children=[left.clone(), b.clone()], closed=True),
                c.clone(),
            ],
            hint=copy.deepcopy(node.hint),
            closed=True,
        )
        rotated.recompute_counts()
        out.append(RewriteCandidate(replacement=rotated, weight=1.0))

    return _limit_candidates(out, max_out=int(params.get("max_candidates", 2)), rng=rng)


def default_init_patterns(device_type_to_ids: Dict[str, List[int]]) -> List[InitPatternSpec]:
    all_counts = _all_device_counts(device_type_to_ids)
    if not all_counts:
        return []

    patterns: List[InitPatternSpec] = []
    seen_names: set[str] = set()

    def add(name: str, stratum: str, root: Dict[str, Any], weight: float = 1.0) -> None:
        if name in seen_names:
            return
        patterns.append(InitPatternSpec(name=name, stratum=stratum, root=root, weight=weight))
        seen_names.add(name)

    # 1) Single closed leaf seeds. Very stable seeds, good for breadth.
    for op_name in ("DP", "PP", "TP"):
        add(
            f"{op_name.lower()}_single_leaf_all",
            "single_leaf",
            {"op": op_name, "device_counts": all_counts, "closed": True},
            1.0 if op_name == "TP" else 0.9,
        )

    # 2) Split by device type: one TP child per type under DP / PP / TP.
    if len(all_counts) >= 2:
        children = [
            {"op": "TP", "device_counts": {dtype: cnt}, "closed": True}
            for dtype, cnt in sorted(all_counts.items())
        ]
        add("dp_split_by_type", "by_type_split", {"op": "DP", "children": children, "closed": True}, 1.2)
        add("tp_split_by_type", "by_type_split", {"op": "TP", "children": children, "closed": True}, 1.1)
        add("pp_split_by_type", "by_type_split", {"op": "PP", "children": children, "closed": True}, 0.8)

    # 3) Closed two-way partitions across all devices. These are the main source
    #    of robust, diverse initial seeds.
    for idx, (left_counts, right_counts) in enumerate(_enumerate_two_way_seed_splits(all_counts)):
        add(
            f"xp_binary_root_{idx}",
            "xp_root",
            {
                "op": "XP",
                "children": [
                    {"op": "TP", "device_counts": left_counts, "closed": True},
                    {"op": "TP", "device_counts": right_counts, "closed": True},
                ],
                "closed": True,
            },
            1.15,
        )
        add(
            f"dp_binary_split_{idx}",
            "binary_split",
            {
                "op": "DP",
                "children": [
                    {"op": "TP", "device_counts": left_counts, "closed": True},
                    {"op": "TP", "device_counts": right_counts, "closed": True},
                ],
                "closed": True,
            },
            1.05,
        )
        add(
            f"pp_binary_split_{idx}",
            "binary_split",
            {
                "op": "PP",
                "children": [
                    {"op": "TP", "device_counts": left_counts, "closed": True},
                    {"op": "TP", "device_counts": right_counts, "closed": True},
                ],
                "closed": True,
            },
            0.95,
        )

    # 4) XP embedded under DP / PP: closed but still structurally richer than pure binary roots.
    dp_xp_tail = _split_counts_for_dp_xp_tail(all_counts)
    if dp_xp_tail is not None:
        xp_left, xp_right, tail = dp_xp_tail
        add(
            "dp_with_embedded_xp",
            "xp_embedded",
            {
                "op": "DP",
                "children": [
                    {
                        "op": "XP",
                        "children": [
                            {"op": "TP", "device_counts": xp_left, "closed": True},
                            {"op": "TP", "device_counts": xp_right, "closed": True},
                        ],
                        "closed": True,
                    },
                    {"op": "TP", "device_counts": tail, "closed": True},
                ],
                "closed": True,
            },
            1.10,
        )
        add(
            "pp_with_embedded_xp",
            "xp_embedded",
            {
                "op": "PP",
                "children": [
                    {
                        "op": "XP",
                        "children": [
                            {"op": "TP", "device_counts": xp_left, "closed": True},
                            {"op": "TP", "device_counts": xp_right, "closed": True},
                        ],
                        "closed": True,
                    },
                    {"op": "TP", "device_counts": tail, "closed": True},
                ],
                "closed": True,
            },
            0.95,
        )

    return patterns



def default_numeric_patterns() -> List[NumericPatternSpec]:
    return [
        NumericPatternSpec(
            name="xp_any_binary",
            match={"op": Parallelism.XP, "arity": 2},
            candidates=[
                NumericAttrCandidate(attrs={"xp_attr": [XpTag.ATTENTION, XpTag.LINEAR]}, weight=1.0),
                NumericAttrCandidate(attrs={"xp_attr": [XpTag.LINEAR, XpTag.ATTENTION]}, weight=1.0),
            ],
            weight=1.0,
        ),
        NumericPatternSpec(
            name="pp_binary_bias",
            match={"op": Parallelism.PP, "arity": 2},
            candidates=[
                NumericAttrCandidate(attrs={"pp_attr": [1.0, 1.0]}, weight=0.8),
                NumericAttrCandidate(attrs={"pp_attr": [1.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"pp_attr": [0.5, 1.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"pp_attr": [2.0, 1.0]}, weight=0.7),
                NumericAttrCandidate(attrs={"pp_attr": [1.0, 2.0]}, weight=0.7),
            ],
            weight=0.9,
        ),
        NumericPatternSpec(
            name="pp_ternary_bias",
            match={"op": Parallelism.PP, "arity": 3},
            candidates=[
                NumericAttrCandidate(attrs={"pp_attr": [1.0, 1.0, 1.0]}, weight=0.8),
                NumericAttrCandidate(attrs={"pp_attr": [1.5, 0.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"pp_attr": [0.5, 1.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"pp_attr": [0.5, 0.5, 1.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"pp_attr": [1.5, 1.0, 0.5]}, weight=0.7),
                NumericAttrCandidate(attrs={"pp_attr": [0.5, 1.0, 1.5]}, weight=0.7),
            ],
            weight=0.8,
        ),
        NumericPatternSpec(
            name="tp_binary_bias",
            match={"op": Parallelism.TP, "arity": 2},
            candidates=[
                NumericAttrCandidate(attrs={"tp_attr": [1.0, 1.0]}, weight=0.8),
                NumericAttrCandidate(attrs={"tp_attr": [1.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"tp_attr": [0.5, 1.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"tp_attr": [2.0, 1.0]}, weight=0.7),
                NumericAttrCandidate(attrs={"tp_attr": [1.0, 2.0]}, weight=0.7),
            ],
            weight=1.0,
        ),
        NumericPatternSpec(
            name="tp_ternary_bias",
            match={"op": Parallelism.TP, "arity": 3},
            candidates=[
                NumericAttrCandidate(attrs={"tp_attr": [1.0, 1.0, 1.0]}, weight=0.8),
                NumericAttrCandidate(attrs={"tp_attr": [1.5, 0.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"tp_attr": [0.5, 1.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"tp_attr": [0.5, 0.5, 1.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"tp_attr": [1.5, 1.0, 0.5]}, weight=0.7),
                NumericAttrCandidate(attrs={"tp_attr": [0.5, 1.0, 1.5]}, weight=0.7),
            ],
            weight=0.9,
        ),
        NumericPatternSpec(
            name="dp_binary_bias",
            match={"op": Parallelism.DP, "arity": 2},
            candidates=[
                # [dp_attr] 一共有 request_type_num 个 list，每个 list 是该 req_type 在各个 child 上的切分比例
                NumericAttrCandidate(attrs={"dp_attr": [[1.0, 0.5], [0.5, 0.0], [0.0, 1.0]]},weight=1.5),
                NumericAttrCandidate(attrs={"dp_attr": [[0.1, 0.5], [0.5, 1.0], [0.5, 1.0]]}, weight=1.5),
                NumericAttrCandidate(attrs={"dp_attr": [[1.0, 1.5], [0.5, 0.5], [0.5, 1.5]]}, weight=1.5),
            ],
            weight=0.9,
        ),
        NumericPatternSpec(
            name="dp_ternary_bias",
            match={"op": Parallelism.DP, "arity": 3},
            candidates=[
                # [dp_attr] 一共有 request_type_num 个 list，每个 list 是该 req_type 在各个 child 上的切分比例
                NumericAttrCandidate(attrs={"dp_attr": [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 0.0, 1.0]]}, weight=1.5),
            ],
            weight=0.9,
        ),
    ]



def default_parameterized_patterns(
    *,
    max_split_children: int = 4,
    partition_budget: int = 12,
    candidate_budget: int = 12,
) -> List[Any]:
    max_split_children = max(2, min(4, int(max_split_children)))
    partition_budget = max(2, int(partition_budget))
    candidate_budget = max(2, int(candidate_budget))

    patterns: List[Any] = []

    for child_count in range(2, max_split_children + 1):
        for op in (Parallelism.DP, Parallelism.PP, Parallelism.TP):
            patterns.append(
                ParametricPatternSpec(
                    name=f"{op.name.lower()}_leaf_to_{child_count}ary_internal",
                    family=RewriteFamily.SKELETON_EXPANSION,
                    match={"op": op, "leaf": True, "closed": True, "min_total_devices": child_count},
                    generator=_generate_leaf_expansion,
                    weight=1.0 if child_count == 2 else 0.8,
                    params={
                        "child_count": child_count,
                        "max_partitions": partition_budget,
                        "max_candidates": candidate_budget,
                    },
                    declared_parallelisms=(op, Parallelism.TP),
                )
            )

    patterns.append(
        ParametricPatternSpec(
            name="xp_leaf_to_binary_internal",
            family=RewriteFamily.SKELETON_EXPANSION,
            match={"op": Parallelism.XP, "leaf": True, "closed": True, "min_total_devices": 2},
            generator=_generate_leaf_expansion,
            weight=1.0,
            params={
                "child_count": 2,
                "max_partitions": partition_budget,
                "max_candidates": candidate_budget,
            },
            declared_parallelisms=(Parallelism.XP, Parallelism.TP),
        )
    )

    for op in (Parallelism.DP, Parallelism.PP, Parallelism.TP, Parallelism.XP):
        patterns.append(
            ParametricPatternSpec(
                name=f"{op.name.lower()}_internal_to_leaf",
                family=RewriteFamily.ROLLBACK,
                match={"op": op, "leaf": False, "closed": True},
                generator=_generate_internal_to_leaf,
                weight=0.9,
                declared_parallelisms=(op,),
            )
        )

    for op in (Parallelism.DP, Parallelism.PP, Parallelism.TP):
        patterns.append(
            ParametricPatternSpec(
                name=f"{op.name.lower()}_nary_to_binaryized",
                family=RewriteFamily.LOCAL_REFINEMENT,
                match={"op": op, "leaf": False, "closed": True, "min_arity": 3},
                generator=_generate_binaryized,
                weight=0.85,
                params={"max_candidates": 2},
                declared_parallelisms=(op,),
            )
        )
        patterns.append(
            ParametricPatternSpec(
                name=f"{op.name.lower()}_binary_to_flattened",
                family=RewriteFamily.LOCAL_REFINEMENT,
                match={"op": op, "leaf": False, "closed": True, "child_count": 2},
                generator=_generate_flattened,
                weight=0.80,
                params={"max_candidates": 2},
                declared_parallelisms=(op,),
            )
        )
        patterns.append(
            ParametricPatternSpec(
                name=f"{op.name.lower()}_same_op_rotation",
                family=RewriteFamily.LOCAL_REFINEMENT,
                match={"op": op, "leaf": False, "closed": True, "child_count": 2},
                generator=_generate_same_op_rotation,
                weight=0.70,
                params={"max_candidates": 2},
                declared_parallelisms=(op,),
            )
        )

    for op in (Parallelism.DP, Parallelism.PP, Parallelism.TP, Parallelism.XP):
        patterns.append(
            ParametricPatternSpec(
                name=f"{op.name.lower()}_repartition",
                family=RewriteFamily.REPARTITION,
                match={"op": op, "leaf": False, "closed": True, "min_arity": 2},
                generator=_generate_repartition,
                weight=1.0,
                params={
                    "max_partitions": partition_budget,
                    "max_candidates": candidate_budget,
                },
                declared_parallelisms=(op,),
            )
        )
        patterns.append(
            ParametricPatternSpec(
                name=f"{op.name.lower()}_child_permutation",
                family=RewriteFamily.LOCAL_REFINEMENT,
                match={"op": op, "leaf": False, "closed": True, "min_arity": 2},
                generator=_generate_child_permutations,
                weight=0.60,
                params={"max_candidates": 2},
                declared_parallelisms=(op,),
            )
        )

    return patterns


def default_patterns() -> List[PatternSpec]:
    xp_hint = {"xp_attr": [XpTag.LINEAR, XpTag.ATTENTION]}

    return [
        # ------------------------------------------------------------------
        # Skeleton expansion rules that DIRECTLY generate closed XP leaves.
        # This matches the current semantics:
        #   - internal XP => exactly 2 parallel children
        #   - leaf XP     => exactly 2 devices
        # ------------------------------------------------------------------
        PatternSpec(
            name="dp_leaf_npu4_pim4_to_dp_xp_tp_tp",
            family=RewriteFamily.SKELETON_EXPANSION,
            match={
                "op": Parallelism.DP,
                "leaf": True,
                "closed": True,
                "device_counts": {"NPU": 4, "PIM": 4},
            },
            rewrite={
                "op": "DP",
                "children": [
                    {"op": "XP", "device_counts": {"NPU": 1, "PIM": 1}, "hint": xp_hint, "closed": True},
                    {"op": "TP", "device_counts": {"NPU": 3}, "closed": True},
                    {"op": "TP", "device_counts": {"PIM": 3}, "closed": True},
                ],
                # [dp_attr] 一共有 request_type_num 个 list，每个 list 是该 req_type 在各个 child 上的切分比例
                "hint": {"dp_attr": [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 1.0, 0.5]]},
                "closed": True,
            },
            weight=1.20,
        ),
        PatternSpec(
            name="tp_leaf_npu4_pim4_to_dp_xp_tp_tp",
            family=RewriteFamily.SKELETON_EXPANSION,
            match={
                "op": Parallelism.TP,
                "leaf": True,
                "closed": True,
                "device_counts": {"NPU": 4, "PIM": 4},
            },
            rewrite={
                "op": "DP",
                "children": [
                    {"op": "XP", "device_counts": {"NPU": 2, "PIM": 2}, "hint": xp_hint, "closed": False},
                    {"op": "TP", "device_counts": {"NPU": 2}, "closed": True},
                    {"op": "TP", "device_counts": {"PIM": 2}, "closed": True},
                ],
                # [dp_attr] 一共有 request_type_num 个 list，每个 list 是该 req_type 在各个 child 上的切分比例
                # "hint": {"dp_attr": [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 1.0, 0.5]]},
                "hint": {"dp_attr": [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]},
                "closed": True,
            },
            weight=1.20,
        ),
        # ------------------------------------------------------------------
        # Repartition / rollback around the new XP-leaf shape.
        # These rules let the search move between:
        #   XP leaf (2 devices)  <->  internal XP with 2 TP children
        # ------------------------------------------------------------------
        PatternSpec(
            name="xp_leaf_npu2_pim2_to_xp_tp_tp",
            family=RewriteFamily.LOCAL_REFINEMENT,
            match={
                "op": Parallelism.XP,
                "leaf": True,
                "closed": False,
                "device_counts": {"NPU": 2, "PIM": 2},
            },
            rewrite={
                "op": "XP",
                "children": [
                    {"op": "TP", "device_counts": {"NPU": 2}, "closed": True},
                    {"op": "TP", "device_counts": {"PIM": 2}, "closed": True},
                ],
                "hint": xp_hint,
                "closed": True,
            },
            weight=0.90,
        ),
        PatternSpec(
            name="xp_leaf_npu1_pim1_to_xp_tp_tp",
            family=RewriteFamily.LOCAL_REFINEMENT,
            match={
                "op": Parallelism.XP,
                "leaf": True,
                "closed": True,
                "device_counts": {"NPU": 1, "PIM": 1},
            },
            rewrite={
                "op": "XP",
                "children": [
                    {"op": "TP", "device_counts": {"NPU": 1}, "closed": True},
                    {"op": "TP", "device_counts": {"PIM": 1}, "closed": True},
                ],
                "hint": xp_hint,
                "closed": True,
            },
            weight=0.90,
        ),
        PatternSpec(
            name="xp_leaf_npu2_to_xp_tp_tp",
            family=RewriteFamily.LOCAL_REFINEMENT,
            match={
                "op": Parallelism.XP,
                "leaf": True,
                "closed": True,
                "device_counts": {"NPU": 2},
            },
            rewrite={
                "op": "XP",
                "children": [
                    {"op": "TP", "device_counts": {"NPU": 1}, "closed": True},
                    {"op": "TP", "device_counts": {"NPU": 1}, "closed": True},
                ],
                "hint": xp_hint,
                "closed": True,
            },
            weight=0.85,
        ),
        PatternSpec(
            name="xp_leaf_pim2_to_xp_tp_tp",
            family=RewriteFamily.LOCAL_REFINEMENT,
            match={
                "op": Parallelism.XP,
                "leaf": True,
                "closed": True,
                "device_counts": {"PIM": 2},
            },
            rewrite={
                "op": "XP",
                "children": [
                    {"op": "TP", "device_counts": {"PIM": 1}, "closed": True},
                    {"op": "TP", "device_counts": {"PIM": 1}, "closed": True},
                ],
                "hint": xp_hint,
                "closed": True,
            },
            weight=0.85,
        )
    ]


# ===============================
# Family-driven rewrite generators
# ===============================

def _fd_partition_binary(counts):
    items = list(counts.items())
    if not items:
        return []
    results = []
    for t, v in items:
        if v >= 2:
            left = dict(counts)
            right = dict(counts)
            left[t] = v // 2
            right[t] = v - left[t]
            results.append((left, right))
    return results


def _fd_all_parallelisms():
    return list(Parallelism)


def fd_skeleton_expansion(node: SymbolicNode, rng=None, params=None):
    if not node.is_leaf():
        return []
    counts = node.device_counts
    parts = _fd_partition_binary(counts)
    out = []
    for left, right in parts:
        for yop in _fd_all_parallelisms():
            for zop in _fd_all_parallelisms():
                repl = SymbolicNode(
                    op=node.op,
                    children=[
                        SymbolicNode(op=yop, device_counts=left),
                        SymbolicNode(op=zop, device_counts=right),
                    ],
                )
                out.append(RewriteCandidate(replacement=repl))
    return out


def fd_relabel(node: SymbolicNode, rng=None, params=None):
    out = []
    for p in _fd_all_parallelisms():
        if p != node.op:
            repl = node.clone()
            repl.op = p
            out.append(RewriteCandidate(replacement=repl))
    return out


def fd_repartition(node: SymbolicNode, rng=None, params=None):
    if node.is_leaf():
        return []
    out = []
    children = node.children
    if len(children) < 2:
        return []
    base = node.device_counts
    parts = _fd_partition_binary(base)
    for left, right in parts:
        new_children = [
            SymbolicNode(op=children[0].op, device_counts=left),
            SymbolicNode(op=children[1].op, device_counts=right),
        ]
        repl = SymbolicNode(op=node.op, children=new_children)
        out.append(RewriteCandidate(replacement=repl))
    return out


def fd_rollback(node: SymbolicNode, rng=None, params=None):
    if node.is_leaf():
        return []
    repl = SymbolicNode(op=node.op, device_counts=node.device_counts)
    return [RewriteCandidate(replacement=repl)]


def default_family_driven_patterns():
    return [
        ParametricPatternSpec(
            name="fd_skeleton_expansion",
            family=RewriteFamily.SKELETON_EXPANSION,
            match={"leaf": True},
            generator=fd_skeleton_expansion,
        ),
        ParametricPatternSpec(
            name="fd_relabel",
            family=RewriteFamily.RELABEL,
            match={},
            generator=fd_relabel,
        ),
        ParametricPatternSpec(
            name="fd_repartition",
            family=RewriteFamily.REPARTITION,
            match={"leaf": False},
            generator=fd_repartition,
        ),
        ParametricPatternSpec(
            name="fd_rollback",
            family=RewriteFamily.ROLLBACK,
            match={"leaf": False},
            generator=fd_rollback,
        ),
    ]
