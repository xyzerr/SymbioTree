from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Set

from parallelism.pnode import Parallelism


_PARALLELISM_NAME_TO_ENUM = {name.upper(): value for name, value in Parallelism.__members__.items()}



def normalize_disabled_parallelisms(values: Optional[Sequence[Any]]) -> Set[Parallelism]:
    out: Set[Parallelism] = set()
    if not values:
        return out

    for raw in values:
        if raw is None:
            continue
        if isinstance(raw, Parallelism):
            out.add(raw)
            continue
        if isinstance(raw, str):
            key = raw.strip().upper()
            if not key:
                continue
            if key not in _PARALLELISM_NAME_TO_ENUM:
                raise ValueError(
                    f"Unknown parallelism in disabled_parallelisms: {raw!r}. "
                    f"Expected one of {sorted(_PARALLELISM_NAME_TO_ENUM.keys())}."
                )
            out.add(_PARALLELISM_NAME_TO_ENUM[key])
            continue
        raise TypeError(
            "disabled_parallelisms only supports Parallelism enums or strings like 'DP'/'PP'/'TP'/'XP'. "
            f"Got {type(raw).__name__}: {raw!r}"
        )
    return out



def allowed_parallelism_types(
    seen_pp: bool,
    seen_tp: bool,
    disabled_parallelisms: Optional[Sequence[Any]] = None,
) -> List[Parallelism]:
    disabled = normalize_disabled_parallelisms(disabled_parallelisms)

    allowed = [Parallelism.XP, Parallelism.TP, Parallelism.PP, Parallelism.DP]
    if seen_pp or seen_tp:
        allowed = [t for t in allowed if t != Parallelism.DP]
    if seen_tp:
        allowed = [t for t in allowed if t != Parallelism.PP]
    return [t for t in allowed if t not in disabled]



def symbolic_contains_disabled_parallelisms(root: Any, disabled_parallelisms: Optional[Sequence[Any]]) -> bool:
    disabled = normalize_disabled_parallelisms(disabled_parallelisms)
    if not disabled or root is None:
        return False

    walk = getattr(root, "walk", None)
    if callable(walk):
        for node in walk():
            if getattr(node, "op", None) in disabled:
                return True
    return False



def individual_contains_disabled_parallelisms(ind: Any, disabled_parallelisms: Optional[Sequence[Any]]) -> bool:
    disabled = normalize_disabled_parallelisms(disabled_parallelisms)
    if not disabled or ind is None:
        return False

    topo = getattr(ind, "topology", None)
    if topo is None:
        return False

    try:
        for nid in topo.iter_dfs():
            if topo.gene(nid).ptype in disabled:
                return True
    except Exception:
        return False
    return False



def filter_init_patterns_by_parallelism(patterns: Sequence[Any], disabled_parallelisms: Optional[Sequence[Any]]) -> List[Any]:
    disabled = normalize_disabled_parallelisms(disabled_parallelisms)
    if not disabled:
        return list(patterns)
    return [pat for pat in patterns if not _spec_contains_disabled_parallelisms(getattr(pat, "root", None), disabled)]



def filter_rewrite_patterns_by_parallelism(patterns: Sequence[Any], disabled_parallelisms: Optional[Sequence[Any]]) -> List[Any]:
    disabled = normalize_disabled_parallelisms(disabled_parallelisms)
    if not disabled:
        return list(patterns)

    out: List[Any] = []
    for pat in patterns:
        declared = getattr(pat, "declared_parallelisms", None)
        if declared:
            try:
                declared_set = {_to_parallelism(value) for value in declared}
                if declared_set & disabled:
                    continue
            except Exception:
                pass

        match_spec = getattr(pat, "match", None)
        rewrite_spec = getattr(pat, "rewrite", None)
        if _spec_contains_disabled_parallelisms(match_spec, disabled):
            continue
        if _spec_contains_disabled_parallelisms(rewrite_spec, disabled):
            continue
        out.append(pat)
    return out



def _spec_contains_disabled_parallelisms(spec: Any, disabled: Set[Parallelism]) -> bool:
    if not disabled or spec is None:
        return False
    return bool(_collect_parallelisms_from_spec(spec) & disabled)



def _collect_parallelisms_from_spec(spec: Any) -> Set[Parallelism]:
    out: Set[Parallelism] = set()

    if isinstance(spec, dict):
        op = spec.get("op")
        if op is not None:
            out.add(_to_parallelism(op))
        for value in spec.values():
            out |= _collect_parallelisms_from_spec(value)
        return out

    if isinstance(spec, (list, tuple)):
        for item in spec:
            out |= _collect_parallelisms_from_spec(item)
        return out

    return out



def _to_parallelism(value: Any) -> Parallelism:
    if isinstance(value, Parallelism):
        return value
    if isinstance(value, str):
        key = value.strip().upper()
        if key in _PARALLELISM_NAME_TO_ENUM:
            return _PARALLELISM_NAME_TO_ENUM[key]
    raise ValueError(f"Unsupported parallelism spec: {value!r}")
