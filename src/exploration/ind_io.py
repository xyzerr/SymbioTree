# ind_io_v3.py
from __future__ import annotations

from typing import Any, Dict, List, Union
import json
import pathlib

from parallelism.pnode import Parallelism, XpTag
from exploration.individual import Individual, Topology, TopologyNodeGene, Attrs, DeviceAssign


def _enum_to_json(v: Any) -> Any:
    if isinstance(v, Parallelism):
        return {"__enum__": "Parallelism", "value": int(v.value)}
    if isinstance(v, XpTag):
        return {"__enum__": "XpTag", "value": int(v.value)}
    return v


def _enum_from_json(obj: Any) -> Any:
    if isinstance(obj, dict) and "__enum__" in obj:
        name = obj["__enum__"]
        val = obj["value"]
        if name == "Parallelism":
            return Parallelism(val)
        if name == "XpTag":
            return XpTag(val)
    return obj


# =========================
# Visualization (printable)
# =========================
def format_topology(
    ind: Individual,
    show_devices: bool,
    show_attrs: bool,
) -> str:

    topo = ind.topology
    lines: List[str] = []


    def node_header(nid: int) -> str:
        t = topo.gene(nid).ptype
        return f"[{nid}] {t.name}"

    def node_attr_str(nid: int) -> str:
        t = topo.gene(nid).ptype
        if not show_attrs:
            return ""
        if t == Parallelism.DP:
            return f"dp_parallel_attr={ind.attrs.dp_attr.get(nid)}"
        if t == Parallelism.PP:
            return f"pp_parallel_attr={ind.attrs.pp_attr.get(nid)}"
        if t == Parallelism.TP:
            return f"tp_parallel_attr={ind.attrs.tp_attr.get(nid)}"
        if t == Parallelism.XP:
            return f"xp_parallel_attr={ind.attrs.xp_attr.get(nid)}"
        return ""

    def node_device_str(nid: int) -> str:
        if not show_devices:
            return ""
        if len(topo.children_of(nid)) != 0:
            return ""
        grp = ind.device_assign.leaf_to_devices.get(nid)
        return f"devices={grp}"

    def node_line(nid: int) -> str:
        parts = [node_header(nid)]
        a = node_attr_str(nid)
        if a:
            parts.append(a)
        d = node_device_str(nid)
        if d:
            parts.append(d)
        return " | ".join(parts)

    def dfs(nid: int, prefix: str, is_last: bool) -> None:
        branch = "└─ " if is_last else "├─ "
        lines.append(prefix + branch + node_line(nid))

        children = topo.children_of(nid)
        new_prefix = prefix + ("   " if is_last else "│  ")
        for i, cid in enumerate(children):
            dfs(cid, new_prefix, i == len(children) - 1)

    rid = topo.root_id
    lines.append(node_line(rid))
    children = topo.children_of(rid)
    for i, cid in enumerate(children):
        dfs(cid, "", i == len(children) - 1)

    return "\n".join(lines)


def format_individual(
    ind: Individual,
    *,
    show_devices: bool = True,
    show_attrs: bool = True,
    show_fitness: bool = True,
) -> str:
    """
    Return a readable multi-line string showing:
      - topology as an indented tree
      - each node's parallel_attr (and for leaf nodes, its device group)
      - summary: uid/fitness/devices/req_type_num

    Fix:
      - root node now prints its attrs/devices the same way as other nodes.
    """
    lines: List[str] = []

    if show_fitness:
        lines.append(f"uid: {ind.uid}")
        lines.append(f"fitness: T={ind.throughput} L={ind.latency} T_req={ind.throughput_req}")
    lines.append(f"req_type_num: {ind.req_type_num}")
    lines.append(f"devices: {list(ind.devices)}")
    lines.append(f"batch_size: {getattr(ind, 'batch_size', None)}")
    lines.append(f"sub_graph_batch_sizes: {getattr(ind, 'sub_graph_batch_sizes', {})}")
    lines.append("")

    lines.append("Topology:")
    lines.append(format_topology(ind, show_devices, show_attrs))

    return "\n".join(lines)


def print_individual(ind: Individual, **kwargs: Any) -> None:
    print(format_individual(ind, **kwargs))


# =========================
# Logging (pareto front and dse scatter point)
# 只保存比较简单的信息，用于日志以及绘图。
# 不同于 save_individual_json 是序列化。
# =========================

def log_individual_json(ind: Individual, path:str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f_dist = getattr(ind, 'f_dist', [])
        p_dist = getattr(ind, 'p_dist', [])
        f.write(json.dumps({"T": ind.throughput,
                            "L": ind.latency,
                            # bs: batch sizes
                            "T_req": ind.throughput_req,
                            "bs": getattr(ind, 'batch_size', None),
                            # sgbs: sub graph batch sizes
                            "sgbs": getattr(ind, 'sub_graph_batch_sizes', {}),
                            # memory feasibility log
                            "mem": getattr(ind, 'memory_feasibility_log', None),
                            # finished_dist
                            "fd": f_dist,
                            # processed_dist
                            "pd": p_dist,
                            # # finished_dist_ratio
                            # "fdr": [round(f / (r * sys_cfg.lam * sys_cfg.t_end), 4) if r != 0 else 0 for f, r in
                            #         zip(f_dist, sys_cfg.req_dist)],
                            # # finished_total_ratio
                            # "ftr": round(sum(f_dist) / (sys_cfg.lam * sys_cfg.t_end), 3),
                            # # processed_total_ratio
                            # "ptr": round(sum(p_dist) / (sys_cfg.lam * sys_cfg.t_end), 3),
                            "info": format_topology(ind, True, True),
                            }, ensure_ascii=False) + "\n")


# =========================
# Storage (JSON serialization)
# =========================

def individual_to_dict(ind: Individual) -> Dict[str, Any]:
    topo = ind.topology
    data: Dict[str, Any] = {
        "uid": ind.uid,
        "devices": list(ind.devices),
        "req_type_num": ind.req_type_num,
        "batch_size": getattr(ind, "batch_size", 1),
        "sub_graph_batch_sizes": {str(k): int(v) for k, v in getattr(ind, "sub_graph_batch_sizes", {}).items()},
        "memory_feasibility_log": getattr(ind, "memory_feasibility_log", None),
        "throughput": ind.throughput,
        "latency": ind.latency,
        "throughput_req": ind.throughput_req,
        "pareto_rank": ind.pareto_rank,
        "crowding": ind.crowding,
        "topology": [
            {
                "node_id": g.node_id,
                "parent_id": g.parent_id,
                "ptype": _enum_to_json(g.ptype),
                "child_slot": g.child_slot,
            }
            for g in topo.nodes
        ],
        "attrs": {
            "dp_attr": ind.attrs.dp_attr,
            "pp_attr": ind.attrs.pp_attr,
            "tp_attr": ind.attrs.tp_attr,
            "xp_attr": {str(k): [_enum_to_json(x) for x in v] for k, v in ind.attrs.xp_attr.items()},
        },
        "device_assign": {"leaf_to_devices": ind.device_assign.leaf_to_devices},
    }
    return data


def individual_from_dict(data: Dict[str, Any]) -> Individual:
    topo_nodes = []
    for item in data["topology"]:
        topo_nodes.append(
            TopologyNodeGene(
                node_id=int(item["node_id"]),
                parent_id=int(item["parent_id"]),
                ptype=_enum_from_json(item["ptype"]),
                child_slot=int(item["child_slot"]),
            )
        )
    topo = Topology(nodes=topo_nodes)

    attrs_raw = data["attrs"]
    dp_attr = {int(k): v for k, v in attrs_raw.get("dp_attr", {}).items()}
    pp_attr = {int(k): v for k, v in attrs_raw.get("pp_attr", {}).items()}
    tp_attr = {int(k): v for k, v in attrs_raw.get("tp_attr", {}).items()}
    xp_attr_raw = attrs_raw.get("xp_attr", {})
    xp_attr = {int(k): [_enum_from_json(x) for x in v] for k, v in xp_attr_raw.items()}

    attrs = Attrs(dp_attr=dp_attr, pp_attr=pp_attr, tp_attr=tp_attr, xp_attr=xp_attr)

    da_raw = data["device_assign"]["leaf_to_devices"]
    da = DeviceAssign(leaf_to_devices={int(k): v for k, v in da_raw.items()})

    ind = Individual(
        topology=topo,
        device_assign=da,
        attrs=attrs,
        devices=list(data["devices"]),
        req_type_num=int(data["req_type_num"]),
        batch_size=int(data.get("batch_size", 1)),
        sub_graph_batch_sizes={int(k): int(v) for k, v in data.get("sub_graph_batch_sizes", {}).items()},
    )
    ind.uid = data.get("uid")
    ind.throughput = data.get("throughput")
    ind.latency = data.get("latency")
    if "memory_feasibility_log" in data:
        setattr(ind, "memory_feasibility_log", data.get("memory_feasibility_log"))
    ind.check_legality()
    return ind


def save_individual_json(ind: Individual, path: Union[str, pathlib.Path], *, indent: int = 2) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(individual_to_dict(ind), f, ensure_ascii=False, indent=indent)


def load_individual_json(path: Union[str, pathlib.Path]) -> Individual:
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return individual_from_dict(data)
