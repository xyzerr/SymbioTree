"""面向用户的部署配置、结构展示与预检查；调用 src 中的原有实现。"""
from __future__ import annotations

import json
import math
from pathlib import Path

HARDWARE_PRESETS = {"intra_package": 8, "intra_node": 9}
DEPLOYMENT_PRESETS = {"gpu_pim_tp8": 16, "gpu_pim_pipeline2_tp4": 17}
FORMAT = "symbiotree.deployment.v1"
PUBLIC_TYPES = {"XP": "OP", "DP": "RP", "PP": "BP", "TP": "TP"}


def resolve_case(section, presets, label, *, extra_keys=()):
    if not isinstance(section, dict):
        raise ValueError(f"{label} 必须是对象")
    selectors = set(section) & {"preset", "case_index"}
    if len(selectors) != 1 or set(section) - selectors - set(extra_keys):
        raise ValueError(f"{label} 必须在 preset 与 case_index 中选择一个，不能同时使用")
    if "preset" in section:
        name = section["preset"]
        if not isinstance(name, str) or name not in presets:
            raise ValueError(f"未知的 {label}.preset，可用名称：{', '.join(presets)}")
        return presets[name]
    value = section["case_index"]
    if type(value) is not int or value < 0:
        raise ValueError(f"{label}.case_index 必须是非负整数")
    return value


def device_label(device):
    return device.name.replace("npu_", "gpu").replace("pim_", "pim")


def positive_int(value, where):
    if type(value) is not int or value < 1:
        raise ValueError(f"{where} 必须是正整数")
    return value


def weights(value, size, where, *, allow_zero=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{where} 需要 {size} 个数值，与子分支或设备数量一致")
    if any(type(v) not in (int, float) or not math.isfinite(v)
           or (v < 0 if allow_zero else v <= 0) for v in value):
        raise ValueError(f"{where} 需要有限的{'非负' if allow_zero else '正'}数")
    if not math.isfinite(sum(value)) or sum(value) <= 0:
        raise ValueError(f"{where} 的总和必须为有限正数")
    return list(value)


def from_user_tree(data, hardware, cfg):
    from exploration.individual import Individual, Topology, TopologyNodeGene, Attrs, DeviceAssign
    from parallelism.pnode import Parallelism, XpTag

    allowed = {"format", "batch_size", "path_batch_sizes", "tree"}
    if set(data) - allowed or data.get("format") != FORMAT:
        raise ValueError(f"部署格式应为 {FORMAT}，支持字段：{', '.join(sorted(allowed))}")
    batch = positive_int(data.get("batch_size", cfg.max_batch_lo), "batch_size")
    aliases = {device_label(d).lower(): int(d.idx) for d in hardware.devices}
    aliases.update({d.name.lower(): int(d.idx) for d in hardware.devices})
    ids = {int(d.idx) for d in hardware.devices}
    genes, attrs, assignment, names, used = [], Attrs(), DeviceAssign(), {}, {}
    type_map = {"OP": Parallelism.XP, "RP": Parallelism.DP,
                "BP": Parallelism.PP, "TP": Parallelism.TP}

    def visit(node, parent=-1, slot=0, path="tree", seen_bp=False, seen_tp=False):
        if not isinstance(node, dict) or not isinstance(node.get("type"), str) or node["type"] not in type_map:
            raise ValueError(f"{path}.type 应为 OP、RP、BP 或 TP")
        kind = node["type"]
        fields = {"type", "name", "children", "devices"} | {
            "OP": {"operators"}, "RP": {"routing"}, "BP": {"weights"}, "TP": {"weights"}
        }[kind]
        if set(node) - fields:
            raise ValueError(f"{path} 有不适用于 {kind} 的字段：{sorted(set(node) - fields)}")
        nid = len(genes)
        name = node.get("name", f"node_{nid}")
        if not isinstance(name, str) or not name.strip() or name in names.values():
            raise ValueError(f"{path}.name 必须是非空且不重复的名称")
        if kind == "RP" and (seen_bp or seen_tp):
            raise ValueError(f"节点 {name} 的 RP 不能位于 BP 或 TP 之下")
        if kind == "BP" and seen_tp:
            raise ValueError(f"节点 {name} 的 BP 不能位于 TP 之下")
        names[nid] = name
        genes.append(TopologyNodeGene(nid, parent, type_map[kind], slot))
        if ("children" in node) == ("devices" in node):
            raise ValueError(f"{name} 必须且只能设置 children 或 devices")
        internal = "children" in node
        children = node.get("children")
        items = children if internal else node["devices"]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{name} 的分支或设备列表不能为空")
        if kind == "RP" and not internal:
            raise ValueError(f"节点 {name} 的 RP 应通过 children 划分执行路径，各路径需包含非 RP 划分节点")
        k = len(items)
        if kind == "OP":
            operators = node.get("operators", ["linear", "attention"])
            if k != 2 or not isinstance(operators, list) or len(operators) != 2 or sorted(str(x) for x in operators) != ["attention", "linear"]:
                raise ValueError(f"{name} 的 OP 需要两个分支，operators 包含 linear 和 attention 各一次")
            attrs.xp_attr[nid] = [XpTag.LINEAR if x == "linear" else XpTag.ATTENTION for x in operators]
        elif kind == "RP":
            routing = node.get("routing")
            if not isinstance(routing, list) or len(routing) != cfg.req_type_num:
                raise ValueError(f"{name}.routing 需要按短、中、长请求给出 {cfg.req_type_num} 行比例")
            attrs.dp_attr[nid] = [weights(row, k, f"{name}.routing[{i}]", allow_zero=True)
                                  for i, row in enumerate(routing)]
        else:
            value = weights(node.get("weights", [1] * k), k, f"{name}.weights")
            (attrs.pp_attr if kind == "BP" else attrs.tp_attr)[nid] = value
        if internal:
            for i, child in enumerate(children):
                visit(child, nid, i, f"{path}.children[{i}]", seen_bp or kind == "BP", seen_tp or kind == "TP")
        else:
            group = []
            for value in items:
                did = aliases.get(value.lower()) if isinstance(value, str) else value
                if type(did) is not int or did not in ids:
                    raise ValueError(f"{name} 使用未知设备 {value!r}；可用名称：{', '.join(aliases)}")
                if did in used:
                    raise ValueError(f"设备 {value} 被 {used[did]} 与 {name} 重复分配")
                used[did] = name
                group.append(did)
            assignment.leaf_to_devices[nid] = group

    visit(data.get("tree"))
    missing = ids - set(used)
    if missing:
        labels = [device_label(d) for d in hardware.devices if d.idx in missing]
        raise ValueError(f"以下设备尚未分配：{', '.join(labels)}；当前部署必须覆盖全部硬件端点")
    ind = Individual(Topology(genes), assignment, attrs,
                     [int(d.idx) for d in hardware.devices], cfg.req_type_num, batch_size=batch)
    by_name = {name: nid for nid, name in names.items()}
    path_batches = data.get("path_batch_sizes", {})
    if not isinstance(path_batches, dict):
        raise ValueError("path_batch_sizes 必须是路径名称到批大小的对象")
    for name, value in path_batches.items():
        if name not in by_name:
            raise ValueError(f"path_batch_sizes 引用了未知节点 {name}")
        ind.sub_graph_batch_sizes[by_name[name]] = positive_int(value, f"path_batch_sizes.{name}")
    ind.ui_names = names
    return ind


def from_preset(hardware, cfg, model):
    from parallelism.ptree import ParallelismTree
    from exploration.seed_from_pcase import individual_from_pcase_root
    tree = ParallelismTree(cfg, model, hardware, case_idx=cfg.pcase_index)
    # 预设叶编号是任务编号；须使用原始 mapper 转为真实硬件编号。
    for leaf in tree.leaf_nodes:
        leaf.idx = int(tree.mapper[leaf.name].idx)
    return individual_from_pcase_root(tree.root_node, devices=[int(d.idx) for d in hardware.devices],
                                      req_type_num=cfg.req_type_num, batch_size=cfg.max_batch_lo)


def prepare_deployment(cfg, model, deployment=None):
    """解析并校验，不运行请求模拟。返回个体、硬件树、DDT 和各路径容量检查。"""
    from hardware.htree import HardwareTree
    from exploration.ind_io import individual_from_dict
    from exploration.decoder import RootInit, decode_to_root, DecodeError
    from exploration.feasibility import FeasibilityConfig, compute_feasible_batch_caps
    from parallelism.ptraversal import derive_from_node, detect_begin_nodes
    from parallelism.pnode import Parallelism

    hardware = HardwareTree(cfg.hcase_index)
    if deployment is None:
        ind = from_preset(hardware, cfg, model)
    else:
        data = json.loads(Path(deployment).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("部署文件必须是 JSON 对象")
        ind = from_user_tree(data, hardware, cfg) if "format" in data or "tree" in data else individual_from_dict(data)
    ind.check_legality()
    if ind.req_type_num != cfg.req_type_num:
        raise ValueError("部署与配置的请求类别数量不一致")
    if set(ind.devices) != {int(d.idx) for d in hardware.devices}:
        raise ValueError("部署与当前硬件的设备编号不匹配")
    positive_int(ind.batch_size, "batch_size")
    # 旧格式也检查数值，防止手改后出现负比例或零分母。
    for nid in ind.topology.iter_dfs():
        kind = ind.topology.gene(nid).ptype
        k = len(ind.topology.children_of(nid)) or len(ind.device_assign.leaf_to_devices[nid])
        if kind == Parallelism.DP:
            for i, row in enumerate(ind.attrs.dp_attr[nid]):
                weights(row, k, f"RP 节点 {nid} 第 {i} 行", allow_zero=True)
        elif kind in (Parallelism.PP, Parallelism.TP):
            value = (ind.attrs.pp_attr if kind == Parallelism.PP else ind.attrs.tp_attr)[nid]
            weights(value, k, f"{PUBLIC_TYPES[kind.name]} 节点 {nid}")
    scope = RootInit([[0., 1.]] * cfg.req_type_num, [0, model.layer_num - 1], [0., 1.])
    try:
        root = decode_to_root(ind, scope)
        derive_from_node(root)
    except (DecodeError, ValueError) as exc:
        if "range is too small" in str(exc):
            raise ValueError("层数不足，无法为每个 BP 分支分配至少一层；请减少阶段数或调整层划分") from exc
        raise
    starts = list(detect_begin_nodes(root))
    if not starts:
        raise ValueError("部署没有可执行的请求路径；请在 RP 子树中设置 TP、BP 或 OP 节点")
    start_ids = {n._topo_node_id for n in starts}
    if set(ind.sub_graph_batch_sizes) - start_ids:
        raise ValueError("path_batch_sizes 只能设置执行路径起点，即沿根部 RP 分支遇到的第一个非 RP 节点")
    for nid, batch in ind.sub_graph_batch_sizes.items():
        positive_int(batch, f"路径 {nid} 的批大小")
    fc = FeasibilityConfig(model, scope,
        {int(d.idx): float(d.meta['mem_cap']) for d in hardware.devices},
        {int(d.idx): int(d.meta.get('byte', 2)) for d in hardware.devices},
        peak_seq_len=cfg.peak_seq_len, runtime_reserve_ratio=cfg.runtime_reserve_ratio)
    upper = max([ind.batch_size, *ind.sub_graph_batch_sizes.values()])
    caps = compute_feasible_batch_caps(ind, fc, default_upper=upper)
    names = getattr(ind, "ui_names", {})
    paths = []
    for node in starts:
        nid = node._topo_node_id
        name = names.get(nid, f"node_{nid}")
        rate = sum((r[1] - r[0]) * p * cfg.lam for r, p in zip(node.dp_attr, cfg.req_dist))
        if rate <= 0:
            raise ValueError(f"路径 {name} 未分到请求，请调整 routing 或 serving.req_dist")
        batch = ind.sub_graph_batch_sizes.get(nid, ind.batch_size)
        cap = caps[nid]
        if cap < 1 or (deployment is not None and batch > cap):
            raise ValueError(f"路径 {name} 的批大小 {batch} 超过当前容量检查允许的 {cap}，请降低批大小或调整配置")
        paths.append({"name": name, "node_id": nid, "arrival_rate": rate,
                      "requested_batch": batch, "effective_batch": min(batch, cap)})
    return ind, hardware, root, paths


def describe(prepared, cfg, model):
    from hardware_config import CUSTOM_CASE, describe_hardware
    ind, hardware, root, paths = prepared
    names = getattr(ind, "ui_names", {})
    labels = {int(d.idx): device_label(d) for d in hardware.devices}
    hardware_name = "自定义硬件" if cfg.hcase_index == CUSTOM_CASE else f"硬件配置 {cfg.hcase_index}"
    lines = [f"模型 {model.model_name}，{model.layer_num} 层；{hardware_name}", "", describe_hardware(hardware)]
    lines.extend(["", "部署树 DDT（层编号从 0 开始）"])

    def ddt(node, indent=""):
        if not hasattr(node, "_topo_node_id"):
            lines.append(indent + labels[node.idx])
            return
        nid = node._topo_node_id
        name = names.get(nid, f"node_{nid}")
        kind = PUBLIC_TYPES[node.type.name]
        values = node.parallel_attr
        if kind == "OP":
            attr = "算子 " + "/".join(x.name.lower() for x in values)
        else:
            attr = ("路由比例 " if kind == "RP" else "划分权重 ") + str(values)
        operators = {"BOTH": "完整算子", "LINEAR": "线性计算", "ATTENTION": "注意力"}
        lines.append(f"{indent}{name} [{kind}] {attr}；层 {node.pp_attr[0]}–{node.pp_attr[1]}，{operators[node.xp_attr.name]}")
        if all(not hasattr(c, "_topo_node_id") for c in node.children):
            for c in node.children:
                lines.append(f"{indent}  {labels[c.idx]} → 层 {c.pp_attr[0]}–{c.pp_attr[1]}，{operators[c.xp_attr.name]}，TP范围 [{c.tp_attr[0]:.4g}, {c.tp_attr[1]:.4g}]")
        else:
            for c in node.children:
                ddt(c, indent + "  ")
    ddt(root)
    lines.extend(["", "执行路径与批大小（采用当前内存可行性模型）"])
    for p in paths:
        batch = str(p['effective_batch'])
        if p['requested_batch'] != p['effective_batch']:
            batch += f"（由 {p['requested_batch']} 按容量下调）"
        lines.append(f"{p['name']}：到达率 {p['arrival_rate']:.6g} 请求/秒，批大小 {batch}")
    lines.append("校验通过，未运行性能模拟。")
    return "\n".join(lines)
