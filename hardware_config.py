"""从用户配置构造硬件树，并在单次运行中接入原有模拟器。"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import re

FORMAT = "symbiotree.hardware.v1"
CUSTOM_CASE = -1
HW_KEYS = {"NPU_FLOPS", "NPU_BW", "NPU_MEM_CAP", "PIM_FLOPS", "PIM_BW", "PIM_MEM_CAP",
           "nvl_bw", "nvl_lat", "mm_bw", "mm_lat"}
DEVICE_FIELDS = {"compute_tflops", "memory_bandwidth_tb_s", "memory_gib", "bytes_per_element"}
LINK_FIELDS = {"bandwidth_gb_s", "latency_ns"}


def _number(value, path, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{path} 必须是有限的{'非负' if zero else '正'}数")
    return value


def _fields(data, allowed, path):
    if not isinstance(data, dict):
        raise ValueError(f"{path} 必须是 JSON 对象")
    if set(data) - allowed:
        raise ValueError(f"{path} 含未知字段：{sorted(set(data) - allowed)}")


def _link(data, path):
    return {"bw": _number(data.get("bandwidth_gb_s"), f"{path}.bandwidth_gb_s"),
            "lat": _number(data.get("latency_ns"), f"{path}.latency_ns", zero=True)}


def build_hardware(data):
    """校验硬件 JSON，返回原生 HHT 根、设备列表和 AllReduce 参数。"""
    from hardware.hnode import HwGroup, HwUnit
    from hardware.htraversal import fill_descendant_set
    from deployment_config import device_label

    _fields(data, {"format", "device_defaults", "all_reduce", "tree"}, "hardware")
    if data.get("format") != FORMAT:
        raise ValueError(f"硬件格式应为 {FORMAT}")
    defaults = data.get("device_defaults", {})
    _fields(defaults, {"GPU", "PIM"}, "device_defaults")
    for kind, values in defaults.items():
        _fields(values, DEVICE_FIELDS, f"device_defaults.{kind}")
        for key, value in values.items():
            _number(value, f"device_defaults.{kind}.{key}")
            if key == "bytes_per_element" and type(value) is not int:
                raise ValueError(f"device_defaults.{kind}.bytes_per_element 必须是正整数")
    collective = data.get("all_reduce")
    _fields(collective, LINK_FIELDS, "all_reduce")
    all_reduce = _link(collective, "all_reduce")
    names, aliases, depths, devices = set(), set(), set(), []
    group_count = 0

    def visit(node, depth=0, path="tree"):
        nonlocal group_count
        if depth > 32:
            raise ValueError("硬件树层级超过 32，请简化组织结构")
        if not isinstance(node, dict) or node.get("type") not in ("group", "GPU", "PIM"):
            raise ValueError(f"{path}.type 应为 group、GPU 或 PIM")
        kind = node["type"]
        _fields(node, {"name", "type"} | ({"children"} | LINK_FIELDS if kind == "group" else DEVICE_FIELDS), path)
        name = node.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise ValueError(f"{path}.name 应以英文字母开头，只含字母、数字、下划线或连字符")
        if name.lower() in names:
            raise ValueError(f"硬件名称重复：{name}（不区分大小写）")
        names.add(name.lower())
        if kind == "group":
            children = node.get("children")
            if not isinstance(children, list) or not children:
                raise ValueError(f"{path}.children 必须是非空列表")
            result = HwGroup(idx=group_count, name=name, meta=_link(node, path))
            group_count += 1
            for i, child in enumerate(children):
                result.add(visit(child, depth + 1, f"{path}.children[{i}]"))
            return result
        if depth == 0:
            raise ValueError("硬件树根节点必须为 group")
        values = {**defaults.get(kind, {}), **{k: v for k, v in node.items() if k in DEVICE_FIELDS}}
        for key in DEVICE_FIELDS:
            _number(values.get(key), f"{path}.{key}")
        if type(values["bytes_per_element"]) is not int:
            raise ValueError(f"{path}.bytes_per_element 必须是正整数")
        result = HwUnit(idx=len(devices), name=name,
                        meta={"type": "NPU" if kind == "GPU" else "PIM",
                              "flops": values["compute_tflops"], "bw": values["memory_bandwidth_tb_s"],
                              "mem_cap": values["memory_gib"], "byte": values["bytes_per_element"]})
        alias = device_label(result).lower()
        if alias in aliases:
            raise ValueError(f"设备名称转换后重复：{alias}，请改用不同名称")
        aliases.add(alias)
        devices.append(result)
        depths.add(depth)
        return result

    root = visit(data.get("tree"))
    if len(depths) != 1:
        raise ValueError("当前通信模型要求所有设备叶节点处于同一深度，请调整各分组层级")
    for device in devices:
        fill_descendant_set(device)
    return root, devices, all_reduce


def resolve_hardware(section, config_dir):
    """解析文件路径并保存自包含配置；文件路径相对于主配置文件。"""
    from deployment_config import HARDWARE_PRESETS, resolve_case
    if not isinstance(section, dict):
        raise ValueError("hardware 必须是对象")
    if "topology_file" in section or "topology" in section:
        if len(section) != 1:
            raise ValueError("自定义硬件只填写 topology_file 或 topology，不与 preset、case_index 或 parameters 混用")
        if "topology_file" in section:
            value = section["topology_file"]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("hardware.topology_file 必须是文件路径")
            path = Path(value)
            if not path.is_absolute():
                path = Path(config_dir) / path
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        else:
            data = deepcopy(section["topology"])
        build_hardware(data)
        return {"topology": data}, CUSTOM_CASE
    case = resolve_case(section, HARDWARE_PRESETS, "hardware", extra_keys=("parameters",))
    parameters = section.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != HW_KEYS:
        raise ValueError(f"hardware.parameters 必须且只能包含以下字段： {', '.join(sorted(HW_KEYS))}")
    for key, value in parameters.items():
        _number(value, f"hardware.parameters.{key}", zero=key.endswith("_lat"))
    return deepcopy(section), case


@contextmanager
def hardware_session(section):
    """临时加载硬件配置；退出后恢复，不改写 src 文件。不同配置应使用不同进程。"""
    from hardware import hcase, hperf
    from hardware.htree import HardwareTree

    old_parameters = {key: getattr(hcase, key) for key in HW_KEYS}
    old_collective = hperf.nvl_bw, hperf.nvl_lat
    old_build = HardwareTree.build_tree_by_case
    old_parallel = None
    parallel_cls = None
    try:
        if "topology" not in section:
            for key, value in section["parameters"].items():
                setattr(hcase, key, value)
            hperf.nvl_bw, hperf.nvl_lat = hcase.nvl_bw, hcase.nvl_lat
        else:
            data = deepcopy(section["topology"])
            _, _, collective = build_hardware(data)
            hperf.nvl_bw, hperf.nvl_lat = collective["bw"], collective["lat"]

            def build_custom(tree, case_idx):
                if case_idx != CUSTOM_CASE:
                    return old_build(tree, case_idx)
                tree.root, tree.devices, _ = build_hardware(data)

            HardwareTree.build_tree_by_case = build_custom
            from parallelism.ptree import ParallelismTree
            parallel_cls = ParallelismTree
            old_parallel = ParallelismTree.build_tree_by_case

            def initialize_custom(tree, case_idx):
                if case_idx != CUSTOM_CASE:
                    return old_parallel(tree, case_idx)
                # 评估器随后用真实候选替换此初始树；不依赖固定端点数的预设。
                from deployment_config import from_user_tree
                from exploration.decoder import RootInit, decode_to_root
                from parallelism.ptraversal import derive_from_node, detect_begin_nodes
                ind = from_user_tree({"format": "symbiotree.deployment.v1", "batch_size": 1,
                                      "tree": {"type": "TP", "devices": [d.idx for d in tree.htree.devices]}},
                                     tree.htree, tree.sys_cfg)
                scope = RootInit([[0., 1.]] * tree.sys_cfg.req_type_num,
                                 [0, tree.model_cfg.layer_num - 1], [0., 1.])
                tree.root_node = decode_to_root(ind, scope)
                tree.leaf_nodes = derive_from_node(tree.root_node)
                tree.begin_nodes = list(detect_begin_nodes(tree.root_node))
                devices = {d.idx: d for d in tree.htree.devices}
                tree.mapper = {leaf.name: devices[leaf.idx] for leaf in tree.leaf_nodes}

            ParallelismTree.build_tree_by_case = initialize_custom
        yield
    finally:
        HardwareTree.build_tree_by_case = old_build
        if old_parallel is not None:
            parallel_cls.build_tree_by_case = old_parallel
        for key, value in old_parameters.items():
            setattr(hcase, key, value)
        hperf.nvl_bw, hperf.nvl_lat = old_collective


def describe_hardware(hardware):
    from deployment_config import device_label
    lines = ["硬件树 HHT"]

    def visit(node, indent=""):
        if node.is_leaf():
            kind = "GPU" if node.meta['type'] == "NPU" else node.meta['type']
            text = (f"{device_label(node)} [设备编号 {node.idx}，{kind}，"
                    f"{node.meta['flops']} TFLOP/s，{node.meta['bw']} TB/s，{node.meta['mem_cap']} GiB]")
        else:
            text = f"{node.name} [互连 {node.meta['bw']} GB/s，{node.meta['lat']} ns]"
        lines.append(indent + text)
        for child in getattr(node, "children", []):
            visit(child, indent + "  ")
    visit(hardware.root)
    from hardware import hperf
    lines.append(f"AllReduce：{hperf.nvl_bw} GB/s，{hperf.nvl_lat} ns")
    return "\n".join(lines)
