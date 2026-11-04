from parallelism.pnode import *
from math import ceil, floor
import copy

def detect_begin_nodes(node: BasicNode):
    if not isinstance(node, BasicParallelismNode):
        return
    if node.type == Parallelism.DP:
        for idx, child in enumerate(node.children):
            node.derive_child_info(child, idx)
            yield from detect_begin_nodes(child)
    else:
        yield node

def derive_from_node(node: BasicNode):
    # 该 DFS 有两个作用：既进行节点信息的推导，同时也会收集 leaf node
    # node.print_info()
    if not isinstance(node, BasicParallelismNode):
        return [node]
    leaf_node_list = []
    for idx, child in enumerate(node.children):
        node.derive_child_info(child, idx)
        leaf_node_list.extend(derive_from_node(child))
    return leaf_node_list

def trigger_leaf_node(layer_idx, module_idx, leaf_nodes):
    active_leaf_idxes = []
    for idx, leaf_node in enumerate(leaf_nodes):
        if leaf_node.pp_attr[0] <= layer_idx <= leaf_node.pp_attr[1]:
            if leaf_node.tp_attr[0] >= 0.0 and leaf_node.tp_attr[1] <= 1.0:
                if leaf_node.xp_attr == XpTag.BOTH or \
                        leaf_node.xp_attr == XpTag.ATTENTION and module_idx == 1 or \
                        leaf_node.xp_attr == XpTag.LINEAR and module_idx != 1:
                    # MegaScale-Infer 中描述的 A/F 分离适用于下面的代码，其中 Attention 包括 QKV、ATTN、PROJ。
                    # if leaf_node.xp_attr == XpTag.BOTH or \
                    #    leaf_node.xp_attr == XpTag.ATTENTION and module_idx <= 2 or \
                    #    leaf_node.xp_attr == XpTag.LINEAR and module_idx == 3:
                    active_leaf_idxes.append(idx)
    return active_leaf_idxes


def print_src_and_dst_info(leaf_nodes: List[BasicHardwareNode],
                           src_node_indexes: List[int], dst_node_indexes: List[int]):
    print("src:")
    for src_idx in src_node_indexes:
        print(src_idx, "dp:", leaf_nodes[src_idx].dp_attr, "tp:", leaf_nodes[src_idx].tp_attr)
    print("dst:")
    for dst_idx in dst_node_indexes:
        print(dst_idx, "dp:", leaf_nodes[dst_idx].dp_attr, "tp:", leaf_nodes[dst_idx].tp_attr)