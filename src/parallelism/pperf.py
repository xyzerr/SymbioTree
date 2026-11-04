from parallelism.pnode import *
from math import ceil, floor
from collections import defaultdict

def interval_intersection(interval_a, interval_b):
    a1, a2 = interval_a
    b1, b2 = interval_b

    # 可选：确保输入顺序正确
    if a1 > a2:
        a1, a2 = a2, a1
    if b1 > b2:
        b1, b2 = b2, b1

    L = max(a1, b1)
    R = min(a2, b2)

    has_intersection = L <= R  # 闭区间：端点相接也算相交
    length = max(0.0, R - L)  # 相交长度（测度）

    inter = [L, R] if has_intersection else None
    return inter, length


def peer_to_peer_communication_pattern(src_node: BasicHardwareNode,
                               dst_node: BasicHardwareNode,
                               req_type_distribution: List[int],
                               full_data_length: int):
    request_type_num = len(src_node.dp_attr)
    tp_comm_interval, tp_comm_ratio = interval_intersection(src_node.tp_attr, dst_node.tp_attr)
    total_comm_size = 0
    if tp_comm_ratio > 0.0:
        for req_idx in range(request_type_num):
            dp_comm_interval, dp_comm_ratio = interval_intersection(src_node.dp_attr[req_idx],
                                                                   dst_node.dp_attr[req_idx])
            # dp_comm_ratio：每个 src 向每个 dst 发送 req_idx 类型数据的比例
            batch_size = ceil(dp_comm_ratio * req_type_distribution[req_idx])
            # tp_comm_ratio：每个 src 向每个 dst 发送的尺寸的比例
            data_length = ceil(full_data_length * tp_comm_ratio)
            total_comm_size += batch_size * data_length
            # print(src_idx, dst_idx, req_idx, dp_comm_ratio, tp_comm_ratio, batch_size, data_length)
    return total_comm_size


def all_reduce_communication_pattern(leaf_nodes: List[BasicHardwareNode],
                             all_reduce_node_indexes: List[int],
                             req_type_distribution: List[int],
                             full_data_length: int):
    # all_reduce_groups 是一个字典，key 是 all reduce group 的终结点，value 是 all reduce group leaf node index 列表。
    all_reduce_groups = defaultdict(list[int])
    for src in all_reduce_node_indexes:
        node = leaf_nodes[src]
        while node.parent is not None and node.parent.type == Parallelism.TP:
            node = node.parent
        assert node is not None
        all_reduce_groups[node.name].append(src)
    # print(all_reduce_groups)

    all_reduce_comm_pattern: List[tuple[List[str], int]] = []
    # 最终返回一个列表，每一项是一个 tuple，包含2个元素：参与 all reduce 的 node name 列表，以及 all reduce 通信量
    for _, all_reduce_group in all_reduce_groups.items():
        all_reduce_leaf_name_list: List[str] = []
        all_reduce_data_length_ratio = 0
        all_reduce_batch_size = 0
        for leaf_node_idx in all_reduce_group:
            leaf_node = leaf_nodes[leaf_node_idx]
            all_reduce_leaf_name_list.append(leaf_node.name)
            all_reduce_data_length_ratio = max(all_reduce_data_length_ratio, leaf_node.tp_attr[1] - leaf_node.tp_attr[0])
            # 同一个 all reduce group 中的 leaf node 的 dp_attr 一定是相同的。
            all_reduce_batch_size = sum([ceil((d[1]-d[0])*n) for d,n in zip(leaf_node.dp_attr, req_type_distribution)])
        all_reduce_data_length = ceil(all_reduce_data_length_ratio * full_data_length)
        all_reduce_data_size = all_reduce_batch_size * all_reduce_data_length
        all_reduce_tuple = (all_reduce_leaf_name_list, all_reduce_data_size)
        all_reduce_comm_pattern.append(all_reduce_tuple)
    return all_reduce_groups, all_reduce_comm_pattern