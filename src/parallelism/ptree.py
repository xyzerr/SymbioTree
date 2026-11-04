from collections import defaultdict
from dataclasses import dataclass, field
import copy
import random

from system.config import SystemConfig, ModelConfig

from serving.request import Request

from parallelism.pcase import *
from parallelism.ptraversal import *
from parallelism.pperf import peer_to_peer_communication_pattern, all_reduce_communication_pattern

from hardware.htree import HardwareTree

from hardware.hperf import peer_to_peer_communication_time_cost, all_reduce_communication_time_cost


@dataclass
class ParallelismRunProfile:
    total_time_ms: float
    per_device_compute_ms: dict[str, float] = field(default_factory=dict)
    per_device_comm_ms: dict[str, float] = field(default_factory=dict)
    per_device_busy_wo_overlap_ms: dict[str, float] = field(default_factory=dict)
    per_device_busy_wi_overlap_ms: dict[str, float] = field(default_factory=dict)

    def scaled(self, factor: float) -> "ParallelismRunProfile":
        if factor == 1.0:
            return ParallelismRunProfile(
                total_time_ms=self.total_time_ms,
                per_device_compute_ms=dict(self.per_device_compute_ms),
                per_device_comm_ms=dict(self.per_device_comm_ms),
                per_device_busy_wo_overlap_ms=dict(self.per_device_busy_wo_overlap_ms),
                per_device_busy_wi_overlap_ms=dict(self.per_device_busy_wi_overlap_ms),
            )
        return ParallelismRunProfile(
            total_time_ms=self.total_time_ms * factor,
            per_device_compute_ms={k: v * factor for k, v in self.per_device_compute_ms.items()},
            per_device_comm_ms={k: v * factor for k, v in self.per_device_comm_ms.items()},
            per_device_busy_wo_overlap_ms={k: v * factor for k, v in self.per_device_busy_wo_overlap_ms.items()},
            per_device_busy_wi_overlap_ms={k: v * factor for k, v in self.per_device_busy_wi_overlap_ms.items()},
        )


class ParallelismTree:

    def __init__(self,
                 sys_cfg: SystemConfig,
                 model_cfg: ModelConfig,
                 htree: HardwareTree,
                 case_idx: int =2):
        self.root_node = None
        self.begin_nodes = None
        self.leaf_nodes = None
        self.mapper = None
        self.sys_cfg = sys_cfg
        self.model_cfg = model_cfg
        self.htree = htree
        self.build_tree_by_case(case_idx)

    def build_tree_by_case(self, case_idx: int):
        BUILD_CASES = {
            0: build_case_0,
            1: build_case_1,
            2: build_case_2,
            3: build_case_3,
            4: build_case_4,
            5: build_case_5,
            6: build_case_6,
            7: build_case_7,
            8: build_case_8,
            9: build_case_9,
            10: build_case_10,
            11: build_case_11,
            12: build_case_12,
            13: build_case_13,
            14: build_case_14,
            15: build_case_15,
            16: build_case_16,
            17: build_case_17,
            18: build_case_18,
            19: build_case_19,
            20: build_case_20,
            21: build_case_21,
            22: build_case_22,
            23: build_case_23,
        }
        try:
            build_fn = BUILD_CASES[case_idx]
        except KeyError:
            raise ValueError(f"Case {case_idx} is not supported.")

        self.root_node, self.leaf_nodes = build_fn(self.sys_cfg.req_type_num, self.model_cfg.layer_num)

        self.begin_nodes = [i for i in detect_begin_nodes(self.root_node)]
        self.hardware_mapping()

        # print("---- leaf node info ----")
        # for begin_node in self.begin_nodes:
        #     leaf_nodes = derive_from_node(begin_node)
        #     for leaf_node in leaf_nodes:
        #         leaf_node.print_info()

    def summarise_layer_info(self, begin_node: BasicHardwareNode):
        leaf_nodes: List[BasicHardwareNode] = derive_from_node(begin_node)
        # for leaf in leaf_nodes:
        #     leaf.print_info()
        total_layer_num = self.model_cfg.layer_num

        # for leaf_node in leaf_nodes:
        #     print(leaf_node.pp_attr)

        # 例子
        # leaf0 pp_attr: [0, 7]
        # leaf1 pp_attr: [4, 15]
        # leaf2 pp_attr: [0, 9]
        # so pp_group_0 = [leaf0,leaf2], layer: [0,4]
        #    pp_group_1 = [leaf0,leaf1,leaf2], layer: [5,7]
        #    pp_group_2 = [leaf1,leaf2], layer: [8,9]
        #    pp_group_3 = [leaf1], layer: [10, 15]
        # 则需要有 4 个 sub batch。
        active_leaf_set = set()
        for layer_idx in range(total_layer_num):
            active_leaf_list = []
            for leaf_node in leaf_nodes:
                if leaf_node.pp_attr[0] <= layer_idx <= leaf_node.pp_attr[1]:
                    active_leaf_list.append(leaf_node.name)
            active_leaf_set.add(frozenset(active_leaf_list))
        sub_batch_num = len(active_leaf_set)
        return sub_batch_num

    def run(self,
            active_queue: List[Request],
            use_pp_sub_batch: bool,
            use_mp_sub_batch: bool):
        req_list_by_type = [[] for _ in range(self.sys_cfg.req_type_num)]
        req_type_distribution = [0] * self.sys_cfg.req_type_num

        req_list_by_type_half1 = [[] for _ in range(self.sys_cfg.req_type_num)]
        req_type_distribution_half1 = [0] * self.sys_cfg.req_type_num
        req_list_by_type_half2 = [[] for _ in range(self.sys_cfg.req_type_num)]
        req_type_distribution_half2 = [0] * self.sys_cfg.req_type_num

        if active_queue is not None:
            # 原始统计
            for r in active_queue:
                req_list_by_type[r.req_type].append(r)
                req_type_distribution[r.req_type] += 1
            # print('req_type_distribution:',req_type_distribution)

            # half 统计
            shuffled_queue = random.sample(active_queue, len(active_queue))
            mid = len(shuffled_queue) // 2
            active_queue_half1 = shuffled_queue[:mid]
            active_queue_half2 = shuffled_queue[mid:]

            # 第一组统计
            for r in active_queue_half1:
                req_list_by_type_half1[r.req_type].append(r)
                req_type_distribution_half1[r.req_type] += 1

            # 第二组统计
            for r in active_queue_half2:
                req_list_by_type_half2[r.req_type].append(r)
                req_type_distribution_half2[r.req_type] += 1

        max_run_time_cost_ms = 0.0
        for begin_node in self.begin_nodes:
            leaf_nodes = derive_from_node(begin_node)
            sub_graph_profile = self.analyse_sub_graph(begin_node,
                                                            leaf_nodes,
                                                            req_list_by_type,
                                                            req_type_distribution,
                                                            use_pp_sub_batch,
                                                            use_mp_sub_batch,
                                                            req_list_by_type_half1,
                                                            req_type_distribution_half1,
                                                            req_list_by_type_half2,
                                                            req_type_distribution_half2)
            max_run_time_cost_ms = max(max_run_time_cost_ms, sub_graph_profile.total_time_ms)
        return max_run_time_cost_ms

    def run_from_begin_node(self,
                            begin_node,
                            active_queue: List[Request],
                            use_pp_sub_batch: bool,
                            use_mp_sub_batch: bool,
                            return_profile: bool = False):
        req_list_by_type = [[] for _ in range(self.sys_cfg.req_type_num)]
        req_type_distribution = [0] * self.sys_cfg.req_type_num

        req_list_by_type_half1 = [[] for _ in range(self.sys_cfg.req_type_num)]
        req_type_distribution_half1 = [0] * self.sys_cfg.req_type_num
        req_list_by_type_half2 = [[] for _ in range(self.sys_cfg.req_type_num)]
        req_type_distribution_half2 = [0] * self.sys_cfg.req_type_num

        if active_queue is not None:
            # 原始统计
            for r in active_queue:
                req_list_by_type[r.req_type].append(r)
                req_type_distribution[r.req_type] += 1

            # half 统计
            shuffled_queue = random.sample(active_queue, len(active_queue))
            mid = len(shuffled_queue) // 2
            active_queue_half1 = shuffled_queue[:mid]
            active_queue_half2 = shuffled_queue[mid:]

            # 第一组统计
            for r in active_queue_half1:
                req_list_by_type_half1[r.req_type].append(r)
                req_type_distribution_half1[r.req_type] += 1

            # 第二组统计
            for r in active_queue_half2:
                req_list_by_type_half2[r.req_type].append(r)
                req_type_distribution_half2[r.req_type] += 1

        leaf_nodes = derive_from_node(begin_node)
        sub_graph_profile = self.analyse_sub_graph(begin_node,
                                                        leaf_nodes,
                                                        req_list_by_type,
                                                        req_type_distribution,
                                                        use_pp_sub_batch,
                                                        use_mp_sub_batch,
                                                        req_list_by_type_half1,
                                                        req_type_distribution_half1,
                                                        req_list_by_type_half2,
                                                        req_type_distribution_half2)
        # print(len(active_queue), sub_graph_profile.total_time_ms)
        return sub_graph_profile if return_profile else sub_graph_profile.total_time_ms


    def analyse_sub_graph(self,
                          root_of_subtree: BasicParallelismNode,
                          leaf_nodes_of_subtree: List[BasicHardwareNode],
                          req_list_by_type: List[List[Request]],
                          req_type_distribution: List[int],
                          use_pp_sub_batch: bool,
                          use_mp_sub_batch: bool,
                          req_list_by_type_half1: List[List[Request]] = None,
                          req_type_distribution_half1: List[int] = None,
                          req_list_by_type_half2: List[List[Request]] = None,
                          req_type_distribution_half2: List[int] = None) -> ParallelismRunProfile:

        for leaf in leaf_nodes_of_subtree:
            assert leaf.is_leaf()
            # print(leaf.name, leaf.pp_attr, leaf.xp_attr)
        previous_module_active_leaf_indexes = []

        # For MP sub batch
        layer_if_use_mp = [False for _ in range(self.model_cfg.layer_num)]

        # 记录每层运行时间
        layer_qkv_time_ms = [0 for _ in range(self.model_cfg.layer_num)]
        layer_attn_time_ms = [0 for _ in range(self.model_cfg.layer_num)]
        layer_proj_time_ms = [0 for _ in range(self.model_cfg.layer_num)]
        layer_ffn_time_ms = [0 for _ in range(self.model_cfg.layer_num)]

        # 记录一半req的时间
        layer_qkv_half_time_ms = [0 for _ in range(self.model_cfg.layer_num)]
        layer_attn_half_time_ms = [0 for _ in range(self.model_cfg.layer_num)]
        layer_proj_half_time_ms = [0 for _ in range(self.model_cfg.layer_num)]
        layer_ffn_half_time_ms = [0 for _ in range(self.model_cfg.layer_num)]

        # 记录每个 layer 都有哪些 device 参与过（不区分模块）
        layer_leaf_info = [[] for _ in range(self.model_cfg.layer_num)]
        for layer_idx in range(self.model_cfg.layer_num):
            for leaf_idx, leaf in enumerate(leaf_nodes_of_subtree):
                if leaf.pp_attr[0] <= layer_idx <= leaf.pp_attr[1]:
                    layer_leaf_info[layer_idx].append(leaf_idx)
                    if leaf.xp_attr is not XpTag.BOTH:
                        layer_if_use_mp[layer_idx] = True

        # print(layer_if_use_mp)
        # print(layer_leaf_info)

        # 记录不使用 PP/MP sub batch 情况下的总运行时间
        original_processing_time_cost_ms = 0.0

        mp_subbatch_processing_time_cost_ms = 0.0

        # 每个 module 都有一个 result cache，用于减少重复的模拟
        result_cache = [dict() for _ in range(4)]
        result_cache_half = [dict() for _ in range(4)]

        # 记录每个 device 上的执行时间，用于 pipeline parallelism
        execution_time_of_leafs_ms = [0 for _ in range(len(leaf_nodes_of_subtree))]

        # 最小改动版：记录每个 device 的原始计算/通信时间，
        # 以及两种 busy 统计（不考虑 PP/MP 调度折叠）
        per_device_compute_ms = defaultdict(float)
        per_device_comm_ms = defaultdict(float)
        per_device_busy_wo_overlap_ms = defaultdict(float)
        per_device_busy_wi_overlap_ms = defaultdict(float)

        for layer_idx in range(self.model_cfg.layer_num):
            for module_idx in range(4):  # QKV / ATTN / PROJ / FFN
                module_active_leaf_indexes = trigger_leaf_node(layer_idx, module_idx, leaf_nodes_of_subtree)
                if frozenset(module_active_leaf_indexes) not in result_cache[module_idx]:
                    computation_time_cost_ms, module_device_compute_ms = self.analyse_computation_pattern(root_of_subtree,
                                                                                                          leaf_nodes_of_subtree,
                                                                                                          module_idx,
                                                                                                          module_active_leaf_indexes,
                                                                                                          req_list_by_type,
                                                                                                          req_type_distribution)
                    communication_time_cost_ms, module_device_comm_ms = self.analyse_communication_pattern(root_of_subtree,
                                                                                                           leaf_nodes_of_subtree,
                                                                                                           layer_idx,
                                                                                                           module_idx,
                                                                                                           previous_module_active_leaf_indexes,
                                                                                                           module_active_leaf_indexes,
                                                                                                           req_type_distribution)
                    computation_half1_time_cost_ms, _ = self.analyse_computation_pattern(root_of_subtree,
                                                                                         leaf_nodes_of_subtree,
                                                                                         module_idx,
                                                                                         module_active_leaf_indexes,
                                                                                         req_list_by_type_half1,
                                                                                         req_type_distribution_half1)
                    communication_half1_time_cost_ms, _ = self.analyse_communication_pattern(root_of_subtree,
                                                                                            leaf_nodes_of_subtree,
                                                                                            layer_idx,
                                                                                            module_idx,
                                                                                            previous_module_active_leaf_indexes,
                                                                                            module_active_leaf_indexes,
                                                                                            req_type_distribution_half1)
                    computation_half2_time_cost_ms, _ = self.analyse_computation_pattern(root_of_subtree,
                                                                                         leaf_nodes_of_subtree,
                                                                                         module_idx,
                                                                                         module_active_leaf_indexes,
                                                                                         req_list_by_type_half2,
                                                                                         req_type_distribution_half2)
                    communication_half2_time_cost_ms, _ = self.analyse_communication_pattern(root_of_subtree,
                                                                                            leaf_nodes_of_subtree,
                                                                                            layer_idx,
                                                                                            module_idx,
                                                                                            previous_module_active_leaf_indexes,
                                                                                            module_active_leaf_indexes,
                                                                                            req_type_distribution_half2)
                    module_time_cost_ms = computation_time_cost_ms + communication_time_cost_ms
                    # module_time_cost_ms = max(computation_time_cost_ms, communication_time_cost_ms)
                    module_half1_time_cost_ms = computation_half1_time_cost_ms + communication_half1_time_cost_ms
                    # module_half1_time_cost_ms = max(computation_half1_time_cost_ms, communication_half1_time_cost_ms)
                    module_half2_time_cost_ms = computation_half2_time_cost_ms + communication_half2_time_cost_ms
                    # module_half2_time_cost_ms = max(computation_half2_time_cost_ms, communication_half2_time_cost_ms)
                    module_half_time_cost_ms = max(module_half1_time_cost_ms, module_half2_time_cost_ms)
                    # print(computation_time_cost_ms, communication_time_cost_ms)
                    # layer_idx = 0 和 module_idx = 0 的情况不能记录进 cache，否则后面的 tp 都没有了
                    if layer_idx != 0 or module_idx != 0:
                        result_cache[module_idx][frozenset(module_active_leaf_indexes)] = (
                            module_time_cost_ms,
                            dict(module_device_compute_ms),
                            dict(module_device_comm_ms),
                        )
                        result_cache_half[module_idx][frozenset(module_active_leaf_indexes)] = module_half_time_cost_ms
                else:
                    module_time_cost_ms, module_device_compute_ms, module_device_comm_ms = result_cache[module_idx][frozenset(module_active_leaf_indexes)]
                    module_half_time_cost_ms = result_cache_half[module_idx][frozenset(module_active_leaf_indexes)]

                for device_name, value in module_device_compute_ms.items():
                    per_device_compute_ms[device_name] += value
                for device_name, value in module_device_comm_ms.items():
                    per_device_comm_ms[device_name] += value

                module_device_names = set(module_device_compute_ms.keys()) | set(module_device_comm_ms.keys())
                for device_name in module_device_names:
                    compute_ms = module_device_compute_ms.get(device_name, 0.0)
                    comm_ms = module_device_comm_ms.get(device_name, 0.0)
                    per_device_busy_wo_overlap_ms[device_name] += compute_ms + comm_ms
                    per_device_busy_wi_overlap_ms[device_name] += max(compute_ms, comm_ms)

                if module_idx == 0:
                    layer_qkv_time_ms[layer_idx] = module_time_cost_ms
                    layer_qkv_half_time_ms[layer_idx] = module_half_time_cost_ms
                elif module_idx == 1:
                    layer_attn_time_ms[layer_idx] = module_time_cost_ms
                    layer_attn_half_time_ms[layer_idx] = module_half_time_cost_ms
                elif module_idx == 2:
                    layer_proj_time_ms[layer_idx] = module_time_cost_ms
                    layer_proj_half_time_ms[layer_idx] = module_half_time_cost_ms
                elif module_idx == 3:
                    layer_ffn_time_ms[layer_idx] = module_time_cost_ms
                    layer_ffn_half_time_ms[layer_idx] = module_half_time_cost_ms

                previous_module_active_leaf_indexes = module_active_leaf_indexes

            # layer_time_cost_ms 这是为了 PP sub batch 设计的。
            # if use_pp_sub_batch:
            #     for leaf_idx in layer_leaf_info[layer_idx]:
            #         execution_time_of_leafs_ms[leaf_idx] += layer_qkv_time_ms[layer_idx] + \
            #                                                 layer_attn_time_ms[layer_idx] + \
            #                                                 layer_proj_time_ms[layer_idx] + \
            #                                                 layer_ffn_time_ms[layer_idx]


        # execution_time_of_leafs_ms 在有 mp 的情况下要记录使用 mp double sub batch 优化的结果。
        # execution_time_of_leafs_ms 在没有 mp 的情况下要记录 original 结果。
        # 给定一个使用 MP Double Sub Batch 的例子：
        # layer0: leaf0, leaf1
        # layer1: leaf0, leaf1
        # layer2: leaf0, leaf1
        # layer3: leaf1, leaf3
        # layer4: leaf1, leaf3
        # layer5: leaf2, leaf3
        # layer6: leaf2, leaf3
        # layer7: leaf2, leaf3
        # layer8: leaf2, leaf3
        # 对于每个 layer 进行遍历，当 previous leaf list 不等于 current 时，说明
        previous_leaf_list = []
        for layer_idx in range(self.model_cfg.layer_num):
            # 原始情况，不考虑 PP 或 MP，直接累加即可
            current_leaf_list = layer_leaf_info[layer_idx]
            full_layer_time_ms = (layer_qkv_time_ms[layer_idx] +
                                  layer_attn_time_ms[layer_idx] +
                                  layer_proj_time_ms[layer_idx] +
                                  layer_ffn_time_ms[layer_idx])
            full_layer_half_time_ms = (layer_qkv_half_time_ms[layer_idx] +
                                       layer_attn_half_time_ms[layer_idx] +
                                       layer_proj_half_time_ms[layer_idx] +
                                       layer_ffn_half_time_ms[layer_idx])
            original_processing_time_cost_ms += full_layer_time_ms
            # 当不考虑 MP 或者该层没有 MP 参与时，按照 original 形式
            if not use_mp_sub_batch or not layer_if_use_mp[layer_idx]:
                if use_pp_sub_batch:
                    for leaf_idx in current_leaf_list:
                        # execution_time_of_leafs_ms[leaf_idx] += full_layer_time_ms
                        execution_time_of_leafs_ms[leaf_idx] += full_layer_half_time_ms
                mp_subbatch_processing_time_cost_ms += full_layer_time_ms
                previous_leaf_list = []
            # 考虑 MP 且该层有 MP 参与，当前层不等于前层，说明遇到新的 double sub batch
            elif current_leaf_list != previous_leaf_list:
                # mp_fix_factor = 1.9
                # first_double_sub_batch_time_ms = (layer_qkv_time_ms[layer_idx] +
                #                                   max(layer_qkv_time_ms[layer_idx], layer_attn_time_ms[layer_idx]) +
                #                                   max(layer_proj_time_ms[layer_idx] + layer_ffn_time_ms[layer_idx], layer_attn_time_ms[layer_idx]) +
                #                                   layer_proj_time_ms[layer_idx] + layer_ffn_time_ms[layer_idx]) / mp_fix_factor
                first_double_sub_batch_time_ms = (layer_qkv_half_time_ms[layer_idx] +
                                                  max(layer_qkv_half_time_ms[layer_idx], layer_attn_half_time_ms[layer_idx]) +
                                                  max(layer_proj_half_time_ms[layer_idx] + layer_ffn_half_time_ms[layer_idx], layer_attn_half_time_ms[layer_idx]) +
                                                  layer_proj_half_time_ms[layer_idx] + layer_ffn_half_time_ms[layer_idx])
                if use_pp_sub_batch:
                    for leaf_idx in current_leaf_list:
                        execution_time_of_leafs_ms[leaf_idx] += first_double_sub_batch_time_ms
                mp_subbatch_processing_time_cost_ms += first_double_sub_batch_time_ms
                previous_leaf_list = current_leaf_list
            # 考虑 MP 且该层有 MP 参与，当前层等于前层，说明未遇到新的 double sub batch
            else: # when current_leaf_list == previous_leaf_list:
                # mp_fix_factor = 1.9
                # next_double_sub_batch_time_ms = 2 * max(layer_qkv_time_ms[layer_idx] + layer_proj_time_ms[layer_idx] + layer_ffn_time_ms[layer_idx],
                #                                         layer_attn_time_ms[layer_idx]) / mp_fix_factor
                next_double_sub_batch_time_ms = 2 * max(layer_qkv_half_time_ms[layer_idx] +
                                                        layer_proj_half_time_ms[layer_idx] +
                                                        layer_ffn_half_time_ms[layer_idx],
                                                        layer_attn_half_time_ms[layer_idx])
                if use_pp_sub_batch:
                    for leaf_idx in current_leaf_list:
                        execution_time_of_leafs_ms[leaf_idx] += next_double_sub_batch_time_ms
                mp_subbatch_processing_time_cost_ms += next_double_sub_batch_time_ms
                previous_leaf_list = current_leaf_list

        total_time_ms = max(execution_time_of_leafs_ms) if use_pp_sub_batch else \
                mp_subbatch_processing_time_cost_ms if use_mp_sub_batch else \
                original_processing_time_cost_ms

        return ParallelismRunProfile(
            total_time_ms=total_time_ms,
            per_device_compute_ms=dict(per_device_compute_ms),
            per_device_comm_ms=dict(per_device_comm_ms),
            per_device_busy_wo_overlap_ms=dict(per_device_busy_wo_overlap_ms),
            per_device_busy_wi_overlap_ms=dict(per_device_busy_wi_overlap_ms),
        )

    def analyse_computation_pattern(self,
                                    root_of_subtree: BasicParallelismNode,
                                    leaf_of_subtree: List[BasicHardwareNode],
                                    module_idx: int,
                                    active_leaf_indexes: List[int],
                                    req_list_by_type: List[List[Request]],
                                    req_type_distribution: List[int],
                                    print_detail = False):
        max_computation_time_ms = 0.0
        per_device_compute_ms = defaultdict(float)

        if module_idx == 0:
            if print_detail:
                print("------ Compute QKV ------")
            for active_leaf_idx in active_leaf_indexes:
                active_leaf = leaf_of_subtree[active_leaf_idx]
                # [M,K] * [K,N] = [M,N]
                M_size = sum([(d[1]-d[0])*n for d,n in zip(active_leaf.dp_attr, req_type_distribution)]) # batch size
                # M_size = sum([1*n for d,n in zip(active_leaf.dp_attr, req_type_distribution)]) # batch size
                K_size = self.model_cfg.hidden_size # input dimension
                N_size = (active_leaf.tp_attr[1] - active_leaf.tp_attr[0]) * (self.model_cfg.hidden_size + 2 * self.model_cfg.kv_hidden_size) # output dimension
                if M_size > 0 and K_size > 0 and N_size > 0:
                    computation_time_ms = self.mapper[active_leaf.name].compute_gemm_time_cost(M_size, N_size, K_size)
                    per_device_compute_ms[self.mapper[active_leaf.name].name] += computation_time_ms
                    max_computation_time_ms = max(max_computation_time_ms, computation_time_ms)
        elif module_idx == 1:
            if print_detail:
                print("------ Compute ATTN ------")
            for active_leaf_idx in active_leaf_indexes:
                active_leaf = leaf_of_subtree[active_leaf_idx]
                query_head_num = ceil(self.model_cfg.query_head_num * (active_leaf.tp_attr[1] - active_leaf.tp_attr[0]))
                kv_head_num = ceil(self.model_cfg.kv_head_num * (active_leaf.tp_attr[1] - active_leaf.tp_attr[0]))
                computation_time_ms = 0.0
                # gqa_ratio = self.model_cfg.query_head_num // self.model_cfg.kv_head_num
                mem_ops = 0
                comp_ops = 0
                for d,n,r in zip(active_leaf.dp_attr, req_type_distribution, req_list_by_type):
                    start_req_idx = floor(d[0] * n)
                    end_req_idx = ceil(d[1] * n)
                    for req_idx in range(start_req_idx, end_req_idx):
                        if req_idx >= len(r):
                            continue
                        sequence_length = r[req_idx].prompt_tokens + r[req_idx].gen_tokens
                        # Q * K^T = S
                        mem_ops += (query_head_num * self.model_cfg.head_dim +
                            kv_head_num * self.model_cfg.head_dim * sequence_length +
                            query_head_num * sequence_length) * 2 # 考虑 float16 类型
                        comp_ops += query_head_num * self.model_cfg.head_dim * sequence_length * 2
                        # S * V = C
                        mem_ops += (query_head_num * self.model_cfg.head_dim +
                            kv_head_num * self.model_cfg.head_dim * sequence_length +
                            query_head_num * sequence_length) * 2 # 考虑 float16 类型
                        comp_ops += query_head_num * self.model_cfg.head_dim * sequence_length * 2
                if mem_ops > 0:
                    computation_time_ms += self.mapper[active_leaf.name].compute_gemm_time_cost_by_ops(comp_ops, mem_ops)
                    per_device_compute_ms[self.mapper[active_leaf.name].name] += computation_time_ms

                max_computation_time_ms = max(max_computation_time_ms, computation_time_ms)

        elif module_idx == 2:
            if print_detail:
                print("------ Compute PROJ ------")
            for active_leaf_idx in active_leaf_indexes:
                active_leaf = leaf_of_subtree[active_leaf_idx]
                # [M,K] * [K,N] = [M,N]
                M_size = sum([(d[1]-d[0])*n for d,n in zip(active_leaf.dp_attr, req_type_distribution)])
                # M_size = sum([1*n for d,n in zip(active_leaf.dp_attr, req_type_distribution)])
                K_size =  (active_leaf.tp_attr[1] - active_leaf.tp_attr[0]) * self.model_cfg.hidden_size
                N_size = self.model_cfg.hidden_size
                if M_size > 0 and K_size > 0 and N_size > 0:
                    computation_time_ms = self.mapper[active_leaf.name].compute_gemm_time_cost(M_size, N_size, K_size)
                    per_device_compute_ms[self.mapper[active_leaf.name].name] += computation_time_ms
                    max_computation_time_ms = max(max_computation_time_ms, computation_time_ms)

        else:
            if print_detail:
                print("------ Compute FFN ------")
            for active_leaf_idx in active_leaf_indexes:
                active_leaf = leaf_of_subtree[active_leaf_idx]
                # 1.[M,K] * [K,N] = [M,N]
                # 2.[M,N] * [N,P] = [M,P]
                M_size = sum([(d[1]-d[0])*n for d,n in zip(active_leaf.dp_attr, req_type_distribution)])
                # M_size = sum([1*n for d,n in zip(active_leaf.dp_attr, req_type_distribution)])
                K_size = self.model_cfg.hidden_size
                N_size = (active_leaf.tp_attr[1] - active_leaf.tp_attr[0]) * self.model_cfg.intermediate_size
                P_size = self.model_cfg.hidden_size
                if M_size > 0 and K_size > 0 and N_size > 0 and P_size > 0:
                    computation_time_ms = self.mapper[active_leaf.name].compute_gemm_time_cost(M_size, N_size, K_size) + \
                                          self.mapper[active_leaf.name].compute_gemm_time_cost(M_size, P_size, N_size)
                    per_device_compute_ms[self.mapper[active_leaf.name].name] += computation_time_ms
                    max_computation_time_ms = max(max_computation_time_ms, computation_time_ms)

        return max_computation_time_ms, dict(per_device_compute_ms)

    def analyse_communication_pattern(self,
                                      root_of_subtree: BasicParallelismNode,
                                      leaf_of_subtree: List[BasicHardwareNode],
                                      layer_idx: int, module_idx: int,
                                      src_leaf_indexes: List[int],
                                      dst_leaf_indexes: List[int],
                                      req_type_distribution: List[int],
                                      print_detail = False):
        max_comm_time_ms = 0.0
        per_device_comm_ms = defaultdict(float)

        # if layer_idx == 0 and module_idx == 0:
        #     if print_detail:
        #         print("------ Broadcast to QKV")

        if module_idx == 0:  # FFN -> QKV
            if print_detail:
                print("------ FFN -> QKV")
            # 可能启动的通信有 D、P、T
            # 先进行发送端的 AllReduce，然后再进行其他操作。总体上就是后面两个操作的联合体。
            # 1. 如果 src_leaf_indexes == dst_leaf_indexes，则没有启动 P 节点，只可能有 D 和 T。
            # 2. 如果 src_leaf_indexes != dst_leaf_indexes，则一定启动了 P 节点，可能有 D 和 T。
            all_reduce_groups, all_reduce_comm_pattern = all_reduce_communication_pattern(leaf_of_subtree, src_leaf_indexes,
                                                         req_type_distribution, self.model_cfg.hidden_size)
            # all reduce 开销
            max_all_reduce_comm_time_ms = 0.0
            for all_reduce_tuple in all_reduce_comm_pattern: # 每个 all reduce group 具有一个 tuple
                all_reduce_leaf_name_list, all_reduce_data_size = all_reduce_tuple
                all_reduce_hardware_list = [self.mapper[name] for name in all_reduce_leaf_name_list]
                all_reduce_comm_time_ms = all_reduce_communication_time_cost(all_reduce_hardware_list, all_reduce_data_size)
                max_all_reduce_comm_time_ms += all_reduce_comm_time_ms
                for hardware in all_reduce_hardware_list:
                    per_device_comm_ms[hardware.name] += all_reduce_comm_time_ms

            # peer to peer 开销
            if src_leaf_indexes != dst_leaf_indexes:
                # print_src_and_dst_info(leaf_nodes, src_leaf_indexes, dst_leaf_indexes)
                for all_reduce_group in all_reduce_groups.values():
                    comm_time_ms = 0.0
                    assert all_reduce_group is not None
                    tmp_src_node_idx = all_reduce_group[0]
                    tmp_src_node = copy.deepcopy(leaf_of_subtree[tmp_src_node_idx])
                    tmp_src_node.tp_attr = [0, 1.0]

                    for dst in dst_leaf_indexes:
                        dst_node = leaf_of_subtree[dst]
                        comm_size_byte = 2 * peer_to_peer_communication_pattern(tmp_src_node,
                                                               dst_node,
                                                               req_type_distribution,
                                                               self.model_cfg.hidden_size)
                        peer_comm_time_ms = peer_to_peer_communication_time_cost(self.mapper[tmp_src_node.name],
                                                                                 self.mapper[dst_node.name],
                                                                                 comm_size_byte)
                        comm_time_ms += peer_comm_time_ms
                        per_device_comm_ms[self.mapper[tmp_src_node.name].name] += peer_comm_time_ms
                        per_device_comm_ms[self.mapper[dst_node.name].name] += peer_comm_time_ms
                    max_comm_time_ms = max(max_comm_time_ms, comm_time_ms)

            max_comm_time_ms += max_all_reduce_comm_time_ms

        elif module_idx == 1 or module_idx == 2:  # QKV -> ATTN and ATTN -> PROJ
            if print_detail:
                if module_idx == 1:
                    print("------ QKV -> ATTN")
                elif module_idx == 2:
                    print("------ ATTN -> PROJ")

            if src_leaf_indexes == dst_leaf_indexes:
                # 未启动 X，则 src == dst；反之也成立，若 src == dst，则未启动 X。
                pass
            else:
                # 启动 X，则 src != dst，需要进行通信，将 src 节点的数据传递到 dst 节点
                # X 节点将 Graph 划分为 Fan-In 路径和 Fan-Out 路径，所以整个通信路径是 Fan-In -> X -> Fan-Out
                # Fan-In 和 Fan-Out 路径上可能会有D、P、T。根据规定，Fan-In 和 Fan-Out 的路径都遵循 D-P-T 的顺序。
                #   Fan-In 和 Fan-Out 路径上的 P 不发挥作用，可不用分析。
                #   因此通信模式可以直接通过 tp_attr 和 dp_attr 来分析得到。

                # print_src_and_dst_info(leaf_nodes, src_leaf_indexes, dst_leaf_indexes)
                for src_idx in src_leaf_indexes:
                    comm_time_ms = 0.0
                    for dst_idx in dst_leaf_indexes:
                        src_node = leaf_of_subtree[src_idx]
                        dst_node = leaf_of_subtree[dst_idx]
                        comm_size_byte = 2 * peer_to_peer_communication_pattern(src_node,
                                                                                  dst_node,
                                                                                  req_type_distribution,
                                                                                  self.model_cfg.hidden_size)
                        peer_comm_time_ms = peer_to_peer_communication_time_cost(self.mapper[src_node.name],
                                                                                 self.mapper[dst_node.name],
                                                                                 comm_size_byte)
                        comm_time_ms += peer_comm_time_ms
                        per_device_comm_ms[self.mapper[src_node.name].name] += peer_comm_time_ms
                        per_device_comm_ms[self.mapper[dst_node.name].name] += peer_comm_time_ms
                    max_comm_time_ms = max(comm_time_ms, max_comm_time_ms)

        elif module_idx == 3:  # PROJ -> FFN
            if print_detail:
                print("------ PROJ -> FFN")
            if src_leaf_indexes == dst_leaf_indexes:  # 启动 AllReduce
                # Active Graph 中肯定有 T，可能有 D，不可能有 P（有 P 的话不会再这个阶段进行通信）。
                # 总体思路就是向上回溯到 level 最高的 TP-node。具有同一个最高 level TP-node 的节点们会进行 All-Reduce。
                _, all_reduce_comm_pattern = all_reduce_communication_pattern(leaf_of_subtree, src_leaf_indexes,
                                                             req_type_distribution, self.model_cfg.hidden_size)
                # all reduce 开销（同前面）
                max_all_reduce_comm_time_ms = 0.0
                for all_reduce_tuple in all_reduce_comm_pattern:
                    all_reduce_leaf_name_list, all_reduce_data_size = all_reduce_tuple
                    all_reduce_hardware_list = [self.mapper[name] for name in all_reduce_leaf_name_list]
                    all_reduce_comm_time_ms = all_reduce_communication_time_cost(all_reduce_hardware_list,
                                                                                 all_reduce_data_size)
                    max_all_reduce_comm_time_ms += all_reduce_comm_time_ms
                    for hardware in all_reduce_hardware_list:
                        per_device_comm_ms[hardware.name] += all_reduce_comm_time_ms
                    # max_all_reduce_comm_time_ms = max(max_all_reduce_comm_time_ms, all_reduce_comm_time_ms)
                max_comm_time_ms = max(max_comm_time_ms, max_all_reduce_comm_time_ms)
            else:
                raise ValueError
        return max_comm_time_ms, dict(per_device_comm_ms)

    def hardware_mapping(self):
        # 将 parallel tree 中的每个 leaf 对应到 hardware 中的 leaf
        # self.mapper's key 是 parallel leaf's name
        # self.mapper's value 就是 device
        print("\n---------hardware mapping---------")
        self.mapper = dict()
        for leaf, device in zip(self.leaf_nodes, self.htree.devices):
            print(leaf.name, device.name)
            self.mapper[leaf.name] = device
        for leaf,device in self.mapper.items():
            print("{} : {}".format(leaf, device))
        print("----------------------------------\n")