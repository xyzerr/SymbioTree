# https://leopard-x.memofun.net/c/69437416-1b78-832c-8215-d2807cabdf0f

from hardware.htraversal import *
from hardware.hcase import *

class HardwareTree:
    def __init__(self, case_idx:int):
        self.root = None
        self.devices = None
        self.build_tree_by_case(case_idx)

    def build_tree_by_case(self, case_idx:int):
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
        }
        try:
            build_fn = BUILD_CASES[case_idx]
        except KeyError:
            raise ValueError(f"Case {case_idx} is not supported.")

        self.root, self.devices = build_fn()

        for device in self.devices:
            fill_descendant_set(device)

        # 遍历 / 查找 / 聚合
        # 1) 打印所有叶子节点路径
        # for u in self.root.iter_leaves():
        #     print(u.path(), u.meta)

        # 2) 按路径找到节点
        # node = self.root.find_by_path("System/rack1/Chassis1/Board1/GPU1")
        # print(node.id, node.path())

        # 3) 统计总功耗
        # total_power = self.root.aggregate(lambda u: float(u.meta.get("power_w", 0.0)))
        # print("Total power:", total_power)

        # 打印所有叶子结点
        print("\n---------hardware tree---------")
        for n in self.root.iter_leaves():
            print(n.path())


if __name__ == "__main__":
    htree = HardwareTree(0)
