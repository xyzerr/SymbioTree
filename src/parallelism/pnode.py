from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Union, Callable, Any, Protocol
from enum import Enum
import math

class Parallelism(Enum):
    NONE = 0
    DP = 1
    PP = 2
    TP = 3
    XP = 4

class XpTag(Enum):
    BOTH = 0
    LINEAR = 1
    ATTENTION = 2


@dataclass
class BasicNode:
    name: str = ""
    value: int = 0
    type: Optional[Parallelism] = None
    parent: Optional[BasicNode] = None
    children: List[BasicNode] = field(default_factory=list)

    # dp 属性：每种类型的 req 都有一个区间，代表其划分比例
    dp_attr: List[List[float]]= field(default_factory=list)
    # pp 属性：需要记录 begin 和 end 的 layer index
    pp_attr: List[int] = field(default_factory=list)
    # tp 属性：记录权重的划分比例
    tp_attr: List[float] = field(default_factory=list)
    # xp 属性：记录该算子的 tag
    xp_attr: XpTag = XpTag.BOTH

    def is_leaf(self) -> bool:
        raise NotImplementedError

    def add_child(self, child: BasicNode):
        child.parent = self
        self.children.append(child)
        return child

    def print_info(self):
        print("name:", self.name)
        print("DP:", self.dp_attr)
        print("PP:", self.pp_attr)
        print("TP:", self.tp_attr)
        print("XP:", self.xp_attr)
        print("----------\n")

@dataclass
class BasicParallelismNode(BasicNode):
    type: Parallelism = Parallelism.NONE

    def is_leaf(self) -> bool:
        return False

    def derive_child_info(self, child: BasicNode, child_idx: int):
        raise NotImplementedError

@dataclass
class DataParallelismNode(BasicParallelismNode):
    type: Parallelism = Parallelism.DP
    parallel_attr: List[List[float]] = field(default_factory=list)

    def split_weight_into_segments(self, req_idx):
        start, end = self.dp_attr[req_idx]
        length = end - start
        total_ratio = sum(self.parallel_attr[req_idx])

        segments = []
        current_point = start

        for ratio in self.parallel_attr[req_idx]:
            next_point = current_point + length * ratio / total_ratio
            segments.append([current_point, next_point])
            current_point = next_point

        return segments

    def derive_child_info(self, child: BasicNode, child_idx: int):
        child.dp_attr = [self.split_weight_into_segments(req_idx)[child_idx] for req_idx in range(len(self.dp_attr))]
        child.pp_attr = self.pp_attr
        child.tp_attr = self.tp_attr
        child.xp_attr = self.xp_attr

@dataclass
class PipelineParallelismNode(BasicParallelismNode):
    type: Parallelism = Parallelism.PP
    parallel_attr: List[float] = field(default_factory=list)

    def split_layers_into_segments(self):
        # 根据 ratio 把 [begin, end] 进行切分，采用最大余数法确保没有遗漏
        begin, end = self.pp_attr
        total = end - begin + 1
        n = len(self.parallel_attr)
        s = sum(self.parallel_attr)
        # 归一化比例
        ratios = [r / s for r in self.parallel_attr]
        # 理想数量
        ideals = [total * r for r in ratios]
        floors = [math.floor(v) for v in ideals]
        remainders = [v - f for v, f in zip(ideals, floors)]

        counts = floors[:]
        leftover = total - sum(counts)

        # 按小数部分从大到小补齐
        order = sorted(range(n), key=lambda i: remainders[i], reverse=True)
        for i in range(leftover):
            counts[order[i % n]] += 1

        # 保证每段非空（在 total >= n 的前提下）
        # 当某个 ratio 太小、在整数切分后变成 0 时，强制从“最多的那一段”里挪 1 个出来，以保证每一段至少有 1 个元素。
        if total < n:
            raise ValueError("range is too small to make all segments non-empty")

        zeros = [i for i, c in enumerate(counts) if c == 0]
        while zeros:
            z = zeros.pop()
            donor = max(range(n), key=lambda i: counts[i])
            counts[donor] -= 1
            counts[z] += 1

        # 转换为 [begin, end] 形式
        segments = []
        cur = begin
        for c in counts:
            seg_begin = cur
            seg_end = cur + c - 1
            segments.append([seg_begin, seg_end])
            cur = seg_end + 1

        return segments

    def derive_child_info(self, child: BasicNode, child_idx: int):
        child.dp_attr = self.dp_attr
        child.pp_attr = self.split_layers_into_segments()[child_idx]
        child.tp_attr = self.tp_attr
        child.xp_attr = self.xp_attr

@dataclass
class TensorParallelismNode(BasicParallelismNode):
    type: Parallelism = Parallelism.TP
    parallel_attr: List[float] = field(default_factory=list)

    def split_weight_into_segments(self):
        start, end = self.tp_attr
        length = end - start
        total_ratio = sum(self.parallel_attr)

        segments = []
        current_point = start

        for ratio in self.parallel_attr:
            next_point = current_point + length * ratio / total_ratio
            segments.append([current_point, next_point])
            current_point = next_point

        return segments

    def derive_child_info(self, child: BasicNode, child_idx: int):
        child.dp_attr = self.dp_attr
        child.pp_attr = self.pp_attr
        child.tp_attr = self.split_weight_into_segments()[child_idx]
        child.xp_attr = self.xp_attr

@dataclass
class ModuleParallelismNode(BasicParallelismNode):
    type: Parallelism = Parallelism.XP
    parallel_attr: List[XpTag] = field(default_factory=list)

    def derive_child_info(self, child: BasicNode, child_idx: int):
        child.dp_attr = self.dp_attr
        child.pp_attr = self.pp_attr
        child.tp_attr = self.tp_attr
        child.xp_attr = self.parallel_attr[child_idx]

@dataclass
class BasicHardwareNode(BasicNode):
    idx: int = 0

    def is_leaf(self) -> bool:
        return True
