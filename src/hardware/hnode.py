
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Iterable, Union, Callable, Any


@dataclass
class HwNode:
    """树节点基类：既可做Group，也可做Unit"""
    idx: int
    name: str
    meta: Dict[str, Any] = field(default_factory=dict)
    parent: Optional["HwGroup"] = field(default=None, repr=False, compare=False)

    def is_leaf(self) -> bool:
        raise NotImplementedError

    def path(self) -> str:
        parts = []
        cur: Optional[HwNode] = self
        while cur is not None:
            parts.append(cur.name)
            cur = cur.parent
        return "/".join(reversed(parts))


@dataclass
class HwGroup(HwNode):
    """非叶子：局部组织（机箱/板卡/簇/分区/子系统…）"""
    children: List[HwNode] = field(default_factory=list)
    descendant_set: Set[str] = field(default_factory=set)

    def is_leaf(self) -> bool:
        return False

    def add(self, child: HwNode) -> None:
        if child.parent is not None:
            raise ValueError(f"child already has parent: {child.parent.name}")
        child.parent = self
        self.children.append(child)

    def remove(self, child_name: str) -> HwNode:
        for i, c in enumerate(self.children):
            if c.name == child_name:
                c.parent = None
                return self.children.pop(i)
        raise KeyError(child_name)

    def iter_nodes(self) -> Iterable[HwNode]:
        """DFS 遍历（包含自己）"""
        yield self
        for c in self.children:
            if isinstance(c, HwGroup):
                yield from c.iter_nodes()
            else:
                yield c

    def iter_leaves(self) -> Iterable[HwUnit]:
        for n in self.iter_nodes():
            if isinstance(n, HwUnit):
                yield n

    def aggregate(self, f: Callable[[HwUnit], float]) -> float:
        """对所有叶子做聚合（例如功耗、算力、成本）"""
        return sum(f(u) for u in self.iter_leaves())

    def to_dict(self) -> Dict[str, Any]:
        """导出为 dict（可 JSON 化）"""
        def node_to_dict(n: HwNode) -> Dict[str, Any]:
            base = {"name": n.name, "meta": n.meta, "type": "unit" if n.is_leaf() else "group"}
            if isinstance(n, HwGroup):
                base["children"] = [node_to_dict(c) for c in n.children]
            return base
        return node_to_dict(self)

    def peer_to_peer_communication(self, comm_byte_size: int):
        assert "bw" in self.meta.keys() and "lat" in self.meta.keys()
        return comm_byte_size / self.meta["bw"] / pow(10,6) + self.meta["lat"] / pow(10, 6) # ms


@dataclass(frozen=True)
class RooflineModel:
    peak_flops: float  # TFLOP/s
    peak_bw: float     # TB/s

    def ai_knee(self) -> float:
        """Arithmetic intensity at the knee point (FLOPs/Byte)."""
        return self.peak_flops / self.peak_bw # 单位是 FLOPs/Byte

    def performance(self, flops_per_byte: float) -> float:
        """
        flops_per_byte: Arithmetic intensity (FLOPs/Byte), scalar float.
        returns: attainable performance in same unit as peak_flops.
        """
        mem_bound = flops_per_byte * self.peak_bw
        return self.peak_flops if mem_bound >= self.peak_flops else mem_bound

    def bound_type(self, flops_per_byte: float) -> str:
        mem_bound = flops_per_byte * self.peak_bw
        return "memory-bound" if mem_bound < self.peak_flops else "compute-bound"

@dataclass
class HwUnit(HwNode):
    """叶子：硬件单元"""

    def __post_init__(self):
        assert self.meta.get("flops") # 单位是 TFLOP/s
        assert self.meta.get("bw")    # 单位是 TB/s
        assert self.meta.get("byte")    # 单位是 TB/s
        self.roofline_model = RooflineModel(self.meta.get("flops"), self.meta.get("bw"))

    def is_leaf(self) -> bool:
        return True

    # def compute_gemm_time_cost(self, M:int, N:int, K:int) -> float:
    #     arithmetic_intensity = (2*M*K*N) / (M*K + M*N + K*N) / self.meta.get("byte")
    #     performance = self.roofline_model.performance(arithmetic_intensity) # TFLOP/s
    #     comp_ops = 2*M*N*K
    #     return comp_ops / (performance*pow(10,12)) * pow(10,3) # ms

    def compute_gemm_time_cost(self, M:int, N:int, K:int) -> float:
        mem_ops = (M*K + M*N + K*N) * self.meta.get("byte")
        comp_ops = 2 * M * N * K
        return self.compute_gemm_time_cost_by_ops(comp_ops, mem_ops)


    def compute_gemm_time_cost_by_ops(self, comp_ops:int, mem_ops:int) -> float:
        arithmetic_intensity = comp_ops / mem_ops
        comp_perf = self.roofline_model.performance(arithmetic_intensity)
        mem_bw = self.meta.get("bw")
        return max(comp_ops / (comp_perf*pow(10,12)) * pow(10,3),
                   mem_ops / (mem_bw*pow(10,12)) * pow(10,3))

if __name__ == "__main__":
    npu_test = HwUnit(idx=0, name="npu", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # pim_test = HwUnit(idx=0, name="pim", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    pim_test = HwUnit(idx=0, name="pim", meta={"type": "PIM", "flops": 128, "bw": 16, "byte": 2})
    print(npu_test.compute_gemm_time_cost(256, 2048, 2048))
    print(pim_test.compute_gemm_time_cost(256, 2048, 2048))
