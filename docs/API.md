# 在 Python 中使用 SymbioTree

[English](API_en.md) · [返回README](../README_zh.md) · [配置说明](CONFIGURATION.md)

以下示例展示如何在自己的Python程序中评估部署、搜索方案，以及调用HHT/DDT和映射模块。第一节设置导入路径、模型、硬件和服务配置，后两节沿用这些设置。

如果只是定义新的OP/RP/BP/TP组合、设备分配或路由比例，可直接使用[树形部署配置](CONFIGURATION.md)，无需编写Python构造函数。`symbiotree.py inspect` 和 `validate` 分别用于查看与检查配置。

## 评估固定部署

在仓库根目录运行以下代码：

```python
import sys
from pathlib import Path

project = Path.cwd()
sys.path.insert(0, str(project))
sys.path.insert(0, str(project / "src"))

from hardware import hcase, hperf

# 必须在构造 HardwareTree 或 System 前设置硬件参数。
hcase.NPU_FLOPS, hcase.NPU_BW, hcase.NPU_MEM_CAP = 300, 1.5, 80
hcase.PIM_FLOPS, hcase.PIM_BW, hcase.PIM_MEM_CAP = 16, 16, 80
hcase.nvl_bw, hcase.nvl_lat = 350, 8000
hcase.mm_bw, hcase.mm_lat = 1500, 10
hperf.nvl_bw, hperf.nvl_lat = hcase.nvl_bw, hcase.nvl_lat

from system.config import ModelConfig, SystemConfig
from system.system import System
from system.metrics import summarize_metrics_data

model = ModelConfig(
    model_name="my-dense-model", hidden_size=4096,
    query_head_num=32, kv_head_num=8, intermediate_size=14336, layer_num=32,
)
serving = SystemConfig(
    hcase_index=9,  # 节点内硬件组织，对应 intra_node
    pcase_index=16,  # GPU/PIM协作且两侧采用TP8，对应 gpu_pim_tp8
    req_type_num=3, req_dist=[1.0, 0.0, 0.0],
    lam=10, t_end=10, max_batch_lo=32,
    peak_seq_len=2048, priority_ratio=0, seed=42,
)
simulation = System(serving, model)
results = simulation.run_system()
throughput, p99_tpot, finished, processed, request_rate = summarize_metrics_data(
    results, serving.t_end
)
print(throughput, p99_tpot, request_rate)
```

硬件配置保存在模块级变量中。并行运行不同配置时，应使用不同进程，不要在多个线程中修改共享的硬件全局变量。命令行入口每个进程运行一份配置。

## 搜索与保存部署

接在第一节的代码后运行。示例使用8个个体、1代搜索，并保存最终前沿上的部署；可修改 `population_size`、`generations` 和 `batch_size_choices` 调整搜索规模。

```python
from datetime import datetime
from hardware.htree import HardwareTree
from exploration.decoder import RootInit
from exploration.evolution_pareto import InitConfig, EvoConfig, evolve
from exploration.feasibility import FeasibilityConfig
from exploration.fitness_adapter import make_fitness_fn
from exploration.ind_io import save_individual_json

hardware = HardwareTree(serving.hcase_index)
scope = RootInit([[0.0, 1.0]] * serving.req_type_num,
                 [0, model.layer_num - 1], [0.0, 1.0])
capacity = FeasibilityConfig(
    model, scope,
    {d.idx: d.meta["mem_cap"] for d in hardware.devices},
    {d.idx: d.meta["byte"] for d in hardware.devices},
    peak_seq_len=serving.peak_seq_len,
    runtime_reserve_ratio=serving.runtime_reserve_ratio,
)
fitness = make_fitness_fn(
    serving, model, req_prob=serving.req_dist,
    hcase_idx=serving.hcase_index, pcase_idx_for_init=serving.pcase_index,
)
_, population = evolve(
    InitConfig(population_size=8, max_depth=3, batch_size_choices=(1, 4, 16, 32)),
    EvoConfig(generations=1, elite_size=2, p_mapping_refine_mut=0.0),
    req_type_num=serving.req_type_num,
    devices=[d.idx for d in hardware.devices], root_init=scope,
    device_type_by_id={d.idx: d.meta["type"] for d in hardware.devices},
    fitness_fn=fitness, feasibility_cfg=capacity, random_seed=42,
)
output = project / "outputs" / datetime.now().strftime("python_search_%Y%m%d_%H%M%S_%f")
output.mkdir(parents=True)
front = [candidate for candidate in population if candidate.pareto_rank == 0]
for i, candidate in enumerate(front):
    path = output / f"point_{i:03d}.json"
    save_individual_json(candidate, path)
    print(path, candidate.throughput, candidate.latency)
```

输出依次为部署文件路径、吞吐量（token/s）和P99 TPOT（ms）。`evolve`、`InitConfig` 和 `EvoConfig` 定义在 [src/exploration/evolution_pareto.py](../src/exploration/evolution_pareto.py)。

## 单独调用HHT、DDT与映射

以下示例沿用第一节的配置，可单独构建和解析部署表示，无需先运行搜索。它构建HHT，在Python字典中定义“GPU执行线性计算、PIM执行注意力”的DDT，并读取任务到设备的映射。

```python
from hardware.htree import HardwareTree
from deployment_config import from_user_tree, device_label
from exploration.decoder import RootInit, decode_to_root
from parallelism.ptraversal import derive_from_node

hardware = HardwareTree(serving.hcase_index)
deployment = {
    "format": "symbiotree.deployment.v1",
    "batch_size": 8,
    "tree": {
        "type": "OP",
        "children": [
            {"type": "TP", "devices": [f"gpu{i}" for i in range(8)]},
            {"type": "TP", "devices": [f"pim{i}" for i in range(8)]},
        ],
    },
}
individual = from_user_tree(deployment, hardware, serving)
individual.check_legality()
scope = RootInit([[0.0, 1.0]] * serving.req_type_num,
                 [0, model.layer_num - 1], [0.0, 1.0])
ddt_root = decode_to_root(individual, scope)
devices = {d.idx: d for d in hardware.devices}
for task in derive_from_node(ddt_root):
    print(task.name, device_label(devices[task.idx]), task.pp_attr, task.xp_attr.name)
```

输出依次为任务名称、设备名称、层范围和算子类型。修改字典中的 `tree`、`devices` 或划分比例即可构建不同部署；`hardware.root`、`individual` 和 `ddt_root` 可直接交给自己的研究代码继续处理。

## 扩展入口

| 希望修改的内容 | 从哪里开始 |
|---|---|
| 稠密 Transformer 的模型维度 | `ModelConfig` 或 JSON 中的模型配置 |
| 设备与链路参数 | JSON 中的硬件配置或 `hardware.hcase` |
| 自定义设备数量和树形硬件组织 | [CONFIGURATION.md](CONFIGURATION.md#自定义硬件结构) 中的硬件树JSON；解析与运行上下文在 `hardware_config.py` |
| 现有原语组成的新部署DDT | 树形部署JSON；外围转换入口在 `deployment_config.py` |
| 新的划分语义或算子类型 | `parallelism.pnode`、`parallelism.ptree` 及相关推导与评估逻辑 |
| 新请求分布或请求轨迹输入 | `serving.simulator.Simulator._sample_req_tokens` 及请求到达处理逻辑 |
| 不同的算子或通信成本模型 | `parallelism.ptree`、`hardware.hnode`、`hardware.hperf` |
| 搜索规则或初始化方式 | `exploration.rewrite_mechanism`、`exploration.evolution_pareto` |

硬件树JSON可修改设备数量、分组和层级。若需支持新的模型类型、非树形互连或请求采样器，应在自己的代码副本中扩展对应模块。

