# 配置说明

[English](CONFIGURATION_en.md) · [返回README](../README_zh.md) · [Python接口](API.md)

本页说明配置字段和自定义方法；安装、运行命令与结果查看见 [README_zh.md](../README_zh.md)。

结果目录的 `config.json` 保存包含命令行覆盖值的配置，可直接作为相同命令的输入再次运行。没有请求完成时，`metrics.json` 中的 `p99_tpot_ms` 为 `null`；Python 指标接口返回 `NaN`。

## 模型

以下六个字段对应本项目 [src/system/config.py](../src/system/config.py) 中的 `ModelConfig` 类：

| 字段 | 含义 |
|---|---|
| `model_name` | 自定义模型名称 |
| `hidden_size` | Transformer 隐藏维度 |
| `query_head_num` | 查询头数量 |
| `kv_head_num` | 键/值头数量 |
| `intermediate_size` | 前馈网络中间层宽度 |
| `layer_num` | 模型层数 |

维度必须是正整数。隐藏维度必须能被查询头数量整除，查询头数量必须能被 KV 头数量整除。每个头的维度和 KV 隐藏维度会自动计算。这些字段只描述模型结构，不会加载或执行对应的神经网络。

## 预设硬件

推荐通过 `hardware.preset` 选择硬件组织。以下两种配置均包含 8 个 GPU 和 8 个 PIM 端点：

| 名称 | 硬件组织 |
|---|---|
| `intra_package` | 每个封装含一个GPU/PIM对，封装之间互连 |
| `intra_node` | GPU和PIM端点连接到同一个节点级互连 |

`hardware.parameters` 需要提供下表中的全部十个参数。程序在实例化硬件树前加载这些参数。本实现中的 `NPU` 表示 GPU 侧。

| 字段 | 单位 |
|---|---|
| `NPU_FLOPS`、`PIM_FLOPS` | 每个端点的算力，TFLOP/s |
| `NPU_BW`、`PIM_BW` | 每个端点的存储带宽，TB/s |
| `NPU_MEM_CAP`、`PIM_MEM_CAP` | 内存可行性检查使用的容量，GiB |
| `nvl_bw`、`mm_bw` | 所建模链路的带宽，GB/s |
| `nvl_lat`、`mm_lat` | 链路延迟，ns |

修改这些参数不会改变端点数量或树结构。需要改变设备数量、分组或层级时，使用下面的自定义硬件结构。

## 自定义硬件结构

从 [examples/custom_hardware.json](../examples/custom_hardware.json) 开始。示例包含两个节点，每个节点有2个GPU和2个PIM。

1. **修改硬件拓扑**：编辑 [examples/hardware/two_nodes.json](../examples/hardware/two_nodes.json)，通过调整 `children` 改变以下内容：

   - **设备数量和类型**：增加或删除GPU、PIM。
   - **设备归属和分组**：决定哪些设备位于同一节点或封装。
   - **连接层级**：增加、删除或重新组织中间分组。

2. **配置部署**：检查并按需修改 [examples/deployments/two_nodes_tp4.json](../examples/deployments/two_nodes_tp4.json)，使部署方案的任务划分和设备分配与修改后的硬件结构一致。设备列表使用硬件文件中的名称，每个设备恰好使用一次。若只想搜索部署，可删除主配置中的 `deployment`，保持 `search.seed_cases` 为 `[]`，由搜索生成匹配新硬件的方案。

```text
# 评估已有部署
python symbiotree.py simulate examples/custom_hardware.json
# 搜索部署方案
python symbiotree.py search examples/custom_hardware.json
```

当前支持树形组织，节点名称须唯一，所有设备叶节点须处于同一深度。AllReduce使用单独配置的统一带宽和延迟，不随树的层级自动变化。

## 请求与服务

模拟器按泊松过程生成请求，`lam` 控制平均到达率，`req_dist` 控制短、中、长请求的比例。每个请求的输入长度和输出长度从对应类别的内置分布中采样。

请求按部署方案分配到执行路径，排队后组成批次进行解码。每轮模拟为批内每个请求生成一个token；完成的请求退出，空出的批位置可接纳等待中的请求。模拟持续到 `t_end`，输出吞吐量和P99 TPOT等指标。

`serving` 对应本项目 [src/system/config.py](../src/system/config.py) 文件中定义的 `SystemConfig` 类，常用字段如下：

| 字段 | 含义 |
|---|---|
| `req_type_num` | 设为 `3`；当前汇总指标固定使用三个请求类别 |
| `req_dist` | 三个类别的概率，均非负且总和为 1 |
| `lam` | 泊松到达过程的平均到达率，单位为请求/秒 |
| `t_end` | 模拟时长，单位为秒 |
| `max_batch_lo` | 固定部署模拟中普通请求的最大批大小 |
| `peak_seq_len` | 内存检查使用的序列长度，不是采样长度的上限 |
| `runtime_reserve_ratio` | 预留内存比例，取值范围为 [0, 1) |
| `seed` | 请求生成的随机种子 |
| `priority_ratio` | 优先请求的比例；设为 0 表示不启用优先级 |

也可以填写 `SystemConfig` 接受的其他字段，例如 `mode`、`max_batch_hi`、`reserve_hi`、`max_wait_s` 和 `max_wait_hi_s`。

内置采样器分别独立采样输入长度与输出长度，采用正态分布并四舍五入，最小值为 1 个 token：

| 类别编号 | 输入长度均值/标准差（token） | 输出长度均值/标准差（token） |
|---|---|---|
| 0 | 256 / 128 | 256 / 128 |
| 1 | 1024 / 512 | 1024 / 512 |
| 2 | 4096 / 2048 | 4096 / 2048 |

例如，`[1, 0, 0]` 表示仅使用短请求，`[0.2, 0.6, 0.2]` 表示混合三类请求。这些均值与标准差目前定义在 `Simulator._sample_req_tokens` 中，不能通过 JSON 修改。

## 选择部署预设

`deployment.preset` 提供两个具名部署预设，均可配合上述两种预设硬件组织（`intra_package` 和 `intra_node`）使用：

| 名称 | 部署含义 | 可编辑模板 |
|---|---|---|
| `gpu_pim_tp8` | GPU执行线性计算，PIM执行注意力，两侧分别采用TP8 | [TP8协作](../examples/deployments/gpu_pim_tp8.json) |
| `gpu_pim_pipeline2_tp4` | 模型分为两个层块，每个阶段内由4个GPU和4个PIM协作 | [两阶段流水](../examples/deployments/gpu_pim_pipeline2_tp4.json) |

这两种部署方案的树结构如下，`OP` 表示算子划分，`BP` 表示层块划分，`TP` 表示张量划分：

```text
gpu_pim_tp8
OP
├─ 线性计算 → TP8 → gpu0 … gpu7
└─ 注意力   → TP8 → pim0 … pim7

gpu_pim_pipeline2_tp4
BP（两个层块，默认权重1:1）
├─ 阶段0 → OP
│          ├─ 线性计算 → TP4 → gpu0 … gpu3
│          └─ 注意力   → TP4 → pim0 … pim3
└─ 阶段1 → OP
           ├─ 线性计算 → TP4 → gpu4 … gpu7
           └─ 注意力   → TP4 → pim4 … pim7
```

对于32层模型，两阶段默认覆盖第0–15层和16–31层。其他层数按划分权重分配为非空整数层块；可用 `inspect` 查看实际分配。

## 修改自己的部署

在上一节中的部署模板，或 [examples/deployments/request_split.json](../examples/deployments/request_split.json) 请求分流模板中，保留 `format` 字段，按需要修改：

- **划分方式**：在 `tree` 中用 `type` 和 `children` 组合OP（算子划分）、RP（请求分流）、BP（层块划分）和TP（张量划分）。
- **设备分配**：修改各分支的 `devices`，使用硬件中的设备名称，每个设备恰好分配一次。
- **划分比例**：RP的 `routing` 三行依次对应短、中、长请求，每列对应一个分支；BP和TP的 `weights` 设置划分权重，省略时等分。
- **批大小**：修改文件顶层的 `batch_size`；请求分流模板中的 `path_batch_sizes` 可分别设置各执行路径的批大小。

当前不支持空闲执行路径，每条路径在 `routing` 与 `serving.req_dist` 共同作用下的配置到达率必须大于0。例如只有短请求时，不能将短请求全部送往GPU而让PIM路径空闲。

修改后的查看、检查和评估方法见 [README_zh.md](../README_zh.md#选择查看和修改部署)。

## 部署搜索

在主配置的 `search` 部分设置搜索参数；使用自定义硬件时，将 `seed_cases` 设为 `[]`。启动搜索、选择结果和重评部署的步骤见 [README_zh.md](../README_zh.md#部署方案生成与设计空间探索)。

`search.initialization` 和 `search.evolution` 分别对应本项目 [src/exploration/evolution_pareto.py](../src/exploration/evolution_pareto.py) 文件中定义的 `InitConfig` 和 `EvoConfig` 类。常用字段如下：

| 字段 | 含义 |
|---|---|
| `initialization.population_size` | 种群个体数量 |
| `initialization.max_depth` / `max_children` | 初始拓扑生成时的深度和子节点数量限制 |
| `initialization.batch_size_choices` | 可选批大小，必须为正整数 |
| `initialization.disabled_parallelisms` | 搜索中禁用的划分原语 |
| `evolution.generations` | 演化代数；设为 0 时仅评估初始化种群 |
| `evolution.elite_size` | 精英个体数量，不能超过种群规模 |
| `evolution.p_rewrite_mut` | 结构突变的概率权重 |
| `evolution.p_numeric_mut` | 数值突变的概率权重 |
| `evolution.p_mapping_refine_mut` | 映射突变的概率权重 |
| `evolution.rewrite_max_steps` | 单次突变中重写链的步数上限 |
| `seed_cases` | 用于初始化搜索的部署预设名称，例如 `["gpu_pim_tp8", "gpu_pim_pipeline2_tp4"]`；`[]` 表示不使用预设种子部署 |
| `seed` | 传给搜索引擎的随机种子 |

入口程序同时接受论文和源码使用的原语名称：

| 概念 | 论文名称/配置别名 | 代码中的名称 |
|---|---|---|
| 算子划分 | OP | XP |
| 请求划分 | RP | DP |
| 流水线/层块划分 | BP | PP |
| 张量划分 | TP | TP |

例如，`"disabled_parallelisms": ["OP", "RP"]` 会禁用这两种划分。种子部署应与当前配置兼容，也可以使用空列表。

有限预算的演化搜索不保证全局最优；增大搜索预算可能改善结果，但不保证一定改善。
