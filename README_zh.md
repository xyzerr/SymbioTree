# SymbioTree

**基于双树表示的GPU+PIM异构LLM解码模拟器与部署搜索框架。**

SymbioTree 面向 GPU+PIM 异构系统上的 LLM 解码服务，用两棵树统一描述硬件结构与部署决策，将算子放置、请求路由和并行方式纳入同一搜索框架，为不同服务场景自动寻找合适的解码部署方案。它推动部署设计从依赖经验的手工设计走向结构化表示与自动综合。

这两棵树分别是异构硬件树（Heterogeneous Hardware Tree，HHT）和部署决策树（Deployment Decision Tree，DDT）。HHT 描述硬件资源及其层次组织，DDT 描述服务工作负载的划分与部署决策，两者之间的显式映射将任务落实到硬件端点。基于这一表示，模拟器对计算、通信和服务过程进行建模，搜索引擎通过演化搜索探索吞吐量与尾延迟之间的权衡。

本项目提供以下三项功能：

1. **部署方案性能评估**：在给定模型、硬件和请求负载下，评估部署方案的吞吐量与P99 TPOT。
2. **部署方案生成与设计空间探索**：生成候选部署，探索算子放置、请求路由和并行方式的组合。
3. **树形形式化表示的模块化调用**：通过Python接口调用HHT、DDT及映射相关模块，构建和解析自定义部署表示，便于集成到自己的研究代码中。

本模拟器对应论文 **From Ad Hoc to Structured: Synthesizing LLM Serving Deployments on Heterogeneous Systems**，已被 **HPCA 2027** 接收。

[English](README.md) · [配置参数与单位](docs/CONFIGURATION.md) · [Python接口](docs/API.md)

## 安装并运行

建议使用Python 3.11或3.12。依赖只有NumPy。在项目根目录执行以下命令。

```text
python -m venv .venv
```

Windows可直接执行，不需要激活虚拟环境：

```text
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe symbiotree.py simulate examples/quickstart.json
```

Linux/macOS先执行`source .venv/bin/activate`，再执行：

```text
python -m pip install -r requirements.txt
python symbiotree.py simulate examples/quickstart.json
```

以下命令在仓库根目录执行。Windows未激活虚拟环境时，将命令中的 `python` 替换为 `.venv\Scripts\python.exe`。

## 性能评估

给定模型、硬件、请求负载及部署方案，模拟其执行过程，评估吞吐量和P99 TPOT，用于比较不同部署或服务条件下的性能。

示例使用一个稠密Transformer、8个GPU和8个PIM端点，模拟10秒短请求负载。
运行结束会打印结果目录、输出token吞吐量、P99 TPOT及已完成请求速率。

常用参数可以直接改：

```text
python symbiotree.py simulate examples/quickstart.json --arrival-rate 20 --batch-size 64 --duration 30
```

这条命令把到达率改为每秒20个请求、最大批大小改为64、模拟时长改为30秒。
`--duration`指模拟服务时间，不是程序运行耗时。若没有请求完成，应延长模拟时间；
此时P99显示为`null`，不是零延迟。

使用自己的模型、硬件和请求负载时，将示例配置文件 [examples/quickstart.json](examples/quickstart.json) 复制为仓库根目录下的 `my_config.json`，再按需要修改以下配置项：

| 配置项 | 可修改的内容 |
|---|---|
| `model` | 层数、隐藏维度、注意力头数、KV头数、FFN宽度 |
| `hardware` | 硬件预设或自定义硬件树，以及算力、带宽、容量和互连参数 |
| `serving` | 到达率、三类请求比例、批大小、时长和随机种子 |
| `deployment` | 部署预设或自定义部署文件 |
| `search` | 搜索种群、代数、批大小候选和禁用的划分原语 |

```text
python symbiotree.py simulate my_config.json --output outputs/my_experiment
```

字段、单位和内置拓扑的说明见[配置文档](docs/CONFIGURATION.md)。
请求长度目前来自源码内置的三类分布；JSON可以改变类别比例，不能直接输入任意trace。

### 选择、查看和修改部署

在配置文件的 `hardware.preset` 中选择硬件组织：`intra_package` 表示每个封装内包含一个GPU/PIM对，`intra_node` 表示GPU和PIM通过节点级互连连接。

需要改变设备数量或组织层级时，可从 [examples/custom_hardware.json](examples/custom_hardware.json) 开始，编辑其引用的硬件树JSON。配置方法见 [CONFIGURATION.md](docs/CONFIGURATION.md#自定义硬件结构) 的“自定义硬件结构”。

在 `deployment.preset` 中选择部署方案：

- `gpu_pim_tp8`：8个GPU执行线性计算，8个PIM执行注意力，两侧分别采用8路张量并行（TP8）。
- `gpu_pim_pipeline2_tp4`：模型分为两个流水阶段，每个阶段内由4个GPU和4个PIM协作，两侧分别采用4路张量并行（TP4）。

先查看示例的硬件树、部署树及实际设备映射：

```text
python symbiotree.py inspect examples/quickstart.json
```

需要自己的部署时，可复制并修改 [TP8协作模板](examples/deployments/gpu_pim_tp8.json)、
[两阶段流水模板](examples/deployments/gpu_pim_pipeline2_tp4.json) 或
[请求分流模板](examples/deployments/request_split.json)。模板直接使用OP、RP、BP、TP与 `gpu0`、`pim0` 等设备名称。

```text
python symbiotree.py validate examples/quickstart.json --deployment examples/deployments/request_split.json
python symbiotree.py simulate examples/quickstart.json --deployment examples/deployments/request_split.json
```

`inspect` 和 `validate` 均不运行性能模拟；前者还展示结构，后者只报告检查结果。
`--deployment` 覆盖配置中的部署预设或部署文件。字段含义和部署树结构见[配置文档](docs/CONFIGURATION.md#修改自己的部署)。

## 部署方案生成与设计空间探索

给定模型、硬件和请求负载，基于HHT/DDT双树表示生成候选部署，通过演化搜索探索算子放置、请求路由和并行方式的组合，输出不同吞吐量与尾延迟权衡下的部署方案。

```text
python symbiotree.py search examples/quickstart.json
```

示例只进行小规模搜索，便于快速理解流程。自己的实验可增大预算：

```text
python symbiotree.py search my_config.json --population 60 --generations 20 --output outputs/my_search
```

结果中的`frontier.json`列出最终前沿候选的吞吐量、P99 TPOT和批大小，
`front/point_*.json`保存完整部署。可以选中一个方案，再更换到达率进行评估：

```text
python symbiotree.py simulate my_config.json --deployment outputs/my_search/front/point_000.json --arrival-rate 20
```

这时沿用部署JSON里的批大小，不能同时使用`--batch-size`。保存的方案也可以通过`inspect`查看和`validate`检查。
设备编号与请求类别数量必须匹配；更换模型结构或硬件拓扑时，建议重新搜索。

## 查看结果

每次运行独立生成`outputs/时间_任务/`，也可通过`--output`指定新目录。
已有目录会被拒绝，防止覆盖旧结果。

- `metrics.json`：模拟结果及单位清晰的指标名称。
- `batch_constraints.json`：固定部署各路径的内存与批大小限制。
- `frontier.json`、`front/`：搜索出的候选指标与部署描述。
- `candidates.jsonl`：搜索实际评估的候选记录。
- `manifest.json`：生效配置、软件版本、源码哈希和运行耗时。
- `run.log`：运行过程日志。

TPOT按已完成请求统计；token吞吐量包含结束时仍在执行请求已生成的token。
已完成请求速率与输入到达率不同。结果是模拟估计，不是GPU实测。

## 扩展和集成

`src/hardware`描述硬件树，`src/parallelism`描述部署树和执行成本，
`src/serving`实现请求生成与离散事件模拟，`src/exploration`实现部署搜索。
也可以在自己的Python程序中直接调用这些模块，见[Python接口示例](docs/API.md)。

当前支持静态部署下的稠密Transformer解码，不包括预填充、MoE执行或真实CUDA部署。

## 作者与引用

Xiaoyu Zhang¹、Xiaotian Sun¹、Xinyu Wang¹、Xinjiang Zhang¹、Chunmeng Dou²、Xueqi Li¹、Xiaoming Chen¹

¹ 中国科学院计算技术研究所  
² 中国科学院微电子研究所

**Xiaoyu Zhang 与 Xiaotian Sun 为共同第一作者，Xiaoming Chen 为通讯作者。**

**通讯作者邮箱：** [chenxiaoming@ict.ac.cn](mailto:chenxiaoming@ict.ac.cn)

使用本模拟器时，请引用上述论文。引用信息见 [CITATION.cff](CITATION.cff)。
