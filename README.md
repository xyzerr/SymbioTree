# SymbioTree

**A dual-tree simulator and deployment search framework for LLM decoding on heterogeneous GPU+PIM systems.**

SymbioTree uses two trees to represent hardware structures and deployment decisions
for LLM decoding on heterogeneous GPU+PIM systems. It brings operator placement,
request routing and parallelism into a single search framework to automatically
find suitable decoding deployment plans for different serving scenarios. The
framework supports a shift from manual design based on experience to structured
representation and automated synthesis.

The two trees are the Heterogeneous Hardware Tree (HHT) and the Deployment Decision
Tree (DDT). The HHT describes hardware resources and their hierarchical
organization, while the DDT describes serving-workload partitions and deployment
decisions. An explicit mapping between them assigns tasks to hardware endpoints.
Based on this representation, the simulator models computation, communication
and serving behavior, and the search engine uses evolutionary search to explore
throughput–tail-latency tradeoffs.

The project provides three capabilities:

1. **Deployment performance evaluation**: estimate throughput and P99 TPOT for a deployment under a specified model, hardware configuration and request workload.
2. **Deployment generation and design space exploration**: generate candidate deployments and explore combinations of operator placement, request routing and parallelism.
3. **Modular access to the formal tree representations**: use the HHT, DDT and mapping modules through Python interfaces to construct and decode custom deployment representations and integrate them into your research code.

This simulator accompanies **From Ad Hoc to Structured: Synthesizing LLM Serving
Deployments on Heterogeneous Systems**, accepted at **HPCA 2027**.

> 🔬 Also check out our paper [SparseX-PIM: Sparse Linear Algebra on Processing-in-Memory](https://dl.acm.org/doi/10.1145/3845814.3850078)! It explores a near-memory architecture for direct solvers, an interesting direction in PIM. Happy reading! 🚀

[中文文档](README_zh.md) · [Configuration guide](docs/CONFIGURATION_en.md) · [Python API](docs/API_en.md)

## Install

Use Python 3.11 or 3.12 and NumPy. From the repository directory:

```bash
python -m venv .venv
```

On Windows, install the dependency without activating the environment:

```text
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the following commands from the repository root. On Windows, if the environment
is not activated, replace `python` with `.venv\Scripts\python.exe`.

## Performance evaluation

Given a model, hardware configuration, request workload and deployment plan,
SymbioTree simulates execution and estimates throughput and P99 TPOT. This
supports performance comparisons across deployments and serving conditions.

```bash
python symbiotree.py simulate examples/quickstart.json
```

The small example describes a dense Transformer, an eight-GPU/eight-PIM topology,
a fixed deployment and 10 simulated seconds of short requests. The command prints
the output folder and a summary containing:

- output-token throughput in tokens/s;
- P99 time per output token (TPOT) in milliseconds;
- completed requests/s and request counts by class.

Every invocation creates a fresh folder under `outputs/`. Detailed metrics are
in `metrics.json`; `run.log` contains the simulator output. `manifest.json`
records the effective configuration, software versions and elapsed wall time.

Change common parameters directly on the command line:

```bash
python symbiotree.py simulate examples/quickstart.json --arrival-rate 20 --batch-size 64 --duration 30
```

`--duration` sets simulated service time, not a wall-clock timeout. Use a longer
simulation when requests do not have enough time to finish. An unavailable P99
is written as `null`, not zero.

### Configure your own experiment

Copy the example configuration [examples/quickstart.json](examples/quickstart.json) to `my_config.json` in the repository root, then edit the following sections for your model, hardware and request workload:

| Section | What you choose |
|---|---|
| `model` | Layer count, hidden width, query/KV heads and FFN width |
| `hardware` | Hardware preset or custom hardware tree, with compute, bandwidth, memory and link parameters |
| `serving` | Request arrival rate, request-class mix, batch limit, duration and seed |
| `deployment` | A deployment preset or custom deployment file |
| `search` | Population, generations, batch choices and allowed partition primitives |

```bash
python symbiotree.py simulate my_config.json --output outputs/my_experiment
```

The output directory must not already exist, so a later run cannot overwrite an
earlier one. See the [configuration guide](docs/CONFIGURATION_en.md) for field names,
units, available topologies and the built-in request length distributions.

### Choose, inspect and customize a deployment

Select the hardware organization with `hardware.preset`: `intra_package` places
a GPU/PIM pair within each package, while `intra_node` connects GPU and PIM
endpoints through a node-level interconnect.

Select the deployment plan with `deployment.preset`:

- `gpu_pim_tp8`: eight GPUs execute linear computation and eight PIM endpoints execute attention, with eight-way tensor parallelism (TP8) on each side.
- `gpu_pim_pipeline2_tp4`: the model is split into two pipeline stages, each using four GPUs and four PIM endpoints collaboratively, with four-way tensor parallelism (TP4) on each side.

Inspect the hardware tree, deployment tree and actual device assignments:

```bash
python symbiotree.py inspect examples/quickstart.json
```

For a custom deployment, copy and edit the [TP8 collaboration template](examples/deployments/gpu_pim_tp8.json),
[two-stage pipeline template](examples/deployments/gpu_pim_pipeline2_tp4.json),
or [request-routing template](examples/deployments/request_split.json).
These use OP/RP/BP/TP and device names such as `gpu0` and `pim0`, without internal
node IDs or enum values. Change the device groups, partition weights, request-class
routing ratios or batch sizes, then validate and simulate:

```bash
python symbiotree.py validate examples/quickstart.json --deployment examples/deployments/request_split.json
python symbiotree.py simulate examples/quickstart.json --deployment examples/deployments/request_split.json
```

`--deployment` overrides the configured deployment preset or file. Both `inspect` and `validate`
check the configuration without running a performance simulation; `inspect`
also displays the trees, layer assignments and execution paths. Checks use the
existing capacity model. Invalid assignments or oversized explicit batch sizes
are rejected. See the [configuration guide](docs/CONFIGURATION_en.md#customize-a-deployment) for configuration fields
and deployment structures. To change device counts or hardware hierarchy,
start with [examples/custom_hardware.json](examples/custom_hardware.json) and edit its hardware-tree JSON.
See “Custom hardware structure” in [CONFIGURATION_en.md](docs/CONFIGURATION_en.md#custom-hardware-structure).

## Deployment generation and design space exploration

Given a model, hardware configuration and request workload, SymbioTree generates
candidate deployments using the HHT/DDT representation. Evolutionary search
explores combinations of operator placement, request routing and parallelism,
returning nondominated deployment candidates with different throughput–tail-latency
tradeoffs.

```bash
python symbiotree.py search examples/quickstart.json
```

The example uses a small population and one generation to demonstrate the
workflow. Increase the budget for your own exploration:

```bash
python symbiotree.py search my_config.json --population 60 --generations 20 --output outputs/my_search
```

`frontier.json` lists the final nondominated candidates and their throughput,
P99 TPOT and batch sizes. `front/point_*.json` stores their complete deployment
descriptions. `candidates.jsonl` records the evaluated search candidates.

Evaluate a saved deployment under the same or a different request arrival rate:

```bash
python symbiotree.py simulate my_config.json --deployment outputs/my_search/front/point_000.json --arrival-rate 20
```

Saved deployments retain their own batch settings; `--batch-size` is only for
fixed-case simulation. The hardware device IDs and request-class count must
match. For a new model or hardware topology, prefer a new search over reusing an
old deployment whose partition attributes were created for different conditions.
Saved search deployments also work with `inspect` and `validate`.

## How it works

The Heterogeneous Hardware Tree (HHT) describes hardware resources and their
organization. The Deployment Decision Tree (DDT) describes operator placement,
request routing and parallelism. A mapping binds tasks to hardware endpoints.
The simulator evaluates deployments; the search engine evolves their structures
and parameters using those evaluations.

The implementation names `XP`, `DP`, `PP` and `TP` correspond to operator
partitioning, request partitioning, pipeline/block partitioning and tensor
partitioning. The configuration guide maps these to the paper's OP/RP/BP/TP
notation. In hardware parameters, `NPU` denotes the GPU side.

## Code layout

```text
symbiotree.py          Configuration-driven simulation and search launcher
deployment_config.py  User deployment format, inspection and validation
hardware_config.py    Custom hardware trees and runtime configuration
examples/             Small editable example configurations
docs/                 Configuration reference and Python usage
src/hardware/         Hardware trees and communication costs
src/parallelism/      Deployment trees, mappings and operator costs
src/serving/          Requests, queues, batching and decode simulation
src/system/           Model/system configurations and aggregate metrics
src/exploration/      IR, transformations, feasibility checks and search
src/run_simulation.py Standalone simulation entry point
src/run_evolution.py  Standalone search entry point
```

See the [Python API guide](docs/API_en.md) for integration examples.

## Authors and citation

This simulator was jointly developed by [Xiaoyu Zhang](https://github.com/xyzerr) and [Xiaotian Sun](https://github.com/sunxt99).

Xiaoyu Zhang¹, Xiaotian Sun¹, Xinyu Wang¹, Xinjiang Zhang¹, Chunmeng Dou²,
Xueqi Li¹, Xiaoming Chen¹

¹ Institute of Computing Technology, Chinese Academy of Sciences  
² Institute of Microelectronics, Chinese Academy of Sciences

**Xiaoyu Zhang and Xiaotian Sun contributed equally and are co-first authors.
Xiaoming Chen is the corresponding author.**

**Corresponding author email:** [chenxiaoming@ict.ac.cn](mailto:chenxiaoming@ict.ac.cn)

If you use SymbioTree, please cite the paper above. Citation metadata is available
in [CITATION.cff](CITATION.cff).
