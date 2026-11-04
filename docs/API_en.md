# Using SymbioTree from Python

[简体中文](API.md) · [README](../README.md) · [Configuration guide](CONFIGURATION_en.md)

These examples show how to evaluate deployments, search for plans, and use the HHT/DDT and mapping modules from your own Python program. The first section sets up imports, the model, hardware and serving configuration; the next two sections reuse that setup.

To define new combinations of OP/RP/BP/TP, device assignments or routing ratios, use the [tree-based deployment configuration](CONFIGURATION_en.md) without writing Python builder functions. Use `symbiotree.py inspect` to view a deployment and `validate` to check it.

## Evaluate a fixed deployment

Run the following code from the repository root:

```python
import sys
from pathlib import Path

project = Path.cwd()
sys.path.insert(0, str(project))
sys.path.insert(0, str(project / "src"))

from hardware import hcase, hperf

# Set hardware parameters before constructing a HardwareTree or System.
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
    hcase_index=9,  # Intra-node hardware organization, corresponding to intra_node
    pcase_index=16,  # GPU/PIM collaboration with TP8 on each side, corresponding to gpu_pim_tp8
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

Hardware configuration is stored in module-level variables. Use separate processes when running different configurations concurrently; do not modify the shared hardware globals from multiple threads. The command-line launcher runs one configuration per process.

## Search and save deployments

Run this after the first section's code. The example searches with eight individuals for one generation and saves deployments on the final frontier. Adjust `population_size`, `generations` and `batch_size_choices` to change the search size.

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

Each line reports the deployment file, throughput (tokens/s) and P99 TPOT (ms). `evolve`, `InitConfig` and `EvoConfig` are defined in [src/exploration/evolution_pareto.py](../src/exploration/evolution_pareto.py).

## Use HHT, DDT and mapping modules independently

This example reuses the first section's setup and needs no prior search. It constructs an HHT, defines a DDT in a Python dictionary with GPUs executing linear computation and PIM executing attention, and reads the task-to-device mapping.

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

Each line reports the task name, device name, layer range and operator type. Edit `tree`, `devices` or partition ratios in the dictionary to construct other deployments. Pass `hardware.root`, `individual` and `ddt_root` directly to your own research code for further processing.

## Extension points

| What to change | Where to start |
|---|---|
| Dense Transformer dimensions | `ModelConfig` or the model section in JSON |
| Device and link parameters | The hardware section in JSON or `hardware.hcase` |
| Custom device counts and tree-structured hardware | Hardware-tree JSON in [CONFIGURATION_en.md](CONFIGURATION_en.md#custom-hardware-structure); parsing and runtime context in `hardware_config.py` |
| New deployment DDT composed of existing primitives | Tree-based deployment JSON; the adapter is in `deployment_config.py` |
| New partition semantics or operator types | `parallelism.pnode`, `parallelism.ptree` and the associated derivation and evaluation logic |
| New request distribution or trace input | `serving.simulator.Simulator._sample_req_tokens` and the request-arrival handling logic |
| Different operator or communication cost models | `parallelism.ptree`, `hardware.hnode`, `hardware.hperf` |
| Search rules or initialization | `exploration.rewrite_mechanism`, `exploration.evolution_pareto` |

Hardware-tree JSON supports changing device counts, groups and hierarchy. For new model types, non-tree interconnects or request samplers, extend the relevant modules in your own code copy.

