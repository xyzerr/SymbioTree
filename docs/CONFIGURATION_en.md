# Configuration guide

[简体中文](CONFIGURATION.md) · [README](../README.md) · [Python API](API_en.md)

This page explains configuration fields and customization. See [README.md](../README.md) for installation, commands and reading results.

The result directory's `config.json` includes command-line overrides and can be reused as input to the same command. If no requests complete, `p99_tpot_ms` in `metrics.json` is `null`; the Python metrics API returns `NaN`.

## Model

The following six fields correspond to the `ModelConfig` class in this project's [src/system/config.py](../src/system/config.py):

| Field | Meaning |
|---|---|
| `model_name` | Custom model name |
| `hidden_size` | Transformer hidden width |
| `query_head_num` | Number of query heads |
| `kv_head_num` | Number of key/value heads |
| `intermediate_size` | Feed-forward network intermediate width |
| `layer_num` | Number of layers |

Dimensions must be positive integers. The hidden width must be divisible by the number of query heads, and the query-head count must be divisible by the KV-head count. The head dimension and KV hidden width are calculated automatically. These fields describe the model structure; they do not load or execute the neural network.

## Hardware presets

Use `hardware.preset` to select a hardware organization. Both presets contain 8 GPU and 8 PIM endpoints:

| Name | Organization |
|---|---|
| `intra_package` | Each package contains a GPU/PIM pair; packages are interconnected |
| `intra_node` | GPU and PIM endpoints share a node-level interconnect |

`hardware.parameters` must supply all ten parameters below. They are loaded before the hardware tree is instantiated. In this implementation, `NPU` denotes the GPU side.

| Field | Unit |
|---|---|
| `NPU_FLOPS`, `PIM_FLOPS` | Compute throughput per endpoint, TFLOP/s |
| `NPU_BW`, `PIM_BW` | Memory bandwidth per endpoint, TB/s |
| `NPU_MEM_CAP`, `PIM_MEM_CAP` | Capacity used by the memory-feasibility checks, GiB |
| `nvl_bw`, `mm_bw` | Modeled link bandwidth, GB/s |
| `nvl_lat`, `mm_lat` | Link latency, ns |

Changing these parameters does not change the endpoint count or tree structure. To change the number of devices, their groups or hierarchy, use a custom hardware structure below.

## Custom hardware structure

Start with [examples/custom_hardware.json](../examples/custom_hardware.json). The example has two nodes, each with two GPUs and two PIM devices.

1. **Edit the hardware topology** in [examples/hardware/two_nodes.json](../examples/hardware/two_nodes.json). Adjust `children` to change:

   - **Device counts and types**: add or remove GPUs and PIM devices.
   - **Device placement and grouping**: choose which devices belong to the same node or package.
   - **Connection hierarchy**: add, remove or reorganize intermediate groups.

2. **Configure the deployment**: review and update [examples/deployments/two_nodes_tp4.json](../examples/deployments/two_nodes_tp4.json) as needed so that task partitions and device assignments match the modified hardware structure. Use names from the hardware file and assign every device exactly once. For search only, remove `deployment` from the main configuration and keep `search.seed_cases` set to `[]`; search will generate deployments for the new hardware.

```text
# Evaluate an existing deployment
python symbiotree.py simulate examples/custom_hardware.json
# Search for deployments
python symbiotree.py search examples/custom_hardware.json
```

The interface supports tree hierarchies with unique node names and all device leaves at the same depth. AllReduce uses separately configured, uniform bandwidth and latency; these do not change automatically with the hierarchy.

## Requests and serving

Requests follow a Poisson arrival process: `lam` sets the mean arrival rate, and `req_dist` sets the proportions of short, medium and long requests. Input and output lengths are sampled from the built-in distributions for each class.

Requests are assigned to execution paths according to the deployment, then queued and admitted into batches for decoding. Each simulated iteration generates one token per active request. Completed requests leave the batch, allowing waiting requests to fill the available slots. The simulation runs until `t_end` and reports throughput, P99 TPOT and other metrics.

`serving` corresponds to the `SystemConfig` class defined in this project's [src/system/config.py](../src/system/config.py). Common fields are:

| Field | Meaning |
|---|---|
| `req_type_num` | Set to `3`; the current aggregate metrics use three request classes |
| `req_dist` | Three nonnegative class probabilities that sum to 1 |
| `lam` | Mean arrival rate of the Poisson process, in requests/s |
| `t_end` | Simulated duration, in seconds |
| `max_batch_lo` | Maximum batch size for ordinary requests in fixed-deployment simulation |
| `peak_seq_len` | Sequence length used for memory checks, not a cap on sampled lengths |
| `runtime_reserve_ratio` | Reserved memory fraction, in [0, 1) |
| `seed` | Request-generation random seed |
| `priority_ratio` | Fraction of priority requests; 0 disables priority handling |

You can also supply other fields accepted by `SystemConfig`, such as `mode`, `max_batch_hi`, `reserve_hi`, `max_wait_s` and `max_wait_hi_s`.

The built-in sampler draws input and output lengths independently from normal distributions, rounds them to integers and clips them to a minimum of 1 token:

| Class index | Input length mean / standard deviation (tokens) | Output length mean / standard deviation (tokens) |
|---|---|---|
| 0 | 256 / 128 | 256 / 128 |
| 1 | 1024 / 512 | 1024 / 512 |
| 2 | 4096 / 2048 | 4096 / 2048 |

For example, `[1, 0, 0]` uses only short requests, while `[0.2, 0.6, 0.2]` mixes all three classes. The means and standard deviations are currently defined in `Simulator._sample_req_tokens` and cannot be changed through JSON.

## Choose a deployment preset

`deployment.preset` provides two named deployment presets. Both work with either hardware preset above (`intra_package` or `intra_node`):

| Name | Deployment | Editable template |
|---|---|---|
| `gpu_pim_tp8` | GPUs execute linear computation and PIM executes attention, with TP8 on each side | [TP8 collaboration](../examples/deployments/gpu_pim_tp8.json) |
| `gpu_pim_pipeline2_tp4` | Two layer blocks, each using four GPUs and four PIM endpoints collaboratively | [Two-stage pipeline](../examples/deployments/gpu_pim_pipeline2_tp4.json) |

The two deployment structures are shown below. `OP` denotes operator partitioning, `BP` denotes block partitioning, and `TP` denotes tensor partitioning:

```text
gpu_pim_tp8
OP
├─ Linear computation → TP8 → gpu0 … gpu7
└─ Attention          → TP8 → pim0 … pim7

gpu_pim_pipeline2_tp4
BP (two layer blocks, default weights 1:1)
├─ Stage 0 → OP
│            ├─ Linear computation → TP4 → gpu0 … gpu3
│            └─ Attention          → TP4 → pim0 … pim3
└─ Stage 1 → OP
             ├─ Linear computation → TP4 → gpu4 … gpu7
             └─ Attention          → TP4 → pim4 … pim7
```

For a 32-layer model, the default stages cover layers 0–15 and 16–31. Other layer counts are divided into nonempty integer layer blocks according to the partition weights. Use `inspect` to see the actual assignment.

## Customize a deployment

In a deployment template from the previous section, or the request-routing template [examples/deployments/request_split.json](../examples/deployments/request_split.json), keep the `format` field and edit:

- **Partition structure**: use `type` and `children` within `tree` to combine OP (operator partitioning), RP (request routing), BP (layer-block partitioning) and TP (tensor partitioning).
- **Device assignments**: edit each branch's `devices`, using names from the hardware configuration and assigning every device exactly once.
- **Partition ratios**: the three `routing` rows for RP correspond to short, medium and long requests, with one column per branch. For BP and TP, `weights` sets partition weights; omitting it gives equal partitions.
- **Batch size**: edit the top-level `batch_size`. In the request-routing template, `path_batch_sizes` sets batch sizes for individual execution paths.

Idle execution paths are currently unsupported: every path must have a positive configured arrival rate under the combination of `routing` and `serving.req_dist`. For example, with short requests only, routing all short requests to the GPU while leaving the PIM path idle is not supported.

See [README.md](../README.md#choose-inspect-and-customize-a-deployment) for inspecting, validating and evaluating the modified deployment.

## Deployment search

Set search parameters in the main configuration's `search` section; for custom hardware, set `seed_cases` to `[]`. See [README.md](../README.md#deployment-generation-and-design-space-exploration) for starting a search, selecting results and reevaluating deployments.

`search.initialization` and `search.evolution` correspond to the `InitConfig` and `EvoConfig` classes, respectively, defined in this project's [src/exploration/evolution_pareto.py](../src/exploration/evolution_pareto.py). Common fields are:

| Field | Meaning |
|---|---|
| `initialization.population_size` | Population size |
| `initialization.max_depth` / `max_children` | Depth and child-count limits for initial topology generation |
| `initialization.batch_size_choices` | Candidate batch sizes, all positive integers |
| `initialization.disabled_parallelisms` | Partition primitives disabled during search |
| `evolution.generations` | Number of generations; 0 evaluates only the initial population |
| `evolution.elite_size` | Number of elite individuals; must not exceed the population size |
| `evolution.p_rewrite_mut` | Probability weight for structural mutation |
| `evolution.p_numeric_mut` | Probability weight for numerical mutation |
| `evolution.p_mapping_refine_mut` | Probability weight for mapping mutation |
| `evolution.rewrite_max_steps` | Maximum rewrite-chain length per mutation |
| `seed_cases` | Deployment preset names used to initialize the search, such as `["gpu_pim_tp8", "gpu_pim_pipeline2_tp4"]`; `[]` disables preset seeds |
| `seed` | Random seed passed to the search engine |

The launcher accepts both the paper's primitive names and the names used in the source code:

| Concept | Paper name / configuration alias | Source-code name |
|---|---|---|
| Operator partitioning | OP | XP |
| Request partitioning | RP | DP |
| Pipeline / block partitioning | BP | PP |
| Tensor partitioning | TP | TP |

For example, `"disabled_parallelisms": ["OP", "RP"]` disables those two primitives. Seed deployments must be compatible with the configuration; the seed list may also be empty.

Evolutionary search with a finite budget does not guarantee a global optimum. Increasing the budget may improve results, but does not guarantee an improvement.
