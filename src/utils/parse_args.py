import argparse

def parse_args():
    p = argparse.ArgumentParser(description="LLM decode batching simulator (no prefill) with priority QoS.")

    # Model
    p.add_argument("--model-index", type = int, required=True, help="The index of the model to run.")

    # hardware
    p.add_argument("--hcase-index", type = int, required=True, help="The index of the hardware example.")
    # TODO: Pareto 会自主搜索 pcase，这个参数在 evo 中只是为了初始化 simulator
    p.add_argument("--pcase-index", type = int, required=True, help="The index of the parallel example.")

    # run time
    p.add_argument("--t-end", type=float, required=True, help="Simulation end time horizon (s).")

    # Request
    p.add_argument("--req-type-num", type=int, required=True, help="Number of the type of requests.")
    p.add_argument("--req-dist", type=str, required=True, help="Distribution of the type of requests.")
    # p.add_argument("--lam", type=float, required=True, help="Arrival rate (req/s).")
    p.add_argument("--lam", type=float, default=100, help="Arrival rate (req/s).")

    # QoS / Priority
    p.add_argument("--priority-ratio", type=float, default=0.0, help="Fraction of priority requests (default 0.05).")
    p.add_argument("--mode", type=str, choices=["preempt", "reserve"], default="preempt",
                   help="QoS mode: preempt (shrink to max-batch-hi when priority exists) or reserve (reserve slots for priority).")

    # Capacity controls
    # TODO: Pareto 会自主搜索batch size，这个参数是用不到的
    p.add_argument("--max-batch-lo", type=int, default=512,
                   help="Max batch size when NO priority exists; also total cap in reserve mode. Default 256.")
    p.add_argument("--max-batch-hi", type=int, default=32,
                   help="Max batch size when priority exists (preempt mode). Default 16.")
    p.add_argument("--reserve-hi", type=int, default=32,
                   help="Reserved slots for priority in reserve mode (reduces normal concurrency). Default 16.")

    p.add_argument("--peak-seq-len", type=int, default=2048,
                   help="Peak sequence length used by the exploration memory-feasibility guard. Default 10240.")
    p.add_argument("--runtime-reserve-ratio", type=float, default=0.0,
                   help="Reserved runtime memory ratio in [0,1) for exploration memory-feasibility. Default 0.0.")

    # Idle batching waits
    p.add_argument("--max-wait-ms", type=float, default=0.0,
                   help="Idle batching wait for normal-only mode (ms). Default 0.")
    p.add_argument("--max-wait-hi-ms", type=float, default=0.0,
                   help="Idle batching wait when priority is waiting (ms). Default 0 (protect priority latency).")


    # Debug and Logging
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--verbose", action="store_true", help="Print verbose logs.")
    p.add_argument("--out", type=str, default=None, help="Output file path")
    p.add_argument("--dse-out", type=str, default=None, help="Output file path of DSE process")

    return p.parse_args()