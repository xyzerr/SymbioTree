import time
import os
import json
import ast

from system.system import System
from system.config import SystemConfig, ModelConfigs
from system.metrics import summarize_metrics, summarize_metrics_data, summarize_device_usage_data

from utils.parse_args import parse_args

def main():
    args = parse_args()

    model_cfg = ModelConfigs[args.model_index]

    sys_cfg = SystemConfig(
        hcase_index=args.hcase_index,
        pcase_index=args.pcase_index,
        req_type_num=args.req_type_num,
        req_dist=ast.literal_eval(args.req_dist),
        lam=args.lam,
        t_end=args.t_end,
        priority_ratio=args.priority_ratio,
        mode=args.mode,
        max_batch_hi=args.max_batch_hi,
        max_batch_lo=args.max_batch_lo,
        reserve_hi=args.reserve_hi,
        peak_seq_len=args.peak_seq_len,
        runtime_reserve_ratio=args.runtime_reserve_ratio,
        max_wait_s=args.max_wait_ms / 1000.0,
        max_wait_hi_s=args.max_wait_hi_ms / 1000.0,
        seed=args.seed,
        verbose=args.verbose,
    )

    start_t = time.perf_counter()

    system = System(sys_cfg, model_cfg)
    result = system.run_system()

    print(summarize_metrics(result, args.t_end))
    T_token, L, f_dist, p_dist, T_req = summarize_metrics_data(result, args.t_end)
    device_usage = summarize_device_usage_data(result, args.t_end)
    if args.out:
        # if os.path.exists(args.out):
        #     os.remove(args.out)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"T":T_token,
                                "L":L,
                                "T_req": T_req,
                                "B":sys_cfg.max_batch_lo,
                                # finished_dist
                                "fd":f_dist,
                                # processed_dist
                                "pd": p_dist,
                                # finished_dist_ratio
                                "fdr": [round(f/(r*sys_cfg.lam*sys_cfg.t_end),4) if r != 0 else 0 for f,r in zip(f_dist, sys_cfg.req_dist)],
                                # finished_total_ratio
                                "ftr": round(sum(f_dist) / (sys_cfg.lam * sys_cfg.t_end),3),
                                # processed_total_ratio
                                "ptr": round(sum(p_dist) / (sys_cfg.lam * sys_cfg.t_end),3),
                                # pcase-path sub-graph batch debug info
                                "peak_seq_len": int(sys_cfg.peak_seq_len),
                                "runtime_reserve_ratio": float(sys_cfg.runtime_reserve_ratio),
                                "subgraph_batch_info": getattr(system, "last_subgraph_batch_info", []),
                                "device_usage": device_usage,
                                }, ensure_ascii=False) + "\n")

    end_t = time.perf_counter()
    print(f"Elapsed: {end_t - start_t:.6f}s")


if __name__ == "__main__":
    main()