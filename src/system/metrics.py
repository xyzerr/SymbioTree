from __future__ import annotations
from typing import List
import numpy as np

from serving.request import Request
from serving.metrics import SimulationResult


def _pct(x, p):
    if len(x) == 0:
        return float("nan")
    return float(np.percentile(x, p))


def _fmt_ms(x_s: float) -> str:
    if np.isnan(x_s):
        return "nan"
    return f"{x_s * 1000.0:.3f}"



def summarize_device_usage_data(res_list: List[SimulationResult],
                                sim_time_end: float):
    horizon_ms = float(sim_time_end) * 1000.0
    merged_compute_ms = {}
    merged_comm_ms = {}
    merged_busy_wo_overlap_ms = {}
    merged_busy_wi_overlap_ms = {}

    for res in res_list:
        for device_name, value in getattr(res, "device_compute_ms", {}).items():
            merged_compute_ms[device_name] = merged_compute_ms.get(device_name, 0.0) + float(value)
        for device_name, value in getattr(res, "device_comm_ms", {}).items():
            merged_comm_ms[device_name] = merged_comm_ms.get(device_name, 0.0) + float(value)
        for device_name, value in getattr(res, "device_busy_wo_overlap_ms", {}).items():
            merged_busy_wo_overlap_ms[device_name] = merged_busy_wo_overlap_ms.get(device_name, 0.0) + float(value)
        for device_name, value in getattr(res, "device_busy_wi_overlap_ms", {}).items():
            merged_busy_wi_overlap_ms[device_name] = merged_busy_wi_overlap_ms.get(device_name, 0.0) + float(value)

    device_names = sorted(
        set(merged_compute_ms.keys())
        | set(merged_comm_ms.keys())
        | set(merged_busy_wo_overlap_ms.keys())
        | set(merged_busy_wi_overlap_ms.keys())
    )
    device_usage = {}
    for device_name in device_names:
        compute_ms = float(merged_compute_ms.get(device_name, 0.0))
        comm_ms = float(merged_comm_ms.get(device_name, 0.0))
        busy_wo_overlap_ms = float(merged_busy_wo_overlap_ms.get(device_name, compute_ms + comm_ms))
        busy_wi_overlap_ms = float(merged_busy_wi_overlap_ms.get(device_name, max(compute_ms, comm_ms)))
        device_usage[device_name] = {
            "compute_ms": round(compute_ms, 6),
            "comm_ms": round(comm_ms, 6),
            "busy_wo_overlap_ms": round(busy_wo_overlap_ms, 6),
            "busy_wi_overlap_ms": round(busy_wi_overlap_ms, 6),
            "compute_ratio": round(compute_ms / horizon_ms, 6) if horizon_ms > 0 else float("nan"),
            "comm_ratio": round(comm_ms / horizon_ms, 6) if horizon_ms > 0 else float("nan"),
            "busy_wo_overlap_ratio": round(busy_wo_overlap_ms / horizon_ms, 6) if horizon_ms > 0 else float("nan"),
            "busy_wi_overlap_ratio": round(busy_wi_overlap_ms / horizon_ms, 6) if horizon_ms > 0 else float("nan"),
            "total_ratio": round(busy_wo_overlap_ms / horizon_ms, 6) if horizon_ms > 0 else float("nan"),
        }
    return device_usage


def summarize_device_usage(res_list: List[SimulationResult],
                           sim_time_end: float) -> str:
    device_usage = summarize_device_usage_data(res_list, sim_time_end)
    if not device_usage:
        return ""

    lines = []
    lines.append("=== Device Timing Summary ===")
    lines.append(f"ratio_denominator={float(sim_time_end) * 1000.0:.3f} ms")
    for device_name, stat in device_usage.items():
        lines.append(
            f"{device_name}: "
            # f"compute_ms={stat['compute_ms']:.6f}, "
            # f"comm_ms={stat['comm_ms']:.6f}, "
            # f"busy_wo_overlap_ms={stat['busy_wo_overlap_ms']:.6f}, "
            # f"busy_wi_overlap_ms={stat['busy_wi_overlap_ms']:.6f}, "
            # f"comp_ratio={stat['compute_ratio']:.3f}, "
            # f"comm_ratio={stat['comm_ratio']:.3f}, "
            # f"busy_wo_overlap_ratio={stat['busy_wo_overlap_ratio']:.3f}, "
            # f"busy_wi_overlap_ratio={stat['busy_wi_overlap_ratio']:.3f}"
            
            f"comp_ratio={stat['compute_ratio']:.3f}, "
            f"comm_ratio={stat['comm_ratio']:.3f}, "
            f"busy_ratio={stat['busy_wo_overlap_ratio']:.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def _summarize_group(name: str, reqs: list[Request], t_end: float) -> str:
    n = len(reqs)
    if n == 0:
        return f"[{name}] finished=0\n"

    lat = np.array([r.finish_time + r.accum_delay_time - r.arrival_time for r in reqs], dtype=float)
    qd = np.array([r.start_time - r.arrival_time for r in reqs], dtype=float)
    st = np.array([r.finish_time + r.accum_delay_time - r.start_time for r in reqs], dtype=float)
    prompt = np.array([r.prompt_tokens for r in reqs], dtype=float)
    gen_tgt = np.array([r.gen_tokens for r in reqs], dtype=float)
    usr_throughput = np.array([r.gen_tokens/(r.finish_time + r.accum_delay_time - r.start_time)/1000 for r in reqs], dtype=float)
    tpot = np.array([(r.finish_time + r.accum_delay_time - r.start_time) / r.gen_tokens for r in reqs], dtype=float)

    lines = []
    lines.append(f"[{name}] finished={n}, throughput={n / t_end:.6f} req/s")
    # lines.append(f"  prompt_tokens mean={float(np.mean(prompt)):.3f}, p50={_pct(prompt, 50):.3f}, p95={_pct(prompt, 95):.3f}")
    # lines.append(f"  gen_tokens mean={float(np.mean(gen_tgt)):.3f}, p50={_pct(gen_tgt, 50):.3f}, p95={_pct(gen_tgt, 95):.3f}")
    # lines.append("  latency_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(lat)))}, p50={_fmt_ms(_pct(lat, 50))}, p95={_fmt_ms(_pct(lat, 95))}, p99={_fmt_ms(_pct(lat, 99))}")
    # lines.append("  queueing_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(qd)))}, p50={_fmt_ms(_pct(qd, 50))}, p95={_fmt_ms(_pct(qd, 95))}, p99={_fmt_ms(_pct(qd, 99))}")
    # lines.append("  service_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(st)))}, p50={_fmt_ms(_pct(st, 50))}, p95={_fmt_ms(_pct(st, 95))}, p99={_fmt_ms(_pct(st, 99))}")
    # lines.append("  user_throughput_token/s: " +
    #              f"mean={_fmt_ms(float(np.mean(usr_throughput)))}, p10={_fmt_ms(_pct(usr_throughput, 10))}, p50={_fmt_ms(_pct(usr_throughput, 50))}, p90={_fmt_ms(_pct(usr_throughput, 90))}")
    # lines.append("  TPOT_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(tpot)))}, p50={_fmt_ms(_pct(tpot, 50))}, p90={_fmt_ms(_pct(tpot, 90))}, p99={_fmt_ms(_pct(tpot, 99))}")
    # lines.append("")

    lines.append("  user_throughput_token/s: " +
                 f"p50={_fmt_ms(_pct(usr_throughput, 50))}, p90={_fmt_ms(_pct(usr_throughput, 90))}, p99={_fmt_ms(_pct(usr_throughput, 99))}")
    lines.append("  latency_ms: " +
                 f"p50={_fmt_ms(_pct(lat, 50))}, p90={_fmt_ms(_pct(lat, 90))}, p99={_fmt_ms(_pct(lat, 99))}")
    lines.append("  queueing_ms: " +
                 f"p50={_fmt_ms(_pct(qd, 50))}, p90={_fmt_ms(_pct(qd, 90))}, p99={_fmt_ms(_pct(qd, 99))}")
    lines.append("  service_ms: " +
                 f"p50={_fmt_ms(_pct(st, 50))}, p90={_fmt_ms(_pct(st, 90))}, p99={_fmt_ms(_pct(st, 99))}")
    lines.append("  TPOT_ms: " +
                 f"p50={_fmt_ms(_pct(tpot, 50))}, p90={_fmt_ms(_pct(tpot, 90))}, p99={_fmt_ms(_pct(tpot, 99))}")
    lines.append("")
    return "\n".join(lines)


def summarize_metrics(res_list: List[SimulationResult],
                      sim_time_end: float) -> str:
    # processed 包括 running 和 finished
    processed = []
    finished = []
    for res in res_list:
        finished.extend(res.finished)
        processed.extend(res.finished)
        for req in res.running:
            req.finish_time = sim_time_end
        processed.extend(res.running)
    finished_num = len(finished)

    busy_time = np.mean([res.busy_time for res in res_list])
    t_end = np.mean([res.t_end for res in res_list])

    util = busy_time / t_end if t_end > 0 else float("nan")

    # prio = [r for r in finished if getattr(r, "is_priority", False)]
    # normal = [r for r in finished if not getattr(r, "is_priority", False)]

    gen_tokens = [r.gen_tokens for r in processed]
    total_gen_tokens = np.sum(gen_tokens)

    lines = []
    lines.append("=== Simulation Summary ===")
    lines.append(f"horizon={t_end:.6f}s, busy_time={busy_time:.6f}s, utilization={util:.6f}")
    lines.append(f"finished_total={finished_num}, throughput_total={finished_num / t_end:.6f} req/s")
    lines.append(f"mean_gen_tokens={np.mean(gen_tokens)} token")
    lines.append(f"throughput={total_gen_tokens/t_end} token/s")
    lines.append("")
    lines.append(_summarize_group("ALL", finished, float(t_end)))
    device_usage_summary = summarize_device_usage(res_list, float(t_end))
    if device_usage_summary:
        lines.append(device_usage_summary)
    # lines.append(_summarize_group("ALL", processed, float(t_end)))

    # lines.append(_summarize_group("PRIORITY", prio, float(t_end)))
    # lines.append(_summarize_group("NORMAL", normal, float(t_end)))

    return "\n".join(lines)


def _summarize_group_data(name: str, reqs: list[Request], t_end: float):
    n = len(reqs)
    if n == 0:
        return float("nan")

    lat = np.array([r.finish_time + r.accum_delay_time - r.arrival_time for r in reqs], dtype=float)
    qd = np.array([r.start_time - r.arrival_time for r in reqs], dtype=float)
    st = np.array([r.finish_time + r.accum_delay_time - r.start_time for r in reqs], dtype=float)
    prompt = np.array([r.prompt_tokens for r in reqs], dtype=float)
    gen_tgt = np.array([r.gen_tokens for r in reqs], dtype=float)
    usr_throughput = np.array([r.gen_tokens/(r.finish_time + r.accum_delay_time - r.start_time)/1000 for r in reqs], dtype=float)
    tpot = np.array([(r.finish_time + r.accum_delay_time - r.start_time) / r.gen_tokens for r in reqs], dtype=float)

    # lines = []
    # lines.append(f"[{name}] finished={n}, throughput={n / t_end:.6f} req/s")
    # lines.append(f"  prompt_tokens mean={float(np.mean(prompt)):.3f}, p50={_pct(prompt, 50):.3f}, p95={_pct(prompt, 95):.3f}")
    # lines.append(f"  gen_tokens mean={float(np.mean(gen_tgt)):.3f}, p50={_pct(gen_tgt, 50):.3f}, p95={_pct(gen_tgt, 95):.3f}")
    # lines.append("  latency_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(lat)))}, p50={_fmt_ms(_pct(lat, 50))}, p95={_fmt_ms(_pct(lat, 95))}, p99={_fmt_ms(_pct(lat, 99))}")
    # lines.append("  queueing_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(qd)))}, p50={_fmt_ms(_pct(qd, 50))}, p95={_fmt_ms(_pct(qd, 95))}, p99={_fmt_ms(_pct(qd, 99))}")
    # lines.append("  service_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(st)))}, p50={_fmt_ms(_pct(st, 50))}, p95={_fmt_ms(_pct(st, 95))}, p99={_fmt_ms(_pct(st, 99))}")
    # lines.append("  user_throughput_token/s: " +
    #              f"mean={_fmt_ms(float(np.mean(usr_throughput)))}, p10={_fmt_ms(_pct(usr_throughput, 10))}, p50={_fmt_ms(_pct(usr_throughput, 50))}, p90={_fmt_ms(_pct(usr_throughput, 90))}")
    # lines.append("  TPOT_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(tpot)))}, p10={_fmt_ms(_pct(tpot, 10))}, p50={_fmt_ms(_pct(tpot, 50))}, p90={_fmt_ms(_pct(tpot, 90))}")
    # lines.append("")
    # return "\n".join(lines)


    # P90/95/99 latency
    # return _fmt_ms(_pct(lat, 90))

    # P90/95/99 TPOT
    return float(_fmt_ms(_pct(tpot, 99)))

def summarize_metrics_data(res_list: List[SimulationResult],
                           sim_time_end: float):
    # processed 包括 running 和 finished
    finished = []
    finished_dist = [0, 0, 0]

    processed = []
    processed_dist = [0, 0, 0]

    for res in res_list:
        for req in res.finished:
            finished_dist[req.req_type] += 1
            processed_dist[req.req_type] += 1
        finished.extend(res.finished)
        processed.extend(res.finished)

        for req in res.running:
            processed_dist[req.req_type] += 1
            req.finish_time = sim_time_end
            req.gen_tokens = max(req.gen_tokens, 1)

        processed.extend(res.running)
    finished_num = len(finished)

    busy_time = np.mean([res.busy_time for res in res_list])
    t_end = np.mean([res.t_end for res in res_list])

    util = busy_time / t_end if t_end > 0 else float("nan")

    # prio = [r for r in finished if getattr(r, "is_priority", False)]
    # normal = [r for r in finished if not getattr(r, "is_priority", False)]

    gen_tokens = np.sum([r.gen_tokens for r in processed], dtype=float)

    # lines = []
    # lines.append("=== Simulation Summary ===")
    # lines.append(f"horizon={t_end:.6f}s, busy_time={busy_time:.6f}s, utilization={util:.6f}")
    # lines.append(f"finished_total={finished_num}, throughput_total={finished_num / t_end:.6f} req/s")
    # lines.append(f"throughput={gen_tokens/t_end} token/s")
    # lines.append("")
    # lines.append(_summarize_group("ALL", finished, float(t_end)))

    # 吞吐是整体的，P99统计的是已完成任务的
    return [gen_tokens/t_end,
            _summarize_group_data("Finished", finished, float(t_end)),
            finished_dist,
            processed_dist,
            finished_num / t_end]

    # 吞吐是整体的，P99统计的是所有已进行任务的
    # return [gen_tokens/t_end, _summarize_group_data("Processed", processed, float(t_end)), finished_dist, processed_dist]
