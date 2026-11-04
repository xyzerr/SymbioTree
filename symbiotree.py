"""使用 JSON 配置运行 SymbioTree 模拟与部署搜索。"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import datetime as dt
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
from deployment_config import DEPLOYMENT_PRESETS, resolve_case, prepare_deployment, describe
from hardware_config import CUSTOM_CASE, resolve_hardware, hardware_session, describe_hardware
PRIMITIVES = {"OP": "XP", "RP": "DP", "BP": "PP", "TP": "TP"}


def serializable(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if hasattr(value, "item"):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_config(path, args):
    from system.config import SystemConfig, ModelConfig
    from serving.simulator import Simulator
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    unknown = set(config) - {"model", "hardware", "serving", "deployment", "search"}
    if unknown:
        raise ValueError(f"未知配置节：{sorted(unknown)}")
    for key in ["hidden_size", "query_head_num", "kv_head_num", "intermediate_size", "layer_num"]:
        value = config["model"][key]
        if type(value) is not int or value < 1:
            raise ValueError(f"model.{key} 必须为正整数")
    model = ModelConfig(**config["model"])
    if model.hidden_size % model.query_head_num or model.query_head_num % model.kv_head_num:
        raise ValueError("hidden_size 必须能被 query_head_num 整除，query_head_num 必须能被 kv_head_num 整除")
    config["hardware"], hardware_case = resolve_hardware(config["hardware"], path.parent)
    deployment_case = CUSTOM_CASE if hardware_case == CUSTOM_CASE else 16
    deployment = config.get("deployment")
    if "deployment" in config and not isinstance(deployment, dict):
        raise ValueError("deployment 必须是对象；不指定部署时请省略该节")
    if deployment is not None:
        if "file" in deployment:
            if set(deployment) != {"file"} or not isinstance(deployment["file"], str) or not deployment["file"].strip():
                raise ValueError("deployment.file 应为部署文件路径，不与 preset 或 case_index 混用")
            source = Path(deployment["file"])
            if not source.is_absolute():
                source = path.parent / source
            config["deployment"] = {"file": str(source.resolve(strict=True))}
        elif hardware_case == CUSTOM_CASE:
            raise ValueError("自定义硬件请使用 deployment.file 或 --deployment；仅搜索时可省略 deployment")
        else:
            deployment_case = resolve_case(deployment, DEPLOYMENT_PRESETS, "deployment")
    if "search" in config and not isinstance(config["search"], dict):
        raise ValueError("search 必须是对象")
    if hardware_case == CUSTOM_CASE and config.get("search", {}).get("seed_cases"):
        raise ValueError("自定义硬件请将 search.seed_cases 设为 []，由搜索生成匹配当前端点的部署")
    serving = dict(config["serving"])
    for arg, key in [("arrival_rate", "lam"), ("duration", "t_end"), ("batch_size", "max_batch_lo")]:
        if getattr(args, arg) is not None:
            serving[key] = getattr(args, arg)
    if serving.get("req_type_num") != 3 or len(serving.get("req_dist", [])) != 3:
        raise ValueError("当前模拟器的请求指标要求配置三个请求类别")
    mix = serving["req_dist"]
    if any(not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for x in mix) or not math.isclose(sum(mix), 1., abs_tol=1e-9):
        raise ValueError("serving.req_dist 必须包含三个非负概率，且总和为 1")
    cfg = SystemConfig(hcase_index=hardware_case, pcase_index=deployment_case, **serving)
    Simulator._validate_sim_cfg(cfg)
    if not 0 <= cfg.runtime_reserve_ratio < 1 or cfg.peak_seq_len <= 0:
        raise ValueError("runtime_reserve_ratio 必须在 [0, 1) 范围内，peak_seq_len 必须为正数")
    return config, model, cfg


def root_scope(model):
    from exploration.decoder import RootInit
    return RootInit([[0., 1.]] * 3, [0, model.layer_num - 1], [0., 1.])


def run_simulation(model, cfg, deployment, output):
    if deployment is None:
        from system.system import System
        from system.metrics import summarize_metrics_data
        system = System(cfg, model)
        results = system.run_system()
        values = summarize_metrics_data(results, cfg.t_end)
        write_json(output / "batch_constraints.json", system.last_subgraph_batch_info)
    else:
        from exploration.ind_io import individual_to_dict
        from exploration.fitness_adapter import SystemEvaluatorV3
        ind, _, root, paths = prepare_deployment(cfg, model, deployment)
        write_json(output / "batch_constraints.json", paths)
        write_json(output / "deployment.json", individual_to_dict(ind))
        (output / "deployment_input.json").write_bytes(deployment.read_bytes())
        evaluator = SystemEvaluatorV3(cfg, model, cfg.req_dist, cfg.hcase_index, cfg.pcase_index, pareto_mode=True)
        values = evaluator.run_system_on_root(root, ind.batch_size)
    t, latency, finished, processed, requests = values
    metrics = {"throughput_tokens_per_s": t, "p99_tpot_ms": latency,
               "completed_requests_per_s": requests, "completed_requests_by_class": finished,
               "processed_requests_by_class": processed, "simulation_duration_s": cfg.t_end}
    write_json(output / "metrics.json", metrics)
    return metrics


def run_search(model, cfg, search, args, output):
    from hardware.htree import HardwareTree
    from parallelism import pcase
    from parallelism.pnode import Parallelism
    from exploration.evolution_pareto import InitConfig, EvoConfig, evolve
    from exploration.feasibility import FeasibilityConfig
    from exploration.fitness_adapter import make_fitness_fn
    from exploration.ind_io import save_individual_json
    search = dict(search)
    unknown = set(search) - {"seed", "seed_cases", "initialization", "evolution"}
    if unknown:
        raise ValueError(f"未知搜索选项：{sorted(unknown)}")
    init = dict(search.get("initialization", {})); evo = dict(search.get("evolution", {}))
    if args.population is not None: init["population_size"] = args.population
    if args.generations is not None: evo["generations"] = args.generations
    init["disabled_parallelisms"] = tuple(Parallelism[PRIMITIVES.get(x, x)] for x in init.get("disabled_parallelisms", []))
    init_cfg, evo_cfg = InitConfig(**init), EvoConfig(**evo)
    if init_cfg.population_size < 2 or evo_cfg.generations < 0 or not 0 <= evo_cfg.elite_size <= init_cfg.population_size:
        raise ValueError("要求 population >= 2、generations >= 0 且 0 <= elite_size <= population")
    if not init_cfg.batch_size_choices or any(type(x) is not int or x < 1 for x in init_cfg.batch_size_choices):
        raise ValueError("batch_size_choices 必须包含正整数")
    hardware = HardwareTree(cfg.hcase_index); scope = root_scope(model)
    seed_roots = []
    for case in search.get("seed_cases", []):
        if isinstance(case, str):
            case = resolve_case({"preset": case}, DEPLOYMENT_PRESETS, "search.seed_cases")
        builder = getattr(pcase, f"build_case_{int(case)}", None)
        if builder is None: raise ValueError(f"未知的种子部署编号 {case}")
        root, _ = builder(cfg.req_type_num, model.layer_num); seed_roots.append(root)
    feasibility = FeasibilityConfig(model, scope,
        {int(d.idx): float(d.meta['mem_cap']) for d in hardware.devices},
        {int(d.idx): int(d.meta.get('byte', 2)) for d in hardware.devices},
        peak_seq_len=cfg.peak_seq_len, runtime_reserve_ratio=cfg.runtime_reserve_ratio)
    write_json(output / "search_config.json", {"initialization": asdict(init_cfg), "evolution": asdict(evo_cfg),
               "seed": search.get("seed", cfg.seed), "seed_cases": search.get("seed_cases", [])})
    best, population = evolve(init_cfg, evo_cfg, req_type_num=cfg.req_type_num,
        devices=[int(d.idx) for d in hardware.devices], root_init=scope,
        device_type_by_id={int(d.idx): str(d.meta.get('type', d.name)) for d in hardware.devices},
        fitness_fn=make_fitness_fn(cfg, model, req_prob=cfg.req_dist, hcase_idx=cfg.hcase_index,
                                  pcase_idx_for_init=cfg.pcase_index, pareto_mode=True),
        feasibility_cfg=feasibility, with_pop_seeds=bool(seed_roots), pop_seed_roots=seed_roots,
        attach_hardware_leaves=True, random_seed=search.get("seed", cfg.seed),
        dse_out=str(output / "candidates.jsonl"))
    save_individual_json(best, output / "best.json")
    front = []
    for i, individual in enumerate(x for x in population if x.pareto_rank == 0):
        path = f"front/point_{i:03d}.json"; save_individual_json(individual, output / path)
        front.append({"deployment": path, "throughput_tokens_per_s": individual.throughput,
                      "p99_tpot_ms": individual.latency, "batch_size": individual.batch_size})
    write_json(output / "frontier.json", front)
    return {"frontier_candidates": len(front)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["simulate", "search", "inspect", "validate"],
                        help="simulate 评估；search 搜索；inspect 查看结构并检查；validate 仅检查")
    parser.add_argument("config", type=Path, help="模型、硬件与服务负载的 JSON 配置文件")
    parser.add_argument("--output", type=Path, help="新结果目录，不能使用已存在的目录")
    parser.add_argument("--arrival-rate", type=float, help="覆盖请求到达率，单位为请求/秒")
    parser.add_argument("--duration", type=float, help="覆盖模拟时长，单位为秒，不是程序运行耗时")
    parser.add_argument("--batch-size", type=int, help="覆盖固定部署的批大小上限")
    parser.add_argument("--deployment", type=Path, help="用户树形配置或搜索保存的部署 JSON（不用于 search）")
    parser.add_argument("--population", type=int, help="search 模式下覆盖种群规模")
    parser.add_argument("--generations", type=int, help="search 模式下覆盖演化代数")
    args = parser.parse_args()
    if args.command == 'search' and args.deployment:
        parser.error('--deployment 用于 simulate、inspect 或 validate 模式')
    if args.command in ('inspect', 'validate') and args.output:
        parser.error('inspect 和 validate 不创建结果目录，不使用 --output')
    if args.command != 'search' and (args.population is not None or args.generations is not None):
        parser.error('--population 和 --generations 仅适用于 search 模式')
    if args.batch_size is not None and (args.command == 'search' or args.deployment):
        parser.error('--batch-size 仅适用于固定部署模拟；搜索与保存部署使用各自的批大小设置')
    try:
        import numpy as np
    except ImportError:
        parser.error('请先安装依赖：python -m pip install -r requirements.txt')
    try:
        config, model, cfg = read_config(args.config.resolve(), args)
        deployment = (args.deployment.resolve(strict=True) if args.deployment else
                      Path(config["deployment"]["file"]) if "file" in config.get("deployment", {}) else None)
        has_deployment = deployment is not None or "deployment" in config
        if args.command == 'simulate' and not has_deployment:
            raise ValueError('性能评估需要部署，请填写 deployment.file 或使用 --deployment；也可先运行 search')
        if args.batch_size is not None and deployment is not None:
            raise ValueError('--batch-size 不用于部署文件，请修改文件中的 batch_size')
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.command in ('inspect', 'validate'):
        import io
        try:
            with hardware_session(config['hardware']):
                if not has_deployment:
                    from hardware.htree import HardwareTree
                    with contextlib.redirect_stdout(io.StringIO()):
                        hardware = HardwareTree(cfg.hcase_index)
                    if args.command == 'inspect':
                        print(describe_hardware(hardware))
                    print(f'硬件检查通过：{len(hardware.devices)} 个端点；未指定部署，未检查 DDT 或运行性能模拟。')
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        prepared = prepare_deployment(cfg, model, deployment)
                    if args.command == 'inspect':
                        print(describe(prepared, cfg, model))
                    else:
                        print(f'校验通过：{len(prepared[1].devices)} 个硬件端点，{len(prepared[3])} 条执行路径；未运行性能模拟。')
                        for p in prepared[3]:
                            if p['requested_batch'] != p['effective_batch']:
                                print(f"路径 {p['name']} 的批大小按容量由 {p['requested_batch']} 下调为 {p['effective_batch']}")
        except Exception as exc:
            parser.error(f'部署检查失败：{exc}')
        return
    out = (args.output or ROOT / 'outputs' / (dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.command)).resolve()
    if out.exists(): parser.error(f'结果目录已存在：{out}')
    out.mkdir(parents=True)
    saved_config = dict(config)
    if deployment is not None:
        saved_config['deployment'] = {'file': 'deployment_input.json'}
        (out / 'deployment_input.json').write_bytes(deployment.read_bytes())
    write_json(out / 'config.json', saved_config)
    if 'topology' in config['hardware']:
        write_json(out / 'hardware.json', config['hardware']['topology'])
    manifest = {'command': args.command, 'python': sys.version, 'numpy': np.__version__,
                'model': asdict(model), 'serving': asdict(cfg), 'hardware': config['hardware'],
                'deployment_input': str(deployment) if deployment else None, 'status': 'running',
                'source_sha256': {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').rglob('*.py')},
                'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'deployment_interface_sha256': hashlib.sha256((ROOT/'deployment_config.py').read_bytes()).hexdigest(),
                'hardware_interface_sha256': hashlib.sha256((ROOT/'hardware_config.py').read_bytes()).hexdigest()}
    write_json(out/'manifest.json', manifest)
    print(f'输出目录：{out}', flush=True)
    started = time.perf_counter(); old_cwd = Path.cwd()
    try:
        os.chdir(out)  # 将搜索过程的调试输出写入当前结果目录。
        with hardware_session(config['hardware']), (out/'run.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = run_simulation(model,cfg,deployment,out) if args.command=='simulate' else run_search(model,cfg,config.get('search',{}),args,out)
        manifest['status']='completed'
        print(json.dumps(serializable(result),indent=2),flush=True)
    except Exception:
        manifest['status']='failed'; manifest['error']=traceback.format_exc()
        raise
    finally:
        os.chdir(old_cwd); manifest['wall_seconds']=time.perf_counter()-started
        write_json(out/'manifest.json',manifest)


if __name__ == '__main__':
    main()
