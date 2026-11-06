"""Compare uniform/hotspot and fixed/mixed traffic on a synthetic 12-node mesh.

Run: python step32_network_traffic.py --no-show
All loads are static request batches, not dynamic Erlang loads.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time

import networkx as nx
import numpy as np
import scipy

from network_scenarios import SCENARIOS, generate_requests, mesh12_case, pair_probabilities
from reference_network import ReferenceNetworkPhysical, result_record
from optical_networking_refactored.network_model import OpticalNetwork
from optical_networking_refactored.physical_model import w_to_dbm


METHODS = ("SP-FF", "KSP-FF", "ILP")
LABELS = {"uniform_fixed": "Uniform OD / fixed rate", "hotspot_fixed": "Hotspot OD / fixed rate",
          "uniform_mixed": "Uniform OD / mixed rates", "hotspot_mixed": "Hotspot OD / mixed rates"}
COLORS = {"SP-FF": "#99743d", "KSP-FF": "#247e93", "ILP": "#6d4c91"}


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["request_count"], row["method"])].append(row)
    summary = []
    for (scenario, count, method), group in sorted(grouped.items()):
        entry = {"scenario": scenario, "request_count": count, "method": method,
                 "seed_count": len(group), "proven_optimal_count": sum(r["optimal"] is True for r in group)}
        for metric in ("admitted_traffic_gbps", "request_blocking_fraction", "bandwidth_blocking_fraction", "elapsed_seconds"):
            values = np.array([r[metric] for r in group])
            entry[metric + "_mean"] = float(values.mean())
            entry[metric + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(entry)
    return summary


def run(args):
    started = time.perf_counter()
    model = ReferenceNetworkPhysical()
    links, state, positions = mesh12_case(args.wavelengths, model.config.channel_count)
    nodes = sorted(positions)
    power = model.optimal_power_w(10)
    # 所有方法共享已预热的物理查表；各自的建图、路径枚举和求解仍计入运行时间
    for spans in range(1, int(sum(links.values()) / model.config.span_length_km) + 1):
        model.snr_db(spans, power)
    warmup_seconds = time.perf_counter() - started
    probabilities = {}
    for name, hotspot in (("uniform", None), ("hotspot", "F")):
        pairs, weights = pair_probabilities(nodes, hotspot, args.hotspot_share)
        probabilities[name] = [{"nodes": list(pair), "probability": float(p)} for pair, p in zip(pairs, weights)]
    payload = {"experiment": "step32_mesh12_traffic", "physical": model.metadata(),
               "topology_name": "Synthetic 3-by-4 grid mesh; no geographic interpretation",
               "nodes": nodes, "positions": positions,
               "links": [{"nodes": list(e), "length_km": length, "available_wavelengths": sorted(state[e])}
                         for e, length in links.items()],
               "physical_channels": model.config.channel_count, "allocatable_wavelengths_per_link": args.wavelengths,
               "resource_scope": "Remaining physical channels are unavailable to these requests; full-load noise is retained",
               "common_power_dbm": w_to_dbm(power), "reference_power_distance_km": 800,
               "snr_margin_db": args.margin_db, "k_candidates": args.k, "seed_values": list(range(args.seeds)),
               "request_counts": args.loads, "hotspot_node": "F", "hotspot_pair_probability": args.hotspot_share,
               "pair_probabilities": probabilities,
               "traffic": {"type": "Static batch without departures; offered load is total requested Gbit/s",
                           "fixed_rate_gbps": 100, "mixed_rates_gbps": [50, 100, 150],
                           "mixed_fractions": [0.25, 0.5, 0.25], "total_demand": "100 * request_count for every batch",
                           "pairing": "Same OD sequence and arrival order for fixed/mixed counterparts; separate OD/rate RNG streams"},
               "ilp_solver_time_limit_seconds": args.ilp_seconds,
               "timing": "Per-instance construction + candidate generation + solve; excludes shared physics warmup and result audit",
               "physical_warmup_seconds": warmup_seconds,
               "limitations": ["Synthetic topology and static batch; blocking fractions are not steady-state probabilities",
                               "ILP optimality is restricted to the K candidate routes and the fixed physical assumptions",
                               "Time-limited ILP incumbents are feasible benchmarks, not certified optima",
                               "Error bars are sample standard deviations across seeds, not confidence intervals"],
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "networkx": nx.__version__},
               "instances": [], "rows": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        for count in args.loads:
            for seed in range(args.seeds):
                requests = generate_requests(nodes, count, seed, scenario, hotspot_share=args.hotspot_share)
                instance_id = f"{scenario}-{count}-{seed}"
                encoded = json.dumps(requests, separators=(",", ":")).encode()
                payload["instances"].append({"id": instance_id, "scenario": scenario, "seed": seed,
                                             "requests": requests, "sha256": hashlib.sha256(encoded).hexdigest()})
                instance_rows = []
                for method in METHODS:
                    start = time.perf_counter()
                    network = OpticalNetwork(model, links, k_candidates=1 if method == "SP-FF" else args.k)
                    if method == "ILP":
                        result = network.solve_ilp(requests, state, power, args.margin_db, time_limit_s=args.ilp_seconds)
                    else:
                        result = network.solve_sequential(requests, state, power, "min", args.margin_db)
                    elapsed = time.perf_counter() - start
                    record = result_record(network, requests, state, power, args.margin_db, result)
                    row = {"instance_id": instance_id, "scenario": scenario, "seed": seed,
                           "method": method, "elapsed_seconds": elapsed, **record}
                    instance_rows.append(row)
                ilp = instance_rows[-1]
                for row in instance_rows[:-1]:
                    gap = ilp["admitted_traffic_gbps"] - row["admitted_traffic_gbps"]
                    if ilp["optimal"] and gap < -1e-7:
                        raise AssertionError("A proven ILP optimum is below a feasible heuristic on the same candidates.")
                    row["gap_to_proven_optimum_gbps"] = gap if ilp["optimal"] else None
                payload["rows"].extend(instance_rows)
            group = [r for r in payload["rows"] if r["scenario"] == scenario and r["request_count"] == count]
            proven = sum(r["optimal"] is True for r in group if r["method"] == "ILP")
            print(f"{scenario:14} requests={count:3}: completed {args.seeds} seeds; ILP proven {proven}/{args.seeds}", flush=True)
            # 每个负载档完成后保存，保留已完成实例与求解状态
            payload["summary"] = summarize(payload["rows"])
            payload["complete"] = False
            save(payload, args.output_dir)
    payload["complete"] = True
    payload["total_elapsed_seconds"] = time.perf_counter() - started
    save(payload, args.output_dir)
    return payload


def save(payload, output_dir):
    (output_dir / "traffic_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    compact = {key: value for key, value in payload.items() if key not in ("rows", "instances")}
    (output_dir / "traffic_summary.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def certify_existing(output_dir, time_limit_s):
    """Retry only unproven ILPs; preserve the initial run and save a derived dataset."""
    started = time.perf_counter()
    data = json.loads((output_dir / "traffic_results.json").read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise ValueError("Finish the initial experiment before certifying its remaining ILPs.")
    # 从保存的配置重建，不能让当前 CLI 默认值改变原实验
    from reference_physical_model import ReferenceParameters, dbm_to_w
    model = ReferenceNetworkPhysical(ReferenceParameters(**data["physical"]["parameters"]),
                                     data["physical"]["fec_overhead"])
    links = {tuple(r["nodes"]): r["length_km"] for r in data["links"]}
    state = {tuple(r["nodes"]): set(r["available_wavelengths"]) for r in data["links"]}
    instances = {r["id"]: r for r in data["instances"]}
    power = dbm_to_w(data["common_power_dbm"])
    for spans in range(1, int(sum(links.values()) / model.config.span_length_km) + 1):
        model.snr_db(spans, power)
    history = []
    for row in data["rows"]:
        if row["method"] != "ILP" or row["optimal"]:
            continue
        before = dict(row)
        requests = instances[row["instance_id"]]["requests"]
        start = time.perf_counter()
        network = OpticalNetwork(model, links, data["k_candidates"])
        result = network.solve_ilp(requests, state, power, data["snr_margin_db"], time_limit_s=time_limit_s)
        elapsed = time.perf_counter() - start
        after = result_record(network, requests, state, power, data["snr_margin_db"], result)
        history.append({"instance_id": row["instance_id"], "initial_result": before,
                        "retry_result": after, "retry_elapsed_seconds": elapsed})
        # 未完成证明时仍保留较好的可行解，并收紧已知上界
        if after["optimal"] or after["admitted_traffic_gbps"] >= row["admitted_traffic_gbps"]:
            row.update(after)
        row["upper_bound_gbps"] = max(row["admitted_traffic_gbps"],
                                       min(before["upper_bound_gbps"], after["upper_bound_gbps"]))
        row["elapsed_seconds"] = before["elapsed_seconds"] + elapsed
        row["mip_gap"] = ((row["upper_bound_gbps"] - row["admitted_traffic_gbps"]) / row["admitted_traffic_gbps"]
                          if row["admitted_traffic_gbps"] else None)
        print(f"Retry {row['instance_id']}: {row['solver_status']}, {row['admitted_traffic_gbps']:g} Gbit/s", flush=True)
    by_instance = {r["instance_id"]: r for r in data["rows"] if r["method"] == "ILP"}
    for row in data["rows"]:
        if row["method"] == "ILP":
            continue
        ilp = by_instance[row["instance_id"]]
        gap = ilp["admitted_traffic_gbps"] - row["admitted_traffic_gbps"]
        if ilp["optimal"] and gap < -1e-7:
            raise AssertionError("Certified optimum is below a feasible heuristic.")
        row["gap_to_proven_optimum_gbps"] = gap if ilp["optimal"] else None
    data["certification"] = {"source": str((output_dir / "traffic_results.json").resolve()),
                             "retry_time_limit_s": time_limit_s, "history": history,
                             "all_ilps_proven": all(r["optimal"] for r in by_instance.values())}
    data["timing"] += "; retried ILPs include initial run + restart time"
    data["initial_total_elapsed_seconds"] = data["total_elapsed_seconds"]
    data["certification_elapsed_seconds"] = time.perf_counter() - started
    data["total_elapsed_seconds"] += data["certification_elapsed_seconds"]
    data["summary"] = summarize(data["rows"])
    destination = output_dir / "certified"
    destination.mkdir(parents=True, exist_ok=True)
    save(data, destination)
    return data, destination


def make_plots(data, output_dir, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    ilps = [r for r in data["rows"] if r["method"] == "ILP"]
    ilp_label = "ILP optimum" if all(r["optimal"] for r in ilps) else "ILP incumbent"
    for metric, ylabel, filename in (("bandwidth_blocking_fraction", "Blocked demand / offered demand", "traffic_comparison.png"),
                                     ("request_blocking_fraction", "Blocked requests / offered requests", "request_blocking.png"),
                                     ("elapsed_seconds", "Construction + allocation time (s)", "runtime.png")):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained", sharex=True, sharey=True)
        for ax, scenario in zip(axes.flat, SCENARIOS):
            for method in METHODS:
                rows = sorted((r for r in data["summary"] if r["scenario"] == scenario and r["method"] == method),
                              key=lambda r: r["request_count"])
                x = [r["request_count"] / 10 for r in rows]
                y = [r[metric + "_mean"] for r in rows]
                label = ilp_label if method == "ILP" else method
                if metric == "elapsed_seconds":
                    # 对数轴上展示每个实际耗时，避免均值减标准差为负时截断误差棒
                    raw = [r for r in data["rows"] if r["scenario"] == scenario and r["method"] == method]
                    ax.scatter([r["request_count"] / 10 for r in raw], [r[metric] for r in raw],
                               color=COLORS[method], alpha=0.25, s=18)
                    ax.plot(x, y, color=COLORS[method], marker="o", label=label)
                else:
                    ax.errorbar(x, y, yerr=[r[metric + "_sd"] for r in rows],
                                color=COLORS[method], marker="o", capsize=3, label=label)
            ax.set(title=LABELS[scenario], xlabel="Offered demand (Tbit/s)", ylabel=ylabel,
                   xticks=[n / 10 for n in data["request_counts"]])
            if metric == "elapsed_seconds":
                ax.set_yscale("log")
            else:
                ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        spread = "dots = individual seeds; lines = means" if metric == "elapsed_seconds" else "bars = sample SD"
        title = (f"12 nodes / 17 links / {data['allocatable_wavelengths_per_link']} allocatable wavelengths per link\n"
                 f"{len(data['seed_values'])} seeds; {spread}; ILP proven optimal: "
                 f"{sum(r['optimal'] for r in ilps)}/{len(ilps)}")
        if metric == "elapsed_seconds" and "certification" in data:
            title += "\nRetried ILPs include initial + retry time"
        fig.suptitle(title, fontsize=12)
        fig.savefig(output_dir / filename, dpi=180)
        if not show:
            plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    graph = nx.Graph()
    graph.add_edges_from(tuple(row["nodes"]) for row in data["links"])
    colors = ["#df9d4f" if node == "F" else "#c9e3eb" for node in graph.nodes]
    nx.draw_networkx(graph, pos=data["positions"], ax=axes[0], node_color=colors, node_size=480, font_size=10)
    nx.draw_networkx_edge_labels(graph, data["positions"], ax=axes[0], font_size=8,
                                edge_labels={tuple(r["nodes"]): f"{r['length_km']:g}" for r in data["links"]})
    axes[0].set_title("Synthetic mesh; edge labels in km\nF is the hotspot")
    axes[0].set_axis_off()
    nodes = data["nodes"]
    vmax = max(r["probability"] for rows in data["pair_probabilities"].values() for r in rows)
    for ax, name in zip(axes[1:], ("uniform", "hotspot")):
        matrix = np.zeros((len(nodes), len(nodes)))
        for row in data["pair_probabilities"][name]:
            i, j = (nodes.index(n) for n in row["nodes"])
            matrix[i, j] = row["probability"]
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        ax.set(xticks=range(len(nodes)), yticks=range(len(nodes)), xticklabels=nodes, yticklabels=nodes,
               title=f"{name.title()} OD probabilities\nUnordered pairs shown once")
    fig.colorbar(im, ax=axes[1:], label="Probability of an unordered pair", shrink=0.8)
    fig.savefig(output_dir / "topology_and_traffic.png", dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--certify-existing", action="store_true",
                        help="Retry only unproven ILPs from output-dir; save derived results in its certified subfolder.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--loads", type=int, nargs="+", default=[24, 48, 72, 96])
    parser.add_argument("--wavelengths", type=int, default=8)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--margin-db", type=float, default=2.0)
    parser.add_argument("--hotspot-share", type=float, default=0.65)
    parser.add_argument("--ilp-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_step32")
    args = parser.parse_args()
    if args.seeds < 1 or args.k < 1 or any(n <= 0 or n % 4 for n in args.loads):
        parser.error("Use positive seeds/k and positive request counts divisible by four.")
    if len(set(args.loads)) != len(args.loads):
        parser.error("Request-count levels must be unique.")
    if not np.isfinite(args.margin_db) or args.margin_db < 0:
        parser.error("Margin must be finite and nonnegative.")
    if not np.isfinite(args.ilp_seconds) or args.ilp_seconds <= 0:
        parser.error("ILP time limit must be finite and positive.")
    if not np.isfinite(args.hotspot_share) or not 0 <= args.hotspot_share <= 1:
        parser.error("Hotspot share must lie in [0, 1].")
    args.loads = sorted(args.loads)
    if args.certify_existing:
        data, destination = certify_existing(args.output_dir, args.ilp_seconds)
    else:
        data, destination = run(args), args.output_dir
    make_plots(data, destination, not args.no_show)
    print(f"Saved {len(data['rows'])} allocation results in {destination.resolve()}")


if __name__ == "__main__":
    main()
