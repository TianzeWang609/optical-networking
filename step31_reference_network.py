"""Recompute the four-node allocation experiments with the 32 GBd reference model.

Run: python step31_reference_network.py --no-show
"""

import argparse
import json
from pathlib import Path
import platform

import numpy as np

from reference_network import ReferenceNetworkPhysical, reference_toy_case, result_record
from reference_physical_model import dbm_to_w
from optical_networking_refactored.physical_model import w_to_dbm
from optical_networking_refactored.network_model import OpticalNetwork, maximum_snr_margin


def build_results():
    model = ReferenceNetworkPhysical()
    power = model.optimal_power_w(10)
    links, state, requests = reference_toy_case()
    network = OpticalNetwork(model, links)
    allocations = []
    for scenario in ("A", "B"):
        _, initial, batch = reference_toy_case(scenario)
        for method in ("min", "max", "exhaustive", "ilp"):
            if method in ("min", "max"):
                result = network.solve_sequential(batch, initial, power, method)
            else:
                result = getattr(network, "solve_" + method)(batch, initial, power)
            allocations.append({"scenario": scenario, "method": method,
                                **result_record(network, batch, initial, power, 0.0, result)})
    routes = []
    for path in network.candidate_routes("A", "D"):
        spans = network.route_spans(path)
        snr = model.snr_db(spans, power)
        modulation = model.best_modulation(snr)
        routes.append({"route": path, "length_km": network.route_length_km(path),
                       "snr_db": snr, "modulation": modulation.name,
                       "net_rate_gbps": model.net_rate_gbps(modulation),
                       "ideal_capacity_gbps": model.ideal_capacity_bps(spans, power) / 1e9})
    powers = np.linspace(-10, 6, 161)
    margins = []
    for margin in (0.0, 2.0, 4.0, 6.0):
        traffic = [network.optimal_admitted_traffic(requests, state, dbm_to_w(p), margin) for p in powers]
        target_samples = powers[np.isclose(traffic, 480.0)]
        margins.append({"margin_db": margin, "traffic_gbps": traffic,
                        "max_traffic_gbps": max(traffic),
                        "target_480_power_range_dbm": [float(target_samples[0]), float(target_samples[-1])]
                        if len(target_samples) else None})
    maximum_margin = maximum_snr_margin(network, requests, state, power, target_gbps=480.0)
    analytic_margin = model.snr_db(20, power) - 10.81
    if not 0 <= analytic_margin - maximum_margin < 1e-4:
        raise AssertionError("Fixed-power ILP margin disagrees with the bottleneck route.")
    distances = []
    for scale in range(1, 7):
        for margin in (0.0, 4.0):
            trial = network.scaled(scale)
            result = trial.solve_ilp(requests, state, power, margin)
            distances.append({"distance_scale": scale, "margin_db": margin,
                              **result_record(trial, requests, state, power, margin, result)})
    old_requests = [(s, t, 1000.0) for s, t, _ in requests]
    old_demand_result = network.solve_ilp(old_requests, state, power)
    assert old_demand_result.admitted_traffic_gbps == 0
    return {"experiment": "step31_reference_network", "physical": model.metadata(),
            "topology": [{"nodes": list(e), "length_km": d, "free_wavelengths": sorted(state[e])}
                         for e, d in links.items()], "requests": requests, "k_candidates": 4,
            "demand_choice": "160 Gbit/s = original 1000 * (32/200); preserves the PM-8QAM service requirement",
            "common_power_dbm": w_to_dbm(power), "power_policy": "Fixed at the 800 km optimum",
            "allocations": allocations, "a_to_d_routes": routes,
            "power_dbm": powers.tolist(), "margin_sweeps": margins,
            "maximum_margin_for_480_gbps": maximum_margin, "analytic_margin_db": analytic_margin,
            "distance_sweep": distances, "old_1000_gbps_demands_admitted_gbps": 0.0,
            "python_version": platform.python_version()}


def make_plot(data, path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for row in data["margin_sweeps"]:
        axes[0].step(data["power_dbm"], row["traffic_gbps"], where="post", label=f"Margin {row['margin_db']:g} dB")
    axes[0].axvline(data["common_power_dbm"], color="gray", linestyle="--", label="800 km reference power")
    axes[0].set(xlabel="Common per-channel power (dBm)", ylabel="Admitted demand (Gbit/s)",
                title="Four-node network: 4 x 160 Gbit/s requests")
    for margin in (0.0, 4.0):
        rows = [r for r in data["distance_sweep"] if r["margin_db"] == margin]
        axes[1].plot([r["distance_scale"] for r in rows], [r["admitted_traffic_gbps"] for r in rows],
                     "o-", label=f"Margin {margin:g} dB")
    axes[1].set(xlabel="Link-distance scale", ylabel="Admitted demand (Gbit/s)",
                title=f"Fixed common power {data['common_power_dbm']:.3f} dBm")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_step31")
    args = parser.parse_args()
    data = build_results()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference_network_results.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in data["allocations"]:
        print(f"{row['scenario']} / {row['method']:10}: {row['admitted_traffic_gbps']:.1f} Gbit/s")
    print(f"Maximum fixed-power margin for 480 Gbit/s: {data['maximum_margin_for_480_gbps']:.6f} dB")
    for row in data["margin_sweeps"]:
        print(f"Margin {row['margin_db']:g} dB: peak {row['max_traffic_gbps']:g} Gbit/s; "
              f"480 Gbit/s power samples {row['target_480_power_range_dbm']}")
    make_plot(data, args.output_dir / "reference_network.png", not args.no_show)
    print(f"Saved results in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
