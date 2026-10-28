"""Run one selected experiment without executing earlier teaching steps."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
import numpy as np

from physical_model import PhysicalModel, dbm_to_w, w_to_dbm
from network_model import OpticalNetwork, maximum_snr_margin, toy_case


def save_results(output_dir, name, model, rows):
    # 同时保存数值和物理参数，便于核对图中的采样点
    links, state, requests = toy_case()
    payload = {
        "experiment": name,
        "physical_config": asdict(model.config),
        "baseline_links": [{"nodes": list(edge), "length_km": length}
                           for edge, length in links.items()],
        "baseline_free_wavelengths": [{"nodes": list(edge), "wavelengths": sorted(waves)}
                                      for edge, waves in state.items()],
        "requests": requests,
        "k_candidates": 4,
        "results": rows,
    }
    path = output_dir / f"{name}_results.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def finish_figure(fig, output_dir, name):
    fig.tight_layout()
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=200)
    print(f"Saved figure: {path.name}")


def experiment_baseline(model, network, state, requests, output_dir):
    power = model.optimal_power_w(10)
    print("\n--- Physical and routing baseline ---")
    print(f"alpha = {model.alpha_per_km:.6f} km^-1")
    print(f"Single-span X_spm = {model.x_spm:.3f} W^-2")
    for spacing_ghz in (400, 800, 1600):
        print(f"Spacing {spacing_ghz:4d} GHz: X_xpm = "
              f"{model.xpm_coefficient(spacing_ghz * 1e9):.3f} W^-2")
    print(f"Single-span XPM sum = {model.x_xpm_sum:.3f} W^-2")
    print(f"Single-span eta = {model.x_spm + model.x_xpm_sum:.3f} W^-2")
    print(f"800 km total ASE = {10 * model.ase_span_w * 1e6:.3f} uW")
    print(f"Analytical optimal power = {w_to_dbm(power):.3f} dBm")
    print(f"Analytical maximum SNR = {model.snr_db(10, power):.3f} dB")
    print(f"Ideal central-channel capacity = {model.ideal_capacity_bps(10, power) / 1e12:.6f} Tbit/s")
    rows = []
    for spans in (10, 20, 30, 40, 50):
        snr = model.snr_db(spans, power)
        modulation = model.best_modulation(snr)
        row = {"length_km": spans * model.config.span_length_km, "snr_db": snr,
               "modulation": modulation.name if modulation else None,
               "net_rate_gbps": model.net_rate_gbps(modulation) if modulation else 0.0}
        rows.append(row)
        print(f"L = {row['length_km']:4.0f} km: SNR = {snr:.3f} dB, "
              f"{row['modulation']}, Net rate = {row['net_rate_gbps']:.1f} Gbit/s")
    for route in network.candidate_routes("A", "D"):
        common = set.intersection(*(state[tuple(sorted((u, v)))]
                                    for u, v in zip(route, route[1:])))
        print(f"{' -> '.join(route)}: L = {network.route_length_km(route):.0f} km, "
              f"Common free wavelengths = {sorted(common)}")
    save_results(output_dir, "baseline", model, rows)


def experiment_physical(model, network, state, requests, output_dir):
    import matplotlib.pyplot as plt
    # 补回前期独立研究的 SNR-P、C-P、C-L 三种物理关系
    power_opt = model.optimal_power_w(10)
    powers_dbm = np.linspace(-10, 16, 261)
    snr = [model.snr_db(10, dbm_to_w(p)) for p in powers_dbm]
    capacities = [model.ideal_capacity_bps(10, dbm_to_w(p)) / 1e12 for p in powers_dbm]
    spans = list(range(1, 101))
    distances = [n * model.config.span_length_km for n in spans]
    capacity_distance = [model.ideal_capacity_bps(n, power_opt) / 1e12 for n in spans]
    print("\n--- Physical-layer relationships ---")
    for name, x, y, xlabel, ylabel in (
        ("snr_vs_power", powers_dbm, snr, "Common per-channel launch power (dBm)", "Central-channel SNR (dB)"),
        ("capacity_vs_power", powers_dbm, capacities, "Common per-channel launch power (dBm)", "Ideal central-channel capacity (Tbit/s)"),
        ("capacity_vs_distance", distances, capacity_distance, "Route length (km)", "Ideal central-channel capacity (Tbit/s)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(x, y)
        if name != "capacity_vs_distance":
            ax.axvline(w_to_dbm(power_opt), color="black", linestyle="--", label="Power maximizing SNR")
            ax.legend()
        ax.set(xlabel=xlabel, ylabel=ylabel,
               title="800 km physical baseline" if name != "capacity_vs_distance"
               else f"Common power = {w_to_dbm(power_opt):.3f} dBm")
        ax.grid(True, alpha=0.3)
        finish_figure(fig, output_dir, name)
    save_results(output_dir, "physical", model, {
        "power_dbm": powers_dbm.tolist(), "snr_db": snr, "capacity_tbit_s": capacities,
        "length_km": distances, "capacity_vs_distance_tbit_s": capacity_distance,
        "fixed_power_dbm": w_to_dbm(power_opt),
    })


def experiment_allocation(model, network, state, requests, output_dir):
    power = model.optimal_power_w(10)
    print("\n--- Candidate route count K: two A-to-D requests ---")
    for k in range(1, 5):
        trial = OpticalNetwork(model, network.links_km, k_candidates=k)
        result = trial.solve_sequential([("A", "D", 1000.0)] * 2, state, power)
        print(f"K = {k}: Accepted = {result.accepted_count}/2, "
              f"Carried traffic = {result.admitted_traffic_gbps:.1f} Gbit/s")
    rows = []
    print("\n--- Sequential rules and offline optima ---")
    for scenario in ("A", "B"):
        _, initial_state, _ = toy_case(scenario)
        for rule in ("min", "max"):
            result = network.solve_sequential(requests, initial_state, power, rule)
            rows.append({"scenario": scenario, "method": rule,
                         "admitted_traffic_gbps": result.admitted_traffic_gbps})
            print(f"Scenario {scenario}, rule = {rule}: Accepted = {result.accepted_count}/4, "
                  f"Carried traffic = {result.admitted_traffic_gbps:.1f} Gbit/s")
        exhaustive = network.solve_exhaustive(requests, initial_state, power)
        ilp = network.solve_ilp(requests, initial_state, power)
        print(f"Scenario {scenario}: Exhaustive optimum = {exhaustive.admitted_traffic_gbps:.1f}, "
              f"ILP optimum = {ilp.admitted_traffic_gbps:.1f} Gbit/s")
        print(f"  Binary variables = {ilp.variable_count}, Constraints = {ilp.constraint_count}")
        for index, option in enumerate(ilp.plan, start=1):
            detail = (f"{' -> '.join(option.route)}, Wavelength = {option.wavelength}, "
                      f"{option.modulation}" if option else "Not admitted")
            print(f"  Request {index}: {detail}")
        rows.append({"scenario": scenario, "method": "ilp",
                     "admitted_traffic_gbps": ilp.admitted_traffic_gbps,
                     "plan": [{"route": list(o.route), "wavelength": o.wavelength} if o else None
                              for o in ilp.plan]})
    save_results(output_dir, "allocation", model, rows)


def scan_power(model, network, state, requests, margin_db):
    powers = np.linspace(-10, 16, 261)
    traffic = np.array([network.optimal_admitted_traffic(requests, state, dbm_to_w(p), margin_db)
                        for p in powers])
    return powers, traffic


def experiment_power(model, network, state, requests, output_dir):
    import matplotlib.pyplot as plt
    powers, traffic = scan_power(model, network, state, requests, 0.0)
    print("\n--- Optimal admitted traffic versus launch power ---")
    for sample in (-10, -5, 0, 5, 10, 12, 13, 14):
        index = np.argmin(abs(powers - sample))
        print(f"P = {powers[index]:5.1f} dBm: Optimal admitted traffic = {traffic[index]:.1f} Gbit/s")
    best = powers[np.isclose(traffic, traffic.max())]
    print(f"Maximum admitted traffic = {traffic.max():.1f} Gbit/s")
    print(f"First/last power samples at maximum = {best[0]:.1f}, {best[-1]:.1f} dBm")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.step(powers, traffic, where="post", label="Optimal admitted traffic")
    ax.axvline(w_to_dbm(model.optimal_power_w(10)), color="black", linestyle="--", label="Reference power (800 km)")
    ax.set(xlabel="Common per-channel launch power (dBm)", ylabel="Optimal admitted traffic (Gbit/s)",
           title="Scenario A: admitted traffic versus launch power")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish_figure(fig, output_dir, "traffic_vs_power")
    save_results(output_dir, "power", model, {"power_dbm": powers.tolist(), "traffic_gbps": traffic.tolist()})


def experiment_margin(model, network, state, requests, output_dir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    rows = []
    print("\n--- Effect of required SNR margin ---")
    for margin in (0.0, 2.0, 4.0, 6.0):
        powers, traffic = scan_power(model, network, state, requests, margin)
        target_powers = powers[np.isclose(traffic, 3000.0)]
        print(f"Margin = {margin:.1f} dB: Maximum admitted traffic = {traffic.max():.1f} Gbit/s")
        if len(target_powers):
            print(f"  First/last power samples at 3000 Gbit/s = {target_powers[0]:.1f}, {target_powers[-1]:.1f} dBm")
        else:
            print("  Target of 3000 Gbit/s is not achievable.")
        ax.step(powers, traffic, where="post", label=f"SNR margin = {margin:.0f} dB")
        rows.append({"margin_db": margin, "power_dbm": powers.tolist(), "traffic_gbps": traffic.tolist()})
    ax.axvline(w_to_dbm(model.optimal_power_w(10)), color="black", linestyle="--", label="Reference power (800 km)")
    ax.set(xlabel="Common per-channel launch power (dBm)", ylabel="Optimal admitted traffic (Gbit/s)",
           title="Scenario A: impact of required SNR margin")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish_figure(fig, output_dir, "snr_margin")
    save_results(output_dir, "margin", model, rows)


def experiment_max_margin(model, network, state, requests, output_dir):
    power = model.optimal_power_w(10)
    margin = maximum_snr_margin(network, requests, state, power)
    print("\n--- Maximum SNR margin at target traffic ---")
    print("Target admitted traffic = 3000.0 Gbit/s")
    print(f"Common per-channel power = {w_to_dbm(power):.3f} dBm")
    print(f"Estimated maximum common SNR margin = {margin:.3f} dB")
    for sample in (4.39, 4.40):
        traffic = network.optimal_admitted_traffic(requests, state, power, sample)
        print(f"Margin = {sample:.2f} dB: Optimal admitted traffic = {traffic:.1f} Gbit/s")
    save_results(output_dir, "max_margin", model, {"power_dbm": w_to_dbm(power),
                 "target_gbps": 3000.0, "maximum_margin_db": margin})


def experiment_distance(model, network, state, requests, output_dir):
    import matplotlib.pyplot as plt
    scales = [1, 2, 3, 4, 5, 6]
    margins = [0.0, 4.0]
    power = model.optimal_power_w(10)
    results = {margin: [] for margin in margins}
    rows = []
    print("\n--- Optimal admitted traffic versus distance scale ---")
    print(f"Common per-channel power = {w_to_dbm(power):.3f} dBm")
    for scale in scales:
        trial = network.scaled(scale)
        print(f"\nDistance scale = {scale}")
        for margin in margins:
            traffic = trial.optimal_admitted_traffic(requests, state, power, margin)
            results[margin].append(traffic)
            rows.append({"distance_scale": scale, "snr_margin_db": margin,
                         "admitted_traffic_gbps": traffic})
            print(f"  SNR margin = {margin:.1f} dB: Optimal admitted traffic = {traffic:.1f} Gbit/s")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    # 连接线仅辅助观察整数距离采样点之间的趋势
    for margin in margins:
        ax.plot(scales, results[margin], marker="o", label=f"SNR margin = {margin:.0f} dB")
    ax.set(xlabel="Network distance scale factor", ylabel="Optimal admitted traffic (Gbit/s)",
           title="Scenario A: admitted traffic versus network distance", xticks=scales)
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish_figure(fig, output_dir, "traffic_vs_distance")
    save_results(output_dir, "distance", model, {"power_dbm": w_to_dbm(power), "samples": rows})


EXPERIMENTS = {
    "baseline": experiment_baseline, "physical": experiment_physical,
    "allocation": experiment_allocation, "power": experiment_power,
    "margin": experiment_margin, "max-margin": experiment_max_margin,
    "distance": experiment_distance,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=[*EXPERIMENTS, "all"], default="distance")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 参数在这里建立，导入模块时不会运行实验、打印或生成图片
    model = PhysicalModel()
    links, state, requests = toy_case()
    network = OpticalNetwork(model, links)
    chosen = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in chosen:
        EXPERIMENTS[name](model, network, state, requests, args.output_dir)
    if args.no_show:
        plt.close("all")
    elif plt.get_fignums():
        plt.show()


if __name__ == "__main__":
    main()
