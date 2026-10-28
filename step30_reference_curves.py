"""Regenerate SNR-P, ideal C-P and ideal C-L using the checked reference model.

Keep reference_physical_model.py beside this file. Earlier step scripts and
their JSON results are not required. All powers are per-channel launch powers.
Run: python step30_reference_curves.py
Batch run: python step30_reference_curves.py --no-show
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy

try:
    from reference_physical_model import ReferencePhysicalModel
except ModuleNotFoundError as exc:
    if exc.name == "reference_physical_model":
        raise SystemExit("Place reference_physical_model.py beside step30_reference_curves.py.") from exc
    raise


POWER_CURVE_SPANS = (10, 20, 50)
REFERENCE_SPANS = 10


def build_results(model):
    """Return power sweeps and distance sweeps with explicitly separated power policies."""
    powers = np.linspace(-12.0, 6.0, 901)
    fixed_power = model.optimal_power_dbm(REFERENCE_SPANS)
    power_curves = []
    for spans in POWER_CURVE_SPANS:
        power_curves.append({"spans": spans, "length_km": spans*model.p.span_length_km,
            "snr_db": model.snr_db(powers, spans).tolist(),
            "ideal_capacity_gbps": model.capacity_gbps(powers, spans).tolist(),
            "analytical_optimum": model.link_summary(spans)})
    distance_rows = []
    for spans in range(1, 101):
        optimum = model.link_summary(spans)
        fixed = model.link_summary(spans, fixed_power)
        incoherent = model.link_summary(spans, accumulation="incoherent")
        distance_rows.append({"spans": spans, "length_km": spans*model.p.span_length_km,
            "coherent_spm_own_optimum": optimum, "coherent_spm_fixed_reference_power": fixed,
            "incoherent_own_optimum": incoherent,
            "capacity_gain_from_retuning_gbps": optimum["ideal_capacity_gbps"]-fixed["ideal_capacity_gbps"],
            "snr_gain_from_retuning_db": optimum["snr_db"]-fixed["snr_db"]})
        if spans % 25 == 0:
            print(f"  Reference distance sweep completed: {spans}/100 span counts.", flush=True)
    return powers, fixed_power, power_curves, distance_rows


def consistency_checks(model, powers, curves, rows, fixed_power):
    """Check physical identities, discrete sweep maxima and distance interpretation."""
    checks = {}
    peak_checks = []
    power_step = float(powers[1] - powers[0])
    for curve in curves:
        snr_index = int(np.argmax(curve["snr_db"]))
        capacity_index = int(np.argmax(curve["ideal_capacity_gbps"]))
        target = curve["analytical_optimum"]["optimal_power_dbm"]
        discrepancy = abs(float(powers[snr_index])-target)
        if snr_index != capacity_index or discrepancy > power_step/2 + 1e-9:
            raise RuntimeError("Power-sweep SNR/capacity maxima disagree with the analytical optimum.")
        peak_checks.append({"spans": curve["spans"], "same_snr_and_capacity_peak": True,
                           "power_grid_step_db": power_step, "grid_vs_analytic_power_difference_db": discrepancy})
    checks["power_sweep_extrema"] = peak_checks

    ratios = [row["coherent_spm_own_optimum"]["noise_w"]["total_nli_w"] /
              row["coherent_spm_own_optimum"]["noise_w"]["ase_w"] for row in rows]
    maximum_ratio_error = max(abs(ratio-0.5) for ratio in ratios)
    if maximum_ratio_error > 1e-12:
        raise RuntimeError("At the analytical optimum, total NLI / ASE must equal 1/2.")
    checks["maximum_nli_over_ase_deviation_from_half"] = maximum_ratio_error

    gaps = [row["capacity_gain_from_retuning_gbps"] for row in rows]
    if min(gaps) < -1e-9 or abs(gaps[REFERENCE_SPANS-1]) > 1e-9:
        raise RuntimeError("The fixed-power / optimized-power capacity comparison is inconsistent.")
    checks["optimized_capacity_never_below_fixed_capacity"] = True
    checks["zero_retuning_gain_at_reference_distance"] = True

    single_span_difference = abs(model.eta_total(1)-model.eta_total(1, accumulation="incoherent"))
    if single_span_difference > 1e-12:
        raise RuntimeError("Coherent and incoherent models must coincide at one span.")
    checks["one_span_coherent_and_incoherent_agree"] = True
    doubled = (model.snr_db(fixed_power, 20, accumulation="incoherent")
               - model.snr_db(fixed_power, 10, accumulation="incoherent"))
    if abs(doubled + 10*math.log10(2)) > 1e-12:
        raise RuntimeError("The incoherent comparison failed its span-doubling identity.")
    checks["incoherent_span_doubling_snr_change_db"] = doubled
    if model.spans_for_distance(800.0) != 10:
        raise RuntimeError("Span-count conversion failed.")
    try:
        model.spans_for_distance(801.0)
    except ValueError:
        checks["partial_span_distance_rejected"] = True
    else:
        raise RuntimeError("A partial-span distance must not be silently rounded.")
    return checks


def make_plot(powers, fixed_power, curves, rows, figure_path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8), layout="constrained")
    axes = axes.ravel()
    blue, orange, green, gray = "#17476b", "#bd682b", "#25867a", "#70777d"
    for curve, color in zip(curves, (blue, orange, green)):
        optimum = curve["analytical_optimum"]
        label = f"{curve['length_km']:.0f} km"
        axes[0].plot(powers, curve["snr_db"], color=color, label=label, linewidth=1.8)
        axes[1].plot(powers, curve["ideal_capacity_gbps"], color=color, label=label, linewidth=1.8)
        axes[0].plot(optimum["optimal_power_dbm"], optimum["snr_db"], "*", color=color, markersize=11)
        axes[1].plot(optimum["optimal_power_dbm"], optimum["ideal_capacity_gbps"], "*", color=color, markersize=11)
    axes[0].set(title="SNR versus launch power", xlabel="Per-channel launch power (dBm)", ylabel="SNR (dB)")
    axes[1].set(title="Ideal capacity versus launch power", xlabel="Per-channel launch power (dBm)",
                ylabel="Ideal capacity per channel (Gbit/s)")
    for ax in axes[:2]:
        ax.legend(title="Stars: analytical optima", fontsize=9, title_fontsize=9)

    distances = [r["length_km"] for r in rows]
    axes[2].plot(distances, [r["incoherent_own_optimum"]["ideal_capacity_gbps"] for r in rows],
                 color=gray, linestyle=":", linewidth=1.7, label="Incoherent comparison, optimized P")
    axes[2].plot(distances, [r["coherent_spm_own_optimum"]["ideal_capacity_gbps"] for r in rows],
                 color=blue, linewidth=2.4, label="Reference model, optimized P(L)")
    axes[2].plot(distances, [r["coherent_spm_fixed_reference_power"]["ideal_capacity_gbps"] for r in rows],
                 color=orange, linestyle="--", linewidth=1.4, label="Reference model, fixed P")
    maximum_gain = max(r["capacity_gain_from_retuning_gbps"] for r in rows)
    axes[2].text(0.97, 0.97, f"Fixed P = {fixed_power:.3f} dBm (800 km optimum)\n"
                  f"Max retuning gain = {maximum_gain:.3f} Gbit/s", transform=axes[2].transAxes,
                  ha="right", va="top", fontsize=8.5,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    axes[2].set(title="Ideal capacity versus distance", xlabel="Distance (km)",
                ylabel="Ideal capacity per channel (Gbit/s)")
    axes[2].legend(loc="lower left", fontsize=8)

    axes[3].plot(distances, [r["coherent_spm_own_optimum"]["optimal_power_dbm"] for r in rows],
                 color=blue, linewidth=1.9, label="Reference model: coherent SPM + linear XPM")
    axes[3].plot(distances, [r["incoherent_own_optimum"]["optimal_power_dbm"] for r in rows],
                 color=gray, linestyle=":", linewidth=1.7, label="Incoherent comparison")
    axes[3].axhline(fixed_power, color=orange, linestyle="--", linewidth=1.5, label="Fixed 800 km reference power")
    axes[3].set(title="Optimal launch power versus distance", xlabel="Distance (km)",
                ylabel="Optimal per-channel power (dBm)")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.23)
    fig.suptitle("Step 30 | Unified reference physical layer\n"
                 "32 GBd, 50 GHz spacing, 80 channels, 80 km spans; central channel", fontsize=13)
    fig.savefig(figure_path, dpi=180)
    if show:
        print("Close the figure window to finish.", flush=True)
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_step30")
    args = parser.parse_args()
    started = time.perf_counter()
    model = ReferencePhysicalModel()
    print("--- Step 30: unified reference physical layer ---", flush=True)
    print("Source: Ives (2015), Table 3.1 and the matched-filter SPM/XPM GN formulation.")
    print("Baseline: numerically integrated coherent SPM + linearly accumulated numerical XPM.")
    print("No fixed epsilon is supplied. All powers are per-channel launch powers.")
    print("C denotes ideal dual-polarization AWGN capacity of one channel, not a net data rate.")
    print("32 GBd, 50 GHz spacing, 80 channels, identical 80 km spans; full-load central channel.")
    print("Preparing coefficients and power/distance curves...", flush=True)
    powers, fixed_power, curves, rows = build_results(model)
    checks = consistency_checks(model, powers, curves, rows, fixed_power)
    print("\n--- Numerical single-span baseline ---")
    print(f"ASE per span = {model.ase_span_w*1e6:.9f} uW")
    print(f"SPM coefficient = {model.spm_coefficient(1):.6f} W^-2")
    print(f"XPM sum = {model.single_span_xpm_sum:.6f} W^-2")
    print(f"Total single-span eta = {model.eta_total(1):.6f} W^-2")

    print("\n--- SNR and ideal capacity versus power at 800 km ---")
    print("Power (dBm)      SNR (dB)    Ideal capacity (Gbit/s)")
    for power in (-10.0, -5.0, fixed_power, 0.0, 3.0, 6.0):
        print(f"{power:11.6f} {model.snr_db(power, REFERENCE_SPANS):13.6f} "
              f"{model.capacity_gbps(power, REFERENCE_SPANS):26.6f}")

    print("\n--- Distance dependence: fixed power versus re-optimized power ---")
    print(f"Fixed power = {fixed_power:.6f} dBm, the coherent-SPM model's 800 km optimum.")
    print("All C values below are per-channel ideal capacities in Gbit/s.")
    print("L (km)   Popt (dBm)   SNR at Popt (dB)   C at Popt   C at fixed P   Retuning gain")
    for spans in (1, 10, 20, 50, 100):
        row = rows[spans-1]
        optimum, fixed = row["coherent_spm_own_optimum"], row["coherent_spm_fixed_reference_power"]
        print(f"{row['length_km']:6.0f} {optimum['optimal_power_dbm']:12.6f} {optimum['snr_db']:18.6f} "
              f"{optimum['ideal_capacity_gbps']:11.6f} {fixed['ideal_capacity_gbps']:14.6f} "
              f"{row['capacity_gain_from_retuning_gbps']:15.6f}")
    maximum_gain_row = max(rows, key=lambda r: r["capacity_gain_from_retuning_gbps"])
    doubled_at_optimum = rows[19]["coherent_spm_own_optimum"]["snr_db"]-rows[9]["coherent_spm_own_optimum"]["snr_db"]
    print(f"Largest retuning gain on this 80..8000 km grid = {maximum_gain_row['capacity_gain_from_retuning_gbps']:.6f} Gbit/s "
          f"at {maximum_gain_row['length_km']:.0f} km.")
    print(f"800 -> 1600 km, each at own optimum: SNR change = {doubled_at_optimum:.6f} dB.")

    print("\n--- Consistency checks ---")
    print("PASS: SNR-P and C-P sweeps have the same maximizing power sample.")
    print("PASS: sweep maxima agree with analytical optima within half the 0.02 dB grid step.")
    print("PASS: total NLI / ASE = 1/2 at each analytical optimum.")
    print("PASS: optimized capacity is never below fixed-power capacity; both coincide at 800 km.")
    print("PASS: coherent and incoherent modes coincide at one span.")
    print("PASS: incoherent span doubling gives -3.010300 dB; partial-span distances are rejected.")
    print("These checks establish model consistency, not measured transmission accuracy.")
    elapsed = time.perf_counter()-started
    print(f"Calculation time: {elapsed:.2f} s.")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"source": "Ives (2015), Table 3.1 and Eqs. (4.15), (4.18)-(4.23), (4.33)",
        "parameters": asdict(model.p), "observed_channel_index": model.channel_index,
        "model": "Numerical coherent SPM + linearly accumulated numerical XPM; no fixed epsilon",
        "capacity_definition": "2 * symbol_rate * log2(1 + linear_SNR), one dual-polarization channel",
        "scope": "Identical contiguous spans, equal channel powers, full-load SPM/XPM decomposition; no additional multi-channel mixing, Raman, ROADM or transceiver terms",
        "reference_span_count": REFERENCE_SPANS, "fixed_reference_power_dbm": fixed_power,
        "power_samples_dbm": powers.tolist(), "power_curves": curves, "distance_rows": rows,
        "maximum_capacity_gain_from_retuning_gbps": maximum_gain_row["capacity_gain_from_retuning_gbps"],
        "distance_at_maximum_retuning_gain_km": maximum_gain_row["length_km"],
        "snr_change_800_to_1600_km_at_own_optima_db": doubled_at_optimum,
        "checks": checks, "quadrature_refinement_target_db": model.tolerance_db,
        "refinement_is_certified_error_bound": False,
        "quadrature": model.quadrature_records(range(1,101)), "calculation_seconds": elapsed,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    json_path = output / "reference_curves_results.json"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure_path = output / "reference_curves.png"
    print(f"Saved numerical results: {json_path}")
    print(f"Figure destination: {figure_path}")
    make_plot(powers, fixed_power, curves, rows, figure_path, show=not args.no_show)


if __name__ == "__main__":
    main()
