"""Check the linear XPM accumulation approximation under thesis parameters.

Source: D. J. Ives (2015), Coherent Optical Fibre Networking in the Nonlinear
Regime, UCL PhD thesis: Eqs. (4.18)-(4.22), pp. 63-65, and Eqs. (4.32)-(4.33),
p. 68. Rectangular spectra and identical contiguous 80 km spans are assumed.
All 79 interferers co-propagate with the observed central channel throughout.

The coherent XPM integral includes the phased-array factor. Its exact 1D
reduction is checked against Step 27 and independent 2D integration.
Both link comparisons retain the same numerically integrated coherent SPM.
This is an SPM/XPM GN-model check, not a full multi-channel GN or experimental
validation; multi-channel mixing terms outside this decomposition are omitted.

Keep the Step 26, Step 27 and Step 28 Python scripts beside this file.
Run: python step29_xpm_coherence.py
Batch run: python step29_xpm_coherence.py --no-show
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy

try:
    from step26_reference_physics import ClosedFormGN, Parameters
    from step27_nli_integral import unit_rule, span_kernel, converged_coefficient
    from step28_spm_coherence import phased_array_factor, converged_spm, link_at_optimum
except ModuleNotFoundError as exc:
    if exc.name in ("step26_reference_physics", "step27_nli_integral", "step28_spm_coherence"):
        raise SystemExit("Place the Step 26, Step 27 and Step 28 scripts beside step29_xpm_coherence.py.") from exc
    raise


SPAN_COUNTS = (1, 2, 5, 10, 20, 50, 100)
PANEL_ORDERS = (8, 16, 32, 64, 128)
TWO_D_ORDERS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


def log1p_remainder(z):
    """Evaluate log(1+z)-z accurately for the XPM overlap weight."""
    result = np.asarray(np.log1p(z) - z)
    small = np.abs(z) < 0.05
    a = z[small]
    # 小参数下采用级数，避免 log1p(z) 与 z 相减导致精度损失
    result[small] = a*a * (-1/2 + a*(1/3 + a*(-1/4 + a*(1/5 + a*(-1/6 +
                       a*(1/7 + a*(-1/8 + a*(1/9 + a*(-1/10 + a/11)))))))))
    return result


def xpm_product_weight(t, delta):
    """Exact product-frequency weight H_delta(t), for delta > 1 and 0 <= t <= delta."""
    if not math.isfinite(delta) or delta <= 1:
        raise ValueError("This reduction requires channel separation / symbol rate > 1.")
    t = np.asarray(t, dtype=float)
    if np.any(t < 0) or np.any(t > delta):
        raise ValueError("Product-frequency coordinates must lie in [0, delta].")
    # u<0 与 u>0 两个分支分别积分；有理化根差保证端点附近稳定
    root_left = np.sqrt((delta - 1)**2 + 4*t)
    root_right = np.sqrt((delta + 1)**2 - 4*t)
    z_left = 2*(delta - t) / (delta*(delta + 1 + root_left))
    z_right = 2*(delta - t) / (delta*(delta - 1 + root_right))
    return (delta*(z_left*z_left + z_right*z_right)
            + (delta - 1)*log1p_remainder(-z_left)
            + (delta + 1)*log1p_remainder(z_right))


def coherent_xpm_1d(model, spacing_hz, spans, order, panel_shift=0.0):
    """Exactly reduced coherent matched-filter XPM integral, in W^-2."""
    if not isinstance(spans, int) or isinstance(spans, bool) or spans < 1:
        raise ValueError("spans must be a positive integer.")
    if not math.isfinite(spacing_hz) or spacing_hz <= model.p.symbol_rate_baud:
        raise ValueError("Channel separation must exceed the symbol rate.")
    if not 0 <= panel_shift < 1:
        raise ValueError("panel_shift must lie in [0, 1).")
    p = model.p
    delta = spacing_hz / p.symbol_rate_baud
    k = 4 * math.pi**2 * p.beta2_abs_s2_per_km * p.symbol_rate_baud**2
    period = 2 * math.pi / (spans * k * p.span_length_km)
    # 默认按相位因子分子零点分段；shift 仅用于独立的分段位置检查
    interior = np.arange(1 + panel_shift, delta / period, 1) * period
    edges = np.r_[0.0, interior[interior < delta], delta]
    nodes, weights = unit_rule(order)
    total = 0.0
    # 限制每次处理的分段数量，使较大 Ns 和信道间隔下仍保持低内存占用
    for start in range(0, len(edges) - 1, 1024):
        stop = min(start + 1024, len(edges) - 1)
        lower = edges[start:stop, None]
        widths = (edges[start + 1:stop + 1] - edges[start:stop])[:, None]
        t = lower + widths * nodes[None, :]
        q = k*t
        integrand = xpm_product_weight(t, delta) * span_kernel(q, model.alpha_per_km, p.span_length_km)
        integrand *= phased_array_factor(q*p.span_length_km, spans)
        total += float(np.sum(widths * weights[None, :] * integrand))
    # XPM 原前因子 32/27，利用 v 正负对称性后再乘 2
    return (64 / 27) * p.gamma_per_w_km**2 * total, len(edges) - 1


def converged_xpm(model, spacing_hz, spans, tolerance_db):
    """Require two stable quadrature refinements; this is not an error bound."""
    history = []
    stable_count = 0
    for order in PANEL_ORDERS:
        value, panels = coherent_xpm_1d(model, spacing_hz, spans, order)
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError("The coherent XPM coefficient must be finite and positive.")
        change = None if not history else abs(10*math.log10(value / history[-1]["coefficient_w_inverse_squared"]))
        history.append({"order_per_panel": order, "panels": panels,
                        "coefficient_w_inverse_squared": value,
                        "absolute_change_from_previous_db": change})
        stable_count = stable_count + 1 if change is not None and change < tolerance_db else 0
        if order >= 32 and stable_count >= 2:
            return {"spans": spans, "spacing_ghz": spacing_hz / 1e9,
                    "coherent_xpm_w_inverse_squared": value, "converged": True,
                    "history": history}
    raise RuntimeError(f"XPM did not converge at {spacing_hz/1e9:g} GHz, Ns={spans}.")


def coherent_xpm_2d(model, spacing_hz, spans, order):
    """Independent integration over the original two-variable overlap domain."""
    p = model.p
    delta = spacing_hz / p.symbol_rate_baud
    k = 4 * math.pi**2 * p.beta2_abs_s2_per_km * p.symbol_rate_baud**2
    nodes, weights = unit_rule(order)
    s = nodes[None, :]
    total = 0.0
    # 保留 u,v 两个变量，不调用一维产品权重或相位分段程序
    for start in range(0, order, 64):
        u = nodes[start:start + 64, None]
        wu = weights[start:start + 64, None]
        remaining = 1 - u
        kernels = np.zeros((len(u), order))
        for sign in (-1, 1):
            q = k*(delta + sign*u)*remaining*s
            kernels += span_kernel(q, model.alpha_per_km, p.span_length_km) * phased_array_factor(q*p.span_length_km, spans)
        total += float(np.sum(2*wu*weights[None, :]*remaining**2*(1 - s)*kernels))
    return (32 / 27)*p.gamma_per_w_km**2*total


def independent_checks(model, coefficients, tolerance_db):
    checks = {"step27_single_span": [], "one_d_vs_two_d": []}
    # 逐个核对实际中心信道使用的全部 40 个不同间隔
    for offset in range(1, model.center_index + 1):
        spacing = offset*model.p.channel_spacing_hz
        reference = converged_coefficient(model, spacing, tolerance_db)
        actual = coefficients[(1, offset)]["coherent_xpm_w_inverse_squared"]
        difference = abs(10*math.log10(actual / reference["numerical_w_inverse_squared"]))
        if difference >= tolerance_db:
            raise RuntimeError(f"Step 27 single-span disagreement at offset {offset}.")
        checks["step27_single_span"].append({"offset": offset, "passed": True,
            "absolute_difference_db": difference,
            "step27_value_w_inverse_squared": reference["numerical_w_inverse_squared"]})

    # 检查近邻多跨，以及远间隔多跨；避免只核对容易积分的单跨情况
    for spans, offset in ((10, 1), (2, 40)):
        history = []
        stable_count = 0
        target = coefficients[(spans, offset)]["coherent_xpm_w_inverse_squared"]
        for order in TWO_D_ORDERS:
            value = coherent_xpm_2d(model, offset*model.p.channel_spacing_hz, spans, order)
            change = None if not history else abs(10*math.log10(value/history[-1]["coefficient_w_inverse_squared"]))
            history.append({"order_per_dimension": order, "coefficient_w_inverse_squared": value,
                            "absolute_change_from_previous_db": change})
            stable_count = stable_count + 1 if change is not None and change < tolerance_db else 0
            if stable_count >= 2:
                difference = abs(10*math.log10(value/target))
                if difference >= tolerance_db:
                    raise RuntimeError(f"Independent 2D XPM disagreement: Ns={spans}, offset={offset}.")
                checks["one_d_vs_two_d"].append({"spans": spans, "offset": offset,
                    "passed": True, "absolute_difference_db": difference, "history": history})
                break
        else:
            raise RuntimeError(f"Independent 2D XPM check did not converge: Ns={spans}, offset={offset}.")

    # 在最大实际间隔和最大主表跨数处，移动分段位置，检查结果是否依赖分段对齐
    target = coefficients[(100, 40)]["coherent_xpm_w_inverse_squared"]
    shifted, panels = coherent_xpm_1d(model, 40*model.p.channel_spacing_hz, 100, 32, panel_shift=0.5)
    difference = abs(10*math.log10(shifted/target))
    if difference >= tolerance_db:
        raise RuntimeError("Shifted-panel quadrature check failed.")
    checks["shifted_panels"] = {"spans": 100, "offset": 40, "panels": panels,
        "passed": True, "absolute_difference_db": difference}

    # 常数 kernel 时，菱形重叠权重积分为 2/3，因此一维 H 积分应为 1/3
    nodes, weights = unit_rule(128)
    normalizations = []
    for delta in (1.5625, 6.25, 62.5):
        integral = float(delta*np.sum(weights*xpm_product_weight(delta*nodes, delta)))
        if abs(integral - 1/3) > 1e-12:
            raise RuntimeError("Product-frequency overlap weight has incorrect normalization.")
        normalizations.append({"delta": delta, "integral": integral, "passed": True})
    checks["weight_normalization"] = normalizations
    return checks


def aggregate_predictions(model, coefficients, multiplicities, tolerance_db):
    rows = []
    single_xpm = sum(count*coefficients[(1, offset)]["coherent_xpm_w_inverse_squared"]
                     for offset, count in multiplicities.items())
    for spans in SPAN_COUNTS:
        spm_result = converged_spm(model, spans, tolerance_db)
        spm = spm_result["coherent_spm_w_inverse_squared"]
        xpm_linear = spans*single_xpm
        xpm_coherent = sum(count*coefficients[(spans, offset)]["coherent_xpm_w_inverse_squared"]
                           for offset, count in multiplicities.items())
        previous = link_at_optimum(model, spans, spm + xpm_linear)
        updated = link_at_optimum(model, spans, spm + xpm_coherent)
        # 固定在 Step 28 模型的最优功率，也计算新模型 SNR 以区分模型变化和功率重调
        power = 1e-3*10**(previous["optimal_power_dbm"]/10)
        snr_fixed = power/(spans*model.ase_span_w + (spm + xpm_coherent)*power**3)
        rows.append({"spans": spans, "length_km": spans*model.p.span_length_km,
            "coherent_spm_w_inverse_squared": spm, "spm_quadrature": spm_result,
            "linear_xpm_sum_w_inverse_squared": xpm_linear,
            "coherent_xpm_sum_w_inverse_squared": xpm_coherent,
            "xpm_coherent_over_linear_db": 10*math.log10(xpm_coherent/xpm_linear),
            "total_eta_coherent_over_previous_db": 10*math.log10((spm + xpm_coherent)/(spm + xpm_linear)),
            "coherent_spm_linear_xpm": previous, "coherent_spm_coherent_xpm": updated,
            "updated_minus_previous_optimal_power_db": updated["optimal_power_dbm"]-previous["optimal_power_dbm"],
            "updated_minus_previous_maximum_snr_db": updated["maximum_snr_db"]-previous["maximum_snr_db"],
            "updated_snr_at_previous_optimal_power_db": 10*math.log10(snr_fixed)})
    return single_xpm, rows


def make_plot(coefficients, rows, figure_path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), layout="constrained")
    axes = axes.ravel()
    colors = ("#17476b", "#bd682b", "#25867a", "#89579e")
    for offset, color in zip((1, 4, 20, 40), colors):
        subset = [coefficients[(n, offset)] for n in SPAN_COUNTS]
        axes[0].semilogx(SPAN_COUNTS, [r["coherent_over_linear_db"] for r in subset],
                         "o-", color=color, markersize=4, label=f"{subset[0]['spacing_ghz']:g} GHz")
    axes[0].set(title="XPM accumulation correction", xlabel="Number of spans",
                ylabel="10 log10[coherent / linear] (dB)")
    axes[0].legend(fontsize=9)

    subset = [coefficients[(100, offset)] for offset in range(1, 41)]
    axes[1].plot([r["spacing_ghz"] for r in subset], [r["effective_epsilon"] for r in subset],
                 color=colors[0], linewidth=1.8)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set(title="Effective XPM exponent at 100 spans", xlabel="Channel separation (GHz)",
                ylabel="Effective coherence exponent")

    distance = [r["length_km"] for r in rows]
    for key, label, color, style in (
        ("coherent_spm_linear_xpm", "Coherent SPM + linear XPM (Step 28)", colors[1], "--"),
        ("coherent_spm_coherent_xpm", "Coherent SPM + coherent XPM", colors[0], "-")):
        axes[2].plot(distance, [r[key]["optimal_power_dbm"] for r in rows], marker="o",
                     markersize=4, color=color, linestyle=style, label=label)
    axes[2].set(title="Optimal per-channel launch power", xlabel="Distance (km)",
                ylabel="Optimal power (dBm)")
    axes[2].legend(fontsize=8)

    axes[3].plot(distance, [-r["updated_minus_previous_maximum_snr_db"] for r in rows],
                 "o-", color=colors[0], markersize=4)
    axes[3].set(title="SNR overestimate from linear XPM", xlabel="Distance (km)",
                ylabel="Previous max SNR - updated max SNR (dB)")
    for ax in axes:
        ax.grid(alpha=0.23)
    fig.suptitle("Step 29 | Checking linear XPM accumulation\n"
                 "80 km identical spans, 32 GBd, 80 channels; coherent SPM in both link models", fontsize=13)
    fig.savefig(figure_path, dpi=180)
    if show:
        print("Close the figure window to finish.", flush=True)
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_step29")
    parser.add_argument("--tolerance-db", type=float, default=1e-4,
                        help="Required change for two successive quadrature refinements.")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance_db) or args.tolerance_db <= 0:
        parser.error("--tolerance-db must be finite and positive.")
    model = ClosedFormGN(Parameters())
    multiplicities = Counter(abs(j - model.center_index) for j in range(model.p.channel_count)
                             if j != model.center_index)
    print("--- Step 29: checking linear XPM accumulation ---", flush=True)
    print("Source: Ives (2015), Eqs. (4.19), (4.21)-(4.22), and (4.32)-(4.33).")
    print("Reference parameters: 32 GBd, 50 GHz spacing, 80 channels, identical 80 km spans.")
    print("All 79 interferers co-propagate across the full link; no add/drop or regeneration.")
    print("Both link models use the same coherent SPM and ASE; only XPM accumulation changes.")
    print(f"Quadrature refinement target = {args.tolerance_db:.1e} dB; not a certified error bound.")
    started = time.perf_counter()
    coefficients = {}
    for spans in SPAN_COUNTS:
        for offset in sorted(multiplicities):
            row = converged_xpm(model, offset*model.p.channel_spacing_hz, spans, args.tolerance_db)
            single = row["coherent_xpm_w_inverse_squared"] if spans == 1 else coefficients[(1, offset)]["coherent_xpm_w_inverse_squared"]
            ratio = row["coherent_xpm_w_inverse_squared"]/(spans*single)
            row.update({"offset": offset, "multiplicity": multiplicities[offset],
                        "linear_xpm_w_inverse_squared": spans*single,
                        "coherent_over_linear_db": 10*math.log10(ratio),
                        "effective_epsilon": None if spans == 1 else math.log(ratio)/math.log(spans)})
            coefficients[(spans, offset)] = row
        print(f"  Completed Ns={spans:3d}: 40 unique separations, 79 interferers.", flush=True)
    print("Checking single-span agreement and independent multi-span 2D integrals...", flush=True)
    checks = independent_checks(model, coefficients, args.tolerance_db)
    single_xpm, rows = aggregate_predictions(model, coefficients, multiplicities, args.tolerance_db)
    # 论文图 4.6 的最近邻最远点另算一次；不把它混入 1..100 跨的全信道链路比较
    long_neighbor = converged_xpm(model, model.p.channel_spacing_hz, 500, args.tolerance_db)
    long_neighbor["coherent_over_linear_db"] = 10*math.log10(long_neighbor["coherent_xpm_w_inverse_squared"] /
        (500*coefficients[(1, 1)]["coherent_xpm_w_inverse_squared"]))
    elapsed = time.perf_counter() - started

    print("\n--- Individual XPM at 100 spans: coefficients in W^-2 ---")
    print("Separation (GHz)      Linear XPM    Coherent XPM   Correction (dB)   epsilon_eff")
    for offset in (1, 2, 4, 8, 16, 32, 40):
        r = coefficients[(100, offset)]
        print(f"{r['spacing_ghz']:16.0f} {r['linear_xpm_w_inverse_squared']:15.6f} "
              f"{r['coherent_xpm_w_inverse_squared']:15.6f} {r['coherent_over_linear_db']:17.6f} "
              f"{r['effective_epsilon']:13.8f}")
    print(f"\nNearest neighbor at 500 spans: coherent/linear correction = {long_neighbor['coherent_over_linear_db']:+.6f} dB")
    print("The thesis reports a closest-neighbor XPM error below 0.13 dB at its largest plotted distance.")
    print("This 500-span point is a separate nearest-neighbor check, not a full-comb link calculation.")

    print("\n--- Full-load central channel: sum of 79 XPM contributions ---")
    print(f"Single-span numerical XPM sum = {single_xpm:.6f} W^-2")
    print("  Ns    Linear XPM sum   Coherent XPM sum   XPM correction (dB)   Total eta change (dB)")
    for r in rows:
        print(f"{r['spans']:4d} {r['linear_xpm_sum_w_inverse_squared']:17.6f} "
              f"{r['coherent_xpm_sum_w_inverse_squared']:18.6f} {r['xpm_coherent_over_linear_db']:21.6f} "
              f"{r['total_eta_coherent_over_previous_db']:23.6f}")

    print("\n--- Link comparison: each model uses its own optimal power ---")
    print("Previous = coherent SPM + linear XPM; updated = coherent SPM + coherent XPM.")
    print("  Ns   Popt previous   Popt updated   SNR previous   SNR updated   SNR change")
    for r in rows:
        a, b = r["coherent_spm_linear_xpm"], r["coherent_spm_coherent_xpm"]
        print(f"{r['spans']:4d} {a['optimal_power_dbm']:15.6f} {b['optimal_power_dbm']:14.6f} "
              f"{a['maximum_snr_db']:14.6f} {b['maximum_snr_db']:13.6f} {r['updated_minus_previous_maximum_snr_db']:+12.6f}")
    print("Power units: dBm per channel. SNR units: dB.")
    print("\n--- Selected ideal capacities (Gbit/s per channel, at own optimum) ---")
    for r in rows:
        if r["spans"] in (10, 20, 50):
            print(f"L = {r['length_km']:.0f} km: Previous = {r['coherent_spm_linear_xpm']['maximum_ideal_capacity_gbps']:.6f}, "
                  f"Updated = {r['coherent_spm_coherent_xpm']['maximum_ideal_capacity_gbps']:.6f}")

    print("\n--- Numerical checks ---")
    maximum = max(c["absolute_difference_db"] for c in checks["step27_single_span"])
    print(f"PASS: all 40 single-span separations versus Step 27; maximum difference = {maximum:.3e} dB.")
    for c in checks["one_d_vs_two_d"]:
        print(f"PASS: Ns={c['spans']}, separation={c['offset']*model.p.channel_spacing_hz/1e9:g} GHz, "
              f"1D versus 2D difference = {c['absolute_difference_db']:.3e} dB.")
    print(f"PASS: shifted phase panels at Ns=100, 2000 GHz; difference = {checks['shifted_panels']['absolute_difference_db']:.3e} dB.")
    print("PASS: product-frequency overlap weight normalization.")
    print(f"All 280 full-comb coefficients and the additional 500-span neighbor met the refinement criterion. Time: {elapsed:.2f} s.")
    print("Agreement checks establish numerical consistency within the SPM/XPM GN assumptions, not measured performance.")
    print("A small correction here does not establish the same accuracy for different fibers, spectra, or add/drop networks.")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # 参数容器沿用 Step 26；其中 epsilon 字段不参与本步骤积分，不作为物理输入导出
    physical_parameters = {key: value for key, value in asdict(model.p).items() if key != "epsilon"}
    results = {"source": "Ives (2015), UCL PhD thesis, Eqs. (4.19), (4.21)-(4.22), (4.32)-(4.33)",
        "parameters": physical_parameters,
        "coherence": {"spm": "Numerically integrated phased-array factor in both link models",
                      "previous_xpm": "Ns times numerical single-span XPM",
                      "updated_xpm": "Numerically integrated phased-array factor",
                      "input_epsilon": None},
        "scope": "SPM/XPM GN, identical contiguous spans, full-load central channel; no multi-channel mixing terms",
        "observed_channel_index": model.center_index, "interferer_count": sum(multiplicities.values()),
        "spacing_multiplicities": dict(sorted(multiplicities.items())),
        "span_counts": list(SPAN_COUNTS), "refinement_target_db": args.tolerance_db,
        "refinement_is_certified_error_bound": False,
        "xpm_coefficients": list(coefficients.values()), "link_predictions": rows,
        "nearest_neighbor_500_spans": long_neighbor, "checks": checks, "calculation_seconds": elapsed,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    json_path = output / "xpm_coherence_results.json"
    json_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure_path = output / "xpm_coherence_comparison.png"
    print(f"Saved numerical results: {json_path}")
    print(f"Figure destination: {figure_path}")
    make_plot(coefficients, rows, figure_path, show=not args.no_show)


if __name__ == "__main__":
    main()
