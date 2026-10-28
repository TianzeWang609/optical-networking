"""Check coherent multi-span SPM accumulation under the Ives reference parameters.

Ives (2015), UCL PhD thesis: phased-array factor, Eq. (4.19), p. 63;
matched-filter SPM integral, Eq. (4.23), p. 65; epsilon definition and
SPM-coherent/XPM-linear approximation, Eqs. (4.31)-(4.33), p. 68.

SPM is numerically integrated with its phased-array factor. XPM is retained as
Ns times the numerically integrated single-span XPM sum. No epsilon is supplied
to the coherent integration; effective exponents are calculated afterwards.
This is a GN-model calculation, not an experimental validation.

Keep step26_reference_physics.py and step27_nli_integral.py in this folder.
Run: python step28_spm_coherence.py
Batch run: python step28_spm_coherence.py --no-show
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy

try:
    from step26_reference_physics import ClosedFormGN, Parameters, w_to_dbm
    from step27_nli_integral import unit_rule, span_kernel, converged_coefficient
except ModuleNotFoundError as exc:
    if exc.name in ("step26_reference_physics", "step27_nli_integral"):
        raise SystemExit("Place the Step 26 and Step 27 Python scripts beside step28_spm_coherence.py.") from exc
    raise


ONE_D_ORDERS = (32, 64, 128, 256, 512, 1024)
TWO_D_ORDERS = (64, 128, 256, 512, 1024, 2048, 4096)
PUBLISHED_EPSILON_100 = 0.2186


def phased_array_factor(phase, spans):
    """Stable |sum(exp(1j*m*phase), m=0..Ns-1)|^2, including removable zeros."""
    if not isinstance(spans, int) or isinstance(spans, bool) or spans < 1:
        raise ValueError("spans must be a positive integer.")
    # 将相位折回 [-pi, pi)，使分母 sinc 不接近零；相位为 2*pi 整数倍时取 Ns^2
    wrapped = (np.asarray(phase) + np.pi) % (2 * np.pi) - np.pi
    x = wrapped / (2 * np.pi)
    return (spans * np.sinc(spans * x) / np.sinc(x)) ** 2


def product_weight(x):
    """H(x^2/4), the exact product-frequency weight for rectangular SPM."""
    s = np.sqrt((1 - x) * (1 + x))
    result = 2 * (np.log1p(s) - np.log(x) - s)
    # x 接近 1 时使用级数，避免两个很接近的数相减
    small = s < 0.05
    z = s[small]
    result[small] = 2 * z**3 * (1 / 3 + z*z * (1 / 5 + z*z * (1 / 7 + z*z *
                                                   (1 / 9 + z*z * (1 / 11 + z*z / 13)))))
    return result


def coherent_spm_1d(model, spans, order):
    """Exactly reduced one-dimensional matched-filter SPM integral, in W^-2."""
    p = model.p
    k = 4 * math.pi**2 * p.beta2_abs_s2_per_km * p.symbol_rate_baud**2
    max_phase = k * p.span_length_km / 4
    # 相位因子分子零点将积分区间分成小段，避免漏掉随 Ns 变窄的峰
    last_zero = int(math.floor(spans * max_phase / (2 * math.pi)))
    t_zeros = 2 * math.pi * np.arange(1, last_zero + 1) / (spans * k * p.span_length_km)
    edges = np.r_[0.0, 2 * np.sqrt(t_zeros[t_zeros < 0.25]), 1.0]
    nodes, weights = unit_rule(order)
    widths = np.diff(edges)[:, None]
    x = edges[:-1, None] + widths * nodes[None, :]
    # 从乘积变量 t=u*v 改为 t=x^2/4；Jacobian 为 x/2
    q = k * x*x / 4
    integrand = (x / 2) * product_weight(x) * span_kernel(q, model.alpha_per_km, p.span_length_km)
    integrand *= phased_array_factor(q * p.span_length_km, spans)
    value = (64 / 27) * p.gamma_per_w_km**2 * float(np.sum(widths * weights[None, :] * integrand))
    return value, len(edges) - 1


def converged_spm(model, spans, tolerance_db):
    history = []
    consecutive = 0
    for order in ONE_D_ORDERS:
        value, panels = coherent_spm_1d(model, spans, order)
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError("The coherent SPM coefficient must be finite and positive.")
        change = None if not history else abs(10 * math.log10(value / history[-1]["coefficient_w_inverse_squared"]))
        history.append({"order_per_panel": order, "panels": panels,
                        "coefficient_w_inverse_squared": value,
                        "absolute_change_from_previous_db": change})
        consecutive = consecutive + 1 if change is not None and change < tolerance_db else 0
        # 至少计算到每段 256 阶，并要求连续两次加密结果稳定
        if order >= 256 and consecutive >= 2:
            return {"spans": spans, "coherent_spm_w_inverse_squared": value,
                    "converged": True, "history": history}
    raise RuntimeError(f"Coherent SPM did not converge for Ns={spans}; no converged value is being reported.")


def coherent_spm_2d(model, spans, order):
    """Independent two-dimensional quadrature of Step 27's SPM domain times chi."""
    p = model.p
    k = 4 * math.pi**2 * p.beta2_abs_s2_per_km * p.symbol_rate_baud**2
    nodes, weights = unit_rule(order)
    s = nodes[None, :]
    total = 0.0
    # 保留 u 和 v 两个变量，不使用一维乘积权重 H
    for start in range(0, order, 64):
        u = nodes[start:start + 64, None]
        wu = weights[start:start + 64, None]
        q = k * u * (1 - u) * s
        values = span_kernel(q, model.alpha_per_km, p.span_length_km)
        values *= phased_array_factor(q * p.span_length_km, spans)
        total += float(np.sum(wu * weights[None, :] * (1 - u)**2 * (1 - s) * values))
    return (64 / 27) * p.gamma_per_w_km**2 * total


def independent_checks(model, rows, tolerance_db):
    # 与直接复指数求和比较，检查相位因子的周期性及可去奇点
    phases = np.array([0, 1e-12, -1e-12, 0.4, math.pi, 2*math.pi,
                       2*math.pi + 1e-10, -2*math.pi, 7.3, 17.0])
    phase_checks = []
    for spans in (1, 2, 5, 10, 100):
        direct = np.abs(np.exp(1j * phases[:, None] * np.arange(spans)[None, :]).sum(axis=1))**2
        stable = phased_array_factor(phases, spans)
        error = float(np.max(np.abs(direct - stable)) / spans**2)
        if error >= 1e-12:
            raise RuntimeError("Stable phased-array formula disagrees with the explicit phasor sum.")
        phase_checks.append({"spans": spans, "maximum_absolute_error_divided_by_ns_squared": error})

    cross_checks = []
    for spans in (1, 10, 100):
        reference = rows[spans - 1]["coherent_spm_w_inverse_squared"]
        history = []
        stable_count = 0
        for order in TWO_D_ORDERS:
            value = coherent_spm_2d(model, spans, order)
            change = None if not history else abs(10 * math.log10(value / history[-1]["coefficient_w_inverse_squared"]))
            history.append({"order_per_dimension": order, "coefficient_w_inverse_squared": value,
                            "absolute_change_from_previous_db": change})
            stable_count = stable_count + 1 if change is not None and change < tolerance_db else 0
            if stable_count >= 2:
                difference = abs(10 * math.log10(value / reference))
                if difference >= tolerance_db:
                    raise RuntimeError(f"Independent 1D/2D integration disagreement at Ns={spans}.")
                cross_checks.append({"spans": spans, "passed": True,
                                     "absolute_1d_vs_2d_difference_db": difference, "two_d_history": history})
                break
        else:
            raise RuntimeError(f"The independent 2D check did not converge at Ns={spans}.")

    single_span = converged_coefficient(model, 0.0, tolerance_db)
    single_difference = abs(10 * math.log10(rows[0]["coherent_spm_w_inverse_squared"] /
                                          single_span["numerical_w_inverse_squared"]))
    if single_difference >= tolerance_db:
        raise RuntimeError("The Ns=1 result disagrees with the Step 27 single-span integral.")
    return {"phased_array_vs_explicit_sum": phase_checks,
            "one_d_vs_two_d": cross_checks,
            "step27_single_span_difference_db": single_difference}


def numerical_xpm_sum(model, tolerance_db):
    offsets = Counter(abs(j - model.center_index) for j in range(model.p.channel_count)
                      if j != model.center_index)
    values = []
    total = 0.0
    for offset, multiplicity in sorted(offsets.items()):
        row = converged_coefficient(model, offset * model.p.channel_spacing_hz, tolerance_db)
        total += multiplicity * row["numerical_w_inverse_squared"]
        values.append({"offset": offset, "multiplicity": multiplicity, "coefficient": row})
    return total, values


def link_at_optimum(model, spans, eta_total):
    ase = spans * model.ase_span_w
    power = (ase / (2 * eta_total)) ** (1 / 3)
    snr = power / (ase + eta_total * power**3)
    return {"optimal_power_dbm": w_to_dbm(power), "maximum_snr_db": 10 * math.log10(snr),
            "maximum_ideal_capacity_gbps": 2 * model.p.symbol_rate_baud * math.log2(1 + snr) / 1e9,
            "eta_total_w_inverse_squared": eta_total, "ase_total_w": ase}


def add_predictions(model, rows, xpm_sum):
    single = rows[0]["coherent_spm_w_inverse_squared"]
    epsilon_100 = math.log(rows[99]["coherent_spm_w_inverse_squared"] / (100 * single)) / math.log(100)
    for row in rows:
        spans = row["spans"]
        coherent = row["coherent_spm_w_inverse_squared"]
        linear = spans * single
        ratio = coherent / linear
        fitted = single * spans ** (1 + epsilon_100)
        row.update({
            "length_km": spans * model.p.span_length_km,
            "linear_spm_w_inverse_squared": linear,
            "spm_coherent_over_linear_ratio": ratio,
            "spm_coherent_over_linear_db": 10 * math.log10(ratio),
            "effective_epsilon": None if spans == 1 else math.log(ratio) / math.log(spans),
            "power_law_spm_w_inverse_squared": fitted,
            "power_law_over_numerical_spm_error_db": 10 * math.log10(fitted / coherent),
            "incoherent_baseline": link_at_optimum(model, spans, linear + spans * xpm_sum),
            "coherent_spm_linear_xpm": link_at_optimum(model, spans, coherent + spans * xpm_sum),
            "epsilon_100_power_law_linear_xpm": link_at_optimum(model, spans, fitted + spans * xpm_sum),
        })
    return epsilon_100


def make_plot(rows, epsilon_100, figure_path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 8.7), layout="constrained")
    axes = axes.ravel()
    spans = [r["spans"] for r in rows]
    distance = [r["length_km"] for r in rows]
    blue, orange, green = "#17476b", "#bd682b", "#25867a"
    axes[0].loglog(spans, [r["coherent_spm_w_inverse_squared"] for r in rows], color=blue,
                   linewidth=2, label="Numerical coherent SPM")
    axes[0].loglog(spans, [r["linear_spm_w_inverse_squared"] for r in rows], color=orange,
                   linewidth=1.7, linestyle="--", label="Incoherent: Ns times single-span SPM")
    axes[0].loglog(spans, [r["power_law_spm_w_inverse_squared"] for r in rows], color=green,
                   linewidth=1.5, linestyle=":", label=f"Power law: epsilon = {epsilon_100:.4f}")
    axes[0].set(title="SPM accumulation", xlabel="Number of spans", ylabel="Accumulated SPM coefficient (W$^{-2}$)")
    axes[0].legend(fontsize=8)

    axes[1].semilogx(spans[1:], [r["effective_epsilon"] for r in rows[1:]], color=blue, linewidth=2,
                     label="Effective exponent from each span count")
    axes[1].axhline(epsilon_100, color=green, linestyle="--", linewidth=1.5,
                    label=f"100-span value: {epsilon_100:.6f}")
    axes[1].set(title="Effective epsilon is not constant", xlabel="Number of spans (Ns > 1)",
                ylabel="Effective SPM coherence exponent")
    axes[1].legend(fontsize=8)

    for key, label, color, style in (
        ("incoherent_baseline", "Incoherent SPM + linear XPM", orange, "--"),
        ("coherent_spm_linear_xpm", "Coherent SPM + linear XPM", blue, "-"),
    ):
        axes[2].plot(distance, [r[key]["optimal_power_dbm"] for r in rows], color=color,
                     linestyle=style, linewidth=1.8, label=label)
        axes[3].plot(distance, [r[key]["maximum_snr_db"] for r in rows], color=color,
                     linestyle=style, linewidth=1.8, label=label)
    axes[2].set(title="Optimal per-channel launch power", xlabel="Distance (km)", ylabel="Optimal power (dBm)")
    axes[3].set(title="Maximum SNR at each model's own optimum", xlabel="Distance (km)", ylabel="Maximum SNR (dB)")
    axes[2].legend(fontsize=8)
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.23)
    fig.suptitle("Step 28 | Coherent multi-span SPM under the thesis reference parameters\n"
                 "80 km spans, 32 GBd, 80 channels; XPM retains linear accumulation", fontsize=13)
    fig.savefig(figure_path, dpi=180)
    if show:
        print("Close the figure window to finish.", flush=True)
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_step28")
    parser.add_argument("--tolerance-db", type=float, default=1e-4,
                        help="Required change for two successive quadrature refinements.")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance_db) or args.tolerance_db <= 0:
        parser.error("--tolerance-db must be finite and positive.")
    model = ClosedFormGN(Parameters())
    print("--- Step 28: coherent multi-span SPM accumulation ---", flush=True)
    print("Source: Ives thesis Eq. (4.19), p. 63, and Eqs. (4.31)-(4.33), p. 68.")
    print("No epsilon is supplied to the coherent SPM integration.")
    print("The link prediction retains linearly accumulated XPM and ideal identical spans.")
    print(f"Quadrature refinement target = {args.tolerance_db:.1e} dB; not a certified error bound.")
    started = time.perf_counter()
    rows = []
    for spans in range(1, 101):
        rows.append(converged_spm(model, spans, args.tolerance_db))
        if spans % 20 == 0:
            print(f"  Coherent SPM completed: {spans}/100 span counts.", flush=True)
    print("Checking phase factors and independent 2D integrals...", flush=True)
    checks = independent_checks(model, rows, args.tolerance_db)
    print("Computing the single-span XPM sum for 79 interferers...", flush=True)
    xpm_sum, xpm_records = numerical_xpm_sum(model, args.tolerance_db)
    epsilon_100 = add_predictions(model, rows, xpm_sum)
    elapsed = time.perf_counter() - started
    samples = [rows[n - 1] for n in (1, 2, 5, 10, 20, 50, 100)]
    largest_error_row = max(rows, key=lambda r: abs(r["power_law_over_numerical_spm_error_db"]))

    print("\n--- SPM accumulation: coefficients in W^-2 ---")
    print(f"{'Ns':>4} {'Distance (km)':>14} {'Linear SPM':>14} {'Coherent SPM':>15} {'Excess (dB)':>13} {'epsilon_eff':>13}")
    for r in samples:
        epsilon_text = "n/a" if r["effective_epsilon"] is None else f"{r['effective_epsilon']:.6f}"
        print(f"{r['spans']:4d} {r['length_km']:14.0f} {r['linear_spm_w_inverse_squared']:14.6f} "
              f"{r['coherent_spm_w_inverse_squared']:15.6f} {r['spm_coherent_over_linear_db']:13.6f} {epsilon_text:>13}")
    print("\n--- 100-span epsilon and the thesis reference ---")
    print(f"Calculated epsilon_100 = {epsilon_100:.9f}")
    print(f"Thesis p. 68 reported epsilon = {PUBLISHED_EPSILON_100:.4f}")
    print(f"Matches the reported four-decimal value: {round(epsilon_100, 4) == PUBLISHED_EPSILON_100}")
    print(f"Largest absolute SPM power-law error over 1..100 spans = "
          f"{abs(largest_error_row['power_law_over_numerical_spm_error_db']):.6f} dB at Ns={largest_error_row['spans']}")
    print("The effective exponent is derived from the integral and need not be constant across distances.")

    print("\n--- Link predictions: each model uses its own optimal power ---")
    print(f"Single-span numerical XPM sum = {xpm_sum:.6f} W^-2")
    print(f"{'Ns':>4} {'Popt linear':>13} {'Popt coherent':>15} {'SNR linear':>13} {'SNR coherent':>15}")
    for r in samples:
        a, b = r["incoherent_baseline"], r["coherent_spm_linear_xpm"]
        print(f"{r['spans']:4d} {a['optimal_power_dbm']:13.6f} {b['optimal_power_dbm']:15.6f} "
              f"{a['maximum_snr_db']:13.6f} {b['maximum_snr_db']:15.6f}")
    print("Power units: dBm per channel. SNR units: dB. XPM is linear in both columns.")
    print("\n--- Selected ideal capacities (Gbit/s per channel, at own optimum) ---")
    for n in (10, 20, 50):
        r = rows[n - 1]
        print(f"L = {r['length_km']:.0f} km: Incoherent = {r['incoherent_baseline']['maximum_ideal_capacity_gbps']:.6f}, "
              f"Coherent SPM = {r['coherent_spm_linear_xpm']['maximum_ideal_capacity_gbps']:.6f}")
    print("\n--- Numerical checks ---")
    print("PASS: stable phased-array factor versus explicit complex phasor sums.")
    print(f"PASS: Ns=1 versus Step 27; difference = {checks['step27_single_span_difference_db']:.3e} dB.")
    for check in checks["one_d_vs_two_d"]:
        print(f"PASS: Ns={check['spans']}, 1D versus 2D integral difference = {check['absolute_1d_vs_2d_difference_db']:.3e} dB.")
    print(f"All 100 span counts met the refinement criterion. Calculation time: {elapsed:.2f} s.")
    print("These checks validate numerical consistency within the GN assumptions, not measured transmission performance.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "spm_coherence_results.json"
    figure_path = output_dir / "spm_coherence_comparison.png"
    results = {
        "source": "Ives (2015), UCL PhD thesis, Eqs. (4.19), (4.23), (4.31)-(4.33), (6.1).",
        "step26_reference_baseline": model.summary(),
        "coherent_spm_method": "Numerical GN integral including the exact phased-array factor; no input epsilon.",
        "xpm_method": "Numerical single-span coefficients multiplied by Ns; XPM coherence is not evaluated.",
        "epsilon_100": epsilon_100,
        "published_epsilon_100": PUBLISHED_EPSILON_100,
        "matches_published_rounding": round(epsilon_100, 4) == PUBLISHED_EPSILON_100,
        "maximum_absolute_power_law_spm_error_db": abs(largest_error_row["power_law_over_numerical_spm_error_db"]),
        "span_count_at_maximum_power_law_error": largest_error_row["spans"],
        "single_span_numerical_spm_w_inverse_squared": rows[0]["coherent_spm_w_inverse_squared"],
        "single_span_numerical_xpm_sum_w_inverse_squared": xpm_sum,
        "span_results": rows, "xpm_coefficients_and_multiplicities": xpm_records,
        "quadrature_tolerance_db": args.tolerance_db, "independent_checks": checks,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "calculation_elapsed_seconds": elapsed,
        "limitations": ["Quadrature refinement changes are not rigorous error bounds.",
                        "epsilon_100 is a model-specific endpoint-calibrated power law, not a material constant.",
                        "The SNR and capacity comparison retains linear XPM, flat parameters, and ideal amplification.",
                        "No Raman scattering, other inter-channel FWM terms, ROADM/transceiver penalties, or experimental calibration."],
    }
    result_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved numerical results: {result_path}")
    print(f"Figure destination: {figure_path}", flush=True)
    make_plot(rows, epsilon_100, figure_path, show=not args.no_show)


if __name__ == "__main__":
    main()
