"""Compare single-span GN integrals with the Step 26 closed-form coefficients.

Source: D. J. Ives, Coherent Optical Fibre Networking in the Nonlinear Regime,
UCL PhD thesis (2015), Eqs. (4.22)-(4.25), pp. 65-67.

For rectangular spectra, the matched-filter triple integral is reduced exactly
to a weighted two-dimensional integral, evaluated by Gauss-Legendre quadrature.
Two cases are also evaluated as direct three-dimensional integrals.
This checks the closed-form approximation within the stated GN assumptions;
it is not validation against split-step propagation or physical experiments.

Keep step26_reference_physics.py in the same folder.
Run: python step27_nli_integral.py
Batch run: python step27_nli_integral.py --no-show
"""

import argparse
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.special import roots_legendre

try:
    from step26_reference_physics import ClosedFormGN, Parameters, w_to_dbm
except ModuleNotFoundError as exc:
    if exc.name == "step26_reference_physics":
        raise SystemExit("Place step26_reference_physics.py beside step27_nli_integral.py.") from exc
    raise


QUADRATURE_ORDERS = (64, 128, 256, 512, 1024, 2048, 4096)


@lru_cache(maxsize=12)
def unit_rule(order):
    """Return Gauss-Legendre nodes and weights on [0, 1]."""
    x, w = roots_legendre(order)
    return (x + 1.0) / 2.0, w / 2.0


def span_kernel(q, alpha, length):
    """Finite-span kernel rho from thesis Eq. (4.18), in km^2."""
    decay = math.exp(-alpha * length)
    # 与 1+exp(-2*alpha*L)-2*exp(-alpha*L)*cos(q*L) 等价，避免小 q 相减损失精度
    numerator = (-math.expm1(-alpha * length)) ** 2 + 4 * decay * np.sin(q * length / 2) ** 2
    return numerator / (alpha**2 + q**2)


def numerical_coefficient_at_order(model, spacing_hz, order):
    """Evaluate the exactly reduced matched-filter GN integral, in W^-2."""
    p = model.p
    if spacing_hz < 0 or (spacing_hz != 0 and spacing_hz < p.symbol_rate_baud):
        raise ValueError("Use spacing = 0 for SPM, or non-overlapping channels for XPM.")
    nodes, weights = unit_rule(order)
    s = nodes[None, :]
    ws = weights[None, :]
    delta = spacing_hz / p.symbol_rate_baud
    phase_scale = 4 * math.pi**2 * p.beta2_abs_s2_per_km * p.symbol_rate_baud**2
    total = 0.0
    # u=(f1-f)/Rs, v=(f2-f)/Rs；接收频率积分的区间长度为 max(0,1-|u|-|v|)
    # 利用 v 的正负对称性，再令 v=(1-u)*s，将菱形区域化为单位正方形
    # 分块计算二维积分，避免大阶数时同时分配完整矩阵
    for start in range(0, order, 64):
        u = nodes[start:start + 64, None]
        wu = weights[start:start + 64, None]
        remaining = 1 - u
        q_plus = phase_scale * (delta + u) * remaining * s
        q_minus = phase_scale * (delta - u) * remaining * s
        kernels = span_kernel(q_plus, model.alpha_per_km, p.span_length_km)
        kernels += span_kernel(q_minus, model.alpha_per_km, p.span_length_km)
        # (1-u)^2*(1-s) 已包含谱重叠权重和 v 变量变换的 Jacobian
        total += float(np.sum(2 * wu * ws * remaining**2 * (1 - s) * kernels))
    # SPM 前因子为 16/27，XPM 为 32/27；不能再次额外乘退化因子
    factor = 16.0 / 27.0 if spacing_hz == 0 else 32.0 / 27.0
    return factor * p.gamma_per_w_km**2 * total


def converged_coefficient(model, spacing_hz, tolerance_db):
    """Require two successive refinements below tolerance; not an error bound."""
    history = []
    stable_count = 0
    for order in QUADRATURE_ORDERS:
        value = numerical_coefficient_at_order(model, spacing_hz, order)
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError("Numerical NLI coefficient must be finite and positive.")
        change_db = None if not history else abs(10 * math.log10(value / history[-1]["coefficient_w_inverse_squared"]))
        history.append({"order_per_dimension": order,
                        "coefficient_w_inverse_squared": value,
                        "absolute_change_from_previous_db": change_db})
        stable_count = stable_count + 1 if change_db is not None and change_db < tolerance_db else 0
        if stable_count >= 2:
            closed = model.coefficient(spacing_hz)
            return {"kind": "SPM" if spacing_hz == 0 else "XPM",
                    "spacing_ghz": spacing_hz / 1e9,
                    "closed_form_w_inverse_squared": closed,
                    "numerical_w_inverse_squared": value,
                    "closed_over_numerical_error_db": 10 * math.log10(closed / value),
                    "final_order_per_dimension": order,
                    "last_refinement_change_db": change_db,
                    "converged": True, "history": history}
    raise RuntimeError(f"No convergence at {spacing_hz / 1e9:.1f} GHz up to order {QUADRATURE_ORDERS[-1]}. "
                       "No converged result is being reported; inspect quadrature resolution.")


def direct_triple_integral(model, spacing_hz, order):
    """Independent domain parameterization of the original three-frequency integral."""
    p = model.p
    nodes, weights = unit_rule(order)
    delta = spacing_hz / p.symbol_rate_baud
    phase_scale = 4 * math.pi**2 * p.beta2_abs_s2_per_km * p.symbol_rate_baud**2
    total = 0.0
    # 直接使用 t=f/Rs、a=f1/Rs、b=f2/Rs，不使用二维积分的三角形权重
    for start in range(0, order, 8):
        t = nodes[start:start + 8, None, None] - 0.5
        wt = weights[start:start + 8, None, None]
        # 在 a=t 处分区，使 b 的上下限在每一区域内光滑
        for side in (-1, 1):
            a_low = np.full_like(t, -0.5) if side < 0 else t
            a_high = t if side < 0 else np.full_like(t, 0.5)
            a = a_low + (a_high - a_low) * nodes[None, :, None]
            b_low = np.maximum(-0.5, t - a - 0.5)
            b_high = np.minimum(0.5, t - a + 0.5)
            b = b_low + (b_high - b_low) * nodes[None, None, :]
            q = phase_scale * (a + delta - t) * (b - t)
            # 直接保留原式中的余弦形式，交叉检查稳定形式的 kernel 实现
            attenuation = math.exp(-model.alpha_per_km * p.span_length_km)
            kernel = ((1 + attenuation**2 - 2 * attenuation * np.cos(q * p.span_length_km))
                      / (model.alpha_per_km**2 + q**2))
            total += float(np.sum(wt * weights[None, :, None] * weights[None, None, :]
                                  * (a_high - a_low) * (b_high - b_low) * kernel))
    return (16.0 / 27.0 if spacing_hz == 0 else 32.0 / 27.0) * p.gamma_per_w_km**2 * total


def check_original_integrals(model, rows, tolerance_db):
    checks = []
    # 对 SPM 和最近邻 XPM 用三重积分独立交叉检查，覆盖两种前因子
    for spacing_ghz in (0.0, model.p.channel_spacing_hz / 1e9):
        row = next(r for r in rows if r["spacing_ghz"] == spacing_ghz)
        history = [{"order_per_dimension": n,
                    "coefficient_w_inverse_squared": direct_triple_integral(model, spacing_ghz * 1e9, n)}
                   for n in (32, 64, 128)]
        values = [h["coefficient_w_inverse_squared"] for h in history]
        refinement_db = abs(10 * math.log10(values[-1] / values[-2]))
        difference_db = abs(10 * math.log10(values[-1] / row["numerical_w_inverse_squared"]))
        passed = refinement_db < tolerance_db and difference_db < tolerance_db
        checks.append({"spacing_ghz": spacing_ghz, "passed": passed,
                       "direct_3d_history": history,
                       "last_3d_refinement_change_db": refinement_db,
                       "absolute_2d_vs_3d_difference_db": difference_db})
        if not passed:
            raise RuntimeError(f"Direct three-dimensional check failed at {spacing_ghz:g} GHz.")
    return checks


def link_prediction(model, eta_span, power_w=None):
    """Illustrative 800 km prediction, retaining epsilon = 0 and all other assumptions."""
    spans = 10
    if power_w is None:
        power_w = (model.ase_span_w / (2 * eta_span)) ** (1 / 3)
    snr = power_w / (spans * model.ase_span_w + spans * eta_span * power_w**3)
    return {"launch_power_dbm": w_to_dbm(power_w), "snr_db": 10 * math.log10(snr),
            "ideal_capacity_gbps": 2 * model.p.symbol_rate_baud * math.log2(1 + snr) / 1e9}


def aggregate_wdm(model, rows):
    # 按实际 80 信道编号逐一计数：中心信道有 79 个干扰信道
    multiplicities = Counter(abs(j - model.center_index) for j in range(model.p.channel_count)
                             if j != model.center_index)
    by_spacing = {round(row["spacing_ghz"], 9): row for row in rows}
    numerical_spm = by_spacing[0.0]["numerical_w_inverse_squared"]
    numerical_xpm = sum(count * by_spacing[round(offset * model.p.channel_spacing_hz / 1e9, 9)]
                        ["numerical_w_inverse_squared"] for offset, count in multiplicities.items())
    eta_numerical = numerical_spm + numerical_xpm
    return {
        "interferer_count": sum(multiplicities.values()),
        "spacing_multiplicities": {str(offset): count for offset, count in sorted(multiplicities.items())},
        "closed_spm_w_inverse_squared": model.x_spm,
        "closed_xpm_sum_w_inverse_squared": model.x_xpm_sum,
        "closed_eta_span_w_inverse_squared": model.eta_span,
        "numerical_spm_w_inverse_squared": numerical_spm,
        "numerical_xpm_sum_w_inverse_squared": numerical_xpm,
        "numerical_eta_span_w_inverse_squared": eta_numerical,
        "eta_closed_over_numerical_error_db": 10 * math.log10(model.eta_span / eta_numerical),
        "closed_form_at_own_optimal_power": link_prediction(model, model.eta_span),
        "numerical_at_own_optimal_power": link_prediction(model, eta_numerical),
        "numerical_at_previous_closed_form_power": link_prediction(model, eta_numerical, model.optimal_power_w()),
    }


def make_plot(rows, tolerance_db, output_path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), layout="constrained")
    xpm = [row for row in rows if row["kind"] == "XPM"]
    spm = rows[0]
    x = [row["spacing_ghz"] for row in xpm]
    colors = ("#17476b", "#c16c2e", "#25867a")
    axes[0].semilogy(x, [r["numerical_w_inverse_squared"] for r in xpm],
                     color=colors[0], linewidth=1.9, label="Numerical integration")
    axes[0].semilogy(x, [r["closed_form_w_inverse_squared"] for r in xpm],
                     color=colors[1], linewidth=1.6, linestyle="--", label="Closed-form approximation")
    axes[0].scatter([0], [spm["numerical_w_inverse_squared"]], color=colors[0], s=34, zorder=4)
    axes[0].scatter([0], [spm["closed_form_w_inverse_squared"]], color=colors[1], marker="x", s=38, zorder=4)
    axes[0].set(title="Single-span SPM / XPM coefficients", xlabel="Channel separation (GHz; 0 = SPM)",
                ylabel="NLI coefficient (W$^{-2}$)", xlim=(-100, 4100))
    axes[0].legend(fontsize=8)

    axes[1].plot(x, [r["closed_over_numerical_error_db"] for r in xpm], color=colors[0], linewidth=1.9)
    axes[1].scatter([0], [spm["closed_over_numerical_error_db"]], color=colors[1], s=38,
                    label=f"SPM: {spm['closed_over_numerical_error_db']:+.3f} dB", zorder=3)
    axes[1].axhline(0, color="#555555", linewidth=1, linestyle="--")
    axes[1].set(title="Closed-form approximation error", xlabel="Channel separation (GHz; 0 = SPM)",
                ylabel="10 log$_{10}$(closed / numerical) (dB)", xlim=(-100, 4100))
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].text(0.97, 0.72, "Positive: coefficient overestimated\nNegative: coefficient underestimated",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=8)

    # 最精细结果作为收敛图的参照；不是严格误差界，零差异只在图中设定显示下限
    for spacing, color in zip((0.0, 50.0, 4000.0), colors):
        row = next(r for r in rows if r["spacing_ghz"] == spacing)
        history = row["history"][:-1]
        errors = [max(abs(10 * math.log10(h["coefficient_w_inverse_squared"] /
                                         row["numerical_w_inverse_squared"])), 1e-12) for h in history]
        axes[2].plot([h["order_per_dimension"] for h in history], errors,
                     marker="x" if spacing == 50.0 else "o", color=color,
                     label="SPM" if spacing == 0 else f"XPM: {spacing:.0f} GHz")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].axhline(tolerance_db, color="#555555", linestyle="--", linewidth=1, label="Refinement target")
    axes[2].set(title="Quadrature resolution check", xlabel="Gauss-Legendre order per dimension",
                ylabel="Difference from finest result (dB)")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].text(0.03, 0.15, "Display floor: 1e-12 dB\nNot a certified error bound",
                 transform=axes[2].transAxes, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.23)
    fig.suptitle("Step 27 | Single-span numerical GN integration versus closed-form approximation\n"
                 "Ives thesis Eqs. (4.22)-(4.25): 80 km, 32 GBd, rectangular spectra", fontsize=12)
    fig.savefig(output_path, dpi=180)
    if show:
        print("Close the figure window to finish.", flush=True)
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_step27")
    parser.add_argument("--tolerance-db", type=float, default=1e-4,
                        help="Required change for each of two successive quadrature refinements.")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance_db) or args.tolerance_db <= 0:
        parser.error("--tolerance-db must be finite and positive.")

    model = ClosedFormGN(Parameters())
    print("--- Step 27: numerical single-span GN integration ---", flush=True)
    print("Reference parameters from Step 26; original Eqs. (4.22)-(4.23) versus closed Eqs. (4.24)-(4.25).")
    print(f"Convergence target: two successive changes below {args.tolerance_db:.1e} dB.")
    print("Computing SPM and XPM at 50 GHz intervals up to 4000 GHz...", flush=True)
    started = time.perf_counter()
    rows = []
    for index, spacing_ghz in enumerate(range(0, 4001, 50)):
        rows.append(converged_coefficient(model, spacing_ghz * 1e9, args.tolerance_db))
        if (index + 1) % 20 == 0 or index == 80:
            print(f"  Completed {index + 1}/81 coefficients.", flush=True)
    checks = check_original_integrals(model, rows, args.tolerance_db)
    aggregate = aggregate_wdm(model, rows)
    elapsed = time.perf_counter() - started

    print("\n--- Coefficient comparison (W^-2) ---")
    print("Error = 10*log10(closed-form / numerical); positive means overestimated.")
    print(f"{'Case':<13} {'Closed form':>14} {'Numerical':>14} {'Error (dB)':>13} {'Final order':>12}")
    for row in rows:
        if row["spacing_ghz"] not in (0, 50, 100, 200, 400, 800, 1600, 2000, 4000):
            continue
        label = "SPM" if row["kind"] == "SPM" else f"XPM {row['spacing_ghz']:.0f} GHz"
        print(f"{label:<13} {row['closed_form_w_inverse_squared']:14.6f} "
              f"{row['numerical_w_inverse_squared']:14.6f} {row['closed_over_numerical_error_db']:+13.6f} "
              f"{row['final_order_per_dimension']:12d}")

    print("\n--- Convergence example: XPM at 4000 GHz ---")
    for h in rows[-1]["history"]:
        change = "n/a" if h["absolute_change_from_previous_db"] is None else f"{h['absolute_change_from_previous_db']:.3e} dB"
        print(f"Order = {h['order_per_dimension']:4d}: X = {h['coefficient_w_inverse_squared']:.9f} W^-2, "
              f"change = {change}")
    print("\n--- Independent direct triple-integral checks ---")
    for check in checks:
        label = "SPM" if check["spacing_ghz"] == 0 else f"XPM {check['spacing_ghz']:.0f} GHz"
        print(f"PASS: {label}, reduced 2D vs direct 3D difference = {check['absolute_2d_vs_3d_difference_db']:.3e} dB")

    print("\n--- Full-load central channel: 80 channels, 79 interferers ---")
    print(f"Closed-form eta_span = {aggregate['closed_eta_span_w_inverse_squared']:.6f} W^-2")
    print(f"Numerical eta_span = {aggregate['numerical_eta_span_w_inverse_squared']:.6f} W^-2")
    print(f"Aggregate coefficient error = {aggregate['eta_closed_over_numerical_error_db']:+.6f} dB")
    print("\n--- Illustrative 800 km comparison, retaining epsilon = 0 ---")
    for key, label in (("closed_form_at_own_optimal_power", "Closed form, own optimum"),
                       ("numerical_at_own_optimal_power", "Numerical, own optimum"),
                       ("numerical_at_previous_closed_form_power", "Numerical, previous power")):
        r = aggregate[key]
        print(f"{label}: P = {r['launch_power_dbm']:.6f} dBm, SNR = {r['snr_db']:.6f} dB, "
              f"Ideal capacity = {r['ideal_capacity_gbps']:.6f} Gbit/s")
    print(f"\nAll 81 coefficients met the refinement criterion. Integration time: {elapsed:.2f} s.")
    print("Refinement differences are not rigorous error bounds.")
    print("This isolates closed-form approximation error; coherent accumulation and experimental validation remain separate.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "nli_integral_results.json"
    figure_path = output_dir / "nli_integral_comparison.png"
    result = {
        "source": "Ives (2015), UCL PhD thesis, Table 3.1 and Eqs. (4.22)-(4.25), pp. 40 and 65-67.",
        "scope": "Single-span matched-filter SPM/XPM GN integrals for rectangular equal-bandwidth channels.",
        "assumptions": ["Same parameters as Step 26", "Both-polarization total powers", "No Raman scattering",
                        "No transceiver or ROADM penalties", "Other inter-channel FWM contributions remain omitted",
                        "The illustrative multi-span prediction retains epsilon=0"],
        "refinement_tolerance_db": args.tolerance_db,
        "convergence_rule": "Two successive order doublings below tolerance; not a rigorous error bound.",
        "parameters_and_closed_form_baseline": model.summary(),
        "coefficients": rows, "direct_3d_cross_checks": checks, "wdm_aggregate": aggregate,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "integration_elapsed_seconds": elapsed,
    }
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved numerical results: {result_path}")
    print(f"Figure destination: {figure_path}", flush=True)
    make_plot(rows, args.tolerance_db, figure_path, show=not args.no_show)


if __name__ == "__main__":
    main()
