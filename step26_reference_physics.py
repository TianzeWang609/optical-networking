"""Check a closed-form GN baseline using the Ives thesis reference parameters.

Reference: D. J. Ives, Coherent Optical Fibre Networking in the Nonlinear
Regime, PhD thesis, University College London, December 2015.
Parameters: Table 3.1 (p. 40). ASE: Eq. (4.15), p. 62.
Single-span coefficients: Eqs. (4.24)-(4.25), p. 67.
Accumulation: Eq. (6.1), p. 99, specialized here to epsilon = 0.

This is a parameter and algebraic consistency check. It is NOT a reproduction
of the thesis's numerically integrated, coherently accumulated GN model.
All launch and noise powers include both polarizations. Capacity is the
equivalent-AWGN estimate per channel, before any discrete modulation or FEC.

Run: python step26_reference_physics.py
Batch run: python step26_reference_physics.py --no-show
Only NumPy and Matplotlib are required. No earlier project scripts are imported.
"""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import cached_property
import json
import math
from pathlib import Path


PLANCK_J_S = 6.62607015e-34


@dataclass(frozen=True)
class Parameters:
    # 论文表 3.1；公式中的 beta2 取绝对值，所有长度统一使用 km
    alpha_db_per_km: float = 0.22
    span_length_km: float = 80.0
    gamma_per_w_km: float = 1.3
    beta2_abs_s2_per_km: float = 21.3e-24
    carrier_frequency_hz: float = 193.5e12
    symbol_rate_baud: float = 32e9
    channel_spacing_hz: float = 50e9
    channel_count: int = 80
    noise_figure_db: float = 5.0
    # 本步骤只检查非相干累积；epsilon=0 是我们的简化，不是论文原值
    epsilon: float = 0.0

    def __post_init__(self):
        positive = (self.alpha_db_per_km, self.span_length_km,
                    self.gamma_per_w_km, self.beta2_abs_s2_per_km,
                    self.carrier_frequency_hz, self.symbol_rate_baud,
                    self.channel_spacing_hz)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Physical parameters must be finite and positive.")
        if (not isinstance(self.channel_count, int)
                or isinstance(self.channel_count, bool) or self.channel_count < 1):
            raise ValueError("channel_count must be a positive integer.")
        if self.channel_spacing_hz < self.symbol_rate_baud:
            raise ValueError("Rectangular signal spectra must not overlap.")
        if not math.isfinite(self.noise_figure_db) or self.noise_figure_db < 0:
            raise ValueError("noise_figure_db must be finite and nonnegative.")
        if self.epsilon != 0:
            raise ValueError("Step 26 checks epsilon = 0 only; use a separate experiment for coherence.")


def dbm_to_w(power_dbm):
    return 10.0 ** ((power_dbm - 30.0) / 10.0)


def w_to_dbm(power_w):
    return 10.0 * math.log10(power_w / 1e-3)


def span_coefficient(alpha, length, gamma, beta2_abs, symbol_rate, spacing_hz=0.0):
    """Evaluate thesis Eq. (4.24) or (4.25) in any consistent length unit."""
    loss_factor = (-math.expm1(-alpha * length)) ** 2
    prefactor = (32.0 / 27.0) * gamma**2 * loss_factor
    prefactor /= 4.0 * math.pi * alpha * beta2_abs * symbol_rate**2
    if spacing_hz == 0:
        # SPM 使用单独的式 (4.25)，不能把 XPM 式直接代入零间隔
        return prefactor * math.asinh(math.pi**2 * beta2_abs * symbol_rate**2 / (2 * alpha))
    if spacing_hz < symbol_rate:
        raise ValueError("XPM spacing must be at least the symbol rate.")
    scale = math.pi**2 * beta2_abs * symbol_rate / alpha
    return prefactor * (math.asinh(scale * (spacing_hz + symbol_rate / 2))
                        - math.asinh(scale * (spacing_hz - symbol_rate / 2)))


class ClosedFormGN:
    def __init__(self, parameters):
        self.p = parameters

    @cached_property
    def alpha_per_km(self):
        # 功率衰减系数：使用 ln(10)/10，不能使用场幅度的 ln(10)/20
        return self.p.alpha_db_per_km * math.log(10.0) / 10.0

    @property
    def center_index(self):
        # 80 个信道有两个等价中心，零起始编号 39 和 40；这里选择 40
        return self.p.channel_count // 2

    def coefficient(self, spacing_hz=0.0):
        p = self.p
        return span_coefficient(self.alpha_per_km, p.span_length_km,
                                p.gamma_per_w_km, p.beta2_abs_s2_per_km,
                                p.symbol_rate_baud, spacing_hz)

    @cached_property
    def x_spm(self):
        return self.coefficient()

    @cached_property
    def x_xpm_by_channel(self):
        p = self.p
        # j 遍历两侧所有实际干扰信道；XPM 系数已包含退化因子，不再额外乘 2
        return tuple(sum(self.coefficient(abs(j - i) * p.channel_spacing_hz)
                         for j in range(p.channel_count) if j != i)
                     for i in range(p.channel_count))

    @property
    def x_xpm_sum(self):
        return self.x_xpm_by_channel[self.center_index]

    @property
    def eta_span(self):
        return self.x_spm + self.x_xpm_sum

    @cached_property
    def ase_span_w(self):
        p = self.p
        # 沿用论文式 (4.15) 的 NF 定义；只补偿光纤损耗，不包含 ROADM 损耗
        gain = 10.0 ** (p.alpha_db_per_km * p.span_length_km / 10.0)
        noise_factor = 10.0 ** (p.noise_figure_db / 10.0)
        return noise_factor * PLANCK_J_S * p.carrier_frequency_hz * p.symbol_rate_baud * gain

    def noise_w(self, spans, power_w):
        if not math.isfinite(spans) or spans < 1 or int(spans) != spans:
            raise ValueError("Use a positive integer number of complete spans.")
        if not math.isfinite(power_w) or power_w <= 0:
            raise ValueError("Launch power must be finite and positive.")
        return spans * self.ase_span_w, spans * self.eta_span * power_w**3

    def snr_linear(self, spans, power_w):
        ase, nli = self.noise_w(spans, power_w)
        return power_w / (ase + nli)

    def snr_db(self, spans, power_w):
        return 10.0 * math.log10(self.snr_linear(spans, power_w))

    def capacity_gbps(self, spans, power_w):
        # 双偏振理想 AWGN 容量估计；不是实际调制净速率或网络业务量
        return 2.0 * self.p.symbol_rate_baud * math.log2(1 + self.snr_linear(spans, power_w)) / 1e9

    def optimal_power_w(self):
        # epsilon=0 且所有跨参数相同时，Ns 约去，最优功率与跨数无关
        return (self.ase_span_w / (2.0 * self.eta_span)) ** (1.0 / 3.0)

    def summary(self, spans=10):
        p = self.p
        power = self.optimal_power_w()
        ase, nli = self.noise_w(spans, power)
        return {
            "parameters": asdict(p), "spans": spans,
            "length_km": spans * p.span_length_km,
            "alpha_per_km": self.alpha_per_km,
            "span_loss_db": p.alpha_db_per_km * p.span_length_km,
            "outer_spectral_width_hz": (p.channel_count - 1) * p.channel_spacing_hz + p.symbol_rate_baud,
            "sum_signal_bandwidths_hz": p.channel_count * p.symbol_rate_baud,
            "nominal_slot_bandwidth_hz": p.channel_count * p.channel_spacing_hz,
            "center_index_zero_based": self.center_index,
            "x_spm_w_inverse_squared": self.x_spm,
            "x_xpm_sum_w_inverse_squared": self.x_xpm_sum,
            "eta_span_w_inverse_squared": self.eta_span,
            "eta_total_w_inverse_squared": spans * self.eta_span,
            "ase_span_w": self.ase_span_w, "ase_total_w": ase,
            "nli_at_optimum_w": nli,
            "optimal_power_w": power, "optimal_power_dbm": w_to_dbm(power),
            "full_load_total_power_dbm": w_to_dbm(p.channel_count * power),
            "maximum_snr_db": self.snr_db(spans, power),
            "maximum_ideal_capacity_gbps": self.capacity_gbps(spans, power),
        }


def consistency_checks(model, power_grid_dbm):
    """Check units, indexing, and analytic identities; not physical accuracy."""
    checks = []

    def record(name, passed, observed, expected):
        checks.append({"name": name, "passed": bool(passed),
                       "observed": observed, "expected": expected})

    p = model.p
    errors = []
    for spacing in (0, p.channel_spacing_hz, (p.channel_count - 1) * p.channel_spacing_hz):
        # 将 alpha、gamma、beta2 换为每米，将跨长换为米，结果应保持不变
        value_m = span_coefficient(model.alpha_per_km / 1000, p.span_length_km * 1000,
                                   p.gamma_per_w_km / 1000, p.beta2_abs_s2_per_km / 1000,
                                   p.symbol_rate_baud, spacing)
        value_km = model.coefficient(spacing)
        errors.append(abs(value_m / value_km - 1.0))
    record("Consistent metre/kilometre units", max(errors) < 1e-11,
           max(errors), "Maximum relative difference < 1e-11")

    values = model.x_xpm_by_channel
    symmetry_error = max(abs(a - b) / max(values) for a, b in zip(values, reversed(values)))
    center_is_worst = math.isclose(model.x_xpm_sum, max(values), rel_tol=1e-12)
    record("Symmetric WDM comb and worst central channel", symmetry_error < 1e-12 and center_is_worst,
           {"symmetry_relative_error": symmetry_error, "center_is_worst": center_is_worst},
           "Mirror symmetry and maximum XPM on a central channel")

    power = model.optimal_power_w()
    ase, nli = model.noise_w(10, power)
    record("NLI / ASE at analytical optimum", math.isclose(nli / ase, 0.5, rel_tol=1e-12),
           nli / ase, 0.5)

    difference = model.snr_db(20, power) - model.snr_db(10, power)
    record("SNR change when span count doubles", math.isclose(difference, -10 * math.log10(2), abs_tol=1e-12),
           difference, -10 * math.log10(2))

    # 网格极值通过直接评价 SNR 获得，不使用解析导数求解
    grid_optima = {}
    for spans in (10, 20, 50):
        best = max(power_grid_dbm, key=lambda x: model.snr_linear(spans, dbm_to_w(float(x))))
        grid_optima[str(spans)] = float(best)
    grid_step = float(power_grid_dbm[1] - power_grid_dbm[0])
    max_power_error = max(abs(x - w_to_dbm(power)) for x in grid_optima.values())
    record("Direct power sweep versus analytical optimum", max_power_error <= grid_step + 1e-10,
           {"grid_optima_dbm_by_spans": grid_optima, "maximum_error_db": max_power_error},
           "Agreement within one grid step; same optimum for all lengths at epsilon = 0")

    failed = [check["name"] for check in checks if not check["passed"]]
    if failed:
        raise RuntimeError("Consistency checks failed: " + "; ".join(failed))
    return checks


def print_summary(summary, checks, distance_samples, previous_summary):
    p = summary["parameters"]
    print("--- Step 26: thesis reference parameters ---")
    print("Source: Ives (2015), Table 3.1; closed-form Eqs. (4.24)-(4.25).")
    print("Accumulation: epsilon = 0 (incoherent baseline, not the full thesis model).")
    print(f"Symbol rate = {p['symbol_rate_baud'] / 1e9:.1f} GBd")
    print(f"Channel spacing = {p['channel_spacing_hz'] / 1e9:.1f} GHz")
    print(f"Channels = {p['channel_count']}; noise figure = {p['noise_figure_db']:.1f} dB")
    print(f"Outer spectral width = {summary['outer_spectral_width_hz'] / 1e12:.3f} THz")
    print(f"Sum of signal bandwidths = {summary['sum_signal_bandwidths_hz'] / 1e12:.3f} THz")
    print(f"Nominal slot bandwidth = {summary['nominal_slot_bandwidth_hz'] / 1e12:.3f} THz")
    print(f"Observed channel index = {summary['center_index_zero_based']} (zero-based)")
    print("\n--- Single-span coefficients ---")
    print(f"alpha = {summary['alpha_per_km']:.9f} km^-1")
    print(f"Span loss = {summary['span_loss_db']:.3f} dB")
    print(f"X_spm = {summary['x_spm_w_inverse_squared']:.6f} W^-2")
    print(f"X_xpm_sum = {summary['x_xpm_sum_w_inverse_squared']:.6f} W^-2")
    print(f"eta_span = {summary['eta_span_w_inverse_squared']:.6f} W^-2")
    print(f"ASE per span = {summary['ase_span_w'] * 1e6:.6f} uW")
    print("\n--- 800 km: full-load central channel ---")
    print(f"eta_total = {summary['eta_total_w_inverse_squared']:.6f} W^-2")
    print(f"Total ASE = {summary['ase_total_w'] * 1e6:.6f} uW")
    print(f"Analytical optimal power = {summary['optimal_power_dbm']:.6f} dBm per channel")
    print(f"Full-load total power = {summary['full_load_total_power_dbm']:.6f} dBm")
    print(f"Maximum SNR = {summary['maximum_snr_db']:.6f} dB")
    print(f"Maximum ideal AWGN capacity = {summary['maximum_ideal_capacity_gbps']:.6f} Gbit/s per channel")
    print("\n--- Distance dependence at fixed optimal power ---")
    for row in distance_samples:
        print(f"L = {row['length_km']:4.0f} km: SNR = {row['snr_db']:.6f} dB, "
              f"Ideal capacity = {row['ideal_capacity_gbps']:.6f} Gbit/s")
    print("\n--- Algebraic and implementation consistency ---")
    for check in checks:
        print(f"PASS: {check['name']}")
    print("These checks do not establish physical-model accuracy.")
    print("\n--- Previous parameter set, for comparison only ---")
    print("200 GBd, 400 GHz spacing, 25 channels, 4.5 dB noise figure; epsilon = 0.")
    print(f"Optimal power = {previous_summary['optimal_power_dbm']:.6f} dBm")
    print(f"Maximum SNR at 800 km = {previous_summary['maximum_snr_db']:.6f} dB")
    print(f"Maximum ideal capacity = {previous_summary['maximum_ideal_capacity_gbps']:.6f} Gbit/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Save results without opening a plot window.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results_step26")
    args = parser.parse_args()

    import numpy as np
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = ClosedFormGN(Parameters())
    # 原配置仅用于数值对照，不修改旧文件，也不运行之前的网络分配实验
    previous = ClosedFormGN(replace(model.p, symbol_rate_baud=200e9,
                                   channel_spacing_hz=400e9, channel_count=25, noise_figure_db=4.5))
    summary = model.summary()
    previous_summary = previous.summary()
    power_grid_dbm = np.linspace(-12.0, 12.0, 2401)
    checks = consistency_checks(model, power_grid_dbm)
    optimum = model.optimal_power_w()
    distance_samples = [{"length_km": ns * model.p.span_length_km,
                         "spans": ns, "snr_db": model.snr_db(ns, optimum),
                         "ideal_capacity_gbps": model.capacity_gbps(ns, optimum)}
                        for ns in (10, 20, 50)]

    power_curves = []
    for spans in (10, 20, 50):
        power_curves.append({
            "spans": spans, "length_km": spans * model.p.span_length_km,
            "snr_db": [model.snr_db(spans, dbm_to_w(float(x))) for x in power_grid_dbm],
            "ideal_capacity_gbps": [model.capacity_gbps(spans, dbm_to_w(float(x))) for x in power_grid_dbm],
        })
    distance_spans = range(1, 101)
    distance_curve = {"length_km": [ns * model.p.span_length_km for ns in distance_spans],
                      "ideal_capacity_gbps": [model.capacity_gbps(ns, optimum) for ns in distance_spans]}

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "scope": "Reference parameters; closed-form GN; epsilon=0; no ROADM or transceiver penalties.",
        "reference": "Ives, UCL PhD thesis, December 2015: Table 3.1 and Eqs. (4.15), (4.24), (4.25), (6.1).",
        "validation_scope": "Parameter transcription, unit conversion, symmetry, analytic identities and grid maximization only.",
        "pending_validation": "Matched-filter numerical GN integration, coherent accumulation, and independent physical validation.",
        "reference_summary": summary, "previous_parameter_summary": previous_summary,
        "consistency_checks": checks, "distance_samples": distance_samples,
        "single_interferer_xpm": [{"spacing_ghz": df, "x_xpm_w_inverse_squared": model.coefficient(df * 1e9)}
                                  for df in (50, 100, 200, 400, 800, 1600, 4000)],
        "power_grid_dbm": power_grid_dbm.tolist(), "power_curves": power_curves,
        "distance_curve": distance_curve,
    }
    json_path = output_dir / "reference_physics_results.json"
    json_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    colors = ("#17476b", "#25867a", "#ba6629")
    for row, color in zip(power_curves, colors):
        label = f"{row['length_km']:.0f} km"
        axes[0].plot(power_grid_dbm, row["snr_db"], color=color, label=label, linewidth=1.8)
        axes[1].plot(power_grid_dbm, row["ideal_capacity_gbps"], color=color, label=label, linewidth=1.8)
    for ax in axes[:2]:
        ax.axvline(summary["optimal_power_dbm"], color="#555555", linestyle="--", linewidth=1.1,
                   label=f"Optimal power: {summary['optimal_power_dbm']:.3f} dBm")
        ax.set_xlabel("Per-channel launch power (dBm)")
        ax.set_xlim(-12, 12)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set(title="SNR versus launch power", ylabel="Symbol SNR (dB)")
    axes[1].set(title="Ideal capacity versus launch power", ylabel="Per-channel capacity (Gbit/s)")
    axes[2].plot(distance_curve["length_km"], distance_curve["ideal_capacity_gbps"],
                 color=colors[0], linewidth=1.8, label="Complete 80 km spans")
    for row in distance_samples:
        axes[2].scatter(row["length_km"], row["ideal_capacity_gbps"], color=colors[0], s=28, zorder=3)
    axes[2].set(title="Ideal capacity versus distance", xlabel="Distance (km)",
                ylabel="Per-channel capacity (Gbit/s)", xlim=(0, 8000))
    axes[2].text(0.97, 0.95, f"Fixed power: {summary['optimal_power_dbm']:.3f} dBm\n"
                 "Lines connect complete-span samples", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle("Thesis reference parameters | 32 GBd / 50 GHz / 80 channels / NF 5 dB\n"
                 "Closed-form GN, full-load central channel, incoherent accumulation (epsilon = 0)", fontsize=12)
    figure_path = output_dir / "reference_physics.png"
    fig.savefig(figure_path, dpi=180)

    print_summary(summary, checks, distance_samples, previous_summary)
    print(f"\nSaved numerical results: {json_path}")
    print(f"Saved figure: {figure_path}")
    if args.no_show:
        plt.close(fig)
    else:
        print("Close the figure window to finish.")
        plt.show()


if __name__ == "__main__":
    main()
