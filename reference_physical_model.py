"""Reusable reference physical layer for the Optical Networking project.

Source: D. J. Ives (2015), Coherent Optical Fibre Networking in the Nonlinear
Regime, UCL PhD thesis: Table 3.1 and Eqs. (4.15), (4.18)-(4.23), (4.33).

Default: numerical coherent SPM plus linearly accumulated numerical XPM.
An incoherent SPM/XPM mode is retained only as a controlled comparison.
No fixed epsilon is fitted or supplied. Capacity is the ideal dual-polarization
AWGN estimate, not a modulation/FEC net rate or network admitted traffic.

The quadrature functions at the end of this module are copied unchanged from
the checked Step 27/28 implementations. Earlier scripts and result files are
not required. Importing this module performs no calculations or printing.
"""

from collections import Counter
from dataclasses import dataclass
from functools import cached_property, lru_cache
import math
from numbers import Integral

import numpy as np
from scipy.special import roots_legendre


PLANCK_J_S = 6.62607015e-34
ONE_D_ORDERS = (32, 64, 128, 256, 512, 1024)
SINGLE_SPAN_ORDERS = (64, 128, 256, 512, 1024, 2048, 4096)
ACCUMULATION_MODES = ("coherent_spm", "incoherent")


@dataclass(frozen=True)
class ReferenceParameters:
    """Physical parameters from thesis Table 3.1; powers include both polarizations."""

    # 长度统一为 km；beta2 使用绝对值，因积分核对其符号不敏感
    alpha_db_per_km: float = 0.22
    span_length_km: float = 80.0
    gamma_per_w_km: float = 1.3
    beta2_abs_s2_per_km: float = 21.3e-24
    carrier_frequency_hz: float = 193.5e12
    symbol_rate_baud: float = 32e9
    channel_spacing_hz: float = 50e9
    channel_count: int = 80
    noise_figure_db: float = 5.0

    def __post_init__(self):
        for value in (self.alpha_db_per_km, self.span_length_km, self.gamma_per_w_km,
                      self.beta2_abs_s2_per_km, self.carrier_frequency_hz,
                      self.symbol_rate_baud, self.channel_spacing_hz):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Physical parameters must be finite and positive.")
        if not isinstance(self.channel_count, Integral) or isinstance(self.channel_count, bool) or self.channel_count < 1:
            raise ValueError("channel_count must be a positive integer.")
        if self.channel_spacing_hz < self.symbol_rate_baud:
            raise ValueError("Rectangular spectra must not overlap.")
        if not math.isfinite(self.noise_figure_db) or self.noise_figure_db < 0:
            raise ValueError("noise_figure_db must be finite and nonnegative.")


def _scalar_or_array(value):
    array = np.asarray(value, dtype=float)
    return float(array) if array.ndim == 0 else array


def dbm_to_w(power_dbm):
    """Convert finite per-channel launch powers in dBm to positive watts."""
    values = np.asarray(power_dbm, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Launch powers must be finite.")
    with np.errstate(over="ignore", under="ignore"):
        power = np.power(10.0, (values - 30.0)/10.0)
    if not np.all(np.isfinite(power) & (power > 0)):
        raise ValueError("Launch power is outside the representable positive range.")
    return _scalar_or_array(power)


class ReferencePhysicalModel:
    """Equal-power, full-load, identical-span SPM/XPM GN reference model.

    Public methods accept a positive integer span count. Use spans_for_distance
    to convert an exact multiple of the span length; partial spans are not rounded.
    SNR/capacity methods accept scalar or array launch powers in dBm per channel.
    """

    def __init__(self, parameters=None, channel_index=None, tolerance_db=1e-4):
        self._parameters = parameters if parameters is not None else ReferenceParameters()
        if not isinstance(self._parameters, ReferenceParameters):
            raise TypeError("parameters must be ReferenceParameters.")
        index = self.p.channel_count // 2 if channel_index is None else channel_index
        if (not isinstance(index, Integral) or isinstance(index, bool)
                or not 0 <= index < self.p.channel_count):
            raise ValueError("channel_index must be a valid zero-based integer index.")
        if not math.isfinite(tolerance_db) or tolerance_db <= 0:
            raise ValueError("tolerance_db must be finite and positive.")
        self._channel_index = int(index)
        self._tolerance_db = float(tolerance_db)

    @property
    def p(self):
        return self._parameters

    @property
    def channel_index(self):
        return self._channel_index

    @property
    def tolerance_db(self):
        return self._tolerance_db

    @cached_property
    def alpha_per_km(self):
        return self.p.alpha_db_per_km * math.log(10.0)/10.0

    @cached_property
    def ase_span_w(self):
        # 沿用论文式 4.15 的噪声系数约定：NF_linear*h*nu*Rs*G
        gain = 10.0**(self.p.alpha_db_per_km*self.p.span_length_km/10.0)
        return (10.0**(self.p.noise_figure_db/10.0) * PLANCK_J_S
                * self.p.carrier_frequency_hz * self.p.symbol_rate_baud * gain)

    @staticmethod
    def _span_count(spans):
        if not isinstance(spans, Integral) or isinstance(spans, bool) or spans < 1:
            raise ValueError("spans must be a positive integer.")
        return int(spans)

    @staticmethod
    def _mode(accumulation):
        if accumulation not in ACCUMULATION_MODES:
            raise ValueError(f"accumulation must be one of {ACCUMULATION_MODES}.")
        return accumulation

    def spans_for_distance(self, length_km):
        """Convert an exact span-length multiple to a span count, without rounding up."""
        if not math.isfinite(length_km) or length_km <= 0:
            raise ValueError("length_km must be finite and positive.")
        ratio = length_km/self.p.span_length_km
        nearest = round(ratio)
        if nearest < 1 or not math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Distance must be an integer multiple of the span length; partial spans are not modeled.")
        return int(nearest)

    @lru_cache(maxsize=None)
    def _spm_result(self, spans):
        return converged_spm(self, spans, self.tolerance_db)

    def spm_coefficient(self, spans, *, accumulation="coherent_spm"):
        """Return accumulated SPM coefficient in W^-2."""
        spans = self._span_count(spans)
        self._mode(accumulation)
        if accumulation == "incoherent":
            return spans*self._spm_result(1)["coherent_spm_w_inverse_squared"]
        return self._spm_result(spans)["coherent_spm_w_inverse_squared"]

    @cached_property
    def _xpm_results(self):
        offsets = Counter(abs(j - self.channel_index) for j in range(self.p.channel_count)
                          if j != self.channel_index)
        results = []
        for offset, multiplicity in sorted(offsets.items()):
            history = []
            stable_count = 0
            for order in SINGLE_SPAN_ORDERS:
                value = numerical_coefficient_at_order(self, offset*self.p.channel_spacing_hz, order)
                if not math.isfinite(value) or value <= 0:
                    raise RuntimeError("Single-span XPM must be finite and positive.")
                change = None if not history else abs(10*math.log10(value/history[-1]["coefficient_w_inverse_squared"]))
                history.append({"order_per_dimension": order, "coefficient_w_inverse_squared": value,
                                "absolute_change_from_previous_db": change})
                stable_count = stable_count + 1 if change is not None and change < self.tolerance_db else 0
                if stable_count >= 2:
                    results.append({"offset": offset, "multiplicity": multiplicity,
                        "spacing_ghz": offset*self.p.channel_spacing_hz/1e9,
                        "coefficient_w_inverse_squared": value, "converged": True, "history": history})
                    break
            else:
                raise RuntimeError(f"Single-span XPM did not converge at offset {offset}.")
        return results

    @cached_property
    def single_span_xpm_sum(self):
        return sum(row["multiplicity"]*row["coefficient_w_inverse_squared"] for row in self._xpm_results)

    def eta_total(self, spans, *, accumulation="coherent_spm"):
        """Return total SPM+XPM coefficient in W^-2; XPM remains linear in both modes."""
        spans = self._span_count(spans)
        return self.spm_coefficient(spans, accumulation=accumulation) + spans*self.single_span_xpm_sum

    def noise_components(self, power_dbm, spans, *, accumulation="coherent_spm"):
        """Return ASE, SPM NLI and XPM NLI in watts for equal per-channel powers."""
        spans = self._span_count(spans)
        power = np.asarray(dbm_to_w(power_dbm))
        with np.errstate(over="ignore", under="ignore"):
            spm = self.spm_coefficient(spans, accumulation=accumulation)*power**3
            xpm = spans*self.single_span_xpm_sum*power**3
        if not np.all(np.isfinite(spm + xpm)):
            raise ValueError("NLI noise overflowed; choose a physically relevant launch power.")
        ase = np.full_like(power, spans*self.ase_span_w)
        return {"signal_w": _scalar_or_array(power), "ase_w": _scalar_or_array(ase),
                "spm_nli_w": _scalar_or_array(spm), "xpm_nli_w": _scalar_or_array(xpm),
                "total_nli_w": _scalar_or_array(spm + xpm),
                "total_noise_w": _scalar_or_array(ase + spm + xpm)}

    def snr_linear(self, power_dbm, spans, *, accumulation="coherent_spm"):
        noise = self.noise_components(power_dbm, spans, accumulation=accumulation)
        return _scalar_or_array(np.asarray(noise["signal_w"])/noise["total_noise_w"])

    def snr_db(self, power_dbm, spans, *, accumulation="coherent_spm"):
        return _scalar_or_array(10*np.log10(self.snr_linear(power_dbm, spans, accumulation=accumulation)))

    def capacity_gbps(self, power_dbm, spans, *, accumulation="coherent_spm"):
        """Ideal AWGN capacity of one dual-polarization channel, in Gbit/s."""
        snr = self.snr_linear(power_dbm, spans, accumulation=accumulation)
        return _scalar_or_array(2*self.p.symbol_rate_baud*np.log1p(snr)/math.log(2.0)/1e9)

    def optimal_power_dbm(self, spans, *, accumulation="coherent_spm"):
        spans = self._span_count(spans)
        power_w = (spans*self.ase_span_w/(2*self.eta_total(spans, accumulation=accumulation)))**(1/3)
        return 10*math.log10(power_w/1e-3)

    def link_summary(self, spans, power_dbm=None, *, accumulation="coherent_spm"):
        """Evaluate one scalar operating point; default power is the model's optimum."""
        spans = self._span_count(spans)
        optimum = self.optimal_power_dbm(spans, accumulation=accumulation)
        launch = optimum if power_dbm is None else float(power_dbm)
        return {"spans": spans, "length_km": spans*self.p.span_length_km,
                "accumulation": accumulation, "launch_power_dbm": launch,
                "optimal_power_dbm": optimum,
                "eta_total_w_inverse_squared": self.eta_total(spans, accumulation=accumulation),
                "snr_db": self.snr_db(launch, spans, accumulation=accumulation),
                "ideal_capacity_gbps": self.capacity_gbps(launch, spans, accumulation=accumulation),
                "noise_w": self.noise_components(launch, spans, accumulation=accumulation)}

    def quadrature_records(self, span_counts):
        """Return JSON-compatible copies of the numerical refinement records."""
        import copy
        spans = [self._span_count(n) for n in span_counts]
        return copy.deepcopy({"single_span_xpm": self._xpm_results,
                              "coherent_spm": [self._spm_result(n) for n in spans]})


# 以下七个数值积分函数从已复核的 Steps 27–28 原样复用

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
