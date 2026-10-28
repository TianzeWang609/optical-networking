"""Physical-layer equations from the step25 teaching baseline."""

from dataclasses import dataclass, field
from functools import cached_property
import math


@dataclass(frozen=True)
class PhysicalConfig:
    # 所有功率均为双偏振合计功率；距离单位为 km
    alpha_db_per_km: float = 0.22
    span_length_km: float = 80.0
    gamma_per_w_km: float = 1.3
    beta2_abs_s2_per_km: float = 21.3e-24
    symbol_rate_baud: float = 200e9
    channel_count: int = 25
    channel_spacing_hz: float = 400e9
    carrier_frequency_hz: float = 193.5e12
    noise_figure_db: float = 4.5
    fec_overhead: float = 0.20
    epsilon: float = 0.0

    def __post_init__(self):
        positive = (
            self.alpha_db_per_km, self.span_length_km, self.gamma_per_w_km,
            self.beta2_abs_s2_per_km, self.symbol_rate_baud,
            self.channel_spacing_hz, self.carrier_frequency_hz,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Physical parameters must be finite and positive.")
        if not isinstance(self.channel_count, int) or self.channel_count < 1:
            raise ValueError("channel_count must be a positive integer.")
        if self.channel_spacing_hz < self.symbol_rate_baud:
            raise ValueError("Channel spectra must not overlap in this model.")
        for value in (self.noise_figure_db, self.fec_overhead, self.epsilon):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Noise figure, FEC overhead and epsilon must be nonnegative.")


@dataclass(frozen=True)
class Modulation:
    name: str
    bits_dp: int
    required_snr_db: float


# 沿用原脚本的门限表；bits_dp 已包含双偏振，计算净速率时不要再乘 2
MODULATIONS = tuple(Modulation(*row) for row in (
    ("PM-BPSK", 2, 3.72), ("PM-QPSK", 4, 6.73),
    ("PM-8QAM", 6, 10.81), ("PM-16QAM", 8, 13.24),
    ("PM-32QAM", 10, 16.22), ("PM-64QAM", 12, 19.01),
    ("PM-128QAM", 14, 21.81), ("PM-256QAM", 16, 24.65),
    ("PM-512QAM", 18, 27.38), ("PM-1024QAM", 20, 30.27),
))


def dbm_to_w(power_dbm):
    return 10.0 ** ((power_dbm - 30.0) / 10.0)


def w_to_dbm(power_w):
    if not math.isfinite(power_w) or power_w <= 0:
        raise ValueError("Launch power must be finite and positive.")
    return 10.0 * math.log10(power_w / 1e-3)


@dataclass(frozen=True)
class PhysicalModel:
    config: PhysicalConfig = field(default_factory=PhysicalConfig)

    @cached_property
    def alpha_per_km(self):
        return self.config.alpha_db_per_km * math.log(10.0) / 10.0

    @cached_property
    def prefactor(self):
        c = self.config
        loss_factor = (1 - math.exp(-self.alpha_per_km * c.span_length_km)) ** 2
        return ((32 / 27) * c.gamma_per_w_km**2 * loss_factor
                / (4 * math.pi * self.alpha_per_km
                   * c.beta2_abs_s2_per_km * c.symbol_rate_baud**2))

    @cached_property
    def x_spm(self):
        c = self.config
        argument = (math.pi**2 * c.beta2_abs_s2_per_km * c.symbol_rate_baud**2
                    / (2 * self.alpha_per_km))
        return self.prefactor * math.asinh(argument)

    def xpm_coefficient(self, delta_f_hz):
        c = self.config
        df = abs(delta_f_hz)
        if not math.isfinite(df) or df < c.symbol_rate_baud:
            raise ValueError("XPM requires distinct, non-overlapping channels.")
        scale = math.pi**2 * c.beta2_abs_s2_per_km * c.symbol_rate_baud / self.alpha_per_km
        return self.prefactor * (
            math.asinh(scale * (df + c.symbol_rate_baud / 2))
            - math.asinh(scale * (df - c.symbol_rate_baud / 2))
        )

    @cached_property
    def x_xpm_sum(self):
        # 始终使用满载 WDM 中间信道的保守估计，不随空闲波长数量改变
        c = self.config
        center = c.channel_count // 2
        return sum(self.xpm_coefficient(abs(j - center) * c.channel_spacing_hz)
                   for j in range(c.channel_count) if j != center)

    @cached_property
    def ase_span_w(self):
        # 保留原脚本的高增益近似 F*h*nu*Rs*G
        c = self.config
        gain = 10 ** (c.alpha_db_per_km * c.span_length_km / 10)
        return (10 ** (c.noise_figure_db / 10) * 6.62607015e-34
                * c.carrier_frequency_hz * c.symbol_rate_baud * gain)

    def spans_for_length(self, length_km):
        ns = length_km / self.config.span_length_km
        if (not math.isfinite(ns) or ns <= 0
                or not math.isclose(ns, round(ns), rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Each link must contain a positive integer number of full spans.")
        return int(round(ns))

    def eta_total(self, spans):
        if not math.isfinite(spans) or spans < 1 or int(spans) != spans:
            raise ValueError("Span count must be a positive integer.")
        return spans ** (1 + self.config.epsilon) * self.x_spm + spans * self.x_xpm_sum

    def optimal_power_w(self, spans=10):
        # epsilon > 0 时，此最优功率依赖路径跨数
        return (spans * self.ase_span_w / (2 * self.eta_total(spans))) ** (1 / 3)

    def snr_linear(self, spans, power_w):
        if not math.isfinite(power_w) or power_w <= 0:
            raise ValueError("Launch power must be finite and positive.")
        eta = self.eta_total(spans)
        return power_w / (spans * self.ase_span_w + eta * power_w**3)

    def snr_db(self, spans, power_w):
        return 10 * math.log10(self.snr_linear(spans, power_w))

    def ideal_capacity_bps(self, spans, power_w):
        # 双偏振 Shannon 容量；与离散调制的净速率、实际承载业务量分开
        return 2 * self.config.symbol_rate_baud * math.log2(1 + self.snr_linear(spans, power_w))

    def net_rate_gbps(self, modulation):
        c = self.config
        return c.symbol_rate_baud * modulation.bits_dp / (1 + c.fec_overhead) / 1e9

    def best_modulation(self, snr_db, snr_margin_db=0.0):
        if not math.isfinite(snr_margin_db) or snr_margin_db < 0:
            raise ValueError("SNR margin must be finite and nonnegative.")
        supported = [m for m in MODULATIONS
                     if snr_db >= m.required_snr_db + snr_margin_db]
        return max(supported, key=lambda m: m.bits_dp, default=None)
