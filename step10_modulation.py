import math

# 光纤与信号参数
alpha_dB = 0.22       # 损耗，dB/km
Lspan_km = 80         # 单跨长度，km
gamma = 1.3          # 光纤非线性参数，1/(W·km)
beta2_abs = 21.3e-24 # 色散参数的绝对值，s²/km
Rs = 200e9           # 符号率，baud；假设零滚降矩形频谱

# 把 dB/km 换为指数衰减公式使用的 1/km
alpha = alpha_dB * math.log(10) / 10

# 论文式 (4.25)：分开计算，便于检查
loss_factor = (1 - math.exp(-alpha * Lspan_km)) ** 2

prefactor = (
    (32 / 27) * gamma**2 * loss_factor
    / (4 * math.pi * alpha * beta2_abs * Rs**2)
)

argument = math.pi**2 * beta2_abs * Rs**2 / (2 * alpha)

X_spm = prefactor * math.asinh(argument)

print(f"alpha = {alpha:.6f} km^-1")
print(f"Single-span X_spm = {X_spm:.3f} W^-2")
# 单跨 XPM 系数：论文式 (4.24)
# delta_f 的单位为 Hz；用于不同且频谱不重叠的信道
def xpm_coefficient(delta_f):
    df = abs(delta_f)

    scale = math.pi**2 * beta2_abs * Rs / alpha
    upper = scale * (df + Rs / 2)
    lower = scale * (df - Rs / 2)

    return prefactor * (
        math.asinh(upper) - math.asinh(lower)
    )


# 先测试三个频率间隔
print("\n--- XPM from a single interfering channel ---")

for spacing_GHz in [400, 800, 1600]:
    X_xpm = xpm_coefficient(spacing_GHz * 1e9)

    print(
        f"Frequency spacing {spacing_GHz:4d} GHz: "
        f"X_xpm = {X_xpm:.3f} W^-2"
    )
# WDM 基线配置：所有信道均开启，且发射功率相同
B_total = 10e12       # 可用频带，Hz
spacing = 400e9      # 相邻信道中心间隔，Hz
Nch = int(B_total / spacing)

# Python 从 0 编号：25 个信道的中间编号为 12
i = Nch // 2

X_xpm_sum = 0.0

for j in range(Nch):
    if j == i:
        continue  # 跳过自身；自身的贡献已由 SPM 计算

    delta_f = abs(j - i) * spacing
    X_xpm_sum += xpm_coefficient(delta_f)

# 被观察信道的单跨总非线性系数
eta_span = X_spm + X_xpm_sum

print("\n--- Central WDM channel: single-span results ---")
print(f"Number of channels: {Nch}")
print(f"Observed channel index: {i} (zero-based)")
print(f"Single-span SPM coefficient: {X_spm:.3f} W^-2")
print(f"Single-span XPM sum: {X_xpm_sum:.3f} W^-2")
print(f"Single-span eta_span: {eta_span:.3f} W^-2")
import numpy as np
import matplotlib.pyplot as plt

# 多跨累积：先采用非相干基线
Nspan = 10
epsilon = 0.0

eta_total = (
    Nspan**(1 + epsilon) * X_spm
    + Nspan * X_xpm_sum
)

# ASE：沿用之前的高增益近似
h = 6.62607015e-34
nu = 193.5e12
NF_dB = 4.5

F = 10**(NF_dB / 10)
G = 10**(alpha_dB * Lspan_km / 10)

ASE_span = F * h * nu * Rs * G
ASE_total = Nspan * ASE_span

# 扫描所有信道共同采用的单信道发射功率
P_dBm_scan = np.linspace(-10, 12, 441)
P_W_scan = 10**((P_dBm_scan - 30) / 10)

NLI_scan = eta_total * P_W_scan**3
SNR_scan = P_W_scan / (ASE_total + NLI_scan)
SNR_ase = P_W_scan / ASE_total

# 解析最优点
P_opt_W = (ASE_total / (2 * eta_total))**(1 / 3)
P_opt_dBm = 10 * np.log10(P_opt_W / 1e-3)

SNR_opt = P_opt_W / (
    ASE_total + eta_total * P_opt_W**3
)
SNR_opt_dB = 10 * np.log10(SNR_opt)

print("\n--- 800 km incoherent GN baseline ---")
print(f"eta_total = {eta_total:.3f} W^-2")
print(f"Total ASE = {ASE_total * 1e6:.3f} μW")
print(f"Analytical optimal power = {P_opt_dBm:.3f} dBm")
print(f"Analytical maximum SNR = {SNR_opt_dB:.3f} dB")

# 绘图
plt.figure(figsize=(7, 4.5))
plt.plot(P_dBm_scan, 10 * np.log10(SNR_ase),
         "--", label="ASE only")
plt.plot(P_dBm_scan, 10 * np.log10(SNR_scan),
         label="ASE + NLI", linewidth=2)
plt.scatter(P_opt_dBm, SNR_opt_dB,
            color="red", label="Maximum SNR", zorder=5)

plt.xlabel("Launch power per channel (dBm)")
plt.ylabel("Central-channel SNR (dB)")
plt.title(
    f"{Nspan * Lspan_km} km, {Nch} channels "
    f"(epsilon={epsilon:g})"
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("step4_snr_vs_power.png", dpi=300)

# --- 第 10 步：路径 SNR → 调制格式 → 单信道净速率 ---

# 论文表 4.2：
# (调制格式, 双偏振每符号比特数, 所需 SNR / dB)
# 使用 pre-FEC BER = 0.015，对应 20% FEC 开销
formats = [
    ("PM-BPSK",     2,  3.72),
    ("PM-QPSK",     4,  6.73),
    ("PM-8QAM",     6, 10.81),
    ("PM-16QAM",    8, 13.24),
    ("PM-32QAM",   10, 16.22),
    ("PM-64QAM",   12, 19.01),
    ("PM-128QAM",  14, 21.81),
    ("PM-256QAM",  16, 24.65),
    ("PM-512QAM",  18, 27.38),
    ("PM-1024QAM", 20, 30.27),
]

fec_overhead = 0.20
P0 = P_opt_W  # 各条路径均使用相同的单信道功率


def route_snr_dB(ns):
    """Return the central-channel SNR for a fully loaded route of ns spans."""
    ase = ns * ASE_span
    eta = ns**(1 + epsilon) * X_spm + ns * X_xpm_sum
    snr = P0 / (ase + eta * P0**3)
    return 10 * np.log10(snr)


print("\n--- Modulation formats and per-channel net rates at different distances ---")
print(f"Common per-channel power: {P_opt_dBm:.3f} dBm")

for ns in [10, 20, 30, 40, 50]:
    length_km = ns * Lspan_km
    snr_dB = route_snr_dB(ns)

    # 格式按阶数升序排列，保留满足门限的最高阶格式
    selected = None
    for name, bits_dp, required_dB in formats:
        if snr_dB >= required_dB:
            selected = (name, bits_dp)

    if selected is None:
        print(f"L = {length_km:4.0f} km: No available modulation format")
    else:
        name, bits_dp = selected
        rate_net = Rs * bits_dp / (1 + fec_overhead)

        print(
            f"L = {length_km:4.0f} km: "
            f"SNR = {snr_dB:.3f} dB, "
            f"{name:8s}, "
            f"Net rate = {rate_net / 1e9:.1f} Gbit/s"
        )

plt.show()