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

# 单信道双偏振容量：使用线性 SNR
C_scan = 2 * Rs * np.log2(1 + SNR_scan)
C_ase = 2 * Rs * np.log2(1 + SNR_ase)
C_opt = 2 * Rs * np.log2(1 + SNR_opt)

print("\n--- Ideal capacity estimate for the central channel ---")
print(f"Power at maximum capacity = {P_opt_dBm:.3f} dBm")
print(f"Maximum capacity = {C_opt / 1e12:.3f} Tbit/s")
print(f"Maximum capacity = {C_opt / 1e9:.3f} Gbit/s")

plt.figure(figsize=(7, 4.5))

plt.plot(P_dBm_scan, C_ase / 1e12,
         "--", label="ASE only")
plt.plot(P_dBm_scan, C_scan / 1e12,
         label="ASE + NLI", linewidth=2)
plt.scatter(P_opt_dBm, C_opt / 1e12,
            color="red", label="Maximum capacity", zorder=5)

plt.xlabel("Launch power per channel (dBm)")
plt.ylabel("Central-channel capacity estimate (Tbit/s)")
plt.title(
    f"{Nspan * Lspan_km} km, {Nch} channels "
    f"(epsilon={epsilon:g})"
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("step5_capacity_vs_power.png", dpi=300)
# 扫描跨数：1～50 跨，即 80～4000 km
Nspan_scan = np.arange(1, 51)
L_km_scan = Nspan_scan * Lspan_km

# 所有距离采用相同的单信道发射功率
P_fixed_W = P_opt_W

# 每个距离对应的 ASE 和非线性系数
ASE_L = Nspan_scan * ASE_span

eta_L = (
    Nspan_scan**(1 + epsilon) * X_spm
    + Nspan_scan * X_xpm_sum
)

# 中间信道的 SNR 和容量
SNR_L = P_fixed_W / (
    ASE_L + eta_L * P_fixed_W**3
)
C_L = 2 * Rs * np.log2(1 + SNR_L)

print("\n--- Capacity versus distance ---")
print(f"Fixed per-channel power: {P_opt_dBm:.3f} dBm")

for n in [10, 20, 50]:
    idx = n - 1
    print(
        f"L = {L_km_scan[idx]:4.0f} km: "
        f"C = {C_L[idx] / 1e12:.3f} Tbit/s"
    )

plt.figure(figsize=(7, 4.5))
plt.plot(L_km_scan, C_L / 1e12, linewidth=2,
         label=f"Fixed power: {P_opt_dBm:.2f} dBm/channel")

plt.xlabel("Total transmission distance (km)")
plt.ylabel("Central-channel capacity estimate (Tbit/s)")
plt.title(f"{Nch} channels (epsilon={epsilon:g})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("step6_capacity_vs_distance.png", dpi=300)
plt.show()