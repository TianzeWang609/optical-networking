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