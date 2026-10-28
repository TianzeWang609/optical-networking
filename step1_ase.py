import math

# 1. 基本常数与信号参数
h = 6.62607015e-34       # 普朗克常数 [J·s]
nu = 193.5e12           # 光载波频率 [Hz]
Rs = 200e9              # 符号率 [baud]
NF_dB = 4.5             # 放大器噪声系数 [dB]

# 2. 光纤链路
alpha_dB = 0.22         # 光纤损耗 [dB/km]
Lspan_km = 80           # 每个跨段的长度 [km]
Nspan = 10              # 跨段数量

# 3. 转换为线性增益和噪声系数
loss_dB = alpha_dB * Lspan_km
G = 10 ** (loss_dB / 10)
F = 10 ** (NF_dB / 10)

# 4. ASE 噪声：采用论文式 (4.15) 的高增益近似
ASE_span = F * h * nu * Rs * G   # 每段噪声 [W]
ASE_total = Nspan * ASE_span    # 总噪声 [W]

# 5. 每信道双偏振总发射功率
P_dBm = 0
P_W = 10 ** ((P_dBm - 30) / 10)  # dBm 转换为 W

# 6. 计算 SNR
SNR_linear = P_W / ASE_total
SNR_dB = 10 * math.log10(SNR_linear)

# 7. 显示结果
print(f"Total distance: {Nspan * Lspan_km:.0f} km")
print(f"ASE per span: {ASE_span * 1e6:.3f} μW")
print(f"Total ASE: {ASE_total * 1e6:.3f} μW")
print(f"SNR: {SNR_dB:.2f} dB")
import numpy as np
import matplotlib.pyplot as plt

# 从 -10 到 +10 dBm，生成 201 个功率点
P_dBm_scan = np.linspace(-10, 10, 201)

# 将每个功率点转换成 W
P_W_scan = 10 ** ((P_dBm_scan - 30) / 10)

# 链路不变，因此 ASE_total 不变
SNR_scan = P_W_scan / ASE_total
SNR_dB_scan = 10 * np.log10(SNR_scan)

# 绘制 SNR 与发射功率的关系
plt.figure(figsize=(7, 4.5))
plt.plot(P_dBm_scan, SNR_dB_scan, label="ASE only", linewidth=2)

plt.xlabel("Launch power per channel (dBm)")
plt.ylabel("SNR (dB)")
plt.title(f"ASE-only link: {Nspan * Lspan_km} km")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 先保存图片，再显示窗口
plt.savefig("step1_snr_vs_power.png", dpi=300)
plt.show()