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

# 1. 扫描每信道功率
P_dBm_scan = np.linspace(-10, 10, 401)
P_W_scan = 10 ** ((P_dBm_scan - 30) / 10)

# 2. 非线性噪声
# 教学示意值：整条 800 km 链路的系数，不是每跨段系数
eta_total_demo = 1e4  # [W^-2]
NLI_scan = eta_total_demo * P_W_scan ** 3

# 3. 比较两种模型
SNR_ase = P_W_scan / ASE_total
SNR_combined = P_W_scan / (ASE_total + NLI_scan)

SNR_ase_dB = 10 * np.log10(SNR_ase)
SNR_combined_dB = 10 * np.log10(SNR_combined)

# 4. 找到扫描曲线的最高点
idx = np.argmax(SNR_combined)
P_best_dBm = P_dBm_scan[idx]
SNR_best_dB = SNR_combined_dB[idx]

print("\n--- Illustrative NLI model ---")
print(f"Optimal power from sweep: {P_best_dBm:.2f} dBm")
print(f"Maximum SNR from sweep: {SNR_best_dB:.2f} dB")
# 解析方法计算最优功率
P_opt_W = (ASE_total / (2 * eta_total_demo)) ** (1 / 3)
P_opt_dBm = 10 * np.log10(P_opt_W / 1e-3)

NLI_opt = eta_total_demo * P_opt_W ** 3
SNR_opt = P_opt_W / (ASE_total + NLI_opt)

print(f"Analytical optimal power: {P_opt_dBm:.3f} dBm")
print(f"Analytical maximum SNR: {10 * np.log10(SNR_opt):.3f} dB")
print(f"NLI/ASE at optimum: {NLI_opt / ASE_total:.3f}")
# 5. 绘图
plt.figure(figsize=(7, 4.5))

plt.plot(P_dBm_scan, SNR_ase_dB,
         "--", label="ASE only", linewidth=2)

plt.plot(P_dBm_scan, SNR_combined_dB,
         label="ASE + NLI (illustrative eta)", linewidth=2)

plt.scatter(P_best_dBm, SNR_best_dB,
            color="red", zorder=5, label="Maximum SNR")

plt.xlabel("Launch power per channel (dBm)")
plt.ylabel("SNR (dB)")
plt.title("800 km link: illustrative nonlinear model")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("step2_snr_nli_demo.png", dpi=300)
plt.show()