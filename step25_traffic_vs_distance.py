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
# --- 第 11 步：计算候选路径的传输能力 ---

# 双向链路：键中的节点按字母顺序排列，数值单位为 km
links_km = {
    ("A", "B"): 400,
    ("B", "D"): 400,
    ("A", "C"): 800,
    ("C", "D"): 800,
    ("B", "C"): 1200,
}

# 本步先手动指定三条候选路径
import networkx as nx
from itertools import islice

# --- 第 14 步：自动生成候选最短路径 ---

# 根据已有链路数据建立无向图，以距离作为边的权重
graph = nx.Graph()

for (u, v), length_km in links_km.items():
    graph.add_edge(u, v, length_km=length_km)

source_node = "A"
target_node = "D"
K = 4

# 按总距离从短到长生成不重复节点的路径，最多取 K 条
path_generator = nx.shortest_simple_paths(
    graph,
    source_node,
    target_node,
    weight="length_km",
)

routes = list(islice(path_generator, K))

print("\n--- Automatically generated candidate routes ---")

for rank, route in enumerate(routes, start=1):
    length_km = sum(
        graph[u][v]["length_km"]
        for u, v in zip(route[:-1], route[1:])
    )

    route_name = " -> ".join(route)

    print(
        f"Route {rank}: {route_name}, "
        f"L = {length_km:.0f} km"
    )

print("\n--- Candidate routes from A to D ---")

for route in routes:
    length_km = 0

    # 依次取出路径中的相邻节点，并累加链路长度
    for u, v in zip(route[:-1], route[1:]):
        key = tuple(sorted((u, v)))
        length_km += links_km[key]

    # 本例每条链路均由完整的 80 km 跨组成
    ns = int(length_km / Lspan_km)
    snr_dB = route_snr_dB(ns)

    # 选择满足 SNR 门限的最高阶调制格式
    selected = None
    for name, bits_dp, required_dB in formats:
        if snr_dB >= required_dB:
            selected = (name, bits_dp)

    route_name = " -> ".join(route)

    if selected is None:
        print(f"{route_name}: No supported modulation format")
    else:
        name, bits_dp = selected
        rate_net = Rs * bits_dp / (1 + fec_overhead)

        print(
            f"{route_name}: "
            f"L = {length_km:.0f} km, "
            f"Ns = {ns}, "
            f"SNR = {snr_dB:.3f} dB, "
            f"{name}, "
            f"Net rate = {rate_net / 1e9:.1f} Gbit/s"
        )

# --- 第 12 步：检查候选路径的波长连续性 ---

# 人为设置各链路的空闲波长编号，其余编号视为已占用
# 波长编号采用 1 到 Nch，物理模型仍保留 Nch = 25
free_wavelengths = {
    ("A", "B"): {1, 2},
    ("B", "D"): {3, 4},
    ("A", "C"): {5, 6},
    ("C", "D"): {6, 7},
    ("B", "C"): {2, 6},
}

print("\n--- Wavelength continuity check ---")

for route in routes:
    # 从全部波长开始，逐段取空闲波长集合的交集
    common_free = set(range(1, Nch + 1))

    for u, v in zip(route[:-1], route[1:]):
        key = tuple(sorted((u, v)))
        common_free = common_free.intersection(
            free_wavelengths[key]
        )

    route_name = " -> ".join(route)

    if common_free:
        print(
            f"{route_name}: "
            f"Common free wavelengths = {sorted(common_free)}"
        )
    else:
        print(
            f"{route_name}: "
            "No common free wavelength"
        )
# --- 第 13 步：接受连接请求并更新波长占用状态 ---

def allocate_connection(demand_gbps, wavelength_rule="max"):
    # 按给定顺序检查候选路径，目前按距离从短到长排列
    for route in routes:
        link_keys = [
            tuple(sorted((u, v)))
            for u, v in zip(route[:-1], route[1:])
        ]

        # 计算路径距离、跨数和保守估计的 SNR
        length_km = sum(links_km[key] for key in link_keys)
        ns = int(length_km / Lspan_km)
        snr_dB = route_snr_dB(ns)

        # 选择满足 SNR 门限的最高阶调制格式
        selected = None
        for name, bits_dp, required_dB in formats:
            if snr_dB >= required_dB:
                selected = (name, bits_dp)

        if selected is None:
            continue

        name, bits_dp = selected
        rate_net = Rs * bits_dp / (1 + fec_overhead)

        # 单信道净速率不足时，继续检查下一条路径
        if rate_net < demand_gbps * 1e9:
            continue

        # 检查整条路径上的共同空闲波长
        common_free = set(range(1, Nch + 1))
        for key in link_keys:
            common_free = common_free.intersection(
                free_wavelengths[key]
            )

        if not common_free:
            continue

        # First-fit：选择编号最小的共同空闲波长
                # 根据参数选择最小或最大编号的共同空闲波长
        if wavelength_rule == "min":
            wavelength = min(common_free)
        elif wavelength_rule == "max":
            wavelength = max(common_free)
        else:
            raise ValueError(
                "wavelength_rule must be 'min' or 'max'."
            )

        # 所有检查通过后，才更新整条路径的资源状态
        for key in link_keys:
            free_wavelengths[key].remove(wavelength)

        return {
            "route": route,
            "wavelength": wavelength,
            "modulation": name,
            "net_rate_gbps": rate_net / 1e9,
        }

    # 所有候选路径均未通过检查
    return None


# 两个请求依次到来，第二个请求会看到第一个请求占用后的状态
requests_gbps = [1000.0, 1000.0]

print("\n--- Sequential connection allocation ---")

for request_id, demand_gbps in enumerate(requests_gbps, start=1):
    print(
        f"Request {request_id}: A -> D, "
        f"Demand = {demand_gbps:.1f} Gbit/s"
    )

    result = allocate_connection(demand_gbps)

    if result is None:
        print("  Blocked: no feasible candidate route.")
    else:
        route_name = " -> ".join(result["route"])
        print(
            f"  Accepted: {route_name}, "
            f"Wavelength = {result['wavelength']}, "
            f"{result['modulation']}, "
            f"Net rate = {result['net_rate_gbps']:.1f} Gbit/s"
        )

print("\n--- Remaining free wavelengths ---")
for key in [("A", "C"), ("C", "D")]:
    print(
        f"{key[0]} - {key[1]}: "
        f"{sorted(free_wavelengths[key])}"
    )
# --- 第 15 步：比较候选路径数量 K 对接入结果的影响 ---

# 重新定义实验的初始状态，避免使用之前已被占用的状态
initial_free_wavelengths = {
    ("A", "B"): {1, 2},
    ("B", "D"): {3, 4},
    ("A", "C"): {5, 6},
    ("C", "D"): {6, 7},
    ("B", "C"): {2, 6},
}

# 保存最多四条候选路径，按距离从短到长排列
all_routes = list(
    islice(
        nx.shortest_simple_paths(
            graph, "A", "D", weight="length_km"
        ),
        4,
    )
)

test_requests_gbps = [1000.0, 1000.0]

print("\n--- Effect of candidate route count K ---")

for k in range(1, len(all_routes) + 1):
    # 当前实验只允许使用前 k 条候选路径
    routes = all_routes[:k]

    # 每组实验都恢复相同的初始资源状态
    # 每个集合也要单独复制，避免修改初始数据
    free_wavelengths = {
        key: wavelengths.copy()
        for key, wavelengths in initial_free_wavelengths.items()
    }

    accepted = 0
    carried_gbps = 0.0

    # 同一组实验内，请求依次到来并更新资源占用
    for demand_gbps in test_requests_gbps:
        result = allocate_connection(demand_gbps)

        if result is not None:
            accepted += 1
            carried_gbps += demand_gbps

    total_requests = len(test_requests_gbps)
    blocked = total_requests - accepted
    blocked_fraction = blocked / total_requests

    print(
        f"K = {k}: "
        f"Accepted = {accepted}/{total_requests}, "
        f"Blocked fraction = {blocked_fraction:.1%}, "
        f"Carried traffic = {carried_gbps:.1f} Gbit/s"
    )
# --- 第 16 步：不同节点对的请求共享网络资源 ---

# 本组实验开始前，恢复原始资源状态
free_wavelengths = {
    key: wavelengths.copy()
    for key, wavelengths in initial_free_wavelengths.items()
}

# 请求依次到来：(起点, 终点, 需求速率 / Gbit/s)
traffic_requests = [
    ("C", "D", 1000.0),
    ("A", "D", 1000.0),
    ("B", "D", 1000.0),
    ("A", "D", 1000.0),
]

k_candidates = 4
accepted_count = 0
carried_gbps = 0.0

print("\n--- Connection allocation for multiple node pairs ---")

for request_id, request in enumerate(traffic_requests, start=1):
    source_node, target_node, demand_gbps = request

    # 为当前请求的起点和终点重新生成候选路径
    routes = list(
        islice(
            nx.shortest_simple_paths(
                graph,
                source_node,
                target_node,
                weight="length_km",
            ),
            k_candidates,
        )
    )

    # 分配函数使用当前请求的 routes 和共享的资源状态
    result = allocate_connection(demand_gbps)

    print(
        f"Request {request_id}: "
        f"{source_node} -> {target_node}, "
        f"Demand = {demand_gbps:.1f} Gbit/s"
    )

    if result is None:
        print("  Blocked: no feasible candidate route.")
    else:
        accepted_count += 1
        carried_gbps += demand_gbps

        route_name = " -> ".join(result["route"])
        print(
            f"  Accepted: {route_name}, "
            f"Wavelength = {result['wavelength']}, "
            f"{result['modulation']}, "
            f"Net rate = {result['net_rate_gbps']:.1f} Gbit/s"
        )

total_requests = len(traffic_requests)
blocked_fraction = (
    total_requests - accepted_count
) / total_requests

print("\n--- Traffic summary ---")
print(f"Accepted requests = {accepted_count}/{total_requests}")
print(f"Blocked fraction = {blocked_fraction:.1%}")
print(f"Carried traffic = {carried_gbps:.1f} Gbit/s")
# --- 第 19 步：在两个资源场景下比较波长选择规则 ---

# 场景 B：只修改 A-C 链路的初始空闲波长
scenario_b = {
    key: wavelengths.copy()
    for key, wavelengths in initial_free_wavelengths.items()
}
scenario_b[("A", "C")] = {5, 7}

scenarios = [
    ("A", initial_free_wavelengths),
    ("B", scenario_b),
]

print("\n--- Comparison of wavelength selection rules ---")

for scenario_name, initial_state in scenarios:
    for rule in ["min", "max"]:
        # 每种规则都从当前场景的同一初始状态开始
        free_wavelengths = {
            key: wavelengths.copy()
            for key, wavelengths in initial_state.items()
        }

        accepted = 0
        carried_gbps = 0.0

        # 使用相同的请求及到达顺序
        for source_node, target_node, demand_gbps in traffic_requests:
            routes = list(
                islice(
                    nx.shortest_simple_paths(
                        graph,
                        source_node,
                        target_node,
                        weight="length_km",
                    ),
                    k_candidates,
                )
            )

            result = allocate_connection(
                demand_gbps,
                wavelength_rule=rule,
            )

            if result is not None:
                accepted += 1
                carried_gbps += demand_gbps

        print(
            f"Scenario {scenario_name}, rule = {rule}: "
            f"Accepted = {accepted}/{len(traffic_requests)}, "
            f"Carried traffic = {carried_gbps:.1f} Gbit/s"
        )
from itertools import product

# --- 第 20 步：穷举求解小网络的离线最优分配 ---

def feasible_options(
    source, target, demand_gbps, initial_state, snr_margin_dB=0.0
):
    # 列出一个请求所有满足传输质量和初始资源约束的方案
    options = []

    candidate_paths = islice(
        nx.shortest_simple_paths(
            graph, source, target, weight="length_km"
        ),
        k_candidates,
    )

    for route in candidate_paths:
        link_keys = [
            tuple(sorted((u, v)))
            for u, v in zip(route[:-1], route[1:])
        ]

        length_km = sum(links_km[key] for key in link_keys)
        ns = int(length_km / Lspan_km)
        snr_dB = route_snr_dB(ns)

        # 计算所有可用调制格式对应的净速率
                # 调制格式除了满足基本门限，还必须保留指定的 SNR 裕量
        supported_rates = [
            Rs * bits_dp / (1 + fec_overhead)
            for _, bits_dp, required_dB in formats
            if snr_dB >= required_dB + snr_margin_dB
        ]

        if not supported_rates:
            continue
        if max(supported_rates) < demand_gbps * 1e9:
            continue

        # 检查初始状态下整条路径的共同空闲波长
        common_free = set(range(1, Nch + 1))
        for key in link_keys:
            common_free.intersection_update(initial_state[key])

        # 每个共同空闲波长都对应一个可选方案
        for wavelength in sorted(common_free):
            options.append({
                "route": route,
                "wavelength": wavelength,
                "resources": {
                    (key, wavelength) for key in link_keys
                },
            })

    return options


print("\n--- Offline optimum by exhaustive search ---")

for scenario_name, initial_state in scenarios:
    # None 表示不接入该请求
    choices = [
        feasible_options(source, target, demand, initial_state)
        + [None]
        for source, target, demand in traffic_requests
    ]

    best_rate = -1.0
    best_plan = None

    # 每个请求各选一个方案，组成一个完整的网络分配方案
    for plan in product(*choices):
        occupied_resources = []

        for option in plan:
            if option is not None:
                occupied_resources.extend(option["resources"])

        # 重复出现的“链路＋波长”意味着资源冲突
        if len(occupied_resources) != len(set(occupied_resources)):
            continue

        carried_gbps = sum(
            request[2]
            for request, option in zip(traffic_requests, plan)
            if option is not None
        )

        if carried_gbps > best_rate:
            best_rate = carried_gbps
            best_plan = plan

    accepted = sum(option is not None for option in best_plan)

    print(
        f"Scenario {scenario_name}: "
        f"Accepted = {accepted}/{len(traffic_requests)}, "
        f"Optimal carried traffic = {best_rate:.1f} Gbit/s"
    )

    for request_id, option in enumerate(best_plan, start=1):
        if option is None:
            print(f"  Request {request_id}: Not admitted")
        else:
            route_name = " -> ".join(option["route"])
            print(
                f"  Request {request_id}: {route_name}, "
                f"Wavelength = {option['wavelength']}"
            )
from scipy.optimize import milp, Bounds, LinearConstraint

# --- 第 21 步：用 0-1 整数线性规划求解联合分配 ---

# --- 第 22 步：共同发射功率对最优接入业务量的影响 ---

def optimal_admitted_traffic(initial_state, snr_margin_dB=0.0):
    # 为当前功率重新计算可行方案
        # 将当前要求的裕量传递给可行方案生成函数
    variables = [
        (request_index, option)
        for request_index, (source, target, demand) in enumerate(
            traffic_requests
        )
        for option in feasible_options(
            source,
            target,
            demand,
            initial_state,
            snr_margin_dB=snr_margin_dB,
        )
    ]

    # 功率过低或过高时，可能没有任何满足速率要求的方案
    if not variables:
        return 0.0

    resources = sorted({
        resource
        for _, option in variables
        for resource in option["resources"]
    })

    n_requests = len(traffic_requests)
    n_variables = len(variables)

    resource_rows = {
        resource: n_requests + row
        for row, resource in enumerate(resources)
    }

    A = np.zeros((
        n_requests + len(resources),
        n_variables,
    ))
    c = np.zeros(n_variables)

    for column, (request_index, option) in enumerate(variables):
        c[column] = -traffic_requests[request_index][2]
        A[request_index, column] = 1.0

        for resource in option["resources"]:
            A[resource_rows[resource], column] = 1.0

    result = milp(
        c=c,
        integrality=np.ones(n_variables, dtype=int),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(A, 0.0, 1.0),
        options={"mip_rel_gap": 0.0},
    )

    if not result.success:
        raise RuntimeError(result.message)

    return -result.fun


# --- 第 25 步：比较网络距离增长后的最优承载业务量 ---

distance_scales = [1, 2, 3, 4, 5, 6]
margin_cases_dB = [0.0, 4.0]

traffic_results = {
    margin_dB: [] for margin_dB in margin_cases_dB
}

# 保存原始状态，避免距离扫描影响后续计算
original_lengths = links_km.copy()
original_graph_lengths = {
    (u, v): graph[u][v]["length_km"]
    for u, v in links_km
}
saved_power = P0

try:
    # 当前非相干模型下，距离增长不会改变最优共同功率
    P0 = P_opt_W

    print("\n--- Optimal admitted traffic versus distance scale ---")
    print(f"Common per-channel power = {P_opt_dBm:.3f} dBm")

    for scale in distance_scales:
        # 同时更新物理层长度与寻路使用的图权重
        # 每次都从原始长度计算，避免连续累乘
        for (u, v), original_length in original_lengths.items():
            scaled_length = scale * original_length
            links_km[(u, v)] = scaled_length
            graph[u][v]["length_km"] = scaled_length

        print(f"\nDistance scale = {scale}")

        for margin_dB in margin_cases_dB:
            admitted_gbps = optimal_admitted_traffic(
                initial_free_wavelengths,
                snr_margin_dB=margin_dB,
            )

            traffic_results[margin_dB].append(admitted_gbps)

            print(
                f"  SNR margin = {margin_dB:.1f} dB: "
                f"Optimal admitted traffic = "
                f"{admitted_gbps:.1f} Gbit/s"
            )

finally:
    # 恢复链路长度、图权重和共同功率
    for (u, v), original_length in original_lengths.items():
        links_km[(u, v)] = original_length
        graph[u][v]["length_km"] = original_graph_lengths[(u, v)]

    P0 = saved_power

# 绘制离散距离采样点；连接线仅用于辅助观察趋势
plt.figure(figsize=(8.5, 5.2))

for margin_dB in margin_cases_dB:
    plt.plot(
        distance_scales,
        traffic_results[margin_dB],
        marker="o",
        label=f"SNR margin = {margin_dB:.0f} dB",
    )

plt.xlabel("Network distance scale factor")
plt.ylabel("Optimal admitted traffic (Gbit/s)")
plt.title("Scenario A: admitted traffic versus network distance")
plt.xticks(distance_scales)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("step25_traffic_vs_distance.png", dpi=300)

plt.show()