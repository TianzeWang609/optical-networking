"""Step 33: replay the Step 32 fixed-rate batches with online congestion-aware routing."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_step32_hotspot import SCENARIOS, diagnose
from optical_networking_refactored.network_model import OpticalNetwork, edge_key
from reference_network import ReferenceNetworkPhysical, result_record
from reference_physical_model import ReferenceParameters, dbm_to_w

ROOT = Path(__file__).resolve().parent
BASE = "KSP-FF"
ONLINE = "KSP-FF (max bottleneck)"
METRICS = ("accepted_count", "admitted_traffic_gbps", "request_blocking_fraction",
           "accepted_hotspot_endpoints", "accepted_hotspot_transits", "used_incident_resources")


def first_change(network, requests, initial_state, baseline, online):
    """Explain the first different route from the identical prefix in both runs."""
    state = {e: set(w) for e, w in initial_state.items()}
    for index, (before, after) in enumerate(zip(baseline.plan, online.plan)):
        if before != after:
            assert before is not None and after is not None
            def snapshot(option):
                path = option.route
                waves = [state[edge_key(u, v)] for u, v in zip(path, path[1:])]
                return {"route": list(path), "km": network.route_length_km(path),
                        "free_per_link": [len(w) for w in waves],
                        "bottleneck": min(map(len, waves)), "wavelength": option.wavelength,
                        "common_free": sorted(set.intersection(*waves))}
            old, new = snapshot(before), snapshot(after)
            assert old["km"] == new["km"] and old["bottleneck"] < new["bottleneck"]
            assert all(s["wavelength"] == s["common_free"][0] for s in (old, new))
            return {"request_index": index, "request": requests[index], "baseline": old, "online": new}
        if before is not None:
            for edge, wave in before.resources:
                state[edge].remove(wave)
    return None


def summarize(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["scenario"], pair["request_count"]].append(pair)
    summary = []
    for (scenario, count), group in sorted(groups.items()):
        gains = [p["delta"]["admitted_traffic_gbps"] for p in group]
        row = {"scenario": scenario, "request_count": count, "n_seeds": len(group),
               "wins": sum(x > 0 for x in gains), "ties": gains.count(0), "losses": sum(x < 0 for x in gains),
               "delta_gbps_sample_sd": stdev(gains) if len(gains) > 1 else 0,
               "delta_gbps_min": min(gains), "delta_gbps_max": max(gains)}
        for label in ("baseline", "online", "ilp", "delta"):
            row[label] = {k: mean(p[label][k] for p in group) for k in METRICS}
        summary.append(row)
    return summary


def run(source):
    raw = source.read_bytes()
    data = json.loads(raw)
    if not data["complete"] or data["k_candidates"] != 4 or data["traffic"]["fixed_rate_gbps"] != 100:
        raise ValueError("Expected completed Step 32 fixed-rate controls with K=4 and 100 Gbit/s.")
    physical = data["physical"]
    model = ReferenceNetworkPhysical(ReferenceParameters(**physical["parameters"]), physical["fec_overhead"])
    if model.metadata()["modulations"] != physical["modulations"]:
        raise ValueError("Modulation thresholds differ from the saved experiment.")
    links = {edge_key(*e["nodes"]): e["length_km"] for e in data["links"]}
    state = {edge_key(*e["nodes"]): set(e["available_wavelengths"]) for e in data["links"]}
    network = OpticalNetwork(model, links, data["k_candidates"])
    power = dbm_to_w(data["common_power_dbm"])
    margin = data["snr_margin_db"]
    saved = {(r["instance_id"], r["method"]): r for r in data["rows"]}
    instances = [i for i in data["instances"] if i["scenario"] in SCENARIOS]
    expected = {(s, n, seed) for s in SCENARIOS for n in data["request_counts"] for seed in data["seed_values"]}
    actual = {(i["scenario"], len(i["requests"]), i["seed"]) for i in instances}
    if actual != expected or len(instances) != len(expected):
        raise ValueError("Incomplete or duplicated fixed-rate input batches.")

    rows, pairs = [], []
    for i in instances:
        iid, requests = i["id"], i["requests"]
        reference, optimum = saved[iid, BASE], saved[iid, "ILP"]
        baseline = network.solve_sequential(requests, state, power, "min", margin)
        check = result_record(network, requests, state, power, margin, baseline)
        if check["plan"] != reference["plan"] or check["admitted_traffic_gbps"] != reference["admitted_traffic_gbps"]:
            raise ValueError(f"Original baseline did not reproduce for {iid}.")
        online = network.solve_sequential(requests, state, power, "min", margin, route_rule="widest_bottleneck")
        record = result_record(network, requests, state, power, margin, online)
        assert optimum["optimal"] and online.admitted_traffic_gbps <= optimum["admitted_traffic_gbps"]
        rows.extend([reference, optimum, {"instance_id": iid, "method": ONLINE, "scenario": i["scenario"],
                                          "seed": i["seed"], **record}])
        pairs.append({"instance_id": iid, "sha256": i["sha256"], "seed": i["seed"], "scenario": i["scenario"],
                      "request_count": len(requests), "online_plan": record["plan"],
                      "first_change": first_change(network, requests, state, baseline, online)})

    # Reuse Step 32's independent plan, hash, per-edge utilisation and hotspot cut audit.
    audit = diagnose({**data, "rows": rows})
    audited = {(r["instance_id"], r["method"]): r for r in audit["records"]}
    for pair in pairs:
        for label, method in (("baseline", BASE), ("online", ONLINE), ("ilp", "ILP")):
            row = audited[pair["instance_id"], method]
            pair[label] = {k: row[k] for k in (*METRICS, "link_utilization")}
        pair["delta"] = {k: pair["online"][k] - pair["baseline"][k] for k in METRICS}
        assert math.isclose(pair["delta"]["admitted_traffic_gbps"], 100 * pair["delta"]["accepted_count"])
    return {"experiment": "step33_online_routing", "source_file": source.name,
            "source_sha256": hashlib.sha256(raw).hexdigest(), "physical": physical,
            "common_power_dbm": data["common_power_dbm"], "snr_margin_db": margin, "links": data["links"],
            "k_candidates": network.k_candidates, "hotspot_node_for_diagnostics": data["hotspot_node"],
            "hotspot_pair_probability": data["hotspot_pair_probability"], "seed_values": data["seed_values"],
            "request_counts": data["request_counts"], "fixed_rate_gbps": 100,
            "rule": "Shortest feasible distance, then largest minimum free wavelength count across links, "
                    "then original candidate order. First-fit on the chosen route.",
            "delta_definition": "Online minus KSP-FF within the same batch",
            "baseline_plans_reproduced": len(pairs), "pairs": pairs, "summary": summarize(pairs)}


def plot(data, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharey=True, layout="constrained")
    for ax, scenario, name in zip(axes, SCENARIOS, ("Uniform endpoints", "Hotspot at F")):
        counts = data["request_counts"]
        means = []
        for n in counts:
            group = sorted((p for p in data["pairs"] if p["scenario"] == scenario and p["request_count"] == n),
                           key=lambda p: p["seed"])
            gains = [p["delta"]["admitted_traffic_gbps"] for p in group]
            offsets = [(j - (len(gains)-1)/2)*0.7 for j in range(len(gains))]
            ax.scatter([n + x for x in offsets], gains, color="#408493", alpha=.75, s=25,
                       label="One paired batch" if n == counts[0] else None)
            means.append(mean(gains))
        ax.axhline(0, color="#64748b", linewidth=.9)
        ax.plot(counts, means, color="#b44c23", marker="D", lw=2, label="Mean paired change")
        ax.set(title=name, xticks=counts, xlabel="Requests per batch")
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Change in admitted traffic (Gbit/s)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Online bottleneck tie-break compared with KSP-FF", fontsize=14)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"results_step32/certified/traffic_results.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results_step33")
    args = parser.parse_args()
    data = run(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [("paired_results.json", data), ("summary.json", {k:v for k,v in data.items() if k != "pairs"})]:
        (args.output_dir/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plot(data, args.output_dir/"paired_gains.png")
    for s in data["summary"]:
        print(s["scenario"], s["request_count"], f"gain {s['delta']['admitted_traffic_gbps']:+g} Gbit/s",
              f"W/T/L {s['wins']}/{s['ties']}/{s['losses']}")


if __name__ == "__main__":
    main()
