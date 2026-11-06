"""Step 34: scan the offered hotspot probability using the two Step 33 online rules."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t

from analyze_step32_hotspot import diagnose
from network_scenarios import generate_requests
from optical_networking_refactored.network_model import AllocationResult, OpticalNetwork, edge_key
from reference_network import ReferenceNetworkPhysical, result_record
from reference_physical_model import ReferenceParameters, dbm_to_w
from step33_online_routing import BASE, ONLINE, METRICS

ROOT = Path(__file__).resolve().parent
UNIFORM = 1 / 6  # 11 of 66 unordered node pairs include F.
SHARES = (UNIFORM, .25, .35, .45, .55, .65, .75, .85, 1.0)
LOADS = (24, 48, 72, 96)
EXISTING = range(10)
FRESH = range(10, 50)
PRIMARY_LOAD = 96
RULES = {BASE: "first", ONLINE: "widest_bottleneck"}
# Nine gains over KSP-FF and eight changes in gain relative to uniform, at n=96.
FAMILY_SIZE = 2 * len(SHARES) - 1


def hash_requests(requests):
    encoded = json.dumps(requests, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def scan_share(share, data, network, state, power, old_baseline, old_pairs):
    scenario = "uniform_fixed" if share == UNIFORM else "hotspot_fixed"
    instances, rows, traces = [], [], []
    anchors = set()
    max_load = max(LOADS)
    for seed in (*EXISTING, *FRESH):
        requests = [list(q) for q in generate_requests(data["nodes"], max_load, seed, scenario,
                                                       hotspot=data["hotspot_node"], hotspot_share=share)]
        # An online algorithm never revisits earlier choices. Use the prefixes of the
        # same 96-request trace, exactly as in the Step 32 fixed-rate generator.
        allocations = {method: network.solve_sequential(requests, state, power, "min",
                       data["snr_margin_db"], route_rule=rule) for method, rule in RULES.items()}
        trace = {"share": share, "seed": seed, "requests": requests, "plans": {}}
        for count in LOADS:
            prefix = requests[:count]
            iid = f"{scenario}-{count}-{seed}"
            digest = hash_requests(prefix)
            instances.append({"id": iid, "scenario": scenario, "seed": seed,
                              "sha256": digest, "requests": prefix})
            is_anchor = seed in EXISTING and share in (UNIFORM, .65)
            for method, allocation in allocations.items():
                plan = allocation.plan[:count]
                traffic = sum(r[2] for r, choice in zip(prefix, plan) if choice is not None)
                prefix_result = AllocationResult(traffic, plan)
                rec = result_record(network, prefix, state, power, data["snr_margin_db"], prefix_result)
                if is_anchor:
                    previous = old_pairs[iid]
                    old_plan = old_baseline[iid]["plan"] if method == BASE else previous["online_plan"]
                    if digest != previous["sha256"] or rec["plan"] != old_plan:
                        raise ValueError(f"Archive replay failed for {iid}, {method}.")
                    anchors.add(iid)
                rows.append({"instance_id": iid, "scenario": scenario, "seed": seed, "method": method, **rec})
                if count == max_load:
                    trace["plans"][method] = rec["plan"]
        traces.append(trace)

    audit = diagnose({**data, "instances": instances, "rows": rows})
    audited = {(r["instance_id"], r["method"]): r for r in audit["records"]}
    pairs = []
    for i in instances:
        base = audited[i["id"], BASE]
        online = audited[i["id"], ONLINE]
        pairs.append({"share": share, "seed": i["seed"], "request_count": len(i["requests"]),
                      "sha256": i["sha256"], "offered_f": base["offered_hotspot_endpoints"],
                      "baseline": {k: base[k] for k in METRICS},
                      "online": {k: online[k] for k in METRICS},
                      "gain_gbps": online["admitted_traffic_gbps"] - base["admitted_traffic_gbps"]})
    return pairs, traces, anchors


def simultaneous_interval(values):
    # Paired observations across independent seeds, using a two-sided t interval.
    halfwidth = t.ppf(1 - .05/(2 * FAMILY_SIZE), len(values)-1) * stdev(values)/math.sqrt(len(values))
    centre = mean(values)
    return [centre - halfwidth, centre + halfwidth]


def summarize(pairs):
    uniform = {(p["seed"], p["request_count"]): p["gain_gbps"] for p in pairs if p["share"] == UNIFORM}
    groups = defaultdict(list)
    for p in pairs:
        p["gain_vs_uniform_gbps"] = p["gain_gbps"] - uniform[p["seed"], p["request_count"]]
        cohort = "new" if p["seed"] in FRESH else "original"
        groups[cohort, p["request_count"], p["share"]].append(p)
    summary = []
    for (cohort, count, share), group in sorted(groups.items()):
        seeds = FRESH if cohort == "new" else EXISTING
        if sorted(p["seed"] for p in group) != list(seeds):
            raise ValueError("Missing or duplicated seed within a comparison group.")
        gains = [p["gain_gbps"] for p in group]
        extra = [p["gain_vs_uniform_gbps"] for p in group]
        row = {"cohort": cohort, "request_count": count, "share": share, "n_seeds": len(group),
               "mean_offered_f": mean(p["offered_f"] for p in group),
               "mean_gain_gbps": mean(gains), "gain_sd_gbps": stdev(gains),
               "mean_gain_vs_uniform_gbps": mean(extra), "gain_vs_uniform_sd_gbps": stdev(extra),
               "wins": sum(g > 0 for g in gains), "ties": gains.count(0), "losses": sum(g < 0 for g in gains)}
        for method in ("baseline", "online"):
            row[method] = {k: mean(p[method][k] for p in group) for k in METRICS}
        if cohort == "new" and count == PRIMARY_LOAD:
            row["gain_simultaneous95_gbps"] = simultaneous_interval(gains)
            if share != UNIFORM:
                row["gain_vs_uniform_simultaneous95_gbps"] = simultaneous_interval(extra)
        summary.append(row)
    return summary


def plot(summary, destination):
    rows = sorted((s for s in summary if s["cohort"] == "new" and s["request_count"] == PRIMARY_LOAD),
                  key=lambda s: s["share"])
    x = [100*s["share"] for s in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.8), layout="constrained")
    for method, title, color, marker in [("baseline", "KSP-FF", "#49627c", "o"),
                                         ("online", "Bottleneck tie-break", "#ba5029", "D")]:
        left.plot(x, [s[method]["admitted_traffic_gbps"] for s in rows], label=title,
                  color=color, marker=marker, linewidth=2)
    left.set(title="Admitted traffic", ylabel="Mean admitted demand (Gbit/s)")
    left.legend(frameon=False)
    gains = [s["mean_gain_gbps"] for s in rows]
    bounds = [s["gain_simultaneous95_gbps"] for s in rows]
    yerr = [[g-b[0] for g,b in zip(gains,bounds)], [b[1]-g for g,b in zip(gains,bounds)]]
    right.axhline(0, linewidth=1, color="#697786")
    right.errorbar(x, gains, yerr=yerr, color="#167876", marker="o", linewidth=1.8, capsize=4)
    right.set(title="Paired gain over KSP-FF", ylabel="Mean difference (Gbit/s)")
    for ax in (left, right):
        ax.set_xlabel("Probability that a request has an endpoint at F (%)")
        ax.set_xticks(x, ["16.7","25","35","45","55","65","75","85","100"])
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Hotspot sweep at 96 requests | 40 new seeds", fontsize=15)
    fig.supxlabel("Whiskers: approximate simultaneous 95% t intervals for the 17 prespecified contrasts.",
                  fontsize=9, color="#475569")
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results_step34")
    args = parser.parse_args()
    source32 = (ROOT/"results_step32/certified/traffic_results.json").read_bytes()
    source33 = (ROOT/"results_step33/paired_results.json").read_bytes()
    data, step33 = json.loads(source32), json.loads(source33)
    if (step33["source_sha256"] != hashlib.sha256(source32).hexdigest()
            or data["request_counts"] != list(LOADS) or data["seed_values"] != list(EXISTING)
            or data["hotspot_node"] != "F" or data["hotspot_pair_probability"] != .65
            or data["traffic"]["fixed_rate_gbps"] != 100 or len(data["nodes"]) != 12):
        raise ValueError("Expected the original Step 32/33 paired mesh experiment.")
    old_baseline = {r["instance_id"]:r for r in data["rows"] if r["method"] == BASE}
    old_pairs = {r["instance_id"]:r for r in step33["pairs"]}
    phys = data["physical"]
    model = ReferenceNetworkPhysical(ReferenceParameters(**phys["parameters"]), phys["fec_overhead"])
    if model.metadata()["modulations"] != phys["modulations"]:
        raise ValueError("Modulation thresholds differ from the source experiment.")
    links = {edge_key(*e["nodes"]): e["length_km"] for e in data["links"]}
    state = {edge_key(*e["nodes"]): set(e["available_wavelengths"]) for e in data["links"]}
    network = OpticalNetwork(model, links, data["k_candidates"])
    power = dbm_to_w(data["common_power_dbm"])
    pairs, traces, anchors = [], [], set()
    for share in SHARES:
        batch, paths, checked = scan_share(share, data, network, state, power, old_baseline, old_pairs)
        pairs.extend(batch)
        traces.extend(paths)
        anchors.update(checked)
        print(f"Completed F probability {share:.2%}: {len(batch)} paired checkpoints.", flush=True)
    if anchors != set(old_pairs):
        raise ValueError("Incomplete archived anchor replay.")
    meta = {"experiment": "step34_hotspot_strength", "shares": list(SHARES), "request_counts": list(LOADS),
            "original_seeds": list(EXISTING), "new_seeds": list(FRESH), "primary_load": PRIMARY_LOAD,
            "source_step32_sha256": hashlib.sha256(source32).hexdigest(),
            "source_step33_sha256": hashlib.sha256(source33).hexdigest(),
            "anchor_pairs_matched": len(anchors), "paired_checkpoints": len(pairs), "max_length_traces": len(traces),
            "pairing": "Shared seed across shares; identical ordered requests between methods at each share. "
                       "Loads are nested prefixes of each 96-request trace.",
            "intervals": "Approximate Bonferroni simultaneous 95% paired t intervals at the primary load "
                         "on new seeds, across 9 gains and 8 differences from uniform. Lower loads are descriptive."}
    summary = summarize(pairs)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out/"summary.json").write_text(json.dumps({**meta, "summary": summary}, indent=2, allow_nan=False)+"\n",
                                    encoding="utf-8")
    encoded = json.dumps({**meta, "traces": traces, "pairs": pairs}, separators=(",", ":"), allow_nan=False).encode()
    (out/"sweep_details.json.gz").write_bytes(gzip.compress(encoded, compresslevel=6, mtime=0))
    plot(summary, out/"hotspot_strength.png")


if __name__ == "__main__":
    main()
