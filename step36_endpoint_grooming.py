"""Step 36: online same-endpoint grooming on the archived mixed-rate Step 32 batches."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optical_networking_refactored.network_model import AllocationOption, AllocationResult, OpticalNetwork, edge_key
from reference_network import ReferenceNetworkPhysical, validate_plan
from reference_physical_model import ReferenceParameters, dbm_to_w

ROOT = Path(__file__).resolve().parent
RATES = (50, 100, 150)
SCENARIOS = ("uniform_mixed", "hotspot_mixed")
TOL = 1e-9


@dataclass
class Lightpath:
    option: AllocationOption
    clients: list[int] = field(default_factory=list)
    used: float = 0.0

    @property
    def remaining(self):
        return self.option.net_rate_gbps - self.used


def allocate(network, requests, initial, power, margin, *, grooming):
    """Return lightpaths in opening order. Each request occupies exactly one or none."""
    free = network._copy_state(initial)
    lightpaths = []
    by_endpoints = defaultdict(list)
    for index, request in enumerate(requests):
        source, target, rate = request
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("A request must have a finite positive rate.")
        endpoints = (source, target)
        chosen = None
        if grooming:
            # Subtracting the same demand from every candidate preserves the order.
            # Ties retain the earlier lightpath in this insertion-ordered list.
            chosen = min((lp for lp in by_endpoints[endpoints] if lp.remaining + TOL >= rate),
                         key=lambda lp: lp.remaining, default=None)
        if chosen is None:
            # feasible_options already enumerates shortest candidates in order and
            # sorts common free wavelengths ascending on each route (Step 32 KSP-FF).
            options = network.feasible_options(request, free, power, margin)
            if not options:
                continue
            chosen = Lightpath(options[0])
            for edge, wave in chosen.option.resources:
                free[edge].remove(wave)
            lightpaths.append(chosen)
            by_endpoints[endpoints].append(chosen)
        chosen.clients.append(index)
        chosen.used += rate
    return lightpaths


def measure(network, requests, initial, power, margin, lightpaths, hotspot):
    """Check client and optical resources separately, then count both layers."""
    assigned = set()
    by_rate = Counter()
    loads, options, details = [], [], []
    endpoint_clients = endpoint_paths = transit_paths = reused = link_waves = 0
    incident_used = 0
    admitted = 0.0
    for lightpath in lightpaths:
        opt = lightpath.option
        clients = lightpath.clients
        assert clients and clients == sorted(set(clients)) and not assigned.intersection(clients)
        assigned.update(clients)
        src, dst = opt.route[0], opt.route[-1]
        assert all(0 <= j < len(requests) and tuple(requests[j][:2]) == (src, dst) for j in clients)
        traffic = sum(requests[j][2] for j in clients)
        assert math.isclose(traffic, lightpath.used) and traffic <= opt.net_rate_gbps + TOL
        loads.append((src, dst, traffic))
        options.append(opt)
        by_rate.update(requests[j][2] for j in clients)
        reused += len(clients) - 1
        link_waves += len(opt.resources)
        incident_used += sum(hotspot in edge for edge, _ in opt.resources)
        if hotspot in (src, dst):
            endpoint_clients += len(clients)
            endpoint_paths += 1
        transit_paths += hotspot in opt.route[1:-1]
        admitted += traffic
        details.append({"open_index": clients[0], "clients": list(clients), "route": list(opt.route),
                        "wavelength": opt.wavelength, "modulation": opt.modulation,
                        "snr_db": opt.snr_db, "net_rate_gbps": opt.net_rate_gbps,
                        "used_gbps": traffic})
    # One audit entry per physical lightpath, with its total client demand. This
    # validates end-to-end continuity, physical capacity and exclusivity across paths.
    validate_plan(network, loads, initial, power, margin, AllocationResult(admitted, tuple(options)))
    assert incident_used == endpoint_paths + 2 * transit_paths
    assert incident_used <= sum(len(w) for edge, w in initial.items() if hotspot in edge)
    offered = sum(q[2] for q in requests)
    metrics = {"accepted": len(assigned), "admitted_gbps": admitted, "lightpaths": len(lightpaths),
               "reused_requests": reused, "used_link_wavelengths": link_waves,
               "hotspot_endpoint_clients": endpoint_clients, "hotspot_endpoint_lightpaths": endpoint_paths,
               "hotspot_transit_lightpaths": transit_paths, "hotspot_used_incident_wavelengths": incident_used,
               "request_blocking": 1 - len(assigned) / len(requests),
               "bandwidth_blocking": 1 - admitted / offered,
               "accepted_by_rate": {rate: by_rate[rate] for rate in sorted({q[2] for q in requests})}}
    return {"metrics": metrics, "lightpaths": details}, assigned


def reproduce_archived(off, archived):
    plan = []
    for lp in off["lightpaths"]:
        assert lp["clients"] == [lp["open_index"]]
        plan.append({"request_index": lp["open_index"], "route": lp["route"], "wavelength": lp["wavelength"],
                     "modulation": lp["modulation"], "snr_db": lp["snr_db"], "net_rate_gbps": lp["net_rate_gbps"]})
    if plan != archived["plan"] or off["metrics"]["admitted_gbps"] != archived["admitted_traffic_gbps"]:
        raise ValueError(f"Step 32 KSP-FF baseline did not reproduce for {archived['instance_id']}.")


def run(source):
    raw = source.read_bytes()
    data = json.loads(raw)
    physical = data["physical"]
    model = ReferenceNetworkPhysical(ReferenceParameters(**physical["parameters"]), physical["fec_overhead"])
    assert data["complete"] and model.metadata()["modulations"] == physical["modulations"]
    links = {edge_key(*e["nodes"]): e["length_km"] for e in data["links"]}
    initial = {edge_key(*e["nodes"]): set(e["available_wavelengths"]) for e in data["links"]}
    network = OpticalNetwork(model, links, data["k_candidates"])
    power, margin = dbm_to_w(data["common_power_dbm"]), data["snr_margin_db"]
    archived = {r["instance_id"]: r for r in data["rows"] if r["method"] == "KSP-FF"}
    pairs = []
    expected = {(s, n, seed) for s in SCENARIOS for n in data["request_counts"] for seed in data["seed_values"]}
    for batch in data["instances"]:
        if batch["scenario"] not in SCENARIOS:
            continue
        iid, requests = batch["id"], batch["requests"]
        n = len(requests)
        assert hash_requests(requests) == batch["sha256"]
        assert Counter(q[2] for q in requests) == {50: n//4, 100: n//2, 150: n//4}
        pair = {"instance_id": iid, "input_sha256": batch["sha256"], "scenario": batch["scenario"],
                "seed": batch["seed"], "request_count": n, "offered_gbps": n * 100,
                "offered_by_rate": dict(Counter(int(q[2]) for q in requests)),
                "requests": requests}
        assignments = {}
        for name, share in (("off", False), ("on", True)):
            lps = allocate(network, requests, initial, power, margin, grooming=share)
            pair[name], assignments[name] = measure(network, requests, initial, power, margin, lps, data["hotspot_node"])
        reproduce_archived(pair["off"], archived[iid])
        gained = sorted(assignments["on"] - assignments["off"])
        lost = sorted(assignments["off"] - assignments["on"])
        pair["newly_served_indices"], pair["displaced_indices"] = gained, lost
        delta = sum(requests[j][2] for j in gained) - sum(requests[j][2] for j in lost)
        assert delta == pair["on"]["metrics"]["admitted_gbps"] - pair["off"]["metrics"]["admitted_gbps"]
        pairs.append(pair)
    actual = {(p["scenario"], p["request_count"], p["seed"]) for p in pairs}
    assert expected == actual and len(pairs) == len(expected)
    return {"source_sha256": hashlib.sha256(raw).hexdigest(), "pair_count": len(pairs),
            "seeds": data["seed_values"], "k_candidates": data["k_candidates"],
            "power_dbm": data["common_power_dbm"], "margin_db": margin,
            "physical_channels": data["physical_channels"], "links": data["links"],
            "hotspot": data["hotspot_node"], "hotspot_share": data["hotspot_pair_probability"],
            "service": "Continuous wavelength per lightpath, no conversion or departures. Reuse only "
                       "the same ordered endpoints. All of a client's rate must fit in one lightpath. "
                       "Tightest residual capacity first, then earliest lightpath. New paths use KSP-FF.",
            "pairs": pairs}


def hash_requests(requests):
    return hashlib.sha256(json.dumps(requests, separators=(",", ":")).encode()).hexdigest()


def summarize(pairs):
    groups = defaultdict(list)
    for p in pairs:
        groups[p["scenario"], p["request_count"]].append(p)
    summary = []
    for (scenario, n), group in sorted(groups.items()):
        gains = [p["on"]["metrics"]["admitted_gbps"] - p["off"]["metrics"]["admitted_gbps"] for p in group]
        row = {"scenario": scenario, "request_count": n, "seed_count": len(group), "gain_gbps": mean(gains),
               "gain_range_gbps": [min(gains), max(gains)], "wins": sum(g > 0 for g in gains),
               "ties": gains.count(0), "losses": sum(g < 0 for g in gains),
               "mean_newly_served": mean(len(p["newly_served_indices"]) for p in group),
               "mean_displaced": mean(len(p["displaced_indices"]) for p in group)}
        for name in ("off", "on"):
            metrics = [p[name]["metrics"] for p in group]
            result = {k: mean(m[k] for m in metrics) for k in metrics[0] if k != "accepted_by_rate"}
            result["class_counts"] = {}
            for rate in RATES:
                accepted = sum(m["accepted_by_rate"][rate] for m in metrics)
                offered = sum(p["offered_by_rate"][rate] for p in group)
                result["class_counts"][rate] = {"accepted": accepted, "offered": offered,
                                                "acceptance_fraction": accepted / offered}
            row[name] = result
        summary.append(row)
    return summary


def plot(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained", sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        rows = sorted((r for r in summary if r["scenario"] == scenario), key=lambda r: r["request_count"])
        counts = [r["request_count"] for r in rows]
        for name, label, colour in (("off", "Dedicated lightpaths", "#416683"),
                                    ("on", "Endpoint grooming", "#b75525")):
            ax.plot(counts, [r[name]["admitted_gbps"] for r in rows], color=colour, marker="o", label=label)
        ax.plot(counts, [100*n for n in counts], linestyle="--", color="#737373", label="Offered")
        ax.set(title="Uniform endpoints" if scenario == SCENARIOS[0] else "Hotspot at F (65%)",
               xticks=counts, xlabel="Requests per batch")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Mean traffic (Gbit/s)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Same-endpoint grooming with mixed rates | 10 paired seeds")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_step36")
    args = parser.parse_args()
    data = run(ROOT / "results_step32/certified/traffic_results.json")
    summary = summarize(data["pairs"])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in data.items() if k != "pairs"}
    (out / "summary.json").write_text(json.dumps({**meta, "summary": summary}, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
    (out / "paired_plans.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
    plot(summary, out / "grooming_comparison.png")
    for r in summary:
        print(r["scenario"], r["request_count"], f"gain {r['gain_gbps']:+g} Gbit/s;",
              f"W/T/L {r['wins']}/{r['ties']}/{r['losses']}")


if __name__ == "__main__":
    main()
