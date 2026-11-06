"""Explain the fixed-rate hotspot experiment using saved Step 32 allocations.

This reads existing results; it does not rerun allocation or change source data.
Run: python analyze_step32_hotspot.py
"""

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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


ROOT = Path(__file__).resolve().parent
SCENARIOS = ("uniform_fixed", "hotspot_fixed")


def diagnose(data):
    hotspot = data["hotspot_node"]
    links = {tuple(sorted(link["nodes"])): set(link["available_wavelengths"])
             for link in data["links"]}
    incident = [edge for edge in links if hotspot in edge]
    cut_capacity = sum(len(links[edge]) for edge in incident)
    instances = {instance["id"]: instance for instance in data["instances"]}
    records = []
    for row in data["rows"]:
        if row["scenario"] not in SCENARIOS:
            continue
        instance = instances[row["instance_id"]]
        requests = instance["requests"]
        encoded = json.dumps(requests, separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == instance["sha256"]
        assert all(request[2] == data["traffic"]["fixed_rate_gbps"] for request in requests)
        occupied = {edge: set() for edge in links}
        endpoints = transit = 0
        seen = set()
        for allocation in row["plan"]:
            index = allocation["request_index"]
            assert index not in seen
            seen.add(index)
            request = requests[index]
            route = allocation["route"]
            assert [route[0], route[-1]] == request[:2]
            assert len(route) == len(set(route))
            endpoints += hotspot in request[:2]
            transit += hotspot in route[1:-1]
            wave = allocation["wavelength"]
            for u, v in zip(route, route[1:]):
                edge = tuple(sorted((u, v)))
                assert wave in links[edge] and wave not in occupied[edge]
                occupied[edge].add(wave)
        usage = {"-".join(edge): len(occupied[edge]) / len(links[edge]) for edge in links}
        assert all(math.isclose(value, row["link_utilization"][edge]) for edge, value in usage.items())
        used_cut = sum(len(occupied[edge]) for edge in incident)
        assert endpoints + 2 * transit == used_cut <= cut_capacity
        assert len(seen) == row["accepted_count"]
        assert math.isclose(sum(requests[i][2] for i in seen), row["admitted_traffic_gbps"])
        if row["method"] == "ILP":
            assert row["optimal"] is True, "Use certified results for the optimal benchmark."
        offered_f = sum(hotspot in request[:2] for request in requests)
        lower_bound = max(0, offered_f - cut_capacity)
        assert len(requests) - len(seen) >= lower_bound
        records.append({
            "instance_id": row["instance_id"], "scenario": row["scenario"],
            "seed": row["seed"], "method": row["method"], "request_count": len(requests),
            "accepted_count": len(seen), "admitted_traffic_gbps": row["admitted_traffic_gbps"],
            "request_blocking_fraction": (len(requests) - len(seen)) / len(requests),
            "offered_hotspot_endpoints": offered_f, "accepted_hotspot_endpoints": endpoints,
            "accepted_hotspot_transits": transit, "used_incident_resources": used_cut,
            "unused_incident_resources": cut_capacity - used_cut,
            "rejection_lower_bound_from_hotspot_cut": lower_bound,
            "link_utilization": usage,
        })
    groups = defaultdict(list)
    for record in records:
        groups[(record["scenario"], record["request_count"], record["method"])].append(record)
    metrics = ("accepted_count", "admitted_traffic_gbps", "request_blocking_fraction",
               "offered_hotspot_endpoints", "accepted_hotspot_endpoints", "accepted_hotspot_transits",
               "used_incident_resources", "unused_incident_resources",
               "rejection_lower_bound_from_hotspot_cut")
    summary = []
    for (scenario, count, method), group in sorted(groups.items()):
        assert len({r["seed"] for r in group}) == len(group)
        item = {"scenario": scenario, "request_count": count, "method": method, "seed_count": len(group)}
        for metric in metrics:
            values = [r[metric] for r in group]
            item[metric + "_mean"] = mean(values)
            item[metric + "_sd"] = stdev(values) if len(values) > 1 else 0.0
        item["link_utilization_mean"] = {edge: mean(r["link_utilization"][edge] for r in group)
                                         for edge in group[0]["link_utilization"]}
        item["all_hotspot_links_full_count"] = sum(
            r["used_incident_resources"] == cut_capacity for r in group)
        summary.append(item)
    return {"hotspot_node": hotspot, "incident_links": [list(edge) for edge in incident],
            "incident_resource_capacity": cut_capacity,
            "identity": "accepted hotspot endpoints + 2 * accepted hotspot transits = used incident resources",
            "scope": "Static batches, simple paths, undirected shared wavelengths, one wavelength per request",
            "records": records, "summary": summary}


def plot_utilization(data, diagnostics, output_dir):
    count = max(record["request_count"] for record in diagnostics["records"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    cmap, norm = plt.get_cmap("YlOrRd"), Normalize(0, 1)
    pos = data["positions"]
    for ax, scenario, title in zip(axes, SCENARIOS, ("Uniform OD", "Hotspot OD: F")):
        entry = next(s for s in diagnostics["summary"] if s["scenario"] == scenario
                     and s["request_count"] == count and s["method"] == "KSP-FF")
        for link in data["links"]:
            u, v = link["nodes"]
            value = entry["link_utilization_mean"]["-".join(sorted((u, v)))]
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color=cmap(norm(value)), linewidth=8, zorder=1)
            ax.text((x1+x2)/2, (y1+y2)/2, f"{value:.1%}", ha="center", va="center",
                    fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=3)
        for node, (x, y) in pos.items():
            ax.scatter(x, y, s=630, facecolor="#dbeafe" if node == data["hotspot_node"] else "white",
                       edgecolor="#334155", linewidth=1.7, zorder=4)
            ax.text(x, y, node, ha="center", va="center", fontsize=12, fontweight="bold", zorder=5)
        ax.set_title(f"{title}\nBlocking: {entry['request_blocking_fraction_mean']:.2%}", fontsize=13, pad=14)
        ax.set(xlim=(-0.3, 3.3), ylim=(-0.35, 2.35), aspect="equal")
        ax.axis("off")
    fig.suptitle(f"KSP-FF | {count} requests x 100 Gbit/s | Mean of {entry['seed_count']} seeds", fontsize=15)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="horizontal",
                       shrink=0.65, fraction=0.07, pad=0.02)
    bar.set_label("Used fraction of the 8 allocatable wavelengths on each link")
    fig.savefig(output_dir / "hotspot_link_utilization.png", dpi=180)
    fig.savefig(output_dir / "hotspot_link_utilization.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results_step32/certified/traffic_results.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input.parent / "hotspot_diagnostics"
    source = args.input.read_bytes()
    data = json.loads(source)
    diagnostics = diagnose(data)
    diagnostics["source"] = str(args.input.resolve())
    diagnostics["source_sha256"] = hashlib.sha256(source).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hotspot_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    plot_utilization(data, diagnostics, output_dir)
    print(f"Audited {len(diagnostics['records'])} saved allocations; hotspot cut capacity = "
          f"{diagnostics['incident_resource_capacity']} wavelength resources.")
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
