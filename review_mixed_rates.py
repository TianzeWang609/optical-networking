"""Paired mixed-rate checks for Steps 33–35; grooming remains disabled."""

from collections import Counter, defaultdict
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

from network_scenarios import generate_requests
from optical_networking_refactored.network_model import OpticalNetwork, edge_key
from reference_network import ReferenceNetworkPhysical
from reference_physical_model import ReferenceParameters, dbm_to_w
from step33_online_routing import BASE, ONLINE
from step34_hotspot_strength import LOADS, RULES, UNIFORM, hash_requests, summarize
from step35_wavelength_conversion import LAYOUTS, exact, unpack
from wavelength_conversion import audit, catalog, online_plan

ROOT = Path(__file__).resolve().parent
RATES = (50, 100, 150)


def load_result(relative_path):
    raw = (ROOT / relative_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw) if relative_path.endswith(".gz") else raw), digest


def sweep_checks(data, sweep, net, paths, state, archived, instances):
    fixed_pairs = {(p["share"], p["seed"], p["request_count"]): p for p in sweep["pairs"]}
    pairs = []
    for trace in sweep["traces"]:
        share, seed = trace["share"], trace["seed"]
        scenario = "uniform_mixed" if share == UNIFORM else "hotspot_mixed"
        for n in LOADS:
            fixed = fixed_pairs[share, seed, n]
            # Shuffle for the actual batch length so each load has the exact 1:2:1 mix.
            qs = [list(q) for q in generate_requests(data["nodes"], n, seed, scenario, "F", share)]
            assert [q[:2] for q in qs] == [q[:2] for q in trace["requests"][:n]]
            assert Counter(q[2] for q in qs) == {50: n//4, 100: n//2, 150: n//4}
            digest = hash_requests(qs)
            anchor = seed < 10 and share in (UNIFORM, .65)
            iid = f"{scenario}-{n}-{seed}"
            row = {"share": share, "seed": seed, "request_count": n, "sha256": digest,
                   "offered_f": fixed["offered_f"], "fixed_gain_gbps": fixed["gain_gbps"]}
            for label, method in (("baseline", BASE), ("online", ONLINE)):
                plan = online_plan(net, qs, paths, state, RULES[method])
                prior = [a for a in trace["plans"][method] if a["request_index"] < n]
                assert plan == unpack(prior)
                stats = audit(plan, qs, paths, state, set())
                assert stats["accepted"] == fixed[label]["accepted_count"]
                row[label] = {**fixed[label], "admitted_traffic_gbps": stats["admitted_gbps"]}
                if anchor:
                    assert digest == instances[iid]["sha256"]
                    if method == BASE:
                        assert plan == unpack(archived[iid, BASE]["plan"])
            row["gain_gbps"] = row["online"]["admitted_traffic_gbps"] - row["baseline"]["admitted_traffic_gbps"]
            pairs.append(row)
    assert len(pairs) == len(fixed_pairs) == 1800
    return pairs


def conversion_checks(instances, archived, fixed_rows, network, paths, state, converters):
    records = []
    for iid, instance in instances.items():
        qs = instance["requests"]
        assert hash_requests(qs) == instance["sha256"] and sum(q[2] for q in qs) == 100 * len(qs)
        fixed_id = iid.replace("_mixed-", "_fixed-")
        plans, proofs = {}, {}
        for layout in LAYOUTS:
            for method, rule in RULES.items():
                plan = online_plan(network, qs, paths[layout], state, rule)
                assert plan == fixed_rows[fixed_id, layout, method]["plan"]
                plans[layout, method] = plan
        ref = archived[iid, "ILP"]
        assert ref["optimal"] and ref["admitted_traffic_gbps"] == ref["upper_bound_gbps"]
        plans["none", "ILP"] = unpack(ref["plan"])
        proofs["none"] = {"upper_gbps": ref["upper_bound_gbps"], "source": "Step 32 mixed certified"}
        cap = fixed_rows[fixed_id, "all", "ILP"]["accepted"]
        # Identical path eligibility makes the certified fixed-rate optimum an upper
        # bound on cardinality. Even the best cap demands cannot offer more than this.
        upper = sum(sorted((q[2] for q in qs), reverse=True)[:cap])
        witnesses = [*plans.values(), fixed_rows[fixed_id, "all", "ILP"]["plan"]]
        full, top, attempts = exact(qs, paths["all"], state, upper, witnesses)
        plans["all", "ILP"] = full
        proofs["all"] = {"upper_gbps": top, "attempts": attempts}
        sparse_witnesses = [plans["none", "ILP"], plans["only_A", BASE], plans["only_A", ONLINE],
                            fixed_rows[fixed_id, "only_A", "ILP"]["plan"]]
        sparse, bound, attempts = exact(qs, paths["only_A"], state, top, sparse_witnesses)
        plans["only_A", "ILP"] = sparse
        proofs["only_A"] = {"upper_gbps": bound, "attempts": attempts}
        for layout in LAYOUTS:
            for method in (*RULES, "ILP"):
                plan = plans[layout, method]
                metrics = audit(plan, qs, paths[layout], state, converters[layout])
                traffic = metrics["admitted_gbps"]
                counts = Counter(int(qs[a["request_index"]][2]) for a in plan)
                assert traffic == sum(k*v for k,v in counts.items())
                assert traffic <= proofs[layout]["upper_gbps"]
                if method == "ILP":
                    assert traffic == proofs[layout]["upper_gbps"]
                records.append({"instance_id": iid, "scenario": instance["scenario"], "seed": instance["seed"],
                                "load": len(qs), "sha256": instance["sha256"], "layout": layout, "method": method,
                                "mixed_gbps": traffic, "mixed_count": len(plan), "accepted_by_rate": dict(counts),
                                "fixed_gbps": fixed_rows[fixed_id,layout,method]["admitted_gbps"],
                                "fixed_count": fixed_rows[fixed_id,layout,method]["accepted"],
                                "proof": proofs[layout] if method == "ILP" else None, "plan": plan})
        if instance["seed"] == 9:
            print(iid, "completed", flush=True)
    assert len(records) == 720
    return records


def conversion_summary(records):
    base = {(r["instance_id"], r["method"]): r for r in records if r["layout"] == "none"}
    groups = defaultdict(list)
    for r in records:
        groups[r["scenario"], r["load"], r["layout"], r["method"]].append(r)
    summary = []
    for (s,n,layout,method), group in sorted(groups.items()):
        row = {"scenario":s, "load":n, "layout":layout, "method":method}
        for field in ("mixed_gbps", "fixed_gbps", "mixed_count", "fixed_count"):
            row[field] = mean(r[field] for r in group)
        for prefix in ("fixed", "mixed"):
            field = prefix + "_gbps"
            deltas = [r[field] - base[r["instance_id"],method][field] for r in group]
            row[prefix + "_gain_vs_none"] = mean(deltas)
            row[prefix + "_wins_ties_losses"] = [sum(x>0 for x in deltas), deltas.count(0), sum(x<0 for x in deltas)]
        row["mean_accepted_by_rate"] = {k:mean(r["accepted_by_rate"].get(k,0) for r in group) for k in RATES}
        summary.append(row)
    return summary


def main():
    data, sha32 = load_result("results_step32/certified/traffic_results.json")
    sweep, sha34 = load_result("results_step34/sweep_details.json.gz")
    fixed35, sha35 = load_result("results_step35/plans.json.gz")
    assert data["complete"] and sweep["source_step32_sha256"] == fixed35["source_step32_sha256"] == sha32
    assert sweep["source_step33_sha256"] == fixed35["source_step33_sha256"]
    phys = data["physical"]
    model = ReferenceNetworkPhysical(ReferenceParameters(**phys["parameters"]), phys["fec_overhead"])
    assert model.metadata()["modulations"] == phys["modulations"]
    links = {edge_key(*e["nodes"]): e["length_km"] for e in data["links"]}
    state = {edge_key(*e["nodes"]): set(e["available_wavelengths"]) for e in data["links"]}
    net = OpticalNetwork(model, links, data["k_candidates"])
    ods = list(combinations(sorted(data["nodes"]), 2))
    keys = [(s,t,rate) for s,t in ods for rate in RATES]
    converters = {k:set(v) for k,v in fixed35["layouts"].items()}
    paths = {k:catalog(net, keys, dbm_to_w(data["common_power_dbm"]), data["snr_margin_db"], v)
             for k,v in converters.items()}
    offered_routes = [p for s,t in ods for p in paths["none"][s,t,150]]
    assert len(offered_routes) == 264
    for s,t in ods:
        route_names = [tuple(path) for path in net.candidate_routes(s,t)]
        for rate in RATES:
            assert [p.nodes for p in paths["none"][s,t,rate]] == route_names
    minimum = min(p.net_rate_gbps for p in offered_routes)
    instances = {i["id"]:i for i in data["instances"] if i["scenario"].endswith("_mixed")}
    archived = {(r["instance_id"],r["method"]):r for r in data["rows"]}
    fixed_rows = {(r["instance_id"],r["layout"],r["method"]):r for r in fixed35["records"]}
    pairs = sweep_checks(data, sweep, net, paths["none"], state, archived, instances)
    print(f"Verified {len(pairs)} mixed online pairs; minimum route rate {minimum}.", flush=True)
    conv = conversion_checks(instances, archived, fixed_rows, net, paths, state, converters)
    sweep_summary = summarize(pairs)
    meta = {"fixed_comparison":"Same OD, order, offered traffic, route sets, wave budget and conversion eligibility",
            "mixed_fraction":{"50":.25,"100":.5,"150":.25}, "grooming":False,
            "source32_sha256":sha32,"source34_sha256":sha34,"source35_sha256":sha35,
            "source33_sha256":sweep["source_step33_sha256"],"route_count":len(offered_routes),
            "min_net_rate_gbps":minimum,"sweep_pairs":len(pairs),"conversion_records":len(conv)}
    out = ROOT / "results_mixed_rate_review"
    out.mkdir(exist_ok=True)
    (out/"summary.json").write_text(json.dumps({**meta, "sweep":sweep_summary,
                            "conversion":conversion_summary(conv)}, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    details = json.dumps({**meta,"sweep_pairs":pairs,"conversion_plans":conv},
                         separators=(",",":"),allow_nan=False).encode()
    (out/"paired_details.json.gz").write_bytes(gzip.compress(details,mtime=0))


if __name__ == "__main__":
    main()
