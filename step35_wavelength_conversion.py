"""Step 35. Paired online and offline routing with ideal wavelength converters."""
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optical_networking_refactored.network_model import OpticalNetwork, edge_key
from reference_network import ReferenceNetworkPhysical
from reference_physical_model import ReferenceParameters, dbm_to_w
from step33_online_routing import BASE, ONLINE, SCENARIOS
from wavelength_conversion import assign_waves, audit, catalog, offline_plan, online_plan

ROOT = Path(__file__).resolve().parent
LAYOUTS = ("none", "only_A", "all")


def unpack(plan):
    return [{"request_index":a["request_index"], "route":a["route"],
             "waves":[a["wavelength"]] * (len(a["route"])-1)} for a in plan]


def value(plan, requests):
    return sum(requests[a["request_index"]][2] for a in plan)


def rejection_causes(plan, requests, candidates, initial):
    state = {e:set(w) for e,w in initial.items()}
    by_request = {a["request_index"]:a for a in plan}
    continuity = saturated = 0
    for j,req in enumerate(requests):
        if j in by_request:
            a = by_request[j]
            for u,v,w in zip(a["route"],a["route"][1:],a["waves"]):
                state[edge_key(u,v)].remove(w)
        else:
            paths = candidates[tuple(req)]
            assert paths and all(assign_waves(p,state) is None for p in paths)
            if any(all(state[e] for e in p.edges) for p in paths):
                continuity += 1
            else:
                saturated += 1
    return {"continuity_blocks":continuity, "saturated_link_blocks":saturated}


def exact(requests, paths, state, upper, witnesses):
    """Use certified bounds across nested layouts; solve only unresolved cases."""
    best = max(witnesses,key=lambda p:value(p,requests))
    rates = [q[2] for q in requests]
    quantum = math.gcd(*(int(r) for r in rates)) if all(float(r).is_integer() for r in rates) else 0
    attempts = []
    for limit in (3,15,60):
        if value(best, requests) >= upper - 1e-6:
            break
        plan, dual, status = offline_plan(requests, paths, state, limit, upper)
        # Only round to a grid implied by the actual demands (50 for the mixed batches).
        upper = min(upper, math.floor((dual+1e-5)/quantum)*quantum if quantum else dual)
        attempts.append({"limit_s":limit, "status":status, "upper_gbps":upper})
        if value(plan,requests) > value(best,requests):
            best = plan
        if upper < value(best,requests)-1e-6:
            raise RuntimeError("Upper bound below a feasible witness.")
    if abs(value(best,requests)-upper)>1e-6:
        raise RuntimeError(f"Unproven optimum: feasible {value(best,requests)}, upper {upper}")
    return best, upper, attempts


def frozen_example(instances, saved, paths, state):
    iid = "uniform_fixed-48-0"
    req = tuple(instances[iid]["requests"][34])
    assert req == ("D","I",100)
    free = {e:set(w) for e,w in state.items()}
    old = unpack(saved[iid,BASE]["plan"])
    assert all(a["request_index"] != 34 for a in old)
    for a in old:
        if a["request_index"]<34:
            for u,v,w in zip(a["route"],a["route"][1:],a["waves"]):
                free[edge_key(u,v)].remove(w)
    route = tuple("DCBAEI")
    example = {}
    for layout in LAYOUTS:
        path = next(p for p in paths[layout][req] if p.nodes==route)
        example[layout] = assign_waves(path,free)
    assert example["none"] is None and example["only_A"] == [44,44,44,41,41]
    return {"instance":iid,"request_index":34,"route":list(route),
            "free_per_edge":[sorted(free[edge_key(u,v)]) for u,v in zip(route,route[1:])],
            "waves":example, "scope":"Original KSP-FF prefix frozen before request 35; same end-to-end physics."}


def run():
    raw32 = (ROOT/"results_step32/certified/traffic_results.json").read_bytes()
    raw33 = (ROOT/"results_step33/paired_results.json").read_bytes()
    d, step33 = json.loads(raw32), json.loads(raw33)
    if hashlib.sha256(raw32).hexdigest() != step33["source_sha256"] or not d["complete"]:
        raise ValueError("Step 32/33 inputs do not match.")
    phys = d["physical"]
    model = ReferenceNetworkPhysical(ReferenceParameters(**phys["parameters"]),phys["fec_overhead"])
    assert model.metadata()["modulations"] == phys["modulations"]
    links = {edge_key(*e["nodes"]):e["length_km"] for e in d["links"]}
    state = {edge_key(*e["nodes"]):set(e["available_wavelengths"]) for e in d["links"]}
    network = OpticalNetwork(model,links,d["k_candidates"])
    archived = {(r["instance_id"],r["method"]):r for r in d["rows"]}
    step33_rows = {p["instance_id"]:p for p in step33["pairs"]}
    instances = {i["id"]:i for i in d["instances"] if i["scenario"] in SCENARIOS}
    assert instances.keys() == step33_rows.keys() and len(instances)==80
    converters = {"none":set(), "only_A":{"A"}, "all":set(d["nodes"])}
    keys = list(dict.fromkeys(tuple(q) for i in instances.values() for q in i["requests"]))
    assert all(q[2] == 100 for q in keys)
    paths = {k:catalog(network,keys,dbm_to_w(d["common_power_dbm"]),d["snr_margin_db"],nodes)
             for k,nodes in converters.items()}
    records = []
    for iid,i in instances.items():
        qs = i["requests"]
        digest = hashlib.sha256(json.dumps(qs,separators=(",", ":")).encode()).hexdigest()
        assert digest == i["sha256"] == step33_rows[iid]["sha256"]
        plans, proofs = {}, {}
        for layout in LAYOUTS:
            for method,rule in ((BASE,"first"),(ONLINE,"widest_bottleneck")):
                plans[layout,method] = online_plan(network,qs,paths[layout],state,rule)
        old = archived[iid,"ILP"]
        assert old["optimal"]
        plans["none","ILP"] = unpack(old["plan"])
        proofs["none"] = {"upper_gbps":old["upper_bound_gbps"],"source":"certified Step 32"}
        assert plans["none",BASE] == unpack(archived[iid,BASE]["plan"])
        assert plans["none",ONLINE] == unpack(step33_rows[iid]["online_plan"])
        # The no-conversion optimum and both online policies are feasible in 'all'.
        all_plan, all_bound, attempts = exact(qs,paths["all"],state,sum(q[2] for q in qs),list(plans.values()))
        plans["all","ILP"] = all_plan
        proofs["all"] = {"upper_gbps":all_bound,"source":"total offered bound / full-conversion MILP",
                         "attempts":attempts}
        sparse_witnesses = [plans["none","ILP"],plans["only_A",BASE],plans["only_A",ONLINE]]
        sparse, sparse_bound, attempts = exact(qs,paths["only_A"],state,all_bound,sparse_witnesses)
        plans["only_A","ILP"] = sparse
        proofs["only_A"] = {"upper_gbps":sparse_bound,"source":"full-conversion bound / sparse MILP",
                            "attempts":attempts}
        for layout in LAYOUTS:
            for method in (BASE,ONLINE,"ILP"):
                plan = plans[layout,method]
                stats = audit(plan,qs,paths[layout],state,converters[layout])
                endpoints = sum(d["hotspot_node"] in (a["route"][0],a["route"][-1]) for a in plan)
                transit = sum(d["hotspot_node"] in a["route"][1:-1] for a in plan)
                assert endpoints+2*transit <= sum(len(w) for e,w in state.items() if d["hotspot_node"] in e)
                if method=="ILP":
                    assert stats["admitted_gbps"] == proofs[layout]["upper_gbps"]
                else:
                    stats.update(rejection_causes(plan,qs,paths[layout],state))
                    assert stats["admitted_gbps"] <= proofs[layout]["upper_gbps"]
                records.append({"instance_id":iid,"input_sha256":digest,"scenario":i["scenario"],"seed":i["seed"],
                                "load":len(qs),"layout":layout,"method":method,"plan":plan,**stats,
                                "F_endpoints":endpoints,"F_transits":transit,
                                "proof":proofs[layout] if method=="ILP" else None})
        if i["seed"]==d["seed_values"][-1]:
            print(f"Completed {i['scenario']}, {len(qs)} requests, {len(d['seed_values'])} seeds.",flush=True)
    return {"source_step32_sha256":hashlib.sha256(raw32).hexdigest(),
            "source_step33_sha256":hashlib.sha256(raw33).hexdigest(),"input_batches":len(instances),
            "assumptions":"Ideal conversion between any available wavelengths at designated intermediate "
                          "nodes, unlimited converters, no added penalty or regeneration; end-to-end SNR is shared.",
            "online_waves":"Minimise changes on the chosen path, then take the lexicographically smallest sequence.",
            "layouts":{k:sorted(nodes) for k,nodes in converters.items()},
            "example":frozen_example(instances,archived,paths,state),"records":records}


def summarize(records):
    baseline = {(r["instance_id"],r["method"]):r for r in records if r["layout"]=="none"}
    groups = defaultdict(list)
    for r in records:
        groups[r["scenario"],r["load"],r["layout"],r["method"]].append(r)
    summary = []
    for (sce,n,layout,method),group in sorted(groups.items()):
        gains = [r["admitted_gbps"]-baseline[r["instance_id"],method]["admitted_gbps"] for r in group]
        item = {"scenario":sce,"load":n,"layout":layout,"method":method,"seeds":len(group),
                "gain_vs_none_gbps":mean(gains),"range_gbps":[min(gains),max(gains)],
                "wins":sum(g>0 for g in gains),"ties":gains.count(0),"losses":sum(g<0 for g in gains)}
        metrics = ["admitted_gbps","accepted","F_endpoints","F_transits"]
        if method!="ILP":
            metrics += ["converted_connections","changes","continuity_blocks","saturated_link_blocks"]
        item.update({key:mean(r[key] for r in group) for key in metrics})
        summary.append(item)
    return summary


def main():
    data = run()
    summary = summarize(data["records"])
    out = ROOT/"results_step35"
    out.mkdir(exist_ok=True)
    (out/"summary.json").write_text(json.dumps({k:v for k,v in data.items() if k!="records"} | {"summary":summary},
                                              indent=2,allow_nan=False)+"\n",encoding="utf-8")
    raw = json.dumps(data,separators=(",",":"),allow_nan=False).encode()
    (out/"plans.json.gz").write_bytes(gzip.compress(raw,mtime=0))
    fig, axes = plt.subplots(1,2,figsize=(11,4.4),layout="constrained",sharey=True)
    for ax,sce in zip(axes,SCENARIOS):
        for method,color in [(BASE,"#3e6985"),(ONLINE,"#ba562c"),("ILP","#387865")]:
            ys = [next(s["admitted_gbps"] for s in summary if (s["scenario"],s["load"],s["layout"],s["method"])
                       == (sce,96,l,method)) for l in LAYOUTS]
            ax.plot(range(3),ys,marker="o",lw=2,color=color,label=method)
        ax.set(xticks=range(3),xticklabels=["None","A only","All nodes"],
               xlabel="Nodes with ideal conversion",title="Uniform endpoints" if sce==SCENARIOS[0] else "Hotspot at F")
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Mean admitted traffic (Gbit/s)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Wavelength conversion at 96 requests | 10 paired seeds")
    fig.savefig(out/"conversion_comparison.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    main()
