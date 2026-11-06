"""Routing with ideal wavelength conversion at designated intermediate nodes."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from optical_networking_refactored.network_model import edge_key


@dataclass(frozen=True)
class CandidateRoute:
    nodes: tuple
    edges: tuple
    segments: tuple
    snr_db: float
    net_rate_gbps: float


def catalog(network, requests, power, margin, converters):
    """Keep the original K routes and end-to-end physics; partition only wavelengths."""
    result = {}
    for key in dict.fromkeys(tuple(r) for r in requests):
        source, target, demand = key
        result[key] = []
        for nodes in network.candidate_routes(source, target):
            snr = network.physical.snr_db(network.route_spans(nodes), power)
            fmt = network.physical.best_modulation(snr, margin)
            if fmt is None or network.physical.net_rate_gbps(fmt) < demand:
                continue
            edges = tuple(edge_key(u, v) for u, v in pairwise(nodes))
            cuts = [0, *(i for i in range(1, len(nodes)-1) if nodes[i] in converters), len(edges)]
            segments = tuple(edges[a:b] for a,b in pairwise(cuts))
            result[key].append(CandidateRoute(tuple(nodes), edges, segments, snr, network.physical.net_rate_gbps(fmt)))
    return result


def assign_waves(path, state):
    """Fewest changes first, then lexicographically smallest wavelength sequence."""
    common = [set.intersection(*(state[e] for e in segment)) for segment in path.segments]
    if any(not waves for waves in common):
        return None
    direct = set.intersection(*common)
    if direct:
        return [min(direct)] * len(path.edges)
    # For each possible last wavelength retain the best feasible prefix.
    best = {w: (0, (w,)) for w in common[0]}
    for waves in common[1:]:
        best = {w: min((cost + (last != w), seq + (w,)) for last,(cost,seq) in best.items())
                for w in waves}
    _, choices = min(best.values())
    return [w for w, segment in zip(choices, path.segments) for _ in segment]


def online_plan(network, requests, candidates, initial, rule):
    if rule not in ("first", "widest_bottleneck"):
        raise ValueError("Unknown routing rule.")
    state = {e:set(w) for e,w in initial.items()}
    plan = []
    for j, request in enumerate(requests):
        choices = []
        for path in candidates[tuple(request)]:
            waves = assign_waves(path, state)
            if waves is None:
                continue
            pressure = min(len(state[e]) for e in path.edges) if rule == "widest_bottleneck" else 0
            choices.append(((network.route_length_km(path.nodes), -pressure), path, waves))
        if choices:
            _, path, waves = min(choices, key=lambda entry: entry[0])
            for edge,w in zip(path.edges, waves):
                state[edge].remove(w)
            plan.append({"request_index":j, "route":list(path.nodes), "waves":waves})
    return plan


def audit(plan, requests, candidates, initial, converters):
    """Replay edge/wavelength conflicts, conversion locations and route eligibility."""
    seen, occupied = set(), set()
    traffic = conversions = converted = 0
    for item in plan:
        j, route, waves = item["request_index"], item["route"], item["waves"]
        assert 0 <= j < len(requests) and j not in seen
        seen.add(j)
        req = requests[j]
        assert route in [list(p.nodes) for p in candidates[tuple(req)]]
        assert route[0] == req[0] and route[-1] == req[1] and len(route) == len(set(route))
        assert len(waves) == len(route)-1
        changes = [route[i] for i in range(1,len(waves)) if waves[i-1] != waves[i]]
        assert set(changes) <= converters
        conversions += len(changes)
        converted += bool(changes)
        for (u,v),w in zip(pairwise(route),waves):
            e = edge_key(u,v)
            assert w in initial[e] and (e,w) not in occupied
            occupied.add((e,w))
        traffic += req[2]
    return {"admitted_gbps":traffic, "accepted":len(seen), "changes":conversions,
            "converted_connections":converted}


def offline_plan(requests, candidates, initial, time_limit, upper_bound):
    """Integer route counts plus colours on transparent segments of two or more links.

    Identical requests are interchangeable offline. A single-link segment can take
    any leftover colour on its edge once longer transparent segments are coloured.
    """
    groups = defaultdict(list)
    for j,r in enumerate(requests):
        groups[tuple(r)].append(j)
    row_keys, lower, upper = {}, [], []
    ri, ci, coefficients, objective, cap = [], [], [], [], []
    def row(key, lo, hi):
        if key not in row_keys:
            row_keys[key] = len(lower)
            lower.append(lo)
            upper.append(hi)
        return row_keys[key]

    def column(cost, bound, entries):
        index = len(objective)
        objective.append(cost)
        cap.append(bound)
        for r, value in entries:
            ri.append(r)
            ci.append(index)
            coefficients.append(value)
        return index

    total_row = row(("total",),0,upper_bound)
    route_columns = []
    for key, indices in groups.items():
        demand = key[2]
        group_row = row(("request",key),0,len(indices))
        for path in candidates[key]:
            pools = [set.intersection(*(initial[e] for e in seg)) for seg in path.segments]
            count_cap = min(len(indices), *(len(pool) for pool in pools))
            if not count_cap:
                continue
            entries = [(group_row,1), (total_row,demand)]
            entries.extend((row(("link",e),0,len(initial[e])),1) for e in path.edges)
            segment_rows = {}
            for s,seg in enumerate(path.segments):
                if len(seg) > 1:
                    eq = row(("segment",key,path.nodes,s),0,0)
                    entries.append((eq,-1))
                    segment_rows[s] = eq
            count_col = column(-demand,count_cap,entries)
            colour_cols = {}
            for s,eq in segment_rows.items():
                colour_cols[s] = {}
                for w in sorted(pools[s]):
                    uses = [(row(("colour",e,w),0,1),1) for e in path.segments[s]]
                    colour_cols[s][w] = column(0,1,[(eq,1), *uses])
            route_columns.append((key,path,count_col,colour_cols))

    if not objective:
        return [], 0.0, "optimal"
    matrix = coo_matrix((coefficients,(ri,ci)),shape=(len(lower),len(objective))).tocsc()
    solution = milp(objective, integrality=np.ones(len(cap),dtype=int), bounds=Bounds(0,cap),
                    constraints=LinearConstraint(matrix,lower,upper),
                    options={"time_limit":time_limit,"mip_rel_gap":0.0})
    if solution.status not in (0,1):
        raise RuntimeError(solution.message)
    dual = getattr(solution,"mip_dual_bound",None)
    bound = upper_bound if dual is None or not math.isfinite(dual) else min(upper_bound,-float(dual))
    if solution.x is None:
        return [], bound, "limit"
    x = np.rint(solution.x).astype(int)
    if (np.max(np.abs(x-solution.x))>1e-5 or np.any(x<0) or np.any(x>cap)
            or np.any(matrix@x < np.array(lower)-1e-6) or np.any(matrix@x > np.array(upper)+1e-6)):
        raise RuntimeError("Invalid integer solution.")
    remaining = {e:set(waves) for e,waves in initial.items()}
    pending = []
    used = defaultdict(int)
    for key,path,count_col,colour_cols in route_columns:
        n = x[count_col]
        if not n:
            continue
        segment_waves = [[None]*n for _ in path.segments]
        for s,columns in colour_cols.items():
            values = [w for w,index in columns.items() if x[index]]
            assert len(values)==n
            segment_waves[s] = values
            for e in path.segments[s]:
                for w in values:
                    remaining[e].remove(w)
        indices = groups[key][used[key]:used[key]+n]
        used[key] += n
        pending.append((path,indices,segment_waves))
    plan = []
    for path,indices,segment_waves in pending:
        for s,seg in enumerate(path.segments):
            if len(seg)==1:
                values = sorted(remaining[seg[0]])[:len(indices)]
                assert len(values)==len(indices)
                remaining[seg[0]].difference_update(values)
                segment_waves[s] = values
        for i,j in enumerate(indices):
            waves = [values[i] for seg,values in zip(path.segments,segment_waves) for _ in seg]
            plan.append({"request_index":j,"route":list(path.nodes),"waves":waves})
    plan.sort(key=lambda a:a["request_index"])
    value = sum(requests[a["request_index"]][2] for a in plan)
    if abs(value+solution.fun)>1e-5 or bound < value-1e-5:
        raise RuntimeError("Inconsistent MILP objective or bound.")
    return plan, value if solution.status==0 else max(value,bound), "optimal" if solution.status==0 else "limit"
