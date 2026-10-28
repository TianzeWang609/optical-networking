"""Routing, wavelength allocation and offline optimization for the toy network."""

from dataclasses import dataclass
from itertools import islice, product
import math

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from physical_model import PhysicalModel


def edge_key(u, v):
    # 保留原模型的无向共享资源约定
    return tuple(sorted((u, v)))


def toy_case(scenario="A"):
    """Return fresh links, free wavelengths and requests for each experiment."""
    links = {("A", "B"): 400, ("B", "D"): 400, ("A", "C"): 800,
             ("C", "D"): 800, ("B", "C"): 1200}
    free = {("A", "B"): {1, 2}, ("B", "D"): {3, 4}, ("A", "C"): {5, 6},
            ("C", "D"): {6, 7}, ("B", "C"): {2, 6}}
    if scenario == "B":
        free[("A", "C")] = {5, 7}
    elif scenario != "A":
        raise ValueError("scenario must be 'A' or 'B'.")
    requests = [("C", "D", 1000.0), ("A", "D", 1000.0),
                ("B", "D", 1000.0), ("A", "D", 1000.0)]
    return links, free, requests


@dataclass(frozen=True)
class AllocationOption:
    route: tuple[str, ...]
    wavelength: int
    resources: frozenset
    snr_db: float
    modulation: str
    net_rate_gbps: float


@dataclass(frozen=True)
class AllocationResult:
    admitted_traffic_gbps: float
    plan: tuple[AllocationOption | None, ...]
    variable_count: int = 0
    constraint_count: int = 0

    @property
    def accepted_count(self):
        return sum(option is not None for option in self.plan)


class OpticalNetwork:
    def __init__(self, physical: PhysicalModel, links_km, k_candidates=4):
        if not isinstance(k_candidates, int) or k_candidates < 1:
            raise ValueError("k_candidates must be a positive integer.")
        self.physical = physical
        self.k_candidates = k_candidates
        self.graph = nx.Graph()
        self.links_km = {}
        for (u, v), length in links_km.items():
            key = edge_key(u, v)
            if u == v or key in self.links_km:
                raise ValueError("Links must be unique undirected edges without self-loops.")
            physical.spans_for_length(length)
            self.links_km[key] = float(length)
            self.graph.add_edge(u, v, length_km=float(length))

    def scaled(self, factor):
        # 构造独立网络，原始长度和图权重都不会被修改
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Distance scale must be finite and positive.")
        return OpticalNetwork(self.physical,
                              {edge: length * factor for edge, length in self.links_km.items()},
                              self.k_candidates)

    def candidate_routes(self, source, target):
        if source == target or source not in self.graph or target not in self.graph:
            raise ValueError("Request endpoints must be distinct nodes in the network.")
        try:
            return list(islice(nx.shortest_simple_paths(
                self.graph, source, target, weight="length_km"), self.k_candidates))
        except nx.NetworkXNoPath:
            return []

    def route_length_km(self, route):
        return sum(self.links_km[edge_key(u, v)] for u, v in zip(route, route[1:]))

    def route_spans(self, route):
        return self.physical.spans_for_length(self.route_length_km(route))

    def _copy_state(self, state):
        normalized = {edge_key(*edge): set(waves) for edge, waves in state.items()}
        if len(normalized) != len(state) or normalized.keys() != self.links_km.keys():
            raise ValueError("Wavelength state must contain exactly one entry per link.")
        for waves in normalized.values():
            if any(not isinstance(w, int) or not 1 <= w <= self.physical.config.channel_count
                   for w in waves):
                raise ValueError("Wavelength identifiers must be integers from 1 to channel_count.")
        return normalized

    def feasible_options(self, request, initial_state, power_w, snr_margin_db=0.0):
        source, target, demand_gbps = request
        if not math.isfinite(demand_gbps) or demand_gbps <= 0:
            raise ValueError("Request demand must be finite and positive.")
        state = self._copy_state(initial_state)
        options = []
        for route in self.candidate_routes(source, target):
            keys = [edge_key(u, v) for u, v in zip(route, route[1:])]
            snr_db = self.physical.snr_db(self.route_spans(route), power_w)
            modulation = self.physical.best_modulation(snr_db, snr_margin_db)
            if modulation is None:
                continue
            rate = self.physical.net_rate_gbps(modulation)
            if rate < demand_gbps:
                continue
            common_free = set.intersection(*(state[key] for key in keys))
            for wavelength in sorted(common_free):
                options.append(AllocationOption(
                    tuple(route), wavelength,
                    frozenset((key, wavelength) for key in keys),
                    snr_db, modulation.name, rate,
                ))
        return options

    @staticmethod
    def _result(requests, plan, variable_count=0, constraint_count=0):
        # 目标是被接纳请求的需求总和，不是所选调制格式的容量总和
        traffic = sum(request[2] for request, option in zip(requests, plan) if option is not None)
        return AllocationResult(float(traffic), tuple(plan), variable_count, constraint_count)

    def solve_sequential(self, requests, initial_state, power_w,
                         wavelength_rule="max", snr_margin_db=0.0):
        if wavelength_rule not in ("min", "max"):
            raise ValueError("wavelength_rule must be 'min' or 'max'.")
        state = self._copy_state(initial_state)
        plan = []
        for request in requests:
            options = self.feasible_options(request, state, power_w, snr_margin_db)
            if not options:
                plan.append(None)
                continue
            # 先选第一条可行路径，再在该路径上选择波长
            first_route_options = [o for o in options if o.route == options[0].route]
            choose = min if wavelength_rule == "min" else max
            chosen = choose(first_route_options, key=lambda o: o.wavelength)
            for edge, wavelength in chosen.resources:
                state[edge].remove(wavelength)
            plan.append(chosen)
        return self._result(requests, plan)

    def solve_exhaustive(self, requests, initial_state, power_w, snr_margin_db=0.0):
        choices = [self.feasible_options(r, initial_state, power_w, snr_margin_db) + [None]
                   for r in requests]
        if math.prod(map(len, choices)) > 1_000_000:
            raise ValueError("Exhaustive search is limited to small examples; use solve_ilp.")
        best = None
        for plan in product(*choices):
            resources = [resource for option in plan if option is not None
                         for resource in option.resources]
            if len(resources) != len(set(resources)):
                continue
            result = self._result(requests, plan)
            if best is None or result.admitted_traffic_gbps > best.admitted_traffic_gbps:
                best = result
        return best

    def solve_ilp(self, requests, initial_state, power_w, snr_margin_db=0.0):
        variables = [(i, option) for i, request in enumerate(requests)
                     for option in self.feasible_options(request, initial_state, power_w, snr_margin_db)]
        if not variables:
            return self._result(requests, [None] * len(requests))
        resources = sorted({resource for _, option in variables for resource in option.resources})
        n_rows = len(requests) + len(resources)
        resource_rows = {resource: len(requests) + i for i, resource in enumerate(resources)}
        matrix = np.zeros((n_rows, len(variables)))
        objective = np.zeros(len(variables))
        for column, (request_index, option) in enumerate(variables):
            objective[column] = -requests[request_index][2]
            matrix[request_index, column] = 1.0
            for resource in option.resources:
                matrix[resource_rows[resource], column] = 1.0
        result = milp(
            c=objective, integrality=np.ones(len(variables), dtype=int),
            bounds=Bounds(0.0, 1.0), constraints=LinearConstraint(matrix, 0.0, 1.0),
            options={"mip_rel_gap": 0.0},
        )
        if not result.success:
            raise RuntimeError(f"ILP optimization failed: {result.message}")
        plan = [None] * len(requests)
        for value, (request_index, option) in zip(result.x, variables):
            if value > 0.5:
                plan[request_index] = option
        return self._result(requests, plan, len(variables), n_rows)

    def optimal_admitted_traffic(self, requests, initial_state, power_w, snr_margin_db=0.0):
        return self.solve_ilp(requests, initial_state, power_w, snr_margin_db).admitted_traffic_gbps


def maximum_snr_margin(network, requests, initial_state, power_w, target_gbps=3000.0,
                       upper_margin_db=10.0, tolerance_db=1e-4):
    """Find the largest feasible margin at the supplied fixed common power."""
    if any(not math.isfinite(v) or v <= 0 for v in (target_gbps, upper_margin_db, tolerance_db)):
        raise ValueError("Target, upper margin and tolerance must be finite and positive.")

    def feasible(margin):
        return network.optimal_admitted_traffic(requests, initial_state, power_w, margin) >= target_gbps - 1e-6

    low, high = 0.0, upper_margin_db
    if not feasible(low):
        raise ValueError("Target traffic is infeasible even at zero SNR margin.")
    if feasible(high):
        raise ValueError("Increase the upper margin bound before searching.")
    while high - low > tolerance_db:
        mid = (low + high) / 2
        if feasible(mid):
            low = mid
        else:
            high = mid
    return low
