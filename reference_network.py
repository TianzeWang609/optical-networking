"""Connect the checked reference GN model to discrete network admission.

The network-facing API uses watts and (spans, power), while the reference
physical API uses dBm and (power, spans). This adapter owns that conversion.
"""

from dataclasses import asdict
from functools import lru_cache
import math

from reference_physical_model import ReferencePhysicalModel, dbm_to_w
from optical_networking_refactored.physical_model import MODULATIONS, w_to_dbm
from optical_networking_refactored.network_model import OpticalNetwork, edge_key, toy_case


class ReferenceNetworkPhysical:
    """Full-load central-channel coherent SPM + linear XPM, plus modulation/FEC."""

    def __init__(self, parameters=None, fec_overhead=0.20):
        if not math.isfinite(fec_overhead) or fec_overhead < 0:
            raise ValueError("fec_overhead must be finite and nonnegative.")
        self.reference = ReferencePhysicalModel(parameters)
        self.config = self.reference.p
        self.fec_overhead = fec_overhead

    def spans_for_length(self, length_km):
        return self.reference.spans_for_distance(length_km)

    def optimal_power_w(self, spans=10):
        return dbm_to_w(self.reference.optimal_power_dbm(spans))

    @lru_cache(maxsize=8192)
    def snr_db(self, spans, power_w):
        return self.reference.snr_db(w_to_dbm(power_w), spans)

    def ideal_capacity_bps(self, spans, power_w):
        return self.reference.capacity_gbps(w_to_dbm(power_w), spans) * 1e9

    def net_rate_gbps(self, modulation):
        return self.config.symbol_rate_baud * modulation.bits_dp / (1 + self.fec_overhead) / 1e9

    def best_modulation(self, snr_db, snr_margin_db=0.0):
        if not math.isfinite(snr_db) or not math.isfinite(snr_margin_db) or snr_margin_db < 0:
            raise ValueError("SNR must be finite and margin must be finite and nonnegative.")
        return max((m for m in MODULATIONS if snr_db >= m.required_snr_db + snr_margin_db),
                   key=lambda m: m.bits_dp, default=None)

    def metadata(self):
        return {"parameters": asdict(self.config), "fec_overhead": self.fec_overhead,
                "modulations": [asdict(m) for m in MODULATIONS],
                "model": "Numerical coherent SPM + linear numerical XPM",
                "snr_policy": "Full-load central-channel estimate, independent of occupancy and wavelength ID",
                "power_unit_in_network_api": "W per channel, both polarizations",
                "rate_definition": "symbol_rate * bits_dp / (1 + fec_overhead)",
                "resource_policy": "Undirected shared resources; one continuous wavelength per admitted request",
                "traffic_policy": "Static batch, no splitting, grooming, regeneration or departures"}


def reference_toy_case(scenario="A", demand_gbps=160.0):
    """Preserve the original resources/order; 160 = 1000 * 32/200 Gbit/s."""
    if not math.isfinite(demand_gbps) or demand_gbps <= 0:
        raise ValueError("demand_gbps must be finite and positive.")
    links, state, requests = toy_case(scenario)
    return links, state, [(s, t, demand_gbps) for s, t, _ in requests]


def validate_plan(network, requests, state, power_w, margin_db, result):
    """Independently replay selected paths and audit resources, rates and traffic."""
    if len(result.plan) != len(requests):
        raise AssertionError("Plan length differs from request count.")
    used, traffic = set(), 0.0
    for request, option in zip(requests, result.plan):
        if option is None:
            continue
        source, target, demand = request
        route = option.route
        if route[0] != source or route[-1] != target or len(set(route)) != len(route):
            raise AssertionError("Invalid path endpoints or repeated nodes.")
        resources = {(edge_key(u, v), option.wavelength) for u, v in zip(route, route[1:])}
        if resources != option.resources or used.intersection(resources):
            raise AssertionError("Resource list mismatch or wavelength collision.")
        if any(edge not in state or wave not in state[edge] for edge, wave in resources):
            raise AssertionError("Plan uses an unavailable resource.")
        modulation = next(m for m in MODULATIONS if m.name == option.modulation)
        snr = network.physical.snr_db(network.route_spans(route), power_w)
        rate = network.physical.net_rate_gbps(modulation)
        if snr + 1e-9 < modulation.required_snr_db + margin_db or rate + 1e-9 < demand:
            raise AssertionError("Plan fails the SNR or net-rate condition.")
        if not math.isclose(snr, option.snr_db, abs_tol=1e-9) or not math.isclose(rate, option.net_rate_gbps):
            raise AssertionError("Reported physical values differ from the chosen path.")
        used.update(resources)
        traffic += demand
    if not math.isclose(traffic, result.admitted_traffic_gbps, abs_tol=1e-9):
        raise AssertionError("Admitted demand was confused with lightpath capacity.")


def result_record(network, requests, state, power_w, margin_db, result):
    validate_plan(network, requests, state, power_w, margin_db, result)
    total = sum(r[2] for r in requests)
    used = [resource for option in result.plan if option for resource in option.resources]
    return {"offered_traffic_gbps": total, "admitted_traffic_gbps": result.admitted_traffic_gbps,
            "accepted_count": result.accepted_count, "request_count": len(requests),
            "request_blocking_fraction": 1 - result.accepted_count / len(requests) if requests else 0.0,
            "bandwidth_blocking_fraction": 1 - result.admitted_traffic_gbps / total if total else 0.0,
            "link_utilization": {"-".join(edge): sum(r[0] == edge for r in used) / len(waves) if waves else 0.0
                                 for edge, waves in state.items()},
            "variable_count": result.variable_count, "constraint_count": result.constraint_count,
            "solver_status": result.solver_status, "optimal": result.optimal,
            "upper_bound_gbps": result.upper_bound_gbps, "mip_gap": result.mip_gap,
            "plan": [{"request_index": i, "route": list(o.route), "wavelength": o.wavelength,
                      "modulation": o.modulation, "net_rate_gbps": o.net_rate_gbps, "snr_db": o.snr_db}
                     for i, o in enumerate(result.plan) if o is not None]}
