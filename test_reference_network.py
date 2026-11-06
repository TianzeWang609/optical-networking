"""Integration checks for reference physics, traffic controls and solver bounds."""

import copy
import math
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from network_scenarios import SCENARIOS, generate_requests, mesh12_case, pair_probabilities
from reference_network import ReferenceNetworkPhysical, reference_toy_case, validate_plan
from optical_networking_refactored.network_model import OpticalNetwork, maximum_snr_margin
from optical_networking_refactored.physical_model import MODULATIONS, w_to_dbm


class ReferenceNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ReferenceNetworkPhysical()
        cls.power = cls.model.optimal_power_w(10)

    def test_reference_units_and_net_rate(self):
        self.assertAlmostEqual(w_to_dbm(self.power), -1.4995629512219573, places=8)
        self.assertAlmostEqual(self.model.snr_db(10, self.power), 18.008654687147416, places=8)
        self.assertAlmostEqual(self.model.ideal_capacity_bps(10, self.power) / 1e9, 384.3191460891319, places=6)
        self.assertAlmostEqual(self.model.net_rate_gbps(next(m for m in MODULATIONS if m.name == "PM-32QAM")),
                               266.6666666666667)
        with self.assertRaises(ValueError):
            self.model.snr_db(10, -1.5)
        with self.assertRaises(ValueError):
            self.model.spans_for_length(801)

    def test_four_node_optimum_and_ordering(self):
        for scenario, expected in (("A", (320, 480)), ("B", (480, 320))):
            links, state, requests = reference_toy_case(scenario)
            original = copy.deepcopy(state)
            network = OpticalNetwork(self.model, links)
            for method, traffic in zip(("min", "max"), expected):
                result = network.solve_sequential(requests, state, self.power, method)
                self.assertEqual(result.admitted_traffic_gbps, traffic)
                validate_plan(network, requests, state, self.power, 0, result)
            exact = network.solve_ilp(requests, state, self.power)
            self.assertTrue(exact.optimal)
            self.assertEqual(exact.admitted_traffic_gbps, 480)
            self.assertEqual(exact.upper_bound_gbps, 480)
            self.assertEqual(exact.admitted_traffic_gbps,
                             network.solve_exhaustive(requests, state, self.power).admitted_traffic_gbps)
            self.assertEqual(state, original)

    def test_old_demands_are_not_shannon_or_multilightpath_rates(self):
        links, state, requests = reference_toy_case(demand_gbps=1000)
        result = OpticalNetwork(self.model, links).solve_ilp(requests, state, self.power)
        self.assertEqual(result.accepted_count, 0)
        self.assertTrue(result.optimal)

    def test_equal_distance_bottleneck_uses_link_counts_then_first_fit(self):
        # Via B: three common wavelengths, weakest link has three. Via C:
        # two common wavelengths, but both links have four. Via E is longer.
        links = {("A", "B"): 80, ("B", "D"): 80, ("A", "C"): 80,
                 ("C", "D"): 80, ("A", "E"): 80, ("E", "D"): 160}
        state = {("A", "B"): {1, 2, 3}, ("B", "D"): {1, 2, 3},
                 ("A", "C"): {4, 5, 6, 7}, ("C", "D"): {1, 2, 4, 6},
                 ("A", "E"): set(range(1, 9)), ("D", "E"): set(range(1, 9))}
        original = copy.deepcopy(state)
        network = OpticalNetwork(self.model, links)
        requests = [("A", "D", 100)]
        baseline = network.solve_sequential(requests, state, self.power, "min", 2)
        online = network.solve_sequential(requests, state, self.power, "min", 2,
                                          route_rule="widest_bottleneck")
        self.assertEqual((baseline.plan[0].route, baseline.plan[0].wavelength), (("A", "B", "D"), 1))
        self.assertEqual((online.plan[0].route, online.plan[0].wavelength), (("A", "C", "D"), 4))
        self.assertEqual(state, original)
        # Equally scored paths keep candidate order. The preferred path must
        # still have a common wavelength, even when each link has more free slots.
        for cd_waves in ({1, 2, 4}, {8, 9, 10, 11}):
            state[("C", "D")] = cd_waves
            result = network.solve_sequential(requests, state, self.power, "min", 2,
                                              route_rule="widest_bottleneck")
            self.assertEqual(result.plan[0].route, ("A", "B", "D"))

    def test_snr_margin_boundary(self):
        links, state, requests = reference_toy_case()
        network = OpticalNetwork(self.model, links)
        analytic = self.model.snr_db(20, self.power) - 10.81
        margin = maximum_snr_margin(network, requests, state, self.power, target_gbps=480)
        self.assertLessEqual(margin, analytic)
        self.assertLess(analytic - margin, 1e-4)
        self.assertEqual(network.solve_ilp(requests, state, self.power, analytic - 1e-5).admitted_traffic_gbps, 480)
        self.assertLess(network.solve_ilp(requests, state, self.power, analytic + 1e-5).admitted_traffic_gbps, 480)

    def test_mixed_weighted_objective_against_exhaustive(self):
        links, state, _ = reference_toy_case()
        requests = [("C", "D", 100), ("A", "D", 200), ("B", "D", 240), ("A", "D", 160)]
        network = OpticalNetwork(self.model, links)
        for margin in (0, 2, 4):
            exact = network.solve_ilp(requests, state, self.power, margin)
            self.assertEqual(exact.admitted_traffic_gbps,
                             network.solve_exhaustive(requests, state, self.power, margin).admitted_traffic_gbps)
            validate_plan(network, requests, state, self.power, margin, exact)

    def test_mesh_and_paired_traffic_controls(self):
        links, state, positions = mesh12_case()
        network = OpticalNetwork(self.model, links)
        self.assertEqual((len(positions), len(links)), (12, 17))
        self.assertTrue(nx.is_connected(network.graph))
        self.assertTrue(all(len(waves) == 8 for waves in state.values()))
        for seed in (0, 1, 9):
            batches = {s: generate_requests(positions, 48, seed, s) for s in SCENARIOS}
            for scenario, batch in batches.items():
                self.assertEqual(sum(r[2] for r in batch), 4800)
                self.assertTrue(all(s != t for s, t, _ in batch))
                self.assertEqual(batch, generate_requests(positions, 48, seed, scenario))
            for prefix in ("uniform", "hotspot"):
                self.assertEqual([r[:2] for r in batches[prefix + "_fixed"]],
                                 [r[:2] for r in batches[prefix + "_mixed"]])
                self.assertEqual(sorted(r[2] for r in batches[prefix + "_mixed"]),
                                 [50] * 12 + [100] * 24 + [150] * 12)

    def test_hotspot_probability_mass(self):
        pairs, weights = pair_probabilities("ABCDEFGHIJKL", "F", 0.65)
        self.assertAlmostEqual(sum(weights), 1)
        self.assertAlmostEqual(sum(w for p, w in zip(pairs, weights) if "F" in p), 0.65)
        _, uniform = pair_probabilities("ABCDEFGHIJKL")
        self.assertTrue(np.allclose(uniform, 1 / 66))

    def test_tiny_mesh_ilp_dominates_heuristics(self):
        links, state, nodes = mesh12_case(2)
        requests = generate_requests(nodes, 12, 5, "hotspot_mixed")
        network = OpticalNetwork(self.model, links)
        exact = network.solve_ilp(requests, state, self.power, 2)
        for k in (1, 4):
            trial = OpticalNetwork(self.model, links, k)
            result = trial.solve_sequential(requests, state, self.power, "min", 2)
            self.assertGreaterEqual(exact.admitted_traffic_gbps, result.admitted_traffic_gbps)
            validate_plan(trial, requests, state, self.power, 2, result)
        validate_plan(network, requests, state, self.power, 2, exact)

    def test_time_limit_keeps_a_feasible_incumbent_without_claiming_optimality(self):
        network = OpticalNetwork(self.model, {("A", "B"): 400})
        stub = SimpleNamespace(status=1, x=np.array([1.0]), mip_dual_bound=-100.0, mip_gap=0.0)
        with patch("optical_networking_refactored.network_model.milp", return_value=stub):
            result = network.solve_ilp([("A", "B", 100)], {("A", "B"): {1}}, self.power, time_limit_s=1)
        self.assertEqual(result.admitted_traffic_gbps, 100)
        self.assertFalse(result.optimal)
        self.assertEqual(result.solver_status, "limit_feasible")

    def test_time_limit_without_incumbent_and_fractional_solution(self):
        network = OpticalNetwork(self.model, {("A", "B"): 400})
        stub = SimpleNamespace(status=1, x=None, mip_dual_bound=-100.0, mip_gap=math.inf)
        with patch("optical_networking_refactored.network_model.milp", return_value=stub):
            result = network.solve_ilp([("A", "B", 100)], {("A", "B"): {1}}, self.power, time_limit_s=1)
        self.assertEqual(result.admitted_traffic_gbps, 0)
        self.assertEqual(result.upper_bound_gbps, 100)
        self.assertIsNone(result.mip_gap)
        self.assertEqual(result.solver_status, "limit_no_incumbent")
        stub.x = np.array([0.5])
        with patch("optical_networking_refactored.network_model.milp", return_value=stub):
            with self.assertRaises(RuntimeError):
                network.solve_ilp([("A", "B", 100)], {("A", "B"): {1}}, self.power)

    def test_certification_preserves_original_inputs_and_timing(self):
        from step32_network_traffic import certify_existing
        from reference_network import result_record
        from optical_networking_refactored.network_model import AllocationResult
        links, state = {("A", "B"): 400}, {("A", "B"): {1}}
        requests = [("A", "B", 100)]
        network = OpticalNetwork(self.model, links)
        heuristic = network.solve_sequential(requests, state, self.power)
        unproven = AllocationResult(0.0, (None,), solver_status="limit_no_incumbent",
                                    optimal=False, upper_bound_gbps=100.0)
        rows = [{"method": method, "instance_id": "sample", "scenario": "uniform_fixed", "seed": 0,
                 "elapsed_seconds": 0.01,
                 **result_record(network, requests, state, self.power, 0.0, result)}
                for method, result in (("KSP-FF", heuristic), ("ILP", unproven))]
        payload = {"complete": True, "physical": self.model.metadata(), "rows": rows,
                   "links": [{"nodes": ["A", "B"], "length_km": 400, "available_wavelengths": [1]}],
                   "instances": [{"id": "sample", "requests": requests}], "k_candidates": 4,
                   "common_power_dbm": w_to_dbm(self.power), "snr_margin_db": 0.0,
                   "timing": "Test timing", "total_elapsed_seconds": 1.0}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "traffic_results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            before = source.read_bytes()
            certified, destination = certify_existing(Path(tmp), 2.0)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(destination, Path(tmp) / "certified")
            self.assertTrue(certified["certification"]["all_ilps_proven"])
            self.assertEqual(certified["rows"][1]["admitted_traffic_gbps"], 100)
            self.assertEqual(certified["rows"][0]["gap_to_proven_optimum_gbps"], 0)
            self.assertGreater(certified["rows"][1]["elapsed_seconds"], 0.01)
            self.assertEqual(certified["certification"]["history"][0]["initial_result"]["elapsed_seconds"], 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
