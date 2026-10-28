"""Regression checks for the original step25 model and earlier reference results."""

import contextlib
import copy
import importlib
import io
import math
import unittest

from physical_model import PhysicalConfig, PhysicalModel, dbm_to_w, w_to_dbm
from network_model import OpticalNetwork, maximum_snr_margin, toy_case


class BaselineRegression(unittest.TestCase):
    def setUp(self):
        self.model = PhysicalModel()
        links, self.state, self.requests = toy_case()
        self.network = OpticalNetwork(self.model, links)
        self.power = self.model.optimal_power_w(10)

    def test_physical_baseline(self):
        self.assertAlmostEqual(self.model.x_spm + self.model.x_xpm_sum, 29.385957178, places=7)
        self.assertAlmostEqual(10 * self.model.ase_span_w * 1e6, 41.5879003285542, places=8)
        self.assertAlmostEqual(w_to_dbm(self.power), 6.165990524656909, places=8)
        self.assertAlmostEqual(self.model.snr_db(10, self.power), 18.21540798958553, places=8)
        self.assertAlmostEqual(self.model.ideal_capacity_bps(10, self.power) / 1e9, 2429.050, places=3)
        self.assertAlmostEqual(self.model.snr_db(10, self.power) - self.model.snr_db(20, self.power),
                               10 * math.log10(2), places=10)

    def test_routes_and_k(self):
        lengths = [self.network.route_length_km(r) for r in self.network.candidate_routes("A", "D")]
        self.assertEqual(lengths, [800, 1600, 2400, 2400])
        for k, expected in ((1, 0), (2, 1000), (3, 1000), (4, 1000)):
            network = OpticalNetwork(self.model, self.network.links_km, k)
            result = network.solve_sequential([("A", "D", 1000.0)] * 2, self.state, self.power)
            self.assertEqual(result.admitted_traffic_gbps, expected)

    def test_rules_and_exact_optima(self):
        for scenario, expected in (("A", (2000, 3000)), ("B", (3000, 2000))):
            _, state, requests = toy_case(scenario)
            for rule, traffic in zip(("min", "max"), expected):
                self.assertEqual(self.network.solve_sequential(requests, state, self.power, rule)
                                 .admitted_traffic_gbps, traffic)
            exhaustive = self.network.solve_exhaustive(requests, state, self.power)
            ilp = self.network.solve_ilp(requests, state, self.power)
            self.assertEqual(ilp.admitted_traffic_gbps, 3000)
            self.assertEqual(ilp.admitted_traffic_gbps, exhaustive.admitted_traffic_gbps)
            self.assertEqual((ilp.variable_count, ilp.constraint_count), (7, 10))
            used = [r for option in ilp.plan if option for r in option.resources]
            self.assertEqual(len(used), len(set(used)))

    def test_power_and_margin_boundaries(self):
        for power_dbm, expected in ((-10, 0), (-5, 1000), (0, 2000), (5, 3000),
                                    (10, 3000), (12, 2000), (13, 1000), (14, 0)):
            self.assertEqual(self.network.optimal_admitted_traffic(
                self.requests, self.state, dbm_to_w(power_dbm)), expected)
        for margin, first, last in ((0, 0.1, 10.5), (2, 2.2, 9.3), (4, 4.8, 7.4)):
            for power_dbm in (first, last):
                self.assertEqual(self.network.optimal_admitted_traffic(
                    self.requests, self.state, dbm_to_w(power_dbm), margin), 3000)
            for power_dbm in (first - 0.1, last + 0.1):
                self.assertLess(self.network.optimal_admitted_traffic(
                    self.requests, self.state, dbm_to_w(power_dbm), margin), 3000)
        self.assertEqual(self.network.optimal_admitted_traffic(self.requests, self.state, self.power, 6), 2000)

    def test_maximum_margin(self):
        margin = maximum_snr_margin(self.network, self.requests, self.state, self.power)
        expected = self.model.snr_db(20, self.power) - 10.81
        self.assertLessEqual(margin, expected)
        self.assertLess(expected - margin, 1e-4)
        self.assertEqual(self.network.optimal_admitted_traffic(self.requests, self.state, self.power, 4.39), 3000)
        self.assertEqual(self.network.optimal_admitted_traffic(self.requests, self.state, self.power, 4.40), 2000)

    def test_distance_scan_and_state_isolation(self):
        original_state = copy.deepcopy(self.state)
        original_links = self.network.links_km.copy()
        expected = {0: [3000, 3000, 2000, 2000, 2000, 1000], 4: [3000, 2000, 1000, 1000, 0, 0]}
        for margin, traffic in expected.items():
            actual = [self.network.scaled(scale).optimal_admitted_traffic(
                self.requests, self.state, self.power, margin) for scale in range(1, 7)]
            self.assertEqual(actual, traffic)
        self.network.solve_sequential(self.requests, self.state, self.power)
        self.assertEqual(self.state, original_state)
        self.assertEqual(self.network.links_km, original_links)

    def test_weighted_objective_against_exhaustive(self):
        # 不同需求可检查 ILP 是否仍最大化业务量，而不是误用接纳数量
        requests = [("C", "D", 500.0), ("A", "D", 1200.0),
                    ("B", "D", 1500.0), ("A", "D", 1000.0)]
        for margin in (0.0, 4.0):
            ilp = self.network.solve_ilp(requests, self.state, self.power, margin)
            exhaustive = self.network.solve_exhaustive(requests, self.state, self.power, margin)
            self.assertEqual(ilp.admitted_traffic_gbps, exhaustive.admitted_traffic_gbps)

    def test_full_span_validation_and_epsilon(self):
        with self.assertRaises(ValueError):
            OpticalNetwork(self.model, {("A", "B"): 401.0})
        coherent = PhysicalModel(PhysicalConfig(epsilon=0.1))
        self.assertLess(coherent.optimal_power_w(20), coherent.optimal_power_w(10))
        self.assertAlmostEqual(self.model.optimal_power_w(20), self.power)

    def test_experiment_import_is_quiet(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            importlib.import_module("run_experiments")
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
