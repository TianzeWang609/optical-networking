"""Regression cases for sharing client capacity while reserving a whole lightpath."""
import unittest

from optical_networking_refactored.network_model import OpticalNetwork
from reference_network import ReferenceNetworkPhysical
from step36_endpoint_grooming import allocate, measure


class EndpointGroomingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ReferenceNetworkPhysical()
        cls.power = cls.model.optimal_power_w(10)
        cls.network = OpticalNetwork(cls.model, {("A", "B"): 1920})

    def test_tightest_fit_and_no_splitting(self):
        # Both lightpaths support 160. The third demand fits either but is
        # tighter in the second. The fourth fits their aggregate spare capacity only.
        requests = [("A","B",d) for d in (90,120,30,75)]
        state = {("A","B"):{37,38}}
        paths = allocate(self.network, requests, state, self.power, 2, grooming=True)
        self.assertEqual([p.clients for p in paths], [[0],[1,2]])
        self.assertEqual([p.remaining for p in paths], [70,10])
        record, accepted = measure(self.network, requests, state, self.power, 2, paths, "A")
        self.assertEqual(accepted, {0,1,2})
        self.assertEqual(record["metrics"]["used_link_wavelengths"], 2)
        self.assertEqual(state, {("A","B"):{37,38}})

    def test_equal_residual_uses_earliest_and_exact_fill_is_allowed(self):
        requests = [("A","B",d) for d in (100,100,60)]
        paths = allocate(self.network, requests, {("A","B"):{37,38}}, self.power, 2, grooming=True)
        self.assertEqual([p.clients for p in paths], [[0,2],[1]])
        self.assertEqual(paths[0].remaining, 0)

    def test_reverse_and_partial_routes_cannot_join(self):
        network = OpticalNetwork(self.model, {("A","B"):960, ("B","C"):960})
        state = {("A","B"):{37}, ("B","C"):{37}}
        requests = [("A","C",50), ("C","A",50), ("B","C",50), ("A","C",100)]
        on = allocate(network, requests, state, self.power, 2, grooming=True)
        off = allocate(network, requests, state, self.power, 2, grooming=False)
        self.assertEqual([lp.clients for lp in on], [[0,3]])
        self.assertEqual([lp.clients for lp in off], [[0]])
        record, _ = measure(network, requests, state, self.power, 2, on, "B")
        self.assertEqual(record["metrics"]["used_link_wavelengths"], 2)
        self.assertEqual(record["metrics"]["hotspot_transit_lightpaths"], 1)
        self.assertEqual(record["metrics"]["admitted_gbps"], 150)


if __name__ == "__main__":
    unittest.main()
