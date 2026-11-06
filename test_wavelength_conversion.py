"""Check segment colouring and the conversion ILP against explicit tiny enumeration."""
from itertools import product
import unittest

from optical_networking_refactored.network_model import OpticalNetwork, edge_key
from reference_network import ReferenceNetworkPhysical
from wavelength_conversion import assign_waves, audit, catalog, offline_plan, online_plan


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ReferenceNetworkPhysical()
        cls.power = cls.model.optimal_power_w(10)

    def test_conversion_only_at_enabled_node_and_change_minimization(self):
        network = OpticalNetwork(self.model,{("P","Q"):80,("Q","R"):80,("R","S"):80,("S","T"):80})
        req = ("P","T",100)
        state = {("P","Q"):{2},("Q","R"):{1,2},("R","S"):{2,3},("S","T"):{3}}
        for conv, wanted in [(set(),None),({"Q"},None),({"S"},[2,2,2,3]),
                             (set("PQRST"),[2,2,2,3])]:
            path = catalog(network,[req],self.power,2,conv)[req][0]
            self.assertEqual(assign_waves(path,state),wanted)
        state["S","T"] = set()
        self.assertIsNone(assign_waves(path,state))

    def test_compact_integer_model_matches_wave_by_wave_search(self):
        links = {("P","X"):80,("X","A"):80,("A","Y"):80,("Y","Q"):80}
        state = dict(zip((edge_key(*e) for e in links),[{1,2},{1,2},{2,3},{2,3}]))
        network = OpticalNetwork(self.model,links)
        repeats = [("P","Q",100)]*2
        for requests in (repeats, repeats+[("P","X",50),("Y","Q",150),("X","Y",100)]):
            for conv in (set(),{"A"},set("PXAYQ")):
                paths = catalog(network,requests,self.power,2,conv)
                options = []
                for request in requests:
                    choices = [(0,set())]
                    for route in paths[request]:
                        for waves in product(*(state[e] for e in route.edges)):
                            changes = {route.nodes[j] for j in range(1,len(waves)) if waves[j]!=waves[j-1]}
                            if changes <= conv:
                                choices.append((request[2],set(zip(route.edges,waves))))
                    options.append(choices)
                optimum = 0
                for selection in product(*options):
                    used = [resource for _,resources in selection for resource in resources]
                    if len(used)==len(set(used)):
                        optimum = max(optimum,sum(rate for rate,_ in selection))
                plan,bound,status = offline_plan(requests,paths,state,5,sum(q[2] for q in requests))
                self.assertEqual(status,"optimal")
                self.assertEqual(bound,optimum)
                self.assertEqual(audit(plan,requests,paths,state,conv)["admitted_gbps"],optimum)
                greedy = online_plan(network,requests,paths,state,"first")
                self.assertLessEqual(audit(greedy,requests,paths,state,conv)["admitted_gbps"],optimum)
                if requests==repeats:
                    self.assertEqual(optimum,200 if "A" in conv else 100)

    def test_proof_preserves_mixed_rate_objective_grid(self):
        from step35_wavelength_conversion import exact
        state = {("P", "Q"): {1}}
        network = OpticalNetwork(self.model, {("P", "Q"): 80})
        requests = [("P", "Q", 100), ("P", "Q", 150)]
        paths = catalog(network, requests, self.power, 2, set())
        plan, bound, _ = exact(requests, paths, state, 250, [[]])
        self.assertEqual(bound, 150)
        self.assertEqual([a["request_index"] for a in plan], [1])


if __name__=="__main__":
    unittest.main()
