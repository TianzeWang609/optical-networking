"""Explicit synthetic topologies and reproducible static traffic batches."""

from itertools import combinations
import math

import numpy as np


SCENARIOS = ("uniform_fixed", "hotspot_fixed", "uniform_mixed", "hotspot_mixed")


def mesh12_case(available_wavelengths=8, physical_channel_count=80):
    """A 3-by-4 grid mesh: 12 nodes, 17 links, 320/480 km horizontal/vertical."""
    if (not isinstance(available_wavelengths, int) or isinstance(available_wavelengths, bool)
            or not 1 <= available_wavelengths <= physical_channel_count):
        raise ValueError("Available wavelengths must lie between 1 and the physical channel count.")
    nodes = tuple("ABCDEFGHIJKL")
    positions = {nodes[4*r+c]: (c, 2-r) for r in range(3) for c in range(4)}
    links = {}
    for r in range(3):
        for c in range(4):
            node = nodes[4*r+c]
            if c < 3:
                links[(node, nodes[4*r+c+1])] = 320.0
            if r < 2:
                links[(node, nodes[4*(r+1)+c])] = 480.0
    # 可分配资源是物理 80 信道中的一个子集；物理噪声仍按满载计算
    first = (physical_channel_count - available_wavelengths) // 2 + 1
    state = {e: set(range(first, first + available_wavelengths)) for e in links}
    return links, state, positions


def pair_probabilities(nodes, hotspot=None, hotspot_share=0.65):
    nodes = tuple(sorted(nodes))
    if len(nodes) < 3 or len(set(nodes)) != len(nodes):
        raise ValueError("Use at least three unique nodes.")
    pairs = list(combinations(nodes, 2))
    if hotspot is None:
        return pairs, np.full(len(pairs), 1 / len(pairs))
    if hotspot not in nodes or not math.isfinite(hotspot_share) or not 0 <= hotspot_share <= 1:
        raise ValueError("Invalid hotspot or hotspot share.")
    is_hot = np.array([hotspot in pair for pair in pairs])
    probabilities = np.where(is_hot, hotspot_share / is_hot.sum(),
                             (1 - hotspot_share) / (~is_hot).sum())
    return pairs, probabilities


def generate_requests(nodes, count, seed, scenario, hotspot="F", hotspot_share=0.65):
    """Generate unordered OD pairs; mixed rates have exactly the same total demand.

    Each batch has count*100 Gbit/s. Mixed batches contain 25% 50, 50% 100,
    and 25% 150 Gbit/s requests, randomly permuted independently of OD draws.
    Fixed/mixed counterparts use identical OD sequences for each seed/count.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or count % 4:
        raise ValueError("Request count must be a positive multiple of four.")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    pairs, probabilities = pair_probabilities(nodes, hotspot if scenario.startswith("hotspot") else None,
                                              hotspot_share)
    pair_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    indices = pair_rng.choice(len(pairs), size=count, p=probabilities)
    rates = np.full(count, 100.0)
    if scenario.endswith("mixed"):
        rates = np.tile([50.0, 100.0, 100.0, 150.0], count // 4)
        np.random.default_rng(np.random.SeedSequence([seed, 1])).shuffle(rates)
    return [(pairs[i][0], pairs[i][1], float(rate)) for i, rate in zip(indices, rates)]
