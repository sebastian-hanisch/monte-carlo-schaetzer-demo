"""Orakel-Tests (ORACLE-PLAYBOOK §5): Kernroutinen gegen einen UNABHÄNGIGEN Rechenweg.

  Zusammenhang          gegen networkx.is_connected auf zufälligen Kantenmengen
  Aufzählung N_k        gegen Matrix-Baum-Satz (Determinante der Laplace-Matrix), gegen Bruchrechnung mit Union-Find (fractions) und für das
                        4x4-Gitter gegen die Zahlen der tutte-polynom-demo (anderer Weg: Frontier/Polynom statt Aufzählung)
  Intervalle            Clopper-Pearson und Wilson gegen scipy.stats.binomtest, Überdeckung gegen eine große Simulation
  Schiefe               Erwartungswert von exp(sigma Z) gegen numerische Integration
Mini-Instanzen, feste Stände (range statt Zufall), insgesamt wenige Sekunden."""

import itertools
import random
from fractions import Fraction

import numpy as np
import pytest

import mcs_algorithm as A
import mcs_constants as C
from mcs_counts import STORED_COUNTS
from mcs_scenario import grid_netz, named_netz


def _spanning_trees(n, edges):
    lap = np.zeros((n, n))
    for u, v in edges:
        lap[u, u] += 1
        lap[v, v] += 1
        lap[u, v] -= 1
        lap[v, u] -= 1
    return round(np.linalg.det(lap[1:, 1:]))


def _random_graph(rng, n, m):
    alle = list(itertools.combinations(range(n), 2))
    return n, tuple(rng.sample(alle, m))


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _connected_unionfind(n, edges, mask):
    parent = list(range(n))
    for e, (u, v) in enumerate(edges):
        if mask >> e & 1:
            parent[_find(parent, u)] = _find(parent, v)
    return len({_find(parent, x) for x in range(n)}) == 1


def _reliability_fractions(n, edges, p: Fraction) -> Fraction:
    m = len(edges)
    total = Fraction(0)
    for mask in range(1 << m):
        if _connected_unionfind(n, edges, mask):
            k = bin(mask).count("1")
            total += p**k * (1 - p) ** (m - k)
    return total


def test_zusammenhang_gegen_networkx_auf_zufaelligen_kantenmengen():
    nx = pytest.importorskip("networkx")
    rng = random.Random(2026)
    for _ in range(40):
        n = rng.randint(2, 7)
        n, edges = _random_graph(rng, n, rng.randint(1, min(len(list(itertools.combinations(range(n), 2))), 12)))
        stats = np.array([[rng.random() < 0.6 for _ in edges] for _ in range(30)])
        ours = A.connected(stats, n, edges)
        for row, got in zip(stats, ours):
            g = nx.Graph()
            g.add_nodes_from(range(n))
            g.add_edges_from(e for e, keep in zip(edges, row) if keep)
            assert nx.is_connected(g) == bool(got)


def test_spannbaumzahl_gegen_matrix_baum_satz_auf_zufallsgraphen():
    rng = random.Random(7)
    for _ in range(25):
        n = rng.randint(3, 6)
        n, edges = _random_graph(rng, n, rng.randint(n - 1, min(10, n * (n - 1) // 2)))
        counts = A.connected_counts(n, edges)
        assert counts[n - 1] == _spanning_trees(n, edges)
        assert counts[: n - 1].sum() == 0                      # unter n-1 Kanten kann nichts zusammenhängen


def test_zuverlaessigkeit_gegen_bruchrechnung_mit_union_find():
    for netz in (grid_netz(2, 3), grid_netz(2, 2)):
        counts = A.connected_counts(netz.n, netz.edges)
        for p in (Fraction(1, 3), Fraction(2, 5), Fraction(9, 10)):
            assert A.reliability_from_counts(counts, float(p)) == pytest.approx(float(_reliability_fractions(netz.n, netz.edges, p)), rel=1e-12)


def test_gespeicherte_zahlen_gegen_matrix_baum_satz_und_kantenregeln():
    for key, counts in STORED_COUNTS.items():
        netz = named_netz(key)
        assert len(counts) == netz.m + 1
        assert counts[netz.n - 1] == _spanning_trees(netz.n, netz.edges)
        assert counts[netz.m] == 1 and counts[netz.m - 1] == netz.m     # Gitter haben keine Brücke: jede einzelne Kante darf fehlen
        assert sum(counts) <= 2**netz.m


def test_viermalvier_gitter_gegen_die_zahlen_der_tutte_demo():
    counts = STORED_COUNTS["4x4"]
    for p, tutte in ((0.5, 0.033092), (0.9, 0.944085), (0.99, 0.999584)):
        assert A.reliability_from_counts(counts, p) == pytest.approx(tutte, abs=5e-7)


def test_live_aufzaehlung_gegen_gespeicherte_zahlen_am_kleinen_gitter():
    """Das 3x5-Gitter ist zu groß für die Live-Aufzählung; geprüft wird die Rechenroutine deshalb an 3x3 gegen den Matrix-Baum-Satz (192)."""
    netz = named_netz("3x3")
    assert A.connected_counts(netz.n, netz.edges)[netz.n - 1] == 192 == _spanning_trees(netz.n, netz.edges)


@pytest.mark.parametrize("k,N", [(0, 20), (1, 20), (7, 20), (19, 20), (20, 20), (0, 500), (3, 500), (250, 500)])
def test_intervalle_gegen_scipy_binomtest(k, N):
    from scipy.stats import binomtest

    ci = binomtest(k, N).proportion_ci(confidence_level=0.95, method="exact")
    assert A.clopper_pearson(k, N) == pytest.approx((ci.low, ci.high), abs=1e-9)
    wi = binomtest(k, N).proportion_ci(confidence_level=0.95, method="wilson")
    assert A.wilson(k, N) == pytest.approx((wi.low, wi.high), abs=1e-9)


@pytest.mark.parametrize("method_name,q,N", [("Wald", 0.1, 30), ("Wilson", 0.03, 100), ("Clopper-Pearson", 0.01, 300)])
def test_exakte_ueberdeckung_gegen_grosse_simulation(method_name, q, N):
    f = A.METHODS[method_name]
    rng = np.random.default_rng(9)
    k = rng.binomial(N, q, size=300_000)
    lo, hi = f(k, N)
    simuliert = float(np.mean((lo <= q) & (q <= hi)))
    assert A.coverage(f, q, N) == pytest.approx(simuliert, abs=0.004)       # 3 Standardfehler bei 300 000: 0,001


def test_lognormal_erwartungswert_gegen_numerische_integration():
    from scipy import integrate

    for sigma in (0.5, 1.0, 2.0):
        dichte = lambda z: np.exp(sigma * z) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)  # noqa: E731
        wert, _ = integrate.quad(dichte, -12, 14)
        assert wert == pytest.approx(np.exp(sigma**2 / 2), rel=1e-6)


def test_mc_rel_sd_gegen_numerische_integration():
    from scipy import integrate

    for d in (1, 2):
        mean = (np.e - 1) ** d
        if d == 1:
            second, _ = integrate.quad(lambda x: np.exp(2 * x), 0, 1)
        else:
            second, _ = integrate.dblquad(lambda y, x: np.exp(2 * (x + y)), 0, 1, 0, 1)
        assert A.mc_rel_sd(d, 1) == pytest.approx(np.sqrt(second / mean**2 - 1), rel=1e-6)
