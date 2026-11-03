"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für ALLE Eingaben gelten müssen, an festen Zufallsständen geprüft."""

import random

import numpy as np
import pytest

import mcs_algorithm as A
import mcs_constants as C
import mcs_evaluation as ev
from mcs_scenario import grid_netz, named_netz, parse_edges


def test_zuverlaessigkeit_waechst_mit_p_und_hat_die_randwerte():
    netz = grid_netz(2, 3)
    counts = A.connected_counts(netz.n, netz.edges)
    werte = [A.reliability_from_counts(counts, p) for p in np.linspace(0, 1, 21)]
    assert werte[0] == 0.0 and werte[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(werte, werte[1:]))


def test_zaehlung_ist_unabhaengig_von_knoten_und_kantenreihenfolge():
    rng = random.Random(5)
    netz = grid_netz(2, 3)
    basis = list(A.connected_counts(netz.n, netz.edges))
    for _ in range(10):
        perm = list(range(netz.n))
        rng.shuffle(perm)
        edges = [(perm[u], perm[v]) for u, v in netz.edges]
        rng.shuffle(edges)
        assert list(A.connected_counts(netz.n, edges)) == basis


def test_gekoppelte_ziehungen_mehr_p_heisst_weniger_oder_gleich_viele_ausfaelle():
    netz = named_netz("3x4")
    ausfaelle = [int(500 - A.draw_connected(77, 500, p, netz.n, netz.edges).sum()) for p in (0.5, 0.6, 0.7, 0.8, 0.9, 0.99)]
    assert all(b <= a for a, b in zip(ausfaelle, ausfaelle[1:]))


def test_gleicher_strom_gleiche_ziehung_und_laengere_folge_beginnt_gleich():
    netz = named_netz("3x3")
    a = A.draw_connected(5, 200, 0.7, netz.n, netz.edges)
    b = A.draw_connected(5, 200, 0.7, netz.n, netz.edges)
    c = A.draw_connected(5, 50, 0.7, netz.n, netz.edges)
    assert np.array_equal(a, b) and np.array_equal(a[:50], c)


def test_schaetzer_ist_erwartungstreu_innerhalb_von_vier_standardfehlern():
    reps, N, q = 4000, 300, 0.07
    est = A.binomial_hits(3, reps, N, q) / N
    se = np.sqrt(q * (1 - q) / N / reps)
    assert abs(est.mean() - q) < 4 * se


def test_clopper_pearson_haelt_in_jeder_zelle_95_prozent():
    for q in C.COVER_Q:
        for N in C.COVER_N:
            assert A.coverage(A.clopper_pearson, q, N) >= 0.95 - 1e-9, (q, N)


def test_intervalle_liegen_zwischen_null_und_eins_und_wald_ist_symmetrisch():
    for N in (1, 5, 30, 1000):
        k = np.arange(N + 1)
        for f in (A.wilson, A.clopper_pearson):
            lo, hi = f(k, N)
            assert (lo >= -1e-12).all() and (hi <= 1 + 1e-12).all() and (lo <= hi).all()
        lo, hi = A.wald(k, N)
        assert (lo <= hi).all()
        assert np.allclose((lo + hi) / 2, k / N)            # Wald ist symmetrisch um die Schätzung


def test_clopper_pearson_enthaelt_immer_die_schaetzung():
    for N in (1, 7, 40):
        k = np.arange(N + 1)
        lo, hi = A.clopper_pearson(k, N)
        assert (lo <= k / N + 1e-12).all() and (hi >= k / N - 1e-12).all()


def test_ueberdeckung_ist_eine_wahrscheinlichkeit():
    for f in A.METHODS.values():
        for q, N in ((0.5, 10), (0.01, 200), (0.3, 1)):
            assert 0.0 <= A.coverage(f, q, N) <= 1.0 + 1e-12


def test_seltener_heisst_mehr_erforderliche_ziehungen():
    qs = [0.3, 0.1, 0.03, 0.01, 0.001]
    n_star = [A.n_required(q) for q in qs]
    assert all(b > a for a, b in zip(n_star, n_star[1:]))


def test_parse_edges_ist_unabhaengig_von_reihenfolge_und_trennzeichen():
    a = parse_edges("0-1\n1-2\n2-0")
    b = parse_edges("2 0\n1,2\n0;1")
    assert sorted(a.edges) == sorted(b.edges) and a.n == b.n == 3


def test_analyse_ist_deterministisch_und_die_intervalle_enthalten_die_zaehlung():
    s = ev.Settings(n=300)
    a, b = ev.analyse(s), ev.analyse(s)
    assert a == b
    assert a.fails == round(a.q_hat * s.n)
    lo, hi = a.intervals["Clopper-Pearson"]
    assert lo <= a.q_hat <= hi
