"""Jede Zahl aus der README und den App-Texten, nachgerechnet über dieselben Auswertungsfunktionen wie die App (DEMO-PLAYBOOK §4).

Ströme sind fest (SplitMix64), die Ganzzahlen (Ausfallzahlen) deshalb plattformunabhängig; Gleitkommazahlen mit Abstand zur Rundung, Verhältnisse
verrauschter Messungen mit Bändern (Rauschen der Verhältnisse bei 400 Wiederholungen: rund 3,5 %). Zeiten stehen nie in einem Test."""

import numpy as np
import pytest

import mcs_algorithm as A
import mcs_constants as C
import mcs_evaluation as ev
from mcs_counts import STORED_COUNTS
from mcs_scenario import named_netz

STANDARD = C.PRESETS["Standardfall (Voreinstellung)"]


def _a(preset):
    return ev.analyse(ev.Settings(**C.PRESETS[preset]))


# --- Die vier Presets ----------------------------------------------------------------------------------------------------------------------

def test_preset_standardfall():
    a = _a("Standardfall (Voreinstellung)")
    assert a.netz.m == 17 and a.netz.n == 12
    assert a.R == pytest.approx(0.746409, abs=5e-7) and a.q == pytest.approx(0.253591, abs=5e-7)
    assert a.fails == 258 and a.q_hat == 0.258
    assert a.intervals["Wald"] == pytest.approx((0.23088, 0.28512), abs=2e-5)
    assert a.intervals["Wilson"] == pytest.approx((0.23184, 0.28601), abs=2e-5)
    assert a.intervals["Clopper-Pearson"] == pytest.approx((0.23113, 0.28631), abs=2e-5)
    assert all(a.covers.values())
    assert a.n_needed == pytest.approx(1131, abs=1)


def test_preset_seltenes_ereignis():
    a = _a("Seltenes Ereignis")
    assert a.R == pytest.approx(0.999583, abs=5e-7) and a.q == pytest.approx(0.000417, abs=5e-7)
    assert a.fails == 0
    assert a.intervals["Wald"] == (0.0, 0.0) and a.covers == {"Wald": False, "Wilson": True, "Clopper-Pearson": True}
    assert a.intervals["Wilson"][1] == pytest.approx(0.00383, abs=2e-5)
    assert a.intervals["Clopper-Pearson"][1] == pytest.approx(0.00368, abs=2e-5)
    assert a.p_no_hit == pytest.approx(0.659, abs=5e-4)
    assert a.n_needed == pytest.approx(920_938, abs=5)
    kind, text = ev.verdict_sample(a, 1000)
    assert kind == "warning" and "Kein Ausfall in 1.000 Ziehungen" in text and "0,66" in text and "920.938" in text


def test_preset_kleines_netz():
    a = _a("Kleines Netz, wenige Ziehungen")
    assert a.netz.m == 7 and a.R == pytest.approx(0.556312, abs=5e-7)
    assert a.fails == 45 and a.n_needed == pytest.approx(482, abs=1)


def test_preset_grosses_netz():
    a = _a("Großes Netz (4×4)")
    assert a.netz.m == 24 and a.R == pytest.approx(0.944085, abs=5e-7)
    assert a.fails == 561 and a.n_needed == pytest.approx(6486, abs=1)
    assert ev.verdict_sample(a, 10_000)[0] == "success"


def test_gitter_kantenzahlen_und_spannbaumzahlen():
    erwartet = {"2x3": (7, 15), "3x3": (12, 192), "2x5": (13, 209), "3x4": (17, 2415), "3x5": (22, 30305), "4x4": (24, 100352)}
    for key, (m, trees) in erwartet.items():
        netz = named_netz(key)
        assert netz.m == m
        assert ev.exact_counts(netz, key)[netz.n - 1] == trees, key
    assert set(STORED_COUNTS) == {"3x5", "4x4"}


# --- Experiment 1: Fehler gegen N -----------------------------------------------------------------------------------------------------------

def test_experiment_1_verhaeltnisse_und_steigung():
    counts = ev.exact_counts(named_netz("3x4"), "3x4")
    e50, e95 = ev.fehler_gegen_n(counts, 0.5), ev.fehler_gegen_n(counts, 0.95)
    assert e50["q"] == pytest.approx(0.5, abs=1e-9) and e95["q"] == pytest.approx(0.05, abs=1e-9)
    r50 = [r["ratio"] for r in e50["rows"]]
    r95 = [r["ratio"] for r in e95["rows"]]
    assert 0.92 <= min(r50) and max(r50) <= 1.10        # gemessen 0,99 bis 1,07
    assert 0.90 <= min(r95) and max(r95) <= 1.10        # gemessen 0,93 bis 1,06
    assert -0.55 <= e50["slope"] <= -0.45 and -0.57 <= e95["slope"] <= -0.43
    assert ev.verdict_fehler(e50)[0] == "success" and ev.verdict_fehler(e95)[0] == "success"


def test_rauschen_der_verhaeltnisse():
    assert 100 / np.sqrt(2 * C.SWEEP_REPS) == pytest.approx(3.54, abs=0.01)
    assert 100 / np.sqrt(2 * C.DIM_REPS) == pytest.approx(4.08, abs=0.01)


def test_binomialziehung_und_graphziehung_haben_dieselben_momente():
    netz = named_netz("3x4")
    counts = ev.exact_counts(netz, "3x4")
    p = 0.9
    q = 1 - A.reliability_from_counts(counts, p)
    graph = np.array([1000 - A.draw_connected(A.stream_seed(8, r), 1000, p, netz.n, netz.edges).sum() for r in range(1000)])
    binom = A.binomial_hits(A.stream_seed(8, 99), 20000, 1000, q)
    soll_sd = np.sqrt(1000 * q * (1 - q))
    assert abs(graph.mean() - 1000 * q) < 4 * soll_sd / np.sqrt(1000)
    assert abs(binom.mean() - 1000 * q) < 4 * soll_sd / np.sqrt(20000)
    assert abs(graph.std() - soll_sd) < 0.09 * soll_sd and abs(binom.std() - soll_sd) < 0.04 * soll_sd


# --- Experiment 2: Dimension ---------------------------------------------------------------------------------------------------------------

def test_experiment_2_streuung_haengt_nicht_an_der_kantenzahl():
    exp = ev.dimension_experiment()
    assert [(r["gitter"], r["m"]) for r in exp["rows"]] == [("2x3", 7), ("3x3", 12), ("2x5", 13), ("3x4", 17)]
    ratios = [r["ratio"] for r in exp["rows"]]
    assert 0.92 <= min(ratios) and max(ratios) <= 1.08                       # gemessen 0,96 bis 1,04
    assert ev.verdict_dimension(exp)[0] == "success"
    assert [r["aufzaehlung"] for r in exp["rows"]] == [128, 4096, 8192, 131072]
    assert ev.breakeven_edges(10_000) == 13


# --- Experiment 3: Überdeckung -----------------------------------------------------------------------------------------------------------

def _cell(grid, q, N):
    return grid[C.COVER_Q.index(q), C.COVER_N.index(N)]


def test_experiment_3_wald_ueberdeckung():
    g = ev.coverage_map("Wald")
    assert g.min() == pytest.approx(0.030, abs=5e-4) and g.max() == pytest.approx(0.953, abs=5e-4)
    assert _cell(g, 0.001, 30) == g.min()
    assert _cell(g, 0.01, 1000) == pytest.approx(0.927, abs=5e-4)
    assert _cell(g, 0.003, 10000) == pytest.approx(0.931, abs=5e-4)
    assert _cell(g, 0.01, 10000) == pytest.approx(0.946, abs=5e-4)


def test_experiment_3_wilson_und_clopper_pearson():
    w = ev.coverage_map("Wilson")
    assert w.min() == pytest.approx(0.905, abs=5e-4) and w.max() == pytest.approx(0.974, abs=5e-4)
    c = ev.coverage_map("Clopper-Pearson")
    assert c.min() == pytest.approx(0.951, abs=5e-4) and c.max() == pytest.approx(1.000, abs=5e-4)
    assert c.size == 30 and ev.verdict_coverage("Clopper-Pearson", c)[0] == "success"


def test_experiment_3_kein_treffer_und_benoetigte_ziehungen():
    assert A.p_no_hit(0.001, 100) == pytest.approx(0.905, abs=5e-4)
    assert A.p_no_hit(0.01, 100) == pytest.approx(0.366, abs=5e-4)
    assert A.p_no_hit(0.003, 300) == pytest.approx(0.406, abs=5e-4)
    rows = {r["q"]: r for r in ev.required_draws_table()}
    assert rows[0.1]["n_needed"] == pytest.approx(3457, abs=1) and rows[0.01]["n_needed"] == pytest.approx(38030, abs=1)
    assert rows[0.001]["n_needed"] == pytest.approx(383_762, abs=1) and rows[1e-6]["n_needed"] == pytest.approx(384_145_498, abs=1)
    assert all(0.945 <= r["prob"] <= 0.955 for r in rows.values() if r["prob"] is not None)
    assert rows[1e-5]["prob"] is None and rows[1e-4]["prob"] is not None


# --- Experiment 4 und 5 ---------------------------------------------------------------------------------------------------------------------

def test_experiment_4_schiefe():
    exp = ev.skew_experiment()
    assert exp["skew"][0.5] == pytest.approx(1.8, abs=0.05) and exp["skew"][1.0] == pytest.approx(6.2, abs=0.05) and exp["skew"][2.0] == pytest.approx(414.4, abs=0.05)
    zwei = [c for c, _ in exp["cov"][2.0]]
    assert zwei == pytest.approx([0.663, 0.757, 0.854, 0.912], abs=0.012)       # Rauschen bei 2000 Wiederholungen: 0,005
    assert [m for _, m in exp["cov"][2.0]] == pytest.approx([-0.279, -0.131, -0.039, -0.007], abs=0.01)
    assert all(m < 0 for _, m in exp["cov"][2.0])
    assert [c for c, _ in exp["cov"][0.5]] == pytest.approx([0.935, 0.951, 0.956, 0.948], abs=0.012)
    assert exp["cov"][1.0][-1][0] == pytest.approx(0.952, abs=0.012)
    assert ev.verdict_skew(exp)[0] == "warning"


def test_experiment_5_quadratur_gegen_monte_carlo():
    exp = ev.quadrature_experiment()
    rows = {r["d"]: r for r in exp["rows"]}
    assert rows[1]["quad"] == pytest.approx(4.17e-10, rel=0.02) and rows[1]["mc_formel"] == pytest.approx(2.86e-3, rel=0.01)
    assert rows[5]["n"] == 6 and rows[5]["evals"] == 7776 and rows[5]["quad"] == pytest.approx(5.77e-3, rel=0.01) and rows[5]["mc_formel"] == pytest.approx(6.95e-3, rel=0.01)
    assert rows[6]["n"] == 4 and rows[6]["evals"] == 4096 and rows[6]["quad"] == pytest.approx(1.55e-2, rel=0.01) and rows[6]["mc_formel"] == pytest.approx(7.77e-3, rel=0.01)
    assert rows[10]["n"] == 2 and rows[10]["evals"] == 1024 and rows[10]["quad"] == pytest.approx(9.87e-2, rel=0.01)
    assert rows[20]["n"] == 1 and rows[20]["quad"] == pytest.approx(0.562, rel=0.01) and rows[20]["mc_formel"] == pytest.approx(1.96e-2, rel=0.01)
    assert ev.crossing_dimension(exp["rows"]) == 6
    assert [rows[d]["sieger"] for d in (5, 6)] == ["Quadratur", "Monte Carlo"]
    for r in exp["rows"]:                                                         # gemessen gegen Formel: Rauschen 5 % bei 200 Wiederholungen
        assert r["mc_gemessen"] == pytest.approx(r["mc_formel"], rel=0.20)
    assert exp["rows"][-1]["mc_formel"] > 5 * exp["rows"][0]["mc_formel"]         # die Konstante wächst mit d: die Rate bleibt 1/sqrt(N)


def test_faustformel_hat_384_als_konstante():
    assert A.n_required(0.2) * 0.2 / 0.8 == pytest.approx(384.146, abs=0.01)         # N* (q / (1 - q)) = 1,96^2 / 0,1^2, unabhängig von q
