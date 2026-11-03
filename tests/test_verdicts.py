"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt gesetzte Werte, keine Seeds. Damit prüft der Test die Grenzen und jede genannte Zahl direkt."""

import numpy as np

import mcs_constants as C
import mcs_evaluation as ev


def _analysis(fails, n, q, n_needed=None, p_no_hit=0.5):
    netz = ev.named_netz("2x3")
    iv = {"Wald": (0.0, 0.0), "Wilson": (0.0, 0.1), "Clopper-Pearson": (0.0, 0.12)}
    return ev.Analysis(netz, (), 1 - q, q, fails, fails / n, iv, {k: False for k in iv}, n_needed, p_no_hit)


def test_zahlenformat_ist_deutsch():
    assert ev.de(1234567.891, 2) == "1.234.567,89"
    assert ev.de(0.05, 3) == "0,050"
    assert ev.de(-0.5, 1) == "−0,5"
    assert ev.de_e(4.17e-10) == "4,2e-10" and ev.de_g(0.5) == "0,5"


def test_sample_ohne_ausfall_ist_eine_warnung_mit_allen_zahlen():
    kind, text = ev.verdict_sample(_analysis(0, 1000, 0.0004, n_needed=960_000, p_no_hit=0.67), 1000)
    assert kind == "warning"
    for teil in ("Kein Ausfall in 1.000 Ziehungen", "[0,0000; 0,0000]", "q = 0,0004", "0,67", "„höchstens 0,1200“", "960.000"):
        assert teil in text, teil


def test_sample_reicht_genau_bei_gleichheit():
    kind, text = ev.verdict_sample(_analysis(25, 1000, 0.025, n_needed=1000.0), 1000)
    assert kind == "success" and "Das reicht." in text and "q = 0,0250" in text and "1.000 Ziehungen" in text


def test_sample_zu_wenig_nennt_den_faktor():
    kind, text = ev.verdict_sample(_analysis(25, 1000, 0.025, n_needed=2500.0), 1000)
    assert kind == "info" and "2,5-mal zu wenige" in text and "2.500" in text


def test_fehler_verdikt_schwellen():
    rows = lambda ratios: [{"ratio": r} for r in ratios]  # noqa: E731
    ok = {"R": 0.5, "slope": -0.5, "rows": rows([0.9, 1.0, 1.1])}
    assert ev.verdict_fehler(ok)[0] == "success"
    assert ev.verdict_fehler({**ok, "rows": rows([0.84, 1.0])})[0] == "warning"        # Untergrenze 0,85
    assert ev.verdict_fehler({**ok, "rows": rows([1.0, 1.16])})[0] == "warning"        # Obergrenze 1,15
    assert ev.verdict_fehler({**ok, "slope": -0.43})[0] == "warning"                   # Steigung weicht um 0,07 ab
    kind, text = ev.verdict_fehler(ok)
    assert "R = 0,50" in text and "0,90- und 1,10-fach" in text and "Soll −0,50" in text and "3,5 %" in text


def test_dimension_verdikt_nennt_bereich_und_breakeven():
    rows = [{"ratio": 0.96, "m": 7}, {"ratio": 1.04, "m": 17}]
    kind, text = ev.verdict_dimension({"rows": rows})
    assert kind == "success" and "7 bis 17 Kanten" in text and "0,96 bis 1,04" in text and "bis m = 13 Kanten" in text
    assert ev.verdict_dimension({"rows": [{"ratio": 0.8, "m": 7}]})[0] == "warning"


def test_breakeven_kanten_gegen_handwerte():
    assert ev.breakeven_edges(10_000) == 13           # 2^13 = 8192 < 10^4 <= 2^14 = 16384
    assert ev.breakeven_edges(8193) == 13 and ev.breakeven_edges(8192) == 12
    assert ev.breakeven_edges(2) == 0


def test_coverage_verdikt_clopper_pearson_und_andere():
    gut = np.full((2, 2), 0.96)
    assert ev.verdict_coverage("Clopper-Pearson", gut)[0] == "success"
    kind, text = ev.verdict_coverage("Clopper-Pearson", np.array([[0.949, 1.0]]))
    assert kind == "error" and "0,949" in text
    kind, text = ev.verdict_coverage("Wald", np.array([[0.03, 0.953]]))
    assert kind == "warning" and "0,030 und 0,953" in text
    assert ev.verdict_coverage("Wald", np.array([[0.94, 0.95]]))[0] == "info"           # Untergrenze 0,93
    assert "garantiert nichts" in ev.verdict_coverage("Wilson", np.array([[0.9, 0.97]]))[1]


def test_skew_verdikt_nennt_die_zellen_und_schwelle():
    cov = {s: [(0.9, 0.0)] * len(C.SKEW_N) for s in C.SKEW_SIGMAS}
    cov[2.0] = [(0.66, -0.28), (0.76, -0.13), (0.85, -0.04), (0.91, -0.01)]
    skew = {s: 2.0 for s in C.SKEW_SIGMAS}
    skew[2.0] = 414.4
    kind, text = ev.verdict_skew({"cov": cov, "skew": skew})
    assert kind == "warning" and "σ = 2" in text and "Schiefe 414" in text and "N = 30 nur 0,66" in text and "N = 10.000 erst 0,91" in text
    cov[2.0][-1] = (0.945, 0.0)
    assert ev.verdict_skew({"cov": cov, "skew": skew})[0] == "success"


def test_quadratur_verdikt_nennt_kreuzung_und_wachstum():
    rows = [
        {"d": 1, "quad": 4.2e-10, "mc_formel": 2.9e-3, "sieger": "Quadratur"},
        {"d": 5, "quad": 5.8e-3, "mc_formel": 7.0e-3, "sieger": "Quadratur"},
        {"d": 6, "quad": 1.6e-2, "mc_formel": 7.8e-3, "sieger": "Monte Carlo"},
        {"d": 20, "quad": 0.56, "mc_formel": 2.0e-2, "sieger": "Monte Carlo"},
    ]
    kind, text = ev.verdict_quadrature({"rows": rows})
    assert kind == "info" and "4,2e-10 gegen 2,9e-03" in text and "ab d = 6 gewinnt Monte Carlo" in text and "2,0e-02 bei d = 20" in text
    assert ev.crossing_dimension(rows) == 6
    rows[1]["sieger"] = "Monte Carlo"
    rows[2]["sieger"] = "Quadratur"                      # Monte Carlo gewinnt später nie dauerhaft vor d = 20
    assert ev.crossing_dimension(rows) == 20
