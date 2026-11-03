"""Kern gegen Handrechnung auf den Mini-Instanzen (conftest.py) und gegen geschlossene Formeln; je Einheit mindestens ein Test."""

import math

import numpy as np
import pytest

import mcs_algorithm as A
import mcs_constants as C


# --- Zufallsstrom --------------------------------------------------------------------------------------------------------------------------

def test_splitmix64_bekannter_startwert():
    assert A.mix_int(A.GAMMA) == 0xE220A8397B1DCDAF       # Seed 0, erste Ausgabe des Verfahrens


@pytest.mark.parametrize("seed", [0, 7, 2**63 + 12345])
def test_vektorisierter_strom_gleich_ganzzahl_referenz(seed):
    ref = [(A.mix_int((seed + i * A.GAMMA) & A.MASK) >> 11) * (1.0 / (1 << 53)) for i in range(1, 301)]
    assert np.array_equal(A.uniforms(seed, 300), np.array(ref))


def test_strom_ist_ein_praefix_und_im_halboffenen_intervall():
    lang, kurz = A.uniforms(99, 1000), A.uniforms(99, 10)
    assert np.array_equal(lang[:10], kurz)
    assert lang.min() >= 0.0 and lang.max() < 1.0


def test_stream_seed_ist_deterministisch_und_unterscheidet_schluessel():
    assert A.stream_seed(1, 2) == A.stream_seed(1, 2)
    assert len({A.stream_seed(1, 2), A.stream_seed(2, 1), A.stream_seed(1), A.stream_seed(1, 3)}) == 4


# --- Zusammenhang und Aufzählung --------------------------------------------------------------------------------------------------------

def test_connected_dreieck_von_hand(dreieck):
    zeilen = np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=bool)
    assert list(A.connected(zeilen, 3, dreieck.edges)) == [True, True, True, False, False]


def test_connected_pfad_zerfaellt_bei_jeder_fehlenden_kante(pfad):
    voll = np.ones((1, 4), dtype=bool)
    assert A.connected(voll, 5, pfad.edges)[0]
    for e in range(4):
        z = voll.copy()
        z[0, e] = False
        assert not A.connected(z, 5, pfad.edges)[0]


def test_connected_ein_knoten_ist_immer_zusammenhaengend():
    assert A.connected(np.zeros((3, 0), dtype=bool), 1, ()).all()


def test_zaehlung_dreieck_kreis_pfad_k4(dreieck, kreis4, pfad, k4):
    assert list(A.connected_counts(dreieck.n, dreieck.edges)) == [0, 0, 3, 1]
    assert list(A.connected_counts(kreis4.n, kreis4.edges)) == [0, 0, 0, 4, 1]
    assert list(A.connected_counts(pfad.n, pfad.edges)) == [0, 0, 0, 0, 1]
    assert list(A.connected_counts(k4.n, k4.edges)) == [0, 0, 0, 16, 15, 6, 1]


def test_zaehlung_in_kleinen_stuecken_gleich_ganzer_aufzaehlung(k4):
    assert list(A.connected_counts(k4.n, k4.edges, chunk=5)) == list(A.connected_counts(k4.n, k4.edges))


def test_zuverlaessigkeit_dreieck_bei_ein_drittel_ist_sieben_siebenundzwanzigstel(dreieck):
    counts = A.connected_counts(dreieck.n, dreieck.edges)
    assert A.reliability_from_counts(counts, 1 / 3) == pytest.approx(7 / 27, rel=1e-12)


def test_zuverlaessigkeit_pfad_ist_p_hoch_vier(pfad):
    counts = A.connected_counts(pfad.n, pfad.edges)
    for p in (0.0, 0.3, 0.9, 1.0):
        assert A.reliability_from_counts(counts, p) == pytest.approx(p**4, abs=1e-15)


def test_p_fuer_zuverlaessigkeit_kehrt_die_zuverlaessigkeit_um(kreis4):
    counts = A.connected_counts(kreis4.n, kreis4.edges)
    for target in (0.1, 0.5, 0.95):
        p = A.p_for_reliability(counts, target)
        assert A.reliability_from_counts(counts, p) == pytest.approx(target, abs=1e-12)


# --- Ziehungen ---------------------------------------------------------------------------------------------------------------------------

def test_edge_states_sind_praefix_und_haengen_nicht_von_p_ab():
    a = A.edge_states(5, 3, 0.7, 4)
    b = A.edge_states(5, 10, 0.7, 4)
    assert np.array_equal(a, b[:3])
    u = A.uniforms(5, 12).reshape(3, 4)
    assert np.array_equal(a, u < 0.7)
    assert np.array_equal(A.edge_states(5, 3, 0.2, 4), u < 0.2)        # gleiche Zufallszahlen, nur die Schwelle wechselt


def test_draw_connected_extreme_p(dreieck):
    assert A.draw_connected(3, 50, 1.0, dreieck.n, dreieck.edges).all()
    assert not A.draw_connected(3, 50, 0.0, dreieck.n, dreieck.edges).any()


def test_draw_connected_gleich_handschleife(dreieck):
    u = A.uniforms(11, 40 * 3).reshape(40, 3)
    erwartet = []
    for zeile in u:
        halten = [x < 0.6 for x in zeile]
        erwartet.append(1 if sum(halten) >= 2 else 0)       # Dreieck: mindestens 2 von 3 Kanten
    assert list(A.draw_connected(11, 40, 0.6, dreieck.n, dreieck.edges)) == erwartet


def test_binomial_hits_mittel_und_streuung():
    hits = A.binomial_hits(4, 4000, 1000, 0.25)
    assert abs(hits.mean() - 250) < 3 * np.sqrt(1000 * 0.25 * 0.75 / 4000) * 1.0 + 1
    assert abs(hits.std() - np.sqrt(1000 * 0.25 * 0.75)) < 0.06 * np.sqrt(1000 * 0.25 * 0.75)


def test_binomial_hits_entartete_wahrscheinlichkeiten():
    assert not A.binomial_hits(1, 20, 100, 0.0).any()
    assert (A.binomial_hits(1, 20, 100, 1.0) == 100).all()


# --- Intervalle --------------------------------------------------------------------------------------------------------------------------

def test_wald_von_hand():
    lo, hi = A.wald(50, 100)                                   # 0,5 +- 1,96 * 0,05
    assert (lo, hi) == pytest.approx((0.5 - 1.959963984540054 * 0.05, 0.5 + 1.959963984540054 * 0.05))
    assert tuple(A.wald(0, 100)) == (0.0, 0.0)               # entartet: ein Punkt


def test_wilson_ist_symmetrisch_bei_halb():
    lo, hi = A.wilson(5, 10)
    assert lo + hi == pytest.approx(1.0)
    lo0, hi0 = A.wilson(0, 30)
    assert lo0 == pytest.approx(0.0, abs=1e-12) and hi0 > 0


def test_clopper_pearson_rand_faelle_geschlossen():
    # k = 0: oben 1 - 0,025^(1/N); k = N: unten 0,025^(1/N)
    lo, hi = A.clopper_pearson(0, 100)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 100))
    lo, hi = A.clopper_pearson(100, 100)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 100))


def test_coverage_von_hand_bei_n_gleich_eins():
    # N = 1, q = 0,5: k = 0 oder 1 mit je 0,5. Wald ist dann [0,0] bzw. [1,1] und enthält q nie.
    assert A.coverage(A.wald, 0.5, 1) == 0.0


def test_p_no_hit_und_n_required():
    assert A.p_no_hit(0.5, 3) == 0.125
    assert A.n_required(0.5) == pytest.approx(1.959963984540054**2 * 0.5 / (0.5 * 0.01))
    assert A.n_required(0.5, rel=1.0) == pytest.approx(1.959963984540054**2 * 0.5 / 0.5)


# --- Schiefe und Quadratur --------------------------------------------------------------------------------------------------------------

def test_lognormal_coverage_bei_kleiner_schiefe_und_vielen_ziehungen_nahe_95_prozent():
    cov, _ = A.lognormal_coverage(0.3, 1000, 400, 1)
    assert 0.90 <= cov <= 0.99


def test_midpoint_von_hand():
    assert A.midpoint(1, 1) == pytest.approx(math.exp(0.5))
    assert A.midpoint(2, 1) == pytest.approx(math.exp(0.5) ** 2)
    assert A.midpoint(1, 1000) == pytest.approx(math.e - 1, rel=1e-6)
    assert A.quad_truth(3) == pytest.approx((math.e - 1) ** 3)


def test_mc_rel_sd_waechst_mit_d_und_faellt_mit_n():
    assert A.mc_rel_sd(2, 100) > A.mc_rel_sd(1, 100)
    assert A.mc_rel_sd(3, 400) == pytest.approx(A.mc_rel_sd(3, 100) / 2)


def test_worst_case_am_regler_maximum_laeuft_durch():
    """Hänger-Netz am Slider-Maximum (kein Zeittest, nur ein Netz): 4x4, 100 000 Ziehungen."""
    from mcs_scenario import named_netz

    netz = named_netz("4x4")
    d = A.draw_connected(1, C.N_MAX, 0.9, netz.n, netz.edges)
    assert d.shape == (C.N_MAX,) and set(np.unique(d)) <= {0, 1}
