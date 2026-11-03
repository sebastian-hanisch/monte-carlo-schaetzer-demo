"""Randfälle: Grenzen der Regler, entartete Netze, Eingabefehler der Spielwiese."""

import pytest

import mcs_algorithm as A
import mcs_constants as C
import mcs_evaluation as ev
from mcs_scenario import Netz, named_netz, parse_edges


def test_kleinste_und_groesste_ziehungszahl_laufen_durch():
    for n in (C.N_MIN, C.N_MAX):
        a = ev.analyse(ev.Settings(n=n))
        assert 0 <= a.fails <= n


def test_hoechstes_p_und_niedrigstes_p_laufen_durch():
    for p in (C.P_MIN, C.P_MAX):
        a = ev.analyse(ev.Settings(p=p))
        assert 0.0 < a.q < 1.0 and 0 <= a.fails <= C.DEFAULT_N


def test_jedes_gitter_hat_eine_zuverlaessigkeit_zwischen_null_und_eins():
    for key in C.GRIDS:
        netz = named_netz(key)
        counts = ev.exact_counts(netz, key)
        assert 0.0 < A.reliability_from_counts(counts, 0.8) < 1.0


def test_nicht_zusammenhaengendes_netz_hat_q_gleich_eins_und_ein_ehrliches_verdikt():
    netz = Netz("zwei Paare", 4, ((0, 1), (2, 3)), tuple((float(i), 0.0) for i in range(4)))
    counts = ev.exact_counts(netz)
    assert sum(counts) == 0
    a = ev.analyse(ev.Settings(n=100), netz=netz, counts=counts)
    assert a.q == 1.0 and a.fails == 100 and a.n_needed == 0.0
    kind, text = ev.verdict_sample(a, 100)
    assert kind == "info" and "immer auseinander" in text


def test_ein_netz_ohne_ausfallmoeglichkeit_hat_q_gleich_null_und_n_needed_none():
    netz = Netz("Kante", 2, ((0, 1),), ((0.0, 0.0), (1.0, 0.0)))
    a = ev.analyse(ev.Settings(p=1.0, n=50), netz=netz)
    assert a.q == 0.0 and a.n_needed is None
    assert ev.verdict_sample(a, 50)[0] == "info"


def test_eine_einzige_kante_ist_ein_munzwurf():
    netz = Netz("Kante", 2, ((0, 1),), ((0.0, 0.0), (1.0, 0.0)))
    counts = ev.exact_counts(netz)
    assert list(counts) == [0, 1]
    assert A.reliability_from_counts(counts, 0.3) == pytest.approx(0.3)


def test_live_aufzaehlung_lehnt_zu_grosse_netze_ab():
    netz = named_netz("3x5")                                   # 22 Kanten: nur gespeichert
    with pytest.raises(ValueError):
        ev.exact_counts(netz)                                  # ohne Gitternamen keine gespeicherten Zahlen
    assert ev.exact_counts(netz, "3x5")[-1] == 1


@pytest.mark.parametrize("text,fragment", [
    ("", "Keine Kanten"),
    ("# nur Kommentar\n\n", "Keine Kanten"),
    ("0-0", "Schleife"),
    ("0-1\n1-0", "Doppelte Kante"),
    ("0-1\nabc", "Zeile 2"),
    ("0-1-2", "Zeile 1"),
    ("\n".join(f"{i}-{i + 1}" for i in range(C.CUSTOM_EDGE_LIMIT + 1)), "Kanten: höchstens"),
    ("\n".join(f"0-{i}" for i in range(1, C.CUSTOM_NODE_LIMIT + 1)), "Knoten: höchstens"),
])
def test_eingabefehler_der_spielwiese_haben_eine_klare_meldung(text, fragment):
    with pytest.raises(ValueError) as err:
        parse_edges(text)
    assert fragment in str(err.value)


def test_knoten_werden_von_null_an_umnummeriert():
    netz = parse_edges("10-20\n20-30")
    assert netz.n == 3 and netz.edges == ((0, 1), (1, 2))


def test_p_no_hit_bei_extremen_q():
    assert A.p_no_hit(0.0, 100) == 1.0 and A.p_no_hit(1.0, 100) == 0.0
