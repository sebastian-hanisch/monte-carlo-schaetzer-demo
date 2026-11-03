"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mcs_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_dim", "streut der Schätzer auf Netzen mit 7 bis 17 Kanten gleich"),
    ("run_skew", "deckt das t-Intervall bei N = 30 nur"),
    ("run_quad", "ab d = 6 gewinnt Monte Carlo"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_quad").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="n_select").set_value(300).run(timeout=TIMEOUT)
    assert not at.exception
    assert "ab d = 6 gewinnt Monte Carlo" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    assert "Das sind 1,1-mal zu wenige" in _texte(at)
    at.sidebar.button(key="preset_Seltenes Ereignis").click().run()
    assert "Kein Ausfall in 1.000 Ziehungen" in _texte(at)
    at.sidebar.button(key="preset_Großes Netz (4×4)").click().run()
    assert "Das reicht." in _texte(at)


def test_spielwiese_gueltiges_netz_und_fehlermeldungen():
    at = _app()
    assert [e.value for e in at.error] == []
    at.text_area(key="eigene_kanten").set_value("0-1\n1-2\n2-0").run(timeout=TIMEOUT)
    assert not at.exception and not at.error
    at.text_area(key="eigene_kanten").set_value("0-0").run(timeout=TIMEOUT)
    assert not at.exception and "Schleife" in " ".join(e.value for e in at.error)
    at.text_area(key="eigene_kanten").set_value("").run(timeout=TIMEOUT)
    assert not at.exception and "Keine Kanten" in " ".join(e.value for e in at.error)


def test_spielwiese_nicht_zusammenhaengendes_netz_laeuft_durch():
    at = _app()
    at.text_area(key="eigene_kanten").set_value("0-1\n2-3").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "immer auseinander" in _texte(at)


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="n_select").set_value(300).run(timeout=TIMEOUT)     # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_ziehungsansicht_bis_zur_zwoelften_ziehung_und_bei_kleinem_n():
    at = _app()
    at.slider(key="ziehung_nr").set_value(C.DRAW_VIEW_MAX).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    at.sidebar.select_slider(key="n_select").set_value(30).run(timeout=TIMEOUT)
    assert not at.exception and at.slider(key="ziehung_nr").max == C.DRAW_VIEW_MAX


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["p"] = "5"
    at.query_params["n"] = "5000"
    at.query_params["g"] = "9x9"            # unbekanntes Gitter: verworfen
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="p_slider").value == C.P_MAX
    assert at.sidebar.select_slider(key="n_select").value == 3000
    assert at.sidebar.selectbox(key="gitter_select").value == C.DEFAULT_GRID
    assert at.sidebar.number_input(key="seed_input").value == 0
