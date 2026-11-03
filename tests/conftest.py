"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcs_scenario import Netz  # noqa: E402


def _netz(name, n, edges):
    return Netz(name, n, tuple(edges), tuple((float(i), 0.0) for i in range(n)))


@pytest.fixture
def dreieck():
    """Dreieck: 3 Knoten, 3 Kanten. Handrechnung: zusammenhängend sind die 3 Mengen mit 2 Kanten und die volle Menge,
    also N = [0, 0, 3, 1] und R(p) = 3 p^2 (1-p) + p^3; bei p = 1/3 ist R = 7/27."""
    return _netz("Dreieck", 3, [(0, 1), (1, 2), (0, 2)])


@pytest.fixture
def kreis4():
    """Kreis mit 4 Kanten: zusammenhängend sind die 4 Mengen mit 3 Kanten und die volle Menge, N = [0,0,0,4,1], R = 4 p^3 (1-p) + p^4."""
    return _netz("Kreis", 4, [(0, 1), (1, 2), (2, 3), (3, 0)])


@pytest.fixture
def pfad():
    """Pfad mit 4 Kanten und 5 Knoten: nur die volle Menge hält zusammen, R = p^4."""
    return _netz("Pfad", 5, [(0, 1), (1, 2), (2, 3), (3, 4)])


@pytest.fixture
def k4():
    """Vollständiger Graph K4: 16 Spannbäume (Cayley: 4^2), N = [0,0,0,16,15,6,1]."""
    return _netz("K4", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
