"""Gespeicherte Zahlen N_k (zusammenhängende Kantenmengen mit k Kanten) für die beiden Gitter, deren Aufzählung live zu lange dauert.

Herkunft: vollständige Aufzählung aller 2^m Kantenmengen (22 Kanten: 18,5 s, 24 Kanten: 69 s auf dem Entwicklungsrechner); die Zahl der Spannbäume
(N_(n-1)) stimmt mit dem Matrix-Baum-Satz überein, das 4x4-Gitter mit den Zahlen der tutte-polynom-demo (tests/test_oracle_mcs.py).
Gitter bis 17 Kanten werden live aufgezählt, nicht gespeichert."""

STORED_COUNTS = {
    "3x5": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30305, 48404, 37626, 18426, 6174, 1440, 227, 22, 1),
    "4x4": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100352, 175264, 151160, 83956, 32888, 9358, 1920, 272, 24, 1),
}
