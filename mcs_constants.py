"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

GRIDS = {"2x3": (2, 3), "3x3": (3, 3), "2x5": (2, 5), "3x4": (3, 4), "3x5": (3, 5), "4x4": (4, 4)}
DEFAULT_GRID = "3x4"
LIVE_EDGE_LIMIT = 17          # Aufzählung live bis hierher: 2^17 = 131 072 Kantenmengen, rund 0,3 s; darüber gespeicherte Zahlen
CUSTOM_EDGE_LIMIT = 17        # eigenes Netz in der Spielwiese: höchstens so viele Kanten
CUSTOM_NODE_LIMIT = 14

P_MIN, P_MAX, DEFAULT_P = 0.50, 0.99, 0.80
N_CHOICES = (30, 100, 300, 1000, 3000, 10000, 30000, 100000)   # Ziehungen; Graphziehung O(N * m * Durchläufe) am Maximum unter 1 s
N_MIN, N_MAX, DEFAULT_N = N_CHOICES[0], N_CHOICES[-1], 1000
SEED_MAX, DEFAULT_SEED = 999999, 35
DRAW_VIEW_MAX = 12            # so viele einzelne Ziehungen zeigt die Ansicht "Ziehungen ansehen" höchstens

Z95 = 1.959963984540054       # Normalquantil für 95 %
REL_ERR = 0.10                # Zielgenauigkeit der Faustformel für die benötigte Ziehungszahl: 10 % relativer Fehler

# Experimente: feste Ströme, nie ein einzelner Seed als Befund
SWEEP_N = (100, 300, 1000, 3000, 10000)
SWEEP_REPS = 400
SWEEP_TARGETS = (0.5, 0.95)                    # Zuverlässigkeit R, bei der Experiment 1 misst
DIM_GRIDS = ("2x3", "3x3", "2x5", "3x4")       # Experiment 2: gleiche Zuverlässigkeit R = 0,5 auf Netzen verschiedener Größe
DIM_N, DIM_REPS, DIM_TARGET = 2000, 300, 0.5
COVER_Q = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
COVER_N = (30, 100, 300, 1000, 10000)
SKEW_SIGMAS = (0.5, 1.0, 2.0)
SKEW_N = (30, 100, 1000, 10000)
SKEW_REPS = 2000
QUAD_DIMS = (1, 2, 3, 4, 5, 6, 8, 10, 20)
QUAD_BUDGET, QUAD_REPS = 10_000, 200

COLOR_OK = "#2F6B65"
COLOR_BAD = "#C0392B"
COLOR_TRUTH = "#222222"
COLOR_LINE = "#4c78a8"
COLOR_OTHER = "rgba(150,150,150,0.55)"
METHOD_COLORS = {"Wald": "#E08E0B", "Wilson": "#4c78a8", "Clopper-Pearson": "#2F6B65"}

PRESETS = {
    "Standardfall (Voreinstellung)": {"gitter": DEFAULT_GRID, "p": DEFAULT_P, "n": DEFAULT_N, "seed": DEFAULT_SEED},
    "Seltenes Ereignis": {"gitter": DEFAULT_GRID, "p": 0.99, "n": 1000, "seed": DEFAULT_SEED},
    "Kleines Netz, wenige Ziehungen": {"gitter": "2x3", "p": 0.70, "n": 100, "seed": DEFAULT_SEED},
    "Großes Netz (4×4)": {"gitter": "4x4", "p": 0.90, "n": 10000, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Gitter 3×4 (12 Knoten, 17 Kanten), jede Kante hält mit 80 %, 1000 Ziehungen: der gutartige Fall, in dem das Intervall hält.",
    "Seltenes Ereignis": "Dasselbe Gitter, Kanten halten mit 99 %: Ausfälle sind selten, das Wald-Intervall wird unbrauchbar (ein Punkt bei null Ausfällen).",
    "Kleines Netz, wenige Ziehungen": "Gitter 2×3 (7 Kanten), 100 Ziehungen: wenig Ziehungen, ein breites Intervall.",
    "Großes Netz (4×4)": "Gitter 4×4 (24 Kanten): die exakte Zuverlässigkeit steht aus gespeicherten Zahlen bereit, geschätzt wird mit 10 000 Ziehungen.",
}
