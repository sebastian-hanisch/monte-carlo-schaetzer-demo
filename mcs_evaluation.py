"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Zielgröße der Demo ist die Ausfallwahrscheinlichkeit q = 1 - R: das Netz fällt auseinander, wenn zu viele Kanten ausfallen. Die Trefferzahl
der Schätzung ist die Zahl der Ziehungen, in denen es auseinanderfällt. Alle Intervalle gelten für q.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import mcs_algorithm as A
import mcs_constants as C
from mcs_counts import STORED_COUNTS
from mcs_scenario import Netz, named_netz


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def de_e(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 4,2e-10."""
    return f"{x:.1e}".replace(".", ",")


def de_g(x: float) -> str:
    return f"{x:g}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    gitter: str = C.DEFAULT_GRID
    p: float = C.DEFAULT_P
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    netz: Netz
    counts: tuple[int, ...]
    R: float                      # exakte Zuverlässigkeit (Wahrheit)
    q: float                      # exakte Ausfallwahrscheinlichkeit 1 - R
    fails: int                    # Ziehungen, in denen das Netz auseinanderfällt
    q_hat: float
    intervals: dict               # Verfahren -> (unten, oben), für q
    covers: dict                  # Verfahren -> Wahrheit im Intervall?
    n_needed: float | None        # Faustformel für 10 % relativen Fehler; None bei q = 0
    p_no_hit: float               # Wahrscheinlichkeit, dass bei N Ziehungen kein Ausfall vorkommt


# --- Wahrheit -----------------------------------------------------------------------------------------------------------------------------

def exact_counts(netz: Netz, gitter: str | None = None) -> tuple[int, ...]:
    """N_k des Netzes: gespeichert für die großen Gitter, sonst live aufgezählt (höchstens C.LIVE_EDGE_LIMIT Kanten)."""
    if gitter in STORED_COUNTS:
        return STORED_COUNTS[gitter]
    if netz.m > C.LIVE_EDGE_LIMIT:
        raise ValueError(f"{netz.m} Kanten: live aufgezählt wird nur bis {C.LIVE_EDGE_LIMIT}")
    return tuple(int(x) for x in A.connected_counts(netz.n, netz.edges))


# --- Eine Schätzung -----------------------------------------------------------------------------------------------------------------------

def analyse(settings: Settings, netz: Netz | None = None, counts=None) -> Analysis:
    """Zieht settings.n Netze aus dem Strom settings.seed, zählt die Ausfälle und bildet die drei Intervalle für q."""
    gitter = settings.gitter if netz is None else None
    netz = netz if netz is not None else named_netz(settings.gitter)
    counts = tuple(counts) if counts is not None else exact_counts(netz, gitter)
    R = A.reliability_from_counts(counts, settings.p)
    q = 1.0 - R
    draws = A.draw_connected(A.stream_seed(settings.seed), settings.n, settings.p, netz.n, netz.edges)
    fails = int(settings.n - draws.sum())
    intervals = {name: tuple(float(x) for x in f(fails, settings.n)) for name, f in A.METHODS.items()}
    covers = {name: lo <= q <= hi for name, (lo, hi) in intervals.items()}
    n_needed = A.n_required(q) if q > 0 else None
    return Analysis(netz, counts, R, q, fails, fails / settings.n, intervals, covers, n_needed, A.p_no_hit(q, settings.n))


# --- Experiment 1: Fehler gegen N ---------------------------------------------------------------------------------------------------------

def fehler_gegen_n(counts, target_r: float) -> dict:
    """RMSE von q-Dach gegen die Formel sqrt(q (1-q) / N) bei der Zuverlässigkeit `target_r`, über SWEEP_REPS feste Ströme je N.

    Die Ausfallzahl wird direkt aus Binomial(N, q) gezogen (gleichwertig zur Graphziehung, siehe A.binomial_hits)."""
    p = A.p_for_reliability(counts, target_r)
    q = 1.0 - A.reliability_from_counts(counts, p)
    rows = []
    for N in C.SWEEP_N:
        fails = A.binomial_hits(A.stream_seed(2, int(round(target_r * 1000)), N), C.SWEEP_REPS, N, q)
        est = fails / N
        rmse = float(np.sqrt(np.mean((est - q) ** 2)))
        theo = float(np.sqrt(q * (1 - q) / N))
        rows.append({"N": N, "mean": float(est.mean()), "rmse": rmse, "formel": theo, "ratio": rmse / theo})
    slope = float(np.polyfit(np.log([r["N"] for r in rows]), np.log([r["rmse"] for r in rows]), 1)[0])
    return {"R": 1.0 - q, "q": q, "p": p, "rows": rows, "slope": slope}


def verdict_fehler(exp: dict) -> tuple[str, str]:
    ratios = [r["ratio"] for r in exp["rows"]]
    lo, hi = min(ratios), max(ratios)
    ok = 0.85 <= lo and hi <= 1.15 and abs(exp["slope"] + 0.5) < 0.06
    text = (f"Bei R = {de(exp['R'], 2)} liegt der gemessene RMSE zwischen {de(lo, 2)}- und {de(hi, 2)}-fach der Formel √(q(1−q)/N); "
            f"die Steigung in der doppelt-logarithmischen Auftragung ist {de(exp['slope'], 2)} (Soll −0,50). "
            f"Das Rauschen der Verhältnisse bei {C.SWEEP_REPS} Wiederholungen beträgt rund {de(100 / np.sqrt(2 * C.SWEEP_REPS), 1)} %.")
    return ("success" if ok else "warning"), text


# --- Experiment 2: Dimension ------------------------------------------------------------------------------------------------------------

def dimension_experiment() -> dict:
    """Streuung des Schätzers bei R = DIM_TARGET auf Gittern verschiedener Kantenzahl (Graphziehungen, feste Ströme)."""
    rows = []
    theo = float(np.sqrt(0.25 / C.DIM_N))
    for key in C.DIM_GRIDS:
        netz = named_netz(key)
        counts = exact_counts(netz, key)
        p = A.p_for_reliability(counts, C.DIM_TARGET)
        est = np.array([
            1.0 - A.draw_connected(A.stream_seed(3, netz.m, rep), C.DIM_N, p, netz.n, netz.edges).mean() for rep in range(C.DIM_REPS)
        ])
        sd = float(est.std(ddof=1))
        rows.append({"gitter": key, "m": netz.m, "p": p, "sd": sd, "formel": theo, "ratio": sd / theo, "aufzaehlung": 2**netz.m})
    return {"rows": rows}


def breakeven_edges(n_draws: int) -> int:
    """Größte Kantenzahl m, bei der die Aufzählung (2^m Prüfungen) noch weniger Netzprüfungen braucht als `n_draws` Ziehungen."""
    m = 0
    while 2 ** (m + 1) < n_draws:
        m += 1
    return m


def verdict_dimension(exp: dict) -> tuple[str, str]:
    ratios = [r["ratio"] for r in exp["rows"]]
    ms = [r["m"] for r in exp["rows"]]
    noise = 100 / np.sqrt(2 * C.DIM_REPS)
    ok = 0.85 <= min(ratios) and max(ratios) <= 1.15
    text = (f"Bei gleichem R = {de(C.DIM_TARGET, 1)} streut der Schätzer auf Netzen mit {min(ms)} bis {max(ms)} Kanten gleich: "
            f"Verhältnis gemessen/Formel {de(min(ratios), 2)} bis {de(max(ratios), 2)} (Rauschen rund {de(noise, 0)} %). "
            f"Die naive Aufzählung braucht dagegen 2^m Prüfungen; gegen 10 000 Ziehungen ist sie (und liefert den exakten Wert) bis m = {breakeven_edges(10_000)} Kanten billiger, danach braucht sie mehr Prüfungen; Frontier-Verfahren verschieben diese Grenze weit nach oben.")
    return ("success" if ok else "warning"), text


# --- Experiment 3: Überdeckung ------------------------------------------------------------------------------------------------------------

def coverage_map(method_name: str) -> np.ndarray:
    """Überdeckung (Zeilen COVER_Q, Spalten COVER_N) des Verfahrens, exakt aus der Binomialverteilung."""
    f = A.METHODS[method_name]
    return np.array([[A.coverage(f, q, N) for N in C.COVER_N] for q in C.COVER_Q])


def no_hit_map() -> np.ndarray:
    return np.array([[A.p_no_hit(q, N) for N in C.COVER_N] for q in C.COVER_Q])


def verdict_coverage(method_name: str, grid: np.ndarray) -> tuple[str, str]:
    lo, hi = float(grid.min()), float(grid.max())
    if method_name == "Clopper-Pearson":
        ok = lo >= 0.95 - 1e-9
        text = f"Clopper-Pearson hält in allen {grid.size} Zellen mindestens 95 % (niedrigster Wert {de(lo, 3)}), ist aber oft deutlich konservativ (höchster Wert {de(hi, 3)})."
        return ("success" if ok else "error"), text
    kind = "warning" if lo < 0.93 else "info"
    if method_name == "Wilson":
        tail = "Das Verfahren garantiert nichts: Der niedrigste Wert steht bei den seltensten Ereignissen; die Überdeckung zackt mit N."
    else:
        tail = "Der niedrigste Wert steht bei den seltensten Ereignissen und den wenigsten Ziehungen."
    return kind, f"{method_name}: Überdeckung zwischen {de(lo, 3)} und {de(hi, 3)} (Soll 0,95). {tail}"


def required_draws_table() -> list[dict]:
    """Faustformel N* für 10 % relativen Fehler und, wo N* klein genug ist, die exakte Wahrscheinlichkeit, damit das Ziel zu treffen."""
    from scipy import stats

    rows = []
    for q in (0.1, 0.03, 0.01, 0.001, 1e-4, 1e-5, 1e-6):
        nstar = A.n_required(q)
        prob = None
        if nstar < 5e6:
            n_int = int(round(nstar))
            k = np.arange(n_int + 1)
            prob = float(np.sum(stats.binom.pmf(k, n_int, q) * (np.abs(k / n_int - q) <= C.REL_ERR * q)))
        rows.append({"q": q, "n_needed": nstar, "prob": prob})
    return rows


# --- Experiment 4: Schiefe --------------------------------------------------------------------------------------------------------------------

def skew_experiment() -> dict:
    cov = {s: [A.lognormal_coverage(s, N, C.SKEW_REPS, 6) for N in C.SKEW_N] for s in C.SKEW_SIGMAS}
    skew = {s: float((np.exp(s**2) + 2) * np.sqrt(np.exp(s**2) - 1)) for s in C.SKEW_SIGMAS}
    return {"cov": cov, "skew": skew}


def verdict_skew(exp: dict) -> tuple[str, str]:
    worst_sigma = max(C.SKEW_SIGMAS)
    last = exp["cov"][worst_sigma][-1][0]
    first = exp["cov"][worst_sigma][0][0]
    text = (f"Bei σ = {de_g(worst_sigma)} (Schiefe {de(exp['skew'][worst_sigma], 0)}) deckt das t-Intervall bei N = {C.SKEW_N[0]} nur {de(first, 2)} "
            f"und selbst bei N = {de(C.SKEW_N[-1], 0)} erst {de(last, 2)} (Soll 0,95); der Median des relativen Fehlers ist negativ: meist wird der Mittelwert unterschätzt, im Erwartungswert ist der Schätzer aber unverzerrt.")
    return ("warning" if last < 0.94 else "success"), text


# --- Experiment 5: Quadratur --------------------------------------------------------------------------------------------------------------

def quadrature_experiment() -> dict:
    rows = []
    for d in C.QUAD_DIMS:
        n = max(1, int(C.QUAD_BUDGET ** (1 / d) + 1e-9))
        quad = abs(A.midpoint(d, n) / A.quad_truth(d) - 1.0)
        formel = A.mc_rel_sd(d, C.QUAD_BUDGET)
        gemessen = A.mc_rel_rmse(d, C.QUAD_BUDGET, C.QUAD_REPS, 5)
        rows.append({"d": d, "n": n, "evals": n**d, "quad": quad, "mc_formel": formel, "mc_gemessen": gemessen, "sieger": "Quadratur" if quad < formel else "Monte Carlo"})
    return {"rows": rows}


def crossing_dimension(rows: list[dict]) -> int | None:
    """Kleinste Dimension, ab der Monte Carlo gewinnt und danach nie mehr die Quadratur."""
    for i, r in enumerate(rows):
        if all(x["sieger"] == "Monte Carlo" for x in rows[i:]):
            return r["d"]
    return None


def verdict_quadrature(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    d_star = crossing_dimension(rows)
    first, last = rows[0], rows[-1]
    text = (f"Bei d = {first['d']} ist die Quadratur um viele Größenordnungen genauer ({de_e(first['quad'])} gegen {de_e(first['mc_formel'])}); "
            f"ab d = {d_star} gewinnt Monte Carlo (bei etwa {de(C.QUAD_BUDGET, 0)} Auswertungen). Die Rate 1/√N hängt nicht von d ab, die Konstante schon: "
            f"der relative Fehler wächst von {de_e(first['mc_formel'])} bei d = {first['d']} auf {de_e(last['mc_formel'])} bei d = {last['d']}.")
    return "info", text


# --- Kernfrage ----------------------------------------------------------------------------------------------------------------------------

def verdict_sample(a: Analysis, n: int) -> tuple[str, str]:
    """Reichen die n Ziehungen für 10 % relativen Fehler an der Ausfallwahrscheinlichkeit, und trägt das Intervall überhaupt?"""
    if a.q == 0:
        return "info", "Das Netz fällt bei diesen Einstellungen nie auseinander (q = 0): es gibt nichts zu schätzen."
    if a.q == 1:
        return "info", f"Das Netz fällt bei diesen Einstellungen immer auseinander (q = 1, zum Beispiel weil es nicht zusammenhängt): alle {de(n, 0)} Ziehungen fallen aus, es gibt nichts zu schätzen."
    if a.fails == 0:
        lo, hi = a.intervals["Wald"]
        cp_hi = a.intervals["Clopper-Pearson"][1]
        return "warning", (f"Kein Ausfall in {de(n, 0)} Ziehungen. Das Wald-Intervall ist [{de(lo, 4)}; {de(hi, 4)}], also ein Punkt, obwohl die Wahrheit q = {de(a.q, 4)} ist; "
                           f"die Wahrscheinlichkeit für „kein Ausfall“ war {de(a.p_no_hit, 2)}. Clopper-Pearson sagt immerhin „höchstens {de(cp_hi, 4)}“. "
                           f"Für 10 % relativen Fehler wären rund {de(a.n_needed, 0)} Ziehungen nötig.")
    reicht = n >= a.n_needed
    text = (f"Für 10 % relativen Fehler an q = {de(a.q, 4)} braucht man bei 95 % Sicherheit rund {de(a.n_needed, 0)} Ziehungen "
            f"(Faustformel 1,96²(1−q)/(q·0,1²)); gezogen sind {de(n, 0)}. ")
    text += "Das reicht." if reicht else f"Das sind {de(a.n_needed / n, 1)}-mal zu wenige."
    return ("success" if reicht else "info"), text
