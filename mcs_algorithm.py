"""Kern der Demo (numpy; scipy nur lazy in den Funktionen, die es brauchen).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom        stream_seed, uniforms                      SplitMix64, zählerbasiert, ein fester Strom je (Einstellung, Wiederholung)
  Zusammenhang        connected                                  kleinste-Marke-Ausbreitung, vektorisiert über viele Ziehungen
  exakte Wahrheit     connected_counts, reliability_from_counts  Aufzählung aller 2^m Kantenmengen
  Ziehung             draw_connected, binomial_hits              Graphziehung; Trefferzahl direkt aus Binomial(N, R)
  Intervalle          wald, wilson, clopper_pearson, coverage    Überdeckung exakt aus der Binomialverteilung
  Faustformeln        n_required, p_no_hit
  Schiefe             lognormal_coverage
  Quadratur           midpoint, mc_rel_sd, mc_rel_rmse
Zufall kommt nur über einen übergebenen Strom (seed), nie aus einem globalen Generator.
"""

from __future__ import annotations

import numpy as np

import mcs_constants as C

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1


# --- Zufallsstrom -------------------------------------------------------------------------------------------------------------------------

def mix_int(z: int) -> int:
    """SplitMix64-Mischfunktion mit Python-Ganzzahlen."""
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys: int) -> int:
    """Seed eines Stroms aus einem Schlüsseltupel (Einstellung, Wiederholung, ...)."""
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed: int, count: int) -> np.ndarray:
    """`count` gleichverteilte Zahlen in [0, 1) (53 Bit) aus dem Strom `seed`; Ausgabe i hängt nur von (seed, i) ab."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


# --- Zusammenhang und exakte Wahrheit -----------------------------------------------------------------------------------------------------

def connected(present: np.ndarray, n: int, edges) -> np.ndarray:
    """Zusammenhang je Zeile: `present` ist (B, m) bool, Rückgabe (B,) bool.

    Kleinste-Marke-Ausbreitung: jeder Knoten trägt eine Marke, vorhandene Kanten gleichen die Marken an, bis nichts mehr wechselt.
    Zusammenhängend genau dann, wenn alle Marken 0 sind (Knoten 0 hat Marke 0)."""
    B = present.shape[0]
    if n <= 1:
        return np.ones(B, dtype=bool)
    lab = np.tile(np.arange(n, dtype=np.int16), (B, 1))
    while True:
        before = int(lab.sum())
        for e, (u, v) in enumerate(edges):
            low = np.minimum(lab[:, u], lab[:, v])
            pe = present[:, e]
            lab[:, u] = np.where(pe, low, lab[:, u])
            lab[:, v] = np.where(pe, low, lab[:, v])
        if int(lab.sum()) == before:
            break
    return lab.max(axis=1) == 0


def connected_counts(n: int, edges, chunk: int = 1 << 20) -> np.ndarray:
    """N_k = Zahl der zusammenhängenden Kantenmengen mit genau k Kanten, durch Aufzählung aller 2^m Kantenmengen (k = 0 bis m)."""
    m = len(edges)
    counts = np.zeros(m + 1, dtype=np.int64)
    shifts = np.arange(m, dtype=np.uint32)
    for start in range(0, 1 << m, chunk):
        masks = np.arange(start, min(start + chunk, 1 << m), dtype=np.uint32)
        present = ((masks[:, None] >> shifts) & 1).astype(bool)
        ok = connected(present, n, edges)
        counts += np.bincount(present.sum(axis=1)[ok], minlength=m + 1)
    return counts


def reliability_from_counts(counts, p: float) -> float:
    """R(p) = Summe_k N_k p^k (1-p)^(m-k): Wahrscheinlichkeit, dass das Netz zusammenhängt, wenn jede Kante mit p hält."""
    counts = np.asarray(counts, dtype=np.float64)
    m = len(counts) - 1
    k = np.arange(m + 1)
    return float(np.sum(counts * p**k * (1.0 - p) ** (m - k)))


def p_for_reliability(counts, target: float) -> float:
    """p mit R(p) = target (Bisektion; R wächst mit p)."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if reliability_from_counts(counts, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- Ziehungen ----------------------------------------------------------------------------------------------------------------------------

def edge_states(seed: int, count: int, p: float, m: int) -> np.ndarray:
    """(count, m) bool: Kante e der Ziehung i hält genau dann, wenn ihre Zufallszahl unter p liegt.

    Die Zahlen hängen nicht von p ab: bei festem Strom sind die Ziehungen für verschiedene p gekoppelt (gemeinsame Zufallszahlen)."""
    return uniforms(seed, count * m).reshape(count, m) < p


def draw_connected(seed: int, count: int, p: float, n: int, edges) -> np.ndarray:
    """`count` unabhängige Ziehungen des Netzes; Rückgabe 0/1 je Ziehung (1 = zusammenhängend)."""
    return connected(edge_states(seed, count, p, len(edges)), n, edges).astype(np.int8)


def binomial_hits(seed: int, reps: int, N: int, R: float) -> np.ndarray:
    """Trefferzahlen von `reps` Wiederholungen mit je N Ziehungen, direkt aus Binomial(N, R) per Umkehrfunktion einer Gleichverteilten.

    Die Zahl der zusammenhängenden Ziehungen unter N unabhängigen ist exakt binomialverteilt; die Graphziehung ist dazu gleichwertig
    (tests/test_claims.py vergleicht beide)."""
    from scipy import stats

    return stats.binom.ppf(uniforms(seed, reps), N, R).astype(np.int64)


# --- Intervalle ---------------------------------------------------------------------------------------------------------------------------

def wald(k, N):
    """Wald-Intervall p-Dach ± z sqrt(p-Dach (1-p-Dach) / N); bei k = 0 oder k = N entartet es zu einem Punkt."""
    k = np.asarray(k, dtype=np.float64)
    ph = k / N
    h = C.Z95 * np.sqrt(ph * (1 - ph) / N)
    return ph - h, ph + h


def wilson(k, N):
    """Wilson-Intervall (Score-Intervall)."""
    k = np.asarray(k, dtype=np.float64)
    ph = k / N
    den = 1 + C.Z95**2 / N
    centre = (ph + C.Z95**2 / (2 * N)) / den
    h = C.Z95 * np.sqrt(ph * (1 - ph) / N + C.Z95**2 / (4 * N**2)) / den
    return centre - h, centre + h


def clopper_pearson(k, N):
    """Exaktes Intervall nach Clopper und Pearson (Beta-Quantile); hält die Überdeckung von 95 % für jedes N und jedes q."""
    from scipy import stats

    k = np.asarray(k)
    lo = np.where(k == 0, 0.0, stats.beta.ppf(0.025, np.maximum(k, 1), N - k + 1))
    hi = np.where(k == N, 1.0, stats.beta.ppf(0.975, k + 1, np.maximum(N - k, 1)))
    return lo, hi


METHODS = {"Wald": wald, "Wilson": wilson, "Clopper-Pearson": clopper_pearson}


def coverage(method, q: float, N: int) -> float:
    """Überdeckung des Verfahrens EXAKT: Summe über k von Bin(k; N, q) mal [q liegt im Intervall zu k]. Kein Zufall."""
    from scipy import stats

    k = np.arange(N + 1)
    lo, hi = method(k, N)
    inside = (lo <= q) & (q <= hi)
    return float(np.sum(stats.binom.pmf(k, N, q) * inside))


def p_no_hit(q: float, N: int) -> float:
    """Wahrscheinlichkeit, dass unter N Ziehungen kein Ereignis der Wahrscheinlichkeit q eintritt: (1-q)^N."""
    return float((1.0 - q) ** N)


def n_required(q: float, rel: float = C.REL_ERR) -> float:
    """Faustformel: Ziehungen für relativen Fehler `rel` bei 95 % Sicherheit, N* = z^2 (1-q) / (q rel^2) (Normalnäherung der Trefferzahl)."""
    return C.Z95**2 * (1.0 - q) / (q * rel**2)


# --- Schiefe Größen -----------------------------------------------------------------------------------------------------------------------

def lognormal_coverage(sigma: float, N: int, reps: int, key: int) -> tuple[float, float]:
    """Überdeckung des t-Intervalls für den Mittelwert von X = exp(sigma Z), Z ~ N(0,1), und Median des relativen Fehlers.

    Wahrheit E(X) = exp(sigma^2 / 2). `reps` feste Ströme; das Normalquantil kommt aus der Umkehrfunktion einer Gleichverteilten."""
    from scipy import stats
    from scipy.special import ndtri

    truth = np.exp(sigma**2 / 2)
    tq = stats.t.ppf(0.975, N - 1)
    hit = 0
    errs = np.empty(reps)
    for rep in range(reps):
        u = uniforms(stream_seed(key, int(round(sigma * 100)), N, rep), N)
        x = np.exp(sigma * ndtri(np.clip(u, 1e-300, 1 - 1e-16)))
        mean, sd = x.mean(), x.std(ddof=1)
        half = tq * sd / np.sqrt(N)
        hit += bool(mean - half <= truth <= mean + half)
        errs[rep] = mean / truth - 1
    return hit / reps, float(np.median(errs))


# --- Quadratur gegen Monte Carlo ----------------------------------------------------------------------------------------------------------

def midpoint(d: int, n: int) -> float:
    """Mittelpunktsregel mit n Punkten je Achse für f(x) = exp(x_1 + ... + x_d) auf [0,1]^d (Produktregel, n^d Auswertungen)."""
    return float(np.mean(np.exp((np.arange(n) + 0.5) / n)) ** d)


def quad_truth(d: int) -> float:
    return float((np.e - 1.0) ** d)


def mc_rel_sd(d: int, N: int) -> float:
    """Relative Streuung des Monte-Carlo-Mittels von f aus N Ziehungen, geschlossen: sqrt((Var f / (E f)^2) / N)."""
    var_ratio = ((np.e**2 - 1.0) / (2.0 * (np.e - 1.0) ** 2)) ** d - 1.0
    return float(np.sqrt(var_ratio / N))


def mc_rel_rmse(d: int, N: int, reps: int, key: int) -> float:
    """Gemessener relativer RMSE des Monte-Carlo-Mittels über `reps` feste Ströme."""
    truth = quad_truth(d)
    errs = np.empty(reps)
    for rep in range(reps):
        x = uniforms(stream_seed(key, d, rep), N * d).reshape(N, d)
        errs[rep] = np.exp(x.sum(axis=1)).mean() / truth - 1.0
    return float(np.sqrt(np.mean(errs**2)))
