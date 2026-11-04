"""Der Monte-Carlo-Schätzer und sein Fehler - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 1 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": eine Wahrscheinlichkeit durch Ziehen schätzen, den Fehler ehrlich angeben und
sehen, wann das Intervall trügt. Wahrheit ist die Zuverlässigkeit eines Gitternetzes, exakt aufgezählt.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import mcs_algorithm as A
import mcs_constants as C
import mcs_evaluation as ev
from mcs_evaluation import Settings, de
from mcs_layout import dunkles_design, kopf_css
from mcs_pdf_export import build_pdf
from mcs_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mcs_scenario import named_netz, parse_edges
from mcs_visualization import (
    build_coverage,
    build_dimension,
    build_error_vs_n,
    build_intervals,
    build_netz,
    build_quadrature,
    build_skew,
)

st.set_page_config(page_title="Der Monte-Carlo-Schätzer und sein Fehler – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Alle Kantenmengen aufzählen …")
def _counts(gitter):
    return ev.exact_counts(named_netz(gitter), gitter)


@st.cache_data(show_spinner=False)
def _analysis(settings, counts):
    return ev.analyse(settings, counts=counts)


@st.cache_data(show_spinner=False)
def _custom(text, settings):
    netz = parse_edges(text)
    counts = ev.exact_counts(netz)
    return netz, ev.analyse(settings, netz=netz, counts=counts)


@st.cache_data(show_spinner=False)
def _fehler(counts, target):
    return ev.fehler_gegen_n(counts, target)


@st.cache_data(show_spinner="Graphziehungen auf vier Netzen …")
def _dimension():
    return ev.dimension_experiment()


@st.cache_data(show_spinner=False)
def _coverage(method):
    return ev.coverage_map(method)


@st.cache_data(show_spinner="Schiefe Größe, 2000 Wiederholungen je Zelle …")
def _skew():
    return ev.skew_experiment()


@st.cache_data(show_spinner="Quadratur und Monte Carlo bis d = 20 …")
def _quadrature():
    return ev.quadrature_experiment()


def _signed(x, nd):
    """Zahl mit Vorzeichen im deutschen Format (+ oder −), ohne Pfeil-Deutung wie bei st.metric-Deltas."""
    return ("+" if x >= 0 else "−") + de(abs(x), nd)


def _gitter_label(key):
    rows, cols = C.GRIDS[key]
    return f"{rows}×{cols} ({rows * (cols - 1) + cols * (rows - 1)} Kanten)"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mcs_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎲 Der Monte-Carlo-Schätzer und sein Fehler")
st.markdown(
    """
Ein **Monte-Carlo-Schätzer** schätzt eine Wahrscheinlichkeit, indem er **zieht und zählt**: Man würfelt das Netz aus, prüft, ob es noch zusammenhängt, und teilt die Zahl der Ausfälle durch die Zahl der Ziehungen.
Sein **Fehler fällt wie 1/√N**, ganz gleich, wie groß das Netz ist (bei gleicher Ausfallwahrscheinlichkeit), aber die Konstante hängt an der Streuung einer Ziehung, also an der **Seltenheit** des Ereignisses und an der **Schiefe** der Größe. Hier ist die Wahrheit bekannt (alle Kantenmengen aufgezählt), also lässt sich jede Schätzung und jedes **Konfidenzintervall** gegen sie messen.

Weiter unten: **🔬 fünf Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Monte-Carlo-Schätzer", expanded=True):
    st.markdown(
        """
1. **Ziehen:** Jede Kante des Netzes fällt unabhängig mit Wahrscheinlichkeit 1 − p aus. Eine Ziehung ist ein ausgewürfeltes Netz; sie zählt als **Ausfall**, wenn es nicht mehr zusammenhängt.
2. **Zählen:** Die **Schätzung** q̂ ist der Anteil der Ausfälle unter N Ziehungen. Sie ist erwartungstreu, ihre Streuung ist √(q(1−q)/N).
3. **Intervall:** Aus der Zahl der Ausfälle folgt ein 95-%-Intervall; drei Verfahren stehen nebeneinander (**Wald**, **Wilson**, **Clopper-Pearson**).
4. **Prüfen:** Die **Wahrheit** q kommt aus der Aufzählung aller 2^m Kantenmengen (m = Zahl der Kanten); nur deshalb lässt sich sehen, ob das Intervall sie enthält.
5. **Warum das funktioniert:** Der Mittelwert unabhängiger Ziehungen konvergiert gegen den Erwartungswert, und seine Streuung ist √(Varianz einer Ziehung / N): Die Rate 1/√N hängt nicht von der Größe des Problems ab, die Varianz einer Ziehung schon.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    gitter = st.selectbox("Gitter", options("gitter_select"), format_func=_gitter_label, key="gitter_select", help="Knoten in Zeilen und Spalten, Kanten waagerecht und senkrecht. Bis 17 Kanten wird die Wahrheit live aufgezählt, die beiden größeren sind gespeichert.")
    p = st.slider("Haltewahrscheinlichkeit p je Kante", *bounds("p_slider"), key="p_slider", step=0.01, help="Jede Kante hält unabhängig mit dieser Wahrscheinlichkeit; bei p nahe 1 sind Ausfälle selten.")
    st.markdown("**Ziehungen**")
    n = st.select_slider("Ziehungen N", options("n_select"), key="n_select", help="Wie oft das Netz ausgewürfelt wird. Bei festem Strom ist die kleinere Folge der Anfang der größeren.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64). Gleicher Strom, anderes p: die Ziehungen sind gekoppelt.")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"gitter_select": gitter, "p_slider": float(p), "n_select": int(n), "seed_input": int(seed)})

settings = Settings(str(gitter), float(p), int(n), int(seed))
counts = _counts(settings.gitter)
a = _analysis(settings, counts)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Wahrheit q", de(a.q, 5), help=f"Exakte Ausfallwahrscheinlichkeit des {a.netz.name}s bei p = {de(settings.p, 2)}, aus der Aufzählung aller Kantenmengen; Zuverlässigkeit R = {de(a.R, 5)}.")
m2.metric("Ausfälle", f"{a.fails}", help=f"So viele der {de(settings.n, 0)} gezogenen Netze fallen auseinander.")
m3.metric("Schätzung q̂", de(a.q_hat, 5), help="Anteil der Ausfälle unter den Ziehungen.")
m4.metric("Fehler q̂ − q", _signed(a.q_hat - a.q, 4), help="Schätzung minus Wahrheit; mit mehr Ziehungen wird er im Mittel kleiner, aber er kann positiv oder negativ sein.")
st.caption("Intervalle, die die Wahrheit enthalten: " + ", ".join(f"{name} {'✅' if ok else '❌'}" for name, ok in a.covers.items()) + ".")
st.plotly_chart(build_intervals(a), width="stretch", key="main_chart")
st.caption("Gestrichelt: die Wahrheit q. Weißer Punkt: die Schätzung q̂. Balken: 95-%-Intervall des jeweiligen Verfahrens.")

if st.button("📄 PDF der Schätzung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="monte-carlo-schaetzung.pdf", mime="application/pdf", key="pdf_download")

with st.expander("🔎 Einzelne Ziehungen ansehen"):
    k_max = min(settings.n, C.DRAW_VIEW_MAX)
    nr = st.slider("Ziehung Nr.", 1, k_max, 1, key="ziehung_nr", help=f"Die ersten {k_max} der {de(settings.n, 0)} Ziehungen; das ist die Folge, aus der die Schätzung oben gebildet wird.")
    states = A.edge_states(A.stream_seed(settings.seed), nr, settings.p, a.netz.m)[nr - 1]
    ok = bool(A.connected(states[None, :], a.netz.n, a.netz.edges)[0])
    st.plotly_chart(build_netz(a.netz, states, ok), width="stretch", key="netz_chart")
    st.caption(f"Ausgefallene Kanten: {int((~states).sum())} von {a.netz.m}. Eine Ziehung ist ein Münzwurf mit Ergebnis „zusammenhängend“ oder „nicht“; die Schätzung ist ihr Anteil.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viele Ziehungen braucht man, und darf man dem Intervall trauen?")
kind, text = ev.verdict_sample(a, settings.n)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Fällt der Fehler wie 1/√N?")
st.caption(
    f"Auf dem gewählten Netz ({a.netz.name}), bei Zuverlässigkeit R = 0,5 und R = 0,95, je {C.SWEEP_REPS} Wiederholungen mit festen Strömen und N = {', '.join(str(x) for x in C.SWEEP_N)}. "
    "Die Ausfallzahl wird dazu direkt aus der Binomialverteilung gezogen (gleichwertig zur Graphziehung, geprüft in tests/test_claims.py)."
)
c1, c2 = st.columns(2)
for col, target in zip((c1, c2), C.SWEEP_TARGETS):
    exp = _fehler(counts, target)
    with col:
        st.plotly_chart(build_error_vs_n(exp), width="stretch", key=f"fehler_chart_{target}")
        kind, text = ev.verdict_fehler(exp)
        getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Hängt die Streuung an der Größe des Netzes?")
st.caption(
    f"Vier Gitter mit {C.DIM_GRIDS[0]} bis {C.DIM_GRIDS[-1]} (7 bis 17 Kanten), jeweils so eingestellt, dass R = {de(C.DIM_TARGET, 1)}; "
    f"{C.DIM_REPS} Wiederholungen mit je {de(C.DIM_N, 0)} echten Graphziehungen. Kostet einige Sekunden."
)
if st.button("▶ Messung starten", key="run_dim"):
    st.session_state["dim_done"] = True
if st.session_state.get("dim_done"):
    dexp = _dimension()
    st.plotly_chart(build_dimension(dexp), width="stretch", key="dim_chart")
    kind, text = ev.verdict_dimension(dexp)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Wann trügt das Konfidenzintervall?")
st.caption(
    "Überdeckung eines 95-%-Intervalls für die Ausfallwahrscheinlichkeit q bei N Ziehungen. Die Zahl der Ausfälle ist binomialverteilt, deshalb lässt sich die Überdeckung "
    "**exakt** ausrechnen (Summe über alle möglichen Ausfallzahlen), ganz ohne Zufall."
)
method = st.radio("Verfahren", list(A.METHODS), horizontal=True, key="cover_method")
grid = _coverage(method)
st.plotly_chart(build_coverage(grid, method), width="stretch", key="cover_chart")
kind, text = ev.verdict_coverage(method, grid)
getattr(st, kind)(text)
rows = ev.required_draws_table()
st.markdown("**Ziehungen für 10 % relativen Fehler** (Faustformel 1,96²(1−q)/(q·0,1²); exakte Binomialwahrscheinlichkeit, das Ziel zu treffen, wo N klein genug ist)")
st.markdown(
    "| q | benötigte Ziehungen N* | Wahrscheinlichkeit, das Ziel zu treffen | Wahrscheinlichkeit „kein Ausfall“ bei N = 100 |\n|---|---|---|---|\n"
    + "\n".join(f"| {de_q} | {de(r['n_needed'], 0)} | {de(r['prob'], 3) if r['prob'] is not None else 'nicht berechnet'} | {de(A.p_no_hit(r['q'], 100), 3)} |"
               for r, de_q in zip(rows, (de(r["q"], 6).rstrip("0").rstrip(",") for r in rows)))
)

st.markdown("---")
st.subheader("🔬 Experiment 4: Was ändert eine schiefe Größe?")
st.caption(
    f"Schätzt wird der Mittelwert von X = exp(σ·Z) mit Z ~ N(0,1) durch das übliche t-Intervall; {C.SKEW_REPS} Wiederholungen je Zelle. Je größer σ, desto schiefer ist X. Kostet einige Sekunden."
)
if st.button("▶ Messung starten", key="run_skew"):
    st.session_state["skew_done"] = True
if st.session_state.get("skew_done"):
    sexp = _skew()
    st.plotly_chart(build_skew(sexp), width="stretch", key="skew_chart")
    kind, text = ev.verdict_skew(sexp)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Quadratur gegen Monte Carlo")
st.caption(
    f"Integrand exp(x₁+…+x_d) auf dem Einheitswürfel, Wahrheit (e−1)^d; Mittelpunktsregel mit ⌊{de(C.QUAD_BUDGET, 0)}^(1/d)⌋ Punkten je Achse gegen {de(C.QUAD_BUDGET, 0)} Ziehungen "
    f"({C.QUAD_REPS} Wiederholungen). Kostet einige Sekunden."
)
if st.button("▶ Messung starten", key="run_quad"):
    st.session_state["quad_done"] = True
if st.session_state.get("quad_done"):
    qexp = _quadrature()
    st.plotly_chart(build_quadrature(qexp), width="stretch", key="quad_chart")
    kind, text = ev.verdict_quadrature(qexp)
    getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein eigenes Netz"):
    st.caption(f"Eine Kante je Zeile, zwei ganze Zahlen (»0-1«). Höchstens {C.CUSTOM_EDGE_LIMIT} Kanten und {C.CUSTOM_NODE_LIMIT} Knoten; p, N und der Strom stammen aus der Seitenleiste.")
    text_in = st.text_area("Kanten", "0-1\n1-2\n2-3\n3-0\n0-2", height=140, key="eigene_kanten")
    try:
        cnetz, ca = _custom(text_in, settings)
    except ValueError as err:
        st.error(str(err))
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric("Wahrheit q", de(ca.q, 5))
        s2.metric("Ausfälle", f"{ca.fails}", help=f"von {de(settings.n, 0)} Ziehungen")
        s3.metric("Fehler q̂ − q", _signed(ca.q_hat - ca.q, 4))
        st.plotly_chart(build_netz(cnetz), width="stretch", key="custom_net")
        st.plotly_chart(build_intervals(ca), width="stretch", key="custom_chart")
        kind, text = ev.verdict_sample(ca, settings.n)
        getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Fehler ist die Streuung eines Mittelwerts unabhängiger Ziehungen** | Wird viel gezogen, aber die Streuung einer Ziehung ist groß, bleibt der Fehler groß (Experiment 4, schiefe Größe). | Stücke 2 bis 6 (Varianzreduktion) |
| **Das Ereignis ist nicht selten** | Das Wald-Intervall versagt bei kleinem N·q (rund unter 10); bei N·q unter etwa 1 ist „kein Treffer“ der Regelfall; die Faustformel verlangt viele Ziehungen (Experiment 3). | Stück 7 (Importance Sampling) |
| **Man kann direkt aus der Verteilung ziehen** | Sonst helfen nur Ketten oder Gewichte; das Stück behandelt Ketten nicht. | Stück 7 und später eine eigene Linie |
| **Die Dimension spielt keine Rolle** | Die Rate 1/√N hängt nicht von der Dimension ab, die Konstante schon (Experiment 5). | Stück 9 (Quasi-Monte-Carlo) |
| **Kanten fallen unabhängig und mit gleicher Wahrscheinlichkeit aus** | Abhängige Ausfälle (gemeinsame Ursache) ändern die Wahrheit und die Streuung; hier nicht modelliert. | – |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Alle Netze, Ausfälle und Größen sind **erfunden**; die Wahrheit ist die exakte Aufzählung oder eine geschlossene Formel. Die Kanten fallen unabhängig und mit gleicher Wahrscheinlichkeit aus. Die Zufallszahlen kommen aus festen Strömen (SplitMix64), nie aus einem einzelnen Seed.

**Grenzen.** Live aufgezählt wird nur bis 17 Kanten; die Wahrheit für das 3×5- und das 4×4-Gitter steht in gespeicherten Zahlen (Aufzählung, mit dem Matrix-Baum-Satz und den Zahlen der tutte-polynom-demo geprüft). Experiment 1 zieht die Ausfallzahl aus der Binomialverteilung statt aus Graphziehungen, Experiment 2 nutzt echte Graphziehungen. Experiment 5 hat nur einen glatten Integranden, Experiment 4 nur die log-normale Größe. Zeiten werden nicht gemessen, nur die Zahl der Netzprüfungen.

**Bekannte Probleme und Unsicherheiten.** Die Verhältnisse in Experiment 1 und 2 rauschen bei 400 beziehungsweise 300 Wiederholungen um rund 3,5 bis 4 %; Abweichungen in dieser Größe sind kein Befund. Die Literaturangaben im Mathematik-Expander stammen aus dem Gedächtnis und sind nicht eingesehen. Die Oberfläche ist mit AppTest geprüft; die Einzelansicht der Ziehungen und die Karten sind im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Ein Netz mit $n$ Knoten und $m$ Kanten; jede Kante hält unabhängig mit Wahrscheinlichkeit $p$. Gesucht ist die Ausfallwahrscheinlichkeit $q = 1 - R(p)$, wobei $R(p)$ die Wahrscheinlichkeit ist, dass das Netz zusammenhängt.

**Wahrheit.** Mit $N_k$ = Zahl der zusammenhängenden Kantenmengen mit $k$ Kanten (Aufzählung aller $2^m$ Mengen) ist $R(p) = \sum_k N_k\, p^k (1-p)^{m-k}$. Kontrolle: $N_{n-1}$ ist die Zahl der Spannbäume (Matrix-Baum-Satz).

**Schätzer.** $N$ unabhängige Ziehungen, $K$ = Zahl der Ausfälle, $\hat q = K/N$. Dann ist $K \sim \mathrm{Bin}(N, q)$, $\mathbb E\,\hat q = q$ und $\operatorname{sd}(\hat q) = \sqrt{q(1-q)/N}$.

**Intervalle** (95 %, $z = 1{,}96$). Wald: $\hat q \pm z\sqrt{\hat q(1-\hat q)/N}$. Wilson: $\dfrac{\hat q + z^2/2N \pm z\sqrt{\hat q(1-\hat q)/N + z^2/4N^2}}{1 + z^2/N}$. Clopper-Pearson: Beta-Quantile von $K$ und $N-K+1$ (unten) beziehungsweise $K+1$ und $N-K$ (oben). Überdeckung: $\sum_k \mathrm{Bin}(k; N, q)\,\mathbf 1[q \in I(k)]$.

**Faustformel.** Relativer Fehler $\varepsilon$ bei 95 % Sicherheit: $N^\ast = z^2 (1-q) / (q\,\varepsilon^2)$, also rund $384/q$ für $\varepsilon = 0{,}1$.

**Schiefe.** $X = e^{\sigma Z}$ hat $\mathbb E X = e^{\sigma^2/2}$ und Schiefe $(e^{\sigma^2}+2)\sqrt{e^{\sigma^2}-1}$.

**Quadratur.** Mittelpunktsregel mit $n$ Punkten je Achse braucht $n^d$ Auswertungen; für $f(x)=e^{x_1+\dots+x_d}$ ist $\int f = (e-1)^d$ und $\operatorname{Var} f/(\mathbb E f)^2 = \bigl(\tfrac{e^2-1}{2(e-1)^2}\bigr)^d - 1$.

**Literatur (aus dem Gedächtnis, nicht eingesehen).** Clopper und Pearson 1934 (exaktes Intervall); Wilson 1927 (Score-Intervall); Brown, Cai und DasGupta 2001, „Interval estimation for a binomial proportion“, *Statistical Science* 16 (Vergleich der Intervalle).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
