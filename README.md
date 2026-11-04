# Der Monte-Carlo-Schätzer und sein Fehler – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-monte-carlo-schaetzer-demo.streamlit.app/)**

Erstes Stück (Wurzel) der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Ein Verfahren, ein wachsendes Beispiel: der **Monte-Carlo-Schätzer**, eine Wahrscheinlichkeit durch Ziehen und Zählen zu schätzen. Dieses Stück fragt nur: **Wie viele Ziehungen braucht man, und wann darf man dem Konfidenzintervall trauen?** Die Folgestücke senken den Fehler (Stücke 2 bis 6), machen Seltenes erreichbar (7 und 8) oder erweitern das Ziel (9 bis 13).

Vehikel ist die **Zuverlässigkeit eines Gitternetzes**: Jede Kante fällt unabhängig mit Wahrscheinlichkeit 1 − p aus, gesucht ist die Wahrscheinlichkeit q, dass das Netz nicht mehr zusammenhängt. Bis 24 Kanten ist q **exakt** aufzählbar; die Wahrheit steht also fest, und jede Schätzung und jedes Intervall lässt sich gegen sie messen.

## Warum dieses Problem

Ein Monte-Carlo-Schätzer wird oft mit zwei Sätzen erklärt: „Der Fehler fällt wie 1/√N“ und „das gilt in jeder Dimension“. Beides stimmt, aber es verschweigt die **Konstante**: Sie hängt an der **Seltenheit** des Ereignisses und an der **Schiefe** der Größe. Wer ein Ausfallrisiko von 0,04 % mit 1000 Ziehungen schätzt, sieht in zwei Dritteln der Fälle gar keinen Ausfall und bekommt vom üblichen Intervall einen Punkt. Das Stück zeigt, wann das passiert und was die drei gängigen Intervalle dann tun.

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat:

- **„Fehler unabhängig von der Dimension“ stimmt nur für die Rate.** Die Rate 1/√N hängt nicht von d ab, die Konstante schon: bei gleichem N wächst der relative Fehler von 2,9·10⁻³ (d = 1) auf 2,0·10⁻² (d = 20). Die App nennt deshalb die Rate und im selben Satz die Konstante, die von der Streuung einer Ziehung abhängt (Seltenheit, Schiefe).
- **Neu: Das Wald-Intervall versagt, wenn N·q klein ist (rund unter 10); bei N·q unter etwa 1 ist „kein Ausfall“ der Regelfall**, kein Randfall (Experiment 3).
- **Neu: Wilson ist gut, aber nicht garantiert** (niedrigste Überdeckung im Raster 0,905, zwischen den Rasterpunkten bis etwa 0,84); von den drei Verfahren hält nur Clopper–Pearson mindestens 95 %, dafür oft weit darüber.
- **Korrektur an der Vormessung selbst:** In der Vormessung stand zunächst „rund 38.400/q“ Ziehungen für 10 % relativen Fehler; richtig ist **384/q** (1,96²/0,1²). Die Tabelle in der App rechnet aus der Formel und prüft sie gegen die exakte Binomialverteilung.
- **Der Orakelweg ist unabhängig:** Die eigene Aufzählung stimmt für das 4×4-Gitter mit den Zahlen der `tutte-polynom-demo` (Abzählen-Linie, Frontier/Polynom statt Aufzählung) auf sechs Stellen überein.

## Modell

- **Netz.** Ein Gitter mit Knoten in Zeilen und Spalten; Kanten waagerecht und senkrecht. Sechs Gitter von 2×3 (7 Kanten) bis 4×4 (24 Kanten). Jede Kante hält unabhängig mit Wahrscheinlichkeit p, das Netz hängt zusammen, wenn alle Knoten verbunden sind (Zuverlässigkeit R = 1 − q).
- **Wahrheit.** Mit N_k = Zahl der zusammenhängenden Kantenmengen mit k Kanten gilt R(p) = Σ N_k p^k (1−p)^(m−k). Bis 17 Kanten wird N_k live aufgezählt (2^17 = 131 072 Kantenmengen); für 3×5 (22 Kanten) und 4×4 (24 Kanten) stehen die Zahlen gespeichert in `mcs_counts.py` (Aufzählung aller Mengen, geprüft mit dem Matrix-Baum-Satz und den Zahlen der Tutte-Demo).
- **Ziehungen.** N Netze aus einem festen Strom (SplitMix64, zählerbasiert: Ausgabe i hängt nur von Strom und i ab). Eine Kante hält, wenn ihre Zufallszahl unter p liegt. Folge: Bei festem Strom sind die Ziehungen für verschiedene p **gekoppelt** (gemeinsame Zufallszahlen), und eine kleinere Ziehungszahl ist der Anfang der größeren.
- **Schätzer und Intervalle.** q̂ = Ausfälle / N; Wald, Wilson und Clopper–Pearson (95 %). Die Überdeckung wird **exakt** aus der Binomialverteilung berechnet (Summe über alle möglichen Ausfallzahlen), nicht durch eine Wiederholungsstudie.
- **Schiefe Größe.** X = exp(σ·Z), Z ~ N(0,1), Mittelwert mit t-Intervall; Wahrheit exp(σ²/2).
- **Quadratur.** f(x) = exp(x₁ + … + x_d) auf dem Einheitswürfel, Wahrheit (e−1)^d, Mittelpunktsregel gegen Monte Carlo bei etwa 10⁴ Auswertungen.
- Alles ist **erfunden**; Netze, Größen und Integranden stehen nicht für Messdaten. Alle Formeln im Expander „📐 Mathematische Formulierung“.

## Methodik

Fünf Experimente, jedes mit live berechnetem Verdikt: (1) Fehler gegen N, (2) Streuung gegen Netzgröße, (3) Überdeckung der Intervalle (Karte q gegen N), (4) schiefe Größe, (5) Quadratur gegen Monte Carlo. Dazu die Ansicht einzelner Ziehungen und eine Spielwiese für ein eigenes Netz (bis 17 Kanten). Experiment 1 zieht die Ausfallzahl aus Binomial(N, q) statt aus Graphziehungen; das ist gleichwertig (Test unten) und macht 2 × 5 × 400 Wiederholungen in Bruchteilen einer Sekunde. Experiment 2 nutzt echte Graphziehungen.

## Befunde (gemessen, keine Behauptungen)

Fast alle Zahlen stehen in `tests/test_claims.py` (mit Toleranzen); Ströme sind fest (nie ein einzelner Seed), Verhältnisse verrauschter Messungen haben Bänder. Das Rauschen der Verhältnisse beträgt bei 400 Wiederholungen rund 3,5 %, bei 300 rund 4,1 %: Abweichungen dieser Größe sind kein Befund.

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (3×4, p = 0,80, N = 1000, Strom 35) | R = 0,746409, q = 0,253591; 258 Ausfälle, q̂ = 0,258; Wald [0,2309; 0,2851], Wilson [0,2318; 0,2860], Clopper–Pearson [0,2311; 0,2863], alle enthalten q; für 10 % relativen Fehler wären etwa 1131 Ziehungen nötig, gezogen sind 1000 | `test_preset_standardfall` |
| **Seltenes Ereignis** (3×4, p = 0,99, N = 1000) | q = 0,000417; **kein Ausfall**; Wald = [0; 0] enthält q nicht, Wilson bis 0,00383 und Clopper–Pearson bis 0,00368 schon; die Wahrscheinlichkeit für „kein Ausfall“ war 0,659; 10 % relativer Fehler brauchen rund 920 938 Ziehungen | `test_preset_seltenes_ereignis` |
| **Kleines Netz** (2×3, p = 0,70, N = 100) | R = 0,556312; 45 Ausfälle; 482 Ziehungen wären für 10 % nötig | `test_preset_kleines_netz` |
| **Großes Netz** (4×4, p = 0,90, N = 10 000) | R = 0,944085 (gleich der Tutte-Demo); 561 Ausfälle; Faustformel 6486, es reicht | `test_preset_grosses_netz` |
| Fällt der Fehler wie 1/√N? (3×4, R = 0,5 und 0,95, je 400 Wiederholungen) | RMSE / √(q(1−q)/N): **0,99 bis 1,07** (R = 0,5) und **0,93 bis 1,06** (R = 0,95); Steigung −0,49 und −0,53 | `test_experiment_1_verhaeltnisse_und_steigung` |
| Binomial statt Graphziehung zulässig? (N = 1000, q = 0,0555) | Mittel 55,33 (Graph) gegen 55,52 (Binomial), Streuung 7,03 gegen 7,23, Soll 55,49 und 7,24 (Unterschiede im Rauschen, der Test prüft mit Toleranz) | `test_binomialziehung_und_graphziehung_haben_dieselben_momente` |
| Hängt die Streuung an der Netzgröße? (R = 0,5, N = 2000, 300 Wdh., 7, 12, 13, 17 Kanten) | Verhältnis gemessen/Formel **1,04 / 1,04 / 0,96 / 1,01** (zwischen 0,96 und 1,04); bei gleichem R haben alle Netze dasselbe q, die gleiche Streuung folgt aus der Binomialverteilung, die Messung bestätigt die Ziehfunktion; die naive Aufzählung braucht 2^m Prüfungen und ist gegen 10 000 Ziehungen bis **13 Kanten** billiger (sie liefert dafür den exakten Wert), ab 14 Kanten braucht sie mehr Prüfungen; Frontier-Verfahren verschieben diese Grenze weit nach oben (Zählung, keine Zeitmessung) | `test_experiment_2_streuung_haengt_nicht_an_der_kantenzahl` |
| Wann trügt Wald? (exakte Überdeckung, Soll 0,95) | q = 0,01, N = 1000: **0,927**; q = 0,003, N = 10 000: **0,931**; q = 0,01, N = 10 000: **0,946**; niedrigster Wert **0,030** (q = 0,001, N = 30); höchster 0,953 | `test_experiment_3_wald_ueberdeckung` |
| Wilson und Clopper–Pearson | Wilson im Raster 0,905 bis 0,974 (keine Garantie; zwischen den Rasterpunkten bis etwa 0,84); Clopper–Pearson **0,951 bis 1,000** in allen 30 Zellen | `test_experiment_3_wilson_und_clopper_pearson` |
| Kein Ausfall in N Ziehungen, (1−q)^N | q = 0,001, N = 100: **0,905**; q = 0,01, N = 100: 0,366; q = 0,003, N = 300: 0,406 | `test_experiment_3_kein_treffer_und_benoetigte_ziehungen` |
| Ziehungen für 10 % relativen Fehler, N* = 1,96²(1−q)/(q·0,01) | q = 0,1: 3457; 0,01: 38 030; 0,001: 383 762; 10⁻⁶: **384 145 498**; bis q = 10⁻⁴ trifft die Faustformel das Ziel mit Wahrscheinlichkeit 0,945 bis 0,955 (exakt aus der Binomialverteilung) | dasselbe, `test_faustformel_hat_384_als_konstante` |
| Schiefe Größe (t-Intervall für den Mittelwert von exp(σZ), 2000 Wdh.) | σ = 0,5 (Schiefe 1,8): 0,935 / 0,951 / 0,956 / 0,948 bei N = 30 / 100 / 1000 / 10 000; σ = 1 (6,2): 0,952 bei N = 10 000; σ = 2 (**414**): **0,663 / 0,757 / 0,854 / 0,912**, Median des relativen Fehlers −0,279 / −0,131 / −0,039 / −0,007 (der Median des Schätzfehlers ist negativ: meist wird der Mittelwert unterschätzt, im Erwartungswert ist der Schätzer aber unverzerrt) | `test_experiment_4_schiefe` |
| Quadratur gegen Monte Carlo (f = exp(Σxᵢ), Budget etwa 10⁴) | relativer Fehler Quadratur / Monte Carlo (Formel): d = 1: 4,2·10⁻¹⁰ / 2,9·10⁻³; d = 5 (6 Punkte je Achse, 7776 Auswertungen): 5,8·10⁻³ / 6,9·10⁻³; d = 6 (4 Punkte, 4096): **1,5·10⁻² / 7,8·10⁻³**; d = 10 (2 Punkte, 1024): 9,9·10⁻² / 1,1·10⁻²; d = 20 (1 Punkt): 0,56 / 2,0·10⁻². **Monte Carlo gewinnt ab d = 6** (bei diesem Integranden und diesem Budget); die Konstante wächst, die Rate nicht | `test_experiment_5_quadratur_gegen_monte_carlo` |
| Gitter | 2×3, 3×3, 2×5, 3×4, 3×5, 4×4 haben 7, 12, 13, 17, 22, 24 Kanten und 15, 192, 209, 2415, 30 305, 100 352 Spannbäume | `test_gitter_kantenzahlen_und_spannbaumzahlen` |

## Ehrliche Grenzen

- **Nur an erfundenen Gittern und einer erfundenen Größe geprüft.** Die Wahrheit ist exakt, aber ob die Aussagen auf reale Netze übertragbar sind, zeigt die Demo nicht.
- **Zwei Vehikel statt eines:** Das Netz trägt die Experimente 1 bis 3, die schiefe Größe und der Integrand tragen 4 und 5. Das ist die ehrliche Abweichung vom Muster „eine Instanz wächst“ (siehe Plan).
- **Experiment 5 hat nur einen glatten Integranden** (exp(Σx)); für unglatte oder schwach von der Dimension abhängige Integranden kann die Kreuzung bei einem anderen d liegen. Quadratur und Monte Carlo haben bei gleichem Budget leicht verschiedene Auswertungszahlen (4096 gegen 10 000 bei d = 6, weil n^d nicht beliebig fein einstellbar ist).
- **Zeiten werden nicht gemessen**, nur die Zahl der Netzprüfungen. Ein Vergleich Aufzählung gegen Ziehung in Sekunden hängt vom Rechner ab; auf dem Entwicklungsrechner dauerte die Aufzählung bei 17 Kanten rund 0,3 s, bei 24 Kanten gut eine Minute (Vormessung, nicht getestet).
- **Die Intervalle gelten für unabhängige Ziehungen.** Ketten (abhängige Ziehungen) gehören nicht in dieses Stück.
- **Kanten fallen unabhängig und mit gleicher Wahrscheinlichkeit aus.** Gemeinsame Ursachen ändern Wahrheit und Streuung und sind nicht modelliert.
- **Literaturangaben im Mathematik-Expander** (Clopper und Pearson 1934, Wilson 1927, Brown, Cai und DasGupta 2001) stammen aus dem Gedächtnis und sind nicht eingesehen. Die Websuche zum Stand der Technik im Plan stützt sich auf Suchauszüge, nicht auf gelesene Volltexte.
- **Mit echtem Mausklick** sind nur die Ziehungsansicht und die Experimentknöpfe im Browser angesehen; AppTest deckt Regler, Presets, Permalink, PDF und Spielwiese ab.

## Verifikation

- **Kern:** Zufallsstrom (SplitMix64-Startwert 0xE220A8397B1DCDAF, vektorisiert gleich einer Ganzzahl-Referenz), Zusammenhang, Aufzählung N_k, Zuverlässigkeit, gekoppelte Ziehungen, Binomialziehung, Wald/Wilson/Clopper–Pearson, exakte Überdeckung, Faustformel, Schiefe und Quadratur gegen Handrechnungen auf Mini-Instanzen (`tests/conftest.py`: Dreieck, Kreis, Pfad, K4) und geschlossene Formeln (`tests/test_algorithm.py`).
- **Orakel** (`tests/test_oracle_mcs.py`): Zusammenhang gegen `networkx.is_connected` auf zufälligen Kantenmengen; Aufzählung gegen Matrix-Baum-Satz auf Zufallsgraphen, gegen Bruchrechnung mit Union-Find und für 4×4 gegen die Zahlen der Tutte-Demo; Clopper–Pearson und Wilson gegen `scipy.stats.binomtest`; exakte Überdeckung gegen eine Simulation mit 300 000 Wiederholungen; Erwartungswert der log-normalen Größe und Streuung des Integranden gegen numerische Integration.
- **Eigenschaften** (`tests/test_properties.py`): Zuverlässigkeit wächst mit p, Zählung unabhängig von Knoten- und Kantenreihenfolge, gekoppelte Ziehungen (mehr p, weniger oder gleich viele Ausfälle), Schätzer erwartungstreu, Clopper–Pearson ≥ 0,95 in jeder Zelle, Intervalle in [0, 1], Analyse deterministisch.
- **Randfälle** (`tests/test_edge_cases.py`): Regler an den Grenzen, nicht zusammenhängendes Netz (q = 1), Netz ohne Ausfallmöglichkeit (q = 0), Eingabefehler der Spielwiese, Worst-Case am Regler-Maximum (4×4, 100 000 Ziehungen).
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten, deutsches Zahlenformat.
- **Presets, Layout, Oberfläche:** Presets vollständig, Werte innerhalb der Grenzen; Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, Regler an Minimum und Maximum, die Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims.**

Zahl der Tests: siehe `pytest --collect-only -q tests` (Stand beim Bau: 136 Tests).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `mcs_algorithm.py` | Kern: Zufallsstrom, Zusammenhang, Aufzählung, Ziehungen, Intervalle, Überdeckung, Schiefe, Quadratur |
| `mcs_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `mcs_counts.py` | Gespeicherte Zahlen N_k für 3×5 und 4×4 |
| `mcs_evaluation.py` | Analyse einer Schätzung, Experimente, Verdikte, Zahlenformat |
| `mcs_layout.py` | Fester Kopf (Stylesheet) |
| `mcs_pdf_export.py` | PDF der aktuellen Schätzung (fpdf2) |
| `mcs_presets.py` | Permalink (Begrenzen, Einrasten), Presets, Zufallsstrom-Knopf |
| `mcs_scenario.py` | Netze: Gitter, eigene Kantenliste mit Eingabeprüfung |
| `mcs_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Mini-Instanzen, Kern, Orakel, Eigenschaften, Randfälle, Verdikte, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Varianzreduktion** (antithetische Paare, Kontrollvariaten, Schichtung, bedingtes Monte Carlo): Stücke 2 bis 6 der Linie.
- **Importance Sampling und Splitting für seltene Ereignisse:** Stück 7 und 8 (Splitting als Verfahren steht in `splitting-demo`, Warteschlangen-Linie).
- **Quasi-Monte-Carlo, Sobol-Indizes, Gradientenschätzer, Multilevel, geschachtelte Erwartungen:** spätere Stücke der Linie.
- **Abbruchregeln, Bootstrap:** Stücke 14 und 15 (die Warteschlangen-Linie nennt beides als „kein Folgestück“).
- **Ketten (MCMC, Metropolis, R̂):** gehören in `posterior-sampling-demo` (Inverse-Linie, Stück 8) und eine spätere eigene Linie.
- **Bayessche Quadratur:** nur als Gegenprobe im Stück 9 vorgesehen; ein kontrollierter Vergleich mit Monte Carlo liegt laut Übersicht nicht vor.
- **Zuverlässigkeit gegen Zufallsziehen als Gegenprobe der exakten Zahl:** `tutte-polynom-demo` (Experiment 5), `random-spanning-tree-demo`.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Monte-Carlo-Schätzung: vom Schätzer zur Wahl](https://sebastianhanisch.net/konzepte-monte-carlo.html).
