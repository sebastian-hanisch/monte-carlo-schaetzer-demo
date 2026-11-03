"""PDF-Export der aktuellen Schätzung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from mcs_evaluation import Analysis, Settings, de, verdict_sample

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~", "σ": "Sigma",
            "∞": "unendlich", "̂": "", "Σ": "Summe", "ζ": "zeta"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Der Monte-Carlo-Schätzer und sein Fehler"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"{a.netz.name}, {a.netz.n} Knoten, {a.netz.m} Kanten; jede Kante hält mit Wahrscheinlichkeit {de(settings.p, 2)}; "
        f"{de(settings.n, 0)} Ziehungen, Strom {settings.seed}. Alles simuliert, keine Messdaten."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(f"Exakte Ausfallwahrscheinlichkeit q = {de(a.q, 5)} (Aufzählung aller Kantenmengen); "
                            f"Ausfälle in den Ziehungen: {a.fails}; Schätzung q̂ = {de(a.q_hat, 5)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 9)
    widths = (45, 40, 40, 40)
    for h, w in zip(("Verfahren", "unten", "oben", "enthält q?"), widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for name, (lo, hi) in a.intervals.items():
        for text, w in zip((name, de(lo, 5), de(hi, 5), "ja" if a.covers[name] else "NEIN"), widths):
            pdf.cell(w, 6, _s(text), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.multi_cell(0, 5, _s(verdict_sample(a, settings.n)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: monte-carlo-schaetzer-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
