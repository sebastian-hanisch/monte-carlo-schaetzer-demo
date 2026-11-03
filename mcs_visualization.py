"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import mcs_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_intervals(a):
    """Die drei Intervalle für q mit der exakten Wahrheit (senkrechte Linie) und der Schätzung (Punkt)."""
    fig = go.Figure()
    names = list(a.intervals)
    hi_max = max(hi for _, hi in a.intervals.values())
    for name in names:
        lo, hi = a.intervals[name]
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[name, name], mode="lines+markers", name=name, line=dict(color=C.METHOD_COLORS[name], width=6),
            marker=dict(symbol="line-ns-open", size=14, color=C.METHOD_COLORS[name]), hovertemplate=f"{name}: [%{{x:.5f}}]<extra></extra>", showlegend=False,
        ))
    for name in names:
        lo, hi = a.intervals[name]
        if hi - lo < 1e-12:     # entartet (Wald bei null Treffern): ein Punkt, ausdrücklich beschriftet statt unsichtbar
            fig.add_trace(go.Scatter(x=[lo], y=[name], mode="markers+text", marker=dict(symbol="diamond", size=13, color=C.METHOD_COLORS[name]),
                                     text=["nur ein Punkt: Breite 0"], textposition="middle right", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[a.q_hat] * len(names), y=names, mode="markers", name="Schätzung q̂", marker=dict(color="white", size=9, line=dict(color="#222", width=2))))
    upper = max(hi_max, a.q) * 1.15 + 1e-4
    fig.add_vline(x=a.q, line=dict(color=C.COLOR_TRUTH, width=2, dash="dash"), annotation_text="Wahrheit q", annotation_position="top")
    fig.update_xaxes(title_text="Ausfallwahrscheinlichkeit q", range=[0, upper])
    fig.update_yaxes(autorange="reversed")
    fig = _base(fig, 260)
    fig.update_layout(showlegend=False)     # Erklärung steht in der Bildunterschrift; die Legende überdeckte die Beschriftung der Wahrheit
    return fig


def build_netz(netz, state_row=None, connected_flag=None):
    """Das Netz mit einer Ziehung: haltende Kanten grün, ausgefallene rot gestrichelt (state_row = bool je Kante, sonst alle schwarz)."""
    fig = go.Figure()
    xs, ys = [c[0] for c in netz.coords], [c[1] for c in netz.coords]
    hold_x, hold_y, fail_x, fail_y = [], [], [], []
    for e, (u, v) in enumerate(netz.edges):
        ok = True if state_row is None else bool(state_row[e])
        tx, ty = (hold_x, hold_y) if ok else (fail_x, fail_y)
        tx += [xs[u], xs[v], None]
        ty += [ys[u], ys[v], None]
    fig.add_trace(go.Scatter(x=hold_x, y=hold_y, mode="lines", line=dict(color=C.COLOR_OK if state_row is not None else "#444", width=4), name="Kante hält", hoverinfo="skip"))
    if fail_x:
        fig.add_trace(go.Scatter(x=fail_x, y=fail_y, mode="lines", line=dict(color=C.COLOR_BAD, width=3, dash="dot"), name="Kante ausgefallen", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=22, color="white", line=dict(color="#222", width=2)), name="Knoten", hoverinfo="skip", showlegend=False))
    for i, (x, y) in enumerate(zip(xs, ys)):
        fig.add_annotation(x=x, y=y, text=str(i), showarrow=False, font=dict(size=10, color="#222"))
    pad = 0.5
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    if connected_flag is not None:
        fig.update_layout(title=dict(text="zusammenhängend" if connected_flag else "NICHT zusammenhängend", font=dict(size=14, color=C.COLOR_OK if connected_flag else C.COLOR_BAD)))
    fig = _base(fig, 320)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=0.0, x=0), margin=dict(l=10, r=10, t=50, b=40))   # Legende unter dem Netz: oben steht die Überschrift
    return fig


def build_error_vs_n(exp):
    rows = exp["rows"]
    fig = go.Figure()
    ns = [r["N"] for r in rows]
    fig.add_trace(go.Scatter(x=ns, y=[r["formel"] for r in rows], mode="lines", name="Formel √(q(1−q)/N)", line=dict(color=C.COLOR_LINE, dash="dash")))
    fig.add_trace(go.Scatter(x=ns, y=[r["rmse"] for r in rows], mode="markers+lines", name="gemessener RMSE", marker=dict(size=9, color=C.COLOR_OK), line=dict(color=C.COLOR_OK)))
    fig.update_xaxes(title_text="Ziehungen N", type="log")
    fig.update_yaxes(title_text="RMSE von q̂", type="log")
    return _base(fig, 320)


def build_dimension(exp):
    rows = exp["rows"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{r['gitter']} ({r['m']} Kanten)" for r in rows], y=[r["ratio"] for r in rows], marker_color=C.COLOR_OK, name="gemessene Streuung / Formel"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_TRUTH, dash="dash"))
    fig.update_yaxes(title_text="Streuung gemessen / Formel", range=[0, 1.5])
    return _base(fig, 300)


def build_coverage(grid, method_name):
    """Überdeckungskarte (Zeilen: q, Spalten: N); Farbe um 0,95 zentriert."""
    fig = go.Figure(go.Heatmap(
        z=grid, x=[str(n) for n in C.COVER_N], y=[str(q) for q in C.COVER_Q], zmin=0.0, zmax=1.0, colorscale=[[0, "#C0392B"], [0.9, "#F7E9B5"], [0.95, "#BFDCD6"], [1, "#2F6B65"]],
        text=[[f"{v:.3f}" for v in row] for row in grid], texttemplate="%{text}", colorbar=dict(title="Überdeckung"),
    ))
    fig.update_xaxes(title_text="Ziehungen N", type="category")
    fig.update_yaxes(title_text="Ausfallwahrscheinlichkeit q", type="category", autorange="reversed")
    fig.update_layout(title=dict(text=f"{method_name}: Überdeckung (Soll 0,95)", font=dict(size=13)))
    return _base(fig, 340)


def build_skew(exp):
    fig = go.Figure()
    for s in C.SKEW_SIGMAS:
        fig.add_trace(go.Scatter(x=list(C.SKEW_N), y=[c for c, _ in exp["cov"][s]], mode="lines+markers", name=f"σ = {s:g} (Schiefe {exp['skew'][s]:.0f})"))
    fig.add_hline(y=0.95, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Soll 0,95")
    fig.update_xaxes(title_text="Ziehungen N", type="log")
    fig.update_yaxes(title_text="Überdeckung des t-Intervalls", range=[0.5, 1.0])
    return _base(fig, 320)


def build_quadrature(exp):
    rows = exp["rows"]
    fig = go.Figure()
    ds = [r["d"] for r in rows]
    fig.add_trace(go.Scatter(x=ds, y=[r["quad"] for r in rows], mode="markers+lines", name="Mittelpunktsregel", line=dict(color=C.COLOR_LINE)))
    fig.add_trace(go.Scatter(x=ds, y=[r["mc_gemessen"] for r in rows], mode="markers+lines", name="Monte Carlo (gemessen)", line=dict(color=C.COLOR_OK)))
    fig.add_trace(go.Scatter(x=ds, y=[r["mc_formel"] for r in rows], mode="lines", name="Monte Carlo (Formel)", line=dict(color=C.COLOR_OK, dash="dot")))
    fig.update_xaxes(title_text="Dimension d", type="log", tickvals=ds, ticktext=[str(d) for d in ds])
    fig.update_yaxes(title_text="relativer Fehler", type="log")
    return _base(fig, 330)
