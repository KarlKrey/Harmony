"""
3D Harmonic Korrelation nach Prof. Krey
========================================
Workflow:
  1. SNA + PgNB eingeben → individuelle Idealwerte werden berechnet
  2. Alle anderen Messwerte des Patienten eingeben
  3. Abweichungen vom Ideal in SD-Einheiten werden dargestellt
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from harmonie_analyse import (
    compute_ideal,
    compute_abweichungen,
    GRUPPEN,
    EINHEITEN,
    STANDARD_ABWEICHUNGEN,
    FORMELN,
)

# ---------------------------------------------------------------------------
# Seitenconfig
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="3D Harmonic Korrelation",
    page_icon="🦷",
    layout="wide",
)

st.title("3D Harmonic Korrelation nach Prof. Krey")
st.caption("Individualisierte kephalometrische Strukturanalyse nach Segner/Hasund")

# ---------------------------------------------------------------------------
# Farben für Status
# ---------------------------------------------------------------------------
STATUS_FARBE = {"ok": "#2ecc71", "grenz": "#f39c12", "auffällig": "#e74c3c"}
STATUS_LABEL = {"ok": "≤ 1 SD", "grenz": "1–2 SD", "auffällig": "> 2 SD"}

# ---------------------------------------------------------------------------
# Schritt 1 – Sidebar: SNA und PgNB als Treiber
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Schritt 1 – Treibervariablen")
    st.markdown("Aus diesen Werten wird das **individuelle Ideal** berechnet.")
    sna   = st.number_input("SNA (°)", min_value=62.0, max_value=103.0, value=82.0, step=0.5)
    pgnb  = st.number_input("PgNB – Kinnprominenz (mm)", min_value=0.0, max_value=10.0, value=2.3, step=0.1)

    st.divider()
    with st.expander("Zahnbogen-Konstanten (Bernabe)"):
        st.markdown("**HZB** = a + b·SNA + b·SNB")
        a_hzb = st.number_input("HZB – a (Achsenabschnitt)", min_value=30.0, max_value=40.0, value=35.0, step=0.5, key="a_hzb")
        b_hzb = st.number_input("HZB – b (Koeffizient)",     min_value=0.10, max_value=0.20,  value=0.20,  step=0.01, key="b_hzb", format="%.2f")
        st.markdown("**VZB** = a + b·SNA + b·SNB")
        a_vzb = st.number_input("VZB – a (Achsenabschnitt)", min_value=20.0, max_value=25.0,  value=22.5,  step=0.5, key="a_vzb")
        b_vzb = st.number_input("VZB – b (Koeffizient)",     min_value=0.10, max_value=0.20,  value=0.20,  step=0.01, key="b_vzb", format="%.2f")

    st.divider()
    st.markdown("#### Legende")
    for status, farbe in STATUS_FARBE.items():
        st.markdown(
            f'<span style="background:{farbe};padding:2px 8px;border-radius:4px;color:white">'
            f'{STATUS_LABEL[status]}</span>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Idealwerte berechnen (immer im Hintergrund)
# ---------------------------------------------------------------------------
ideal = compute_ideal(sna, pgnb_mm=pgnb, a_hzb=a_hzb, b_hzb=b_hzb, a_vzb=a_vzb, b_vzb=b_vzb)

# ---------------------------------------------------------------------------
# Schritt 2 – Patientenmesswerte eingeben
# ---------------------------------------------------------------------------
st.subheader("Schritt 2 – Patientenmesswerte eingeben")
st.markdown(
    "Die Felder sind mit den **Idealwerten** vorbelegt. "
    "Tragen Sie die tatsächlich gemessenen Werte des Patienten ein."
)

gemessen: dict[str, float] = {}
cols_main = st.columns(len(GRUPPEN))

for col, (gruppe, variablen) in zip(cols_main, GRUPPEN.items()):
    with col:
        st.markdown(f"**{gruppe}**")
        for var in variablen:
            einheit = EINHEITEN.get(var, "")
            gemessen[var] = st.number_input(
                f"{var} ({einheit})",
                value=float(ideal[var]),
                step=0.5,
                key=f"input_{var}",
                format="%.2f",
            )

st.divider()

# ---------------------------------------------------------------------------
# Schritt 3 – Abweichungsanalyse
# ---------------------------------------------------------------------------
abweichungen = compute_abweichungen(gemessen, ideal)
df_abw = pd.DataFrame([{
    "Variable":   a.variable,
    "Ideal":      a.ideal,
    "Gemessen":   a.gemessen,
    "Δ absolut":  round(a.delta, 2),
    "Δ in SD":    round(a.delta_sd, 2),
    "Einheit":    a.einheit,
    "Status":     a.status,
    "Farbe":      STATUS_FARBE[a.status],
} for a in abweichungen])

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Abweichungsübersicht", "🧊 3D Harmonieraum", "📋 Wertetabelle", "📐 Formeln"]
)

# ── Tab 1: Abweichungs-Balkendiagramm ──────────────────────────────────────
with tab1:
    st.subheader("Abweichung vom individuellen Ideal (in SD-Einheiten)")
    st.caption("Grün ≤ 1 SD  |  Orange 1–2 SD  |  Rot > 2 SD  |  Gestrichelt = ±2 SD-Grenze")

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=df_abw["Δ in SD"],
        y=df_abw["Variable"],
        orientation="h",
        marker_color=df_abw["Farbe"].tolist(),
        text=[
            f"{row['Δ absolut']:+.2f} {row['Einheit']}  ({row['Δ in SD']:+.2f} SD)"
            for _, row in df_abw.iterrows()
        ],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Ideal: %{customdata[0]}<br>"
            "Gemessen: %{customdata[1]}<br>"
            "Δ: %{customdata[2]:+.2f} SD"
            "<extra></extra>"
        ),
        customdata=df_abw[["Ideal", "Gemessen", "Δ in SD"]].values,
    ))

    # ±1 SD und ±2 SD Linien
    for x_val, dash, color in [(-2, "dash", "red"), (-1, "dot", "gray"),
                                 (1, "dot", "gray"), (2, "dash", "red")]:
        fig_bar.add_vline(x=x_val, line_dash=dash, line_color=color, line_width=1.5)

    fig_bar.add_vrect(x0=-1, x1=1, fillcolor="#2ecc71", opacity=0.07, line_width=0)
    fig_bar.add_vrect(x0=-2, x1=-1, fillcolor="#f39c12", opacity=0.07, line_width=0)
    fig_bar.add_vrect(x0=1, x1=2, fillcolor="#f39c12", opacity=0.07, line_width=0)

    fig_bar.update_layout(
        height=550,
        xaxis_title="Abweichung in Standardabweichungen (SD)",
        yaxis_title="",
        xaxis=dict(range=[-4, 4], zeroline=True, zerolinewidth=2, zerolinecolor="black"),
        plot_bgcolor="white",
        margin=dict(l=10, r=180, t=30, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Kennzahlen-Zusammenfassung
    c1, c2, c3 = st.columns(3)
    n_ok    = (df_abw["Status"] == "ok").sum()
    n_grenz = (df_abw["Status"] == "grenz").sum()
    n_auf   = (df_abw["Status"] == "auffällig").sum()
    c1.metric("Im Normbereich (≤1 SD)", n_ok)
    c2.metric("Grenzbereich (1–2 SD)", n_grenz)
    c3.metric("Auffällig (>2 SD)", n_auf)

# ── Tab 2: 3D Harmonieraum ──────────────────────────────────────────────────
with tab2:
    st.subheader("3D Harmonieraum – Patient im SD-Koordinatensystem")
    st.caption(
        "Jede Achse zeigt die Abweichung einer Variable vom Ideal in SD-Einheiten. "
        "Der Ursprung (0,0,0) = perfekte Harmonie."
    )

    alle_vars = df_abw["Variable"].tolist()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        x_var = st.selectbox("X-Achse", alle_vars, index=0)
    with col_b:
        y_var = st.selectbox("Y-Achse", alle_vars, index=2)
    with col_c:
        z_var = st.selectbox("Z-Achse", alle_vars, index=4)

    def get_delta_sd(var: str) -> float:
        row = df_abw[df_abw["Variable"] == var].iloc[0]
        return float(row["Δ in SD"])

    px_val = get_delta_sd(x_var)
    py_val = get_delta_sd(y_var)
    pz_val = get_delta_sd(z_var)

    # SD-Kugeln (±1 SD und ±2 SD) als Drahtgitter-Sphären
    def sphere_surface(r: float, n: int = 30):
        u = np.linspace(0, 2 * np.pi, n)
        v = np.linspace(0, np.pi, n)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(n), np.cos(v))
        return x, y, z

    fig3d = go.Figure()

    for r, farbe, name, opacity in [
        (1.0, "#2ecc71", "±1 SD", 0.08),
        (2.0, "#e74c3c", "±2 SD", 0.05),
    ]:
        sx, sy, sz = sphere_surface(r)
        fig3d.add_trace(go.Surface(
            x=sx, y=sy, z=sz,
            opacity=opacity,
            colorscale=[[0, farbe], [1, farbe]],
            showscale=False,
            name=name,
            hoverinfo="skip",
        ))

    # Achsenkreuz im Ursprung (Ideal)
    for axis in [([-3, 3], [0, 0], [0, 0]),
                 ([0, 0], [-3, 3], [0, 0]),
                 ([0, 0], [0, 0], [-3, 3])]:
        fig3d.add_trace(go.Scatter3d(
            x=axis[0], y=axis[1], z=axis[2],
            mode="lines",
            line=dict(color="lightgray", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Patientenpunkt
    patient_farbe = STATUS_FARBE[
        "auffällig" if abs(px_val) > 2 or abs(py_val) > 2 or abs(pz_val) > 2
        else "grenz" if abs(px_val) > 1 or abs(py_val) > 1 or abs(pz_val) > 1
        else "ok"
    ]
    fig3d.add_trace(go.Scatter3d(
        x=[px_val], y=[py_val], z=[pz_val],
        mode="markers+text",
        marker=dict(size=12, color=patient_farbe, symbol="diamond"),
        text=["Patient"],
        textposition="top center",
        name="Patient",
        hovertemplate=(
            f"<b>Patient</b><br>"
            f"{x_var}: {px_val:+.2f} SD<br>"
            f"{y_var}: {py_val:+.2f} SD<br>"
            f"{z_var}: {pz_val:+.2f} SD<extra></extra>"
        ),
    ))

    # Linie vom Ideal zum Patienten
    fig3d.add_trace(go.Scatter3d(
        x=[0, px_val], y=[0, py_val], z=[0, pz_val],
        mode="lines",
        line=dict(color=patient_farbe, width=4, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Idealpunkt
    fig3d.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=8, color="navy"),
        name="Ideal (Harmonie)",
    ))

    fig3d.update_layout(
        height=650,
        scene=dict(
            xaxis_title=f"{x_var} (SD)",
            yaxis_title=f"{y_var} (SD)",
            zaxis_title=f"{z_var} (SD)",
            xaxis=dict(range=[-3.5, 3.5]),
            yaxis=dict(range=[-3.5, 3.5]),
            zaxis=dict(range=[-3.5, 3.5]),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig3d, use_container_width=True)

# ── Tab 3: Wertetabelle ────────────────────────────────────────────────────
with tab3:
    st.subheader("Wertetabelle – Ideal vs. Gemessen")
    display_df = df_abw[["Variable", "Einheit", "Ideal", "Gemessen", "Δ absolut", "Δ in SD", "Status"]].copy()
    display_df["Δ in SD"] = display_df["Δ in SD"].map(lambda x: f"{x:+.2f}")
    display_df["Δ absolut"] = display_df["Δ absolut"].map(lambda x: f"{x:+.2f}")

    def highlight_status(row):
        farbe = STATUS_FARBE.get(row["Status"], "white")
        return [f"background-color: {farbe}22" if col == "Status" else "" for col in row.index]

    st.dataframe(
        display_df.style.apply(highlight_status, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇ Als CSV herunterladen",
        data=df_abw.to_csv(index=False).encode("utf-8"),
        file_name="harmonie_analyse.csv",
        mime="text/csv",
    )

# ── Tab 4: Formeln ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("Verwendete Regressionsformeln (Segner/Hasund)")
    st.markdown(f"**SNA = {sna}°  |  PgNB = {pgnb} mm**")
    st.divider()
    col_f1, col_f2 = st.columns(2)
    items = list(FORMELN.items())
    for i, (name, formel) in enumerate(items):
        col = col_f1 if i < len(items) // 2 else col_f2
        sd  = STANDARD_ABWEICHUNGEN.get(name, "–")
        col.markdown(f"**{name}** = `{formel}`  \n_SD = ±{sd} {EINHEITEN.get(name, '')}_")
