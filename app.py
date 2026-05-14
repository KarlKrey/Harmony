"""
Harmony – Interaktives Streamlit-Dashboard
==========================================
Individualisierte kephalometrische Strukturanalyse nach Segner/Hasund.
Berechnet Normwerte und zeigt 3D-Korrelationsplots.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from harmonie_analyse import compute_harmonie_normwerte, generate_normwerttabelle, FORMELN

# ---------------------------------------------------------------------------
# Seitenconfig
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Harmony – Kephalometrie",
    page_icon="🦷",
    layout="wide",
)

st.title("3D Harmonic Korrelation nach Prof. Krey")
st.caption("Individualisierte kephalometrische Strukturanalyse nach Segner/Hasund – Normwertberechnung und 3D-Korrelationsvisualisierung")

# ---------------------------------------------------------------------------
# Sidebar: Patientenwerte
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Patientenwerte")
    sna = st.slider("SNA (°)", min_value=62.0, max_value=103.0, value=82.0, step=0.5)
    pgnb = st.slider("PgNB – Kinnprominenz (mm)", min_value=0.0, max_value=10.0, value=2.3, step=0.1)
    jochbogen = st.number_input("Jochbogenbreite (mm, optional)", min_value=0.0, max_value=200.0, value=0.0, step=0.5)
    jochbogen_val = jochbogen if jochbogen > 0 else None

    st.divider()
    st.header("3D-Plot-Achsen")
    alle_variablen = [
        "SNA", "SNB", "ANB",
        "NL-NSL", "NSBa", "ML-NSL", "ML-NL",
        "1-NA_mm", "1-NA_deg", "1-NB_mm", "1-NB_deg",
        "H-Winkel", "Nasolabialwinkel", "Z-Winkel",
        "HZB", "VZB", "Eckzahn-OK", "Pont-SI-OK", "Pont-SI-UK",
    ]
    x_var = st.selectbox("X-Achse", alle_variablen, index=0)
    y_var = st.selectbox("Y-Achse", alle_variablen, index=2)
    z_var = st.selectbox("Z-Achse", alle_variablen, index=11)
    color_var = st.selectbox("Farb-Codierung", alle_variablen, index=1)

# ---------------------------------------------------------------------------
# Berechnungen
# ---------------------------------------------------------------------------
patient = compute_harmonie_normwerte(sna, pgnb_mm=pgnb, jochbogenbreite_mm=jochbogen_val)

# Normwerttabelle für den gesamten SNA-Bereich (Grundlage für Plots)
tabelle = generate_normwerttabelle(62, 103, pgnb_mm=pgnb)
df = pd.DataFrame(tabelle)

# ---------------------------------------------------------------------------
# Tab-Layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Normwerte", "🧊 3D-Korrelation", "📈 2D-Verläufe", "📋 Formeln"])

# ── Tab 1: Normwerte des Patienten ─────────────────────────────────────────
with tab1:
    st.subheader(f"Normwerte für SNA = {sna}°  |  PgNB = {pgnb} mm")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Skelettale Basis**")
        for key in ["SNA", "SNB", "ANB", "NL-NSL", "NSBa", "ML-NSL", "ML-NL"]:
            val = patient[key]
            unit = "°"
            st.metric(key, f"{val} {unit}")

    with col2:
        st.markdown("**Dentale Normwerte**")
        for key in ["1-NA_mm", "1-NA_deg", "1-NB_mm", "1-NB_deg", "H-Winkel", "Nasolabialwinkel", "Z-Winkel"]:
            val = patient[key]
            unit = "mm" if key.endswith("_mm") else "°"
            st.metric(key, f"{val} {unit}")

    with col3:
        st.markdown("**Zahnbogen & Indices**")
        for key in ["HZB", "VZB", "Eckzahn-OK", "Pont-SI-OK", "Pont-SI-UK"]:
            val = patient[key]
            st.metric(key, f"{val} mm")
        if patient["Izard-Index"] is not None:
            st.metric("Izard-Index", f"{patient['Izard-Index']:.4f}")

# ── Tab 2: 3D-Korrelation ───────────────────────────────────────────────────
with tab2:
    st.subheader("3D-Korrelationsplot über den gesamten SNA-Bereich (62°–103°)")
    st.caption(f"Achsen: {x_var} × {y_var} × {z_var}  |  Farbe: {color_var}")

    # PgNB-Varianz: erzeuge Datenpunkte über SNA × PgNB-Raster
    pgnb_werte = np.arange(0.5, 6.0, 0.5)
    rows = []
    for sna_i in range(62, 104):
        for pg_i in pgnb_werte:
            r = compute_harmonie_normwerte(float(sna_i), pgnb_mm=float(pg_i))
            r["PgNB_label"] = f"{pg_i:.1f}"
            rows.append(r)
    df_3d = pd.DataFrame(rows)

    fig3d = px.scatter_3d(
        df_3d,
        x=x_var,
        y=y_var,
        z=z_var,
        color=color_var,
        opacity=0.6,
        color_continuous_scale="Viridis",
        hover_data=["SNA", "ANB", "PgNB_mm"],
        labels={x_var: x_var, y_var: y_var, z_var: z_var},
    )
    # Patientenpunkt hervorheben
    fig3d.add_trace(go.Scatter3d(
        x=[patient[x_var]],
        y=[patient[y_var]],
        z=[patient[z_var]],
        mode="markers",
        marker=dict(size=10, color="red", symbol="diamond"),
        name=f"Patient (SNA={sna}°)",
    ))
    fig3d.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig3d, use_container_width=True)

# ── Tab 3: 2D-Verläufe ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Normwertverlauf über SNA (62°–103°)")
    y_auswahl = st.multiselect(
        "Variablen anzeigen",
        alle_variablen,
        default=["SNB", "ANB", "ML-NSL", "H-Winkel"],
    )
    if y_auswahl:
        fig2d = px.line(
            df,
            x="SNA",
            y=y_auswahl,
            markers=True,
            labels={"value": "Wert", "variable": "Variable"},
        )
        # Patientenwert markieren
        fig2d.add_vline(x=sna, line_dash="dash", line_color="red", annotation_text=f"Patient SNA={sna}°")
        fig2d.update_layout(height=500)
        st.plotly_chart(fig2d, use_container_width=True)

# ── Tab 4: Formeln ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("Verwendete Regressionsformeln")
    for name, formel in FORMELN.items():
        st.markdown(f"**{name}** = `{formel}`")
