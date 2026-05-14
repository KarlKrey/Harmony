"""
3D Harmonic Korrelation nach Prof. Krey
========================================
Workflow:
  1. SNA + PgNB + optionale Paddenberg-Eingaben → individuelles Ideal berechnen
  2. Alle Patientenmesswerte eingeben
  3. Abweichungen vom Ideal in SD-Einheiten visualisieren:
     - Balkendiagramm (alle Variablen)
     - 3D-Oberflächen (je eine Fläche pro Kategorie, Harmonielinie Y=0)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from harmonie_analyse import (
    compute_ideal,
    compute_abweichungen,
    Abweichung,
    GRUPPEN,
    EINHEITEN,
    STANDARD_ABWEICHUNGEN,
    FORMELN,
)
from paddenberg_floating_norms import paddenberg_analyse, EMPIRISCHE_NORMEN

# ---------------------------------------------------------------------------
# Seitenconfig
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="3D Harmonic Korrelation",
    page_icon="🦷",
    layout="wide",
)
st.title("3D Harmonic Korrelation nach Prof. Krey")
st.caption("Individualisierte kephalometrische Strukturanalyse nach Segner/Hasund & Paddenberg")

# ---------------------------------------------------------------------------
# Statische Definitionen
# ---------------------------------------------------------------------------
STATUS_FARBE = {"ok": "#2ecc71", "grenz": "#f39c12", "auffällig": "#e74c3c"}
STATUS_LABEL = {"ok": "≤ 1 SD", "grenz": "1–2 SD", "auffällig": "> 2 SD"}

# 4 Kategorien für die 3D-Oberflächen (Z = Höhe der Ebene)
KATEGORIEN_3D_BASIS = [
    {"name": "FRS (Fernröntgen)",  "z":  0, "color": "rgba(100,149,237,0.18)", "line": "#4169E1",
     "vars": ["SNB", "ANB", "NL-NSL", "NSBa", "ML-NSL", "ML-NL"]},
    {"name": "Dentale Variablen", "z":  7, "color": "rgba(60,179,113,0.18)",  "line": "#2E8B57",
     "vars": ["1-NA_deg", "1-NA_mm", "1-NB_deg", "1-NB_mm", "H-Winkel"]},
    {"name": "Weichgewebe",        "z": 14, "color": "rgba(255,140,0,0.18)",   "line": "#CC7000",
     "vars": ["Nasolabialwinkel", "Z-Winkel"]},
    {"name": "Modell",             "z": 21, "color": "rgba(147,112,219,0.18)", "line": "#6A0DAD",
     "vars": ["HZB", "VZB", "Eckzahn-OK", "Pont-SI-OK", "Pont-SI-UK"]},
]

# ---------------------------------------------------------------------------
# Sidebar: Treibervariablen
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Schritt 1 – Treibervariablen")
    sna  = st.number_input("SNA (°)", min_value=62.0, max_value=103.0, value=82.0, step=0.5)
    pgnb = st.number_input("PgNB – Kinnprominenz (mm)", min_value=0.0, max_value=10.0, value=2.3, step=0.1)

    with st.expander("Zahnbogen-Konstanten (Bernabe)"):
        st.markdown("**HZB** = a + b·SNA + b·SNB")
        a_hzb = st.number_input("HZB – a", min_value=30.0, max_value=40.0, value=35.0, step=0.5, key="a_hzb")
        b_hzb = st.number_input("HZB – b", min_value=0.10, max_value=0.20, value=0.20, step=0.01, key="b_hzb", format="%.2f")
        st.markdown("**VZB** = a + b·SNA + b·SNB")
        a_vzb = st.number_input("VZB – a", min_value=20.0, max_value=25.0, value=22.5, step=0.5, key="a_vzb")
        b_vzb = st.number_input("VZB – b", min_value=0.10, max_value=0.20, value=0.20, step=0.01, key="b_vzb", format="%.2f")

    st.divider()
    with st.expander("Paddenberg Floating Norms"):
        pdb_aktiv = st.checkbox("Paddenberg-Analyse aktivieren", value=False)
        if pdb_aktiv:
            sn_occl       = st.number_input("SN-Occl (°)",          value=14.5, step=0.5)
            wits_gemessen = st.number_input("Wits gemessen (mm)",    value=0.0,  step=0.5)
            ml_nsl_pdb    = st.number_input("ML-NSL für Paddenberg (°)", value=32.0, step=0.5)
            st.markdown("_Optional – für erweitertes Modell B/D:_")
            idx_hasund    = st.number_input("Index Hasund (%)",       value=0.0,  step=1.0)
            fazialachse   = st.number_input("Fazialachse Ricketts (°)", value=0.0, step=0.5)
            geschlecht    = st.radio("Geschlecht", ["weiblich", "männlich"], horizontal=True)

    st.divider()
    st.markdown("**Legende**")
    for status, farbe in STATUS_FARBE.items():
        st.markdown(
            f'<span style="background:{farbe};padding:2px 8px;border-radius:4px;color:white">'
            f'{STATUS_LABEL[status]}</span>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Idealwerte berechnen
# ---------------------------------------------------------------------------
ideal = compute_ideal(sna, pgnb_mm=pgnb, a_hzb=a_hzb, b_hzb=b_hzb, a_vzb=a_vzb, b_vzb=b_vzb)

# ---------------------------------------------------------------------------
# Patientenmesswerte eingeben (Schritt 2)
# ---------------------------------------------------------------------------
st.subheader("Schritt 2 – Patientenmesswerte")
st.caption("Felder sind mit Idealwerten vorbelegt – bitte tatsächliche Messwerte eintragen.")

gemessen: dict[str, float] = {}
cols_inp = st.columns(len(GRUPPEN))
for col, (gruppe, variablen) in zip(cols_inp, GRUPPEN.items()):
    with col:
        st.markdown(f"**{gruppe}**")
        for var in variablen:
            einh = EINHEITEN.get(var, "")
            gemessen[var] = st.number_input(
                f"{var} ({einh})", value=float(ideal[var]),
                step=0.5, key=f"inp_{var}", format="%.2f",
            )

# Paddenberg-Zusatzvariablen
pdb_erg = None
if pdb_aktiv:
    anb_g = gemessen.get("ANB", ideal["ANB"])
    nsba_g = gemessen.get("NSBa", ideal["NSBa"])
    nl_nsl_g = gemessen.get("NL-NSL", ideal["NL-NSL"])
    idx_h = idx_hasund if idx_hasund > 0 else None
    faz   = fazialachse if fazialachse > 0 else None
    pdb_erg = paddenberg_analyse(
        sna=sna, anb_gemessen=anb_g, ml_nsl=ml_nsl_pdb,
        sn_occl=sn_occl, nsba=nsba_g, nl_nsl=nl_nsl_g,
        index_hasund=idx_h, fazialachse=faz,
    )

st.divider()

# ---------------------------------------------------------------------------
# Abweichungsanalyse zusammenbauen
# ---------------------------------------------------------------------------
abweichungen = compute_abweichungen(gemessen, ideal)

# Paddenberg-Variablen anhängen
if pdb_erg is not None:
    wits_sd = 2.0 if geschlecht == "weiblich" else 2.0
    abweichungen.append(Abweichung(
        variable="ANB (Paddenberg)",
        ideal=pdb_erg.anb_ideal_A,
        gemessen=gemessen.get("ANB", ideal["ANB"]),
        sd=2.0,
        einheit="°",
    ))
    abweichungen.append(Abweichung(
        variable="Wits",
        ideal=pdb_erg.wits_ideal_C,
        gemessen=wits_gemessen,
        sd=wits_sd,
        einheit="mm",
    ))

df_abw = pd.DataFrame([{
    "Variable":  a.variable,
    "Ideal":     a.ideal,
    "Gemessen":  a.gemessen,
    "Δ absolut": round(a.delta, 2),
    "Δ in SD":   round(a.delta_sd, 2),
    "Einheit":   a.einheit,
    "Status":    a.status,
    "Farbe":     STATUS_FARBE[a.status],
} for a in abweichungen])

# Paddenberg-Variablen der FRS-Kategorie hinzufügen
kategorien_3d = [dict(k, vars=list(k["vars"])) for k in KATEGORIEN_3D_BASIS]
if pdb_erg is not None:
    kategorien_3d[0]["vars"] = ["ANB (Paddenberg)", "Wits"] + kategorien_3d[0]["vars"]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Abweichungsübersicht", "🧊 3D Harmonieraum", "📋 Wertetabelle", "📐 Formeln"]
)

# ── Tab 1: Abweichungsbalken ────────────────────────────────────────────────
with tab1:
    st.subheader("Abweichung vom individuellen Ideal (in SD-Einheiten)")
    st.caption("Grün ≤ 1 SD  |  Orange 1–2 SD  |  Rot > 2 SD")

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_abw["Δ in SD"],
        y=df_abw["Variable"],
        orientation="h",
        marker_color=df_abw["Farbe"].tolist(),
        text=[
            f"{r['Δ absolut']:+.2f} {r['Einheit']}  ({r['Δ in SD']:+.2f} SD)"
            for _, r in df_abw.iterrows()
        ],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>Ideal: %{customdata[0]:.2f}  "
            "Gemessen: %{customdata[1]:.2f}<br>"
            "Δ: %{customdata[2]:+.2f} SD<extra></extra>"
        ),
        customdata=df_abw[["Ideal", "Gemessen", "Δ in SD"]].values,
    ))
    for xv, dash, col in [(-2,"dash","red"),(-1,"dot","gray"),(1,"dot","gray"),(2,"dash","red")]:
        fig_bar.add_vline(x=xv, line_dash=dash, line_color=col, line_width=1.5)
    fig_bar.add_vrect(x0=-1, x1=1, fillcolor="#2ecc71", opacity=0.07, line_width=0)
    fig_bar.add_vrect(x0=-2, x1=-1, fillcolor="#f39c12", opacity=0.07, line_width=0)
    fig_bar.add_vrect(x0= 1, x1= 2, fillcolor="#f39c12", opacity=0.07, line_width=0)
    fig_bar.update_layout(
        height=max(400, len(abweichungen) * 28),
        xaxis=dict(range=[-4, 4], zeroline=True, zerolinewidth=2, zerolinecolor="black",
                   title="Abweichung in SD"),
        yaxis_title="",
        plot_bgcolor="white",
        margin=dict(l=10, r=200, t=30, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Im Normbereich (≤1 SD)", (df_abw["Status"] == "ok").sum())
    c2.metric("Grenzbereich (1–2 SD)",  (df_abw["Status"] == "grenz").sum())
    c3.metric("Auffällig (>2 SD)",      (df_abw["Status"] == "auffällig").sum())

# ── Tab 2: 3D Harmonieraum ──────────────────────────────────────────────────
with tab2:
    st.subheader("3D Harmonieraum – 4 Messflächen")
    st.caption(
        "Jede horizontale Fläche = eine Messkategorie. "
        "Y-Achse = Abweichung in SD. "
        "Die **Harmonielinie** (Y=0) verbindet alle Flächen – "
        "bei perfekter Harmonie liegen alle Punkte exakt darauf."
    )

    fig3d = go.Figure()
    y_lim = 3.8

    harmony_x, harmony_y, harmony_z = [], [], []  # für die Harmonielinie

    for kat in kategorien_3d:
        z_cat = float(kat["z"])
        vars_kat = kat["vars"]
        n = len(vars_kat)
        color_surf = kat["color"]
        color_line = kat["line"]

        # ── Transparente Fläche ──────────────────────────────────────────
        xp = np.linspace(-0.6, n - 0.4, 12)
        yp = np.linspace(-y_lim, y_lim, 12)
        Xp, Yp = np.meshgrid(xp, yp)
        Zp = np.full_like(Xp, z_cat)
        fig3d.add_trace(go.Surface(
            x=Xp, y=Yp, z=Zp,
            colorscale=[[0, color_surf], [1, color_surf]],
            showscale=False,
            hoverinfo="skip",
            name=kat["name"],
        ))

        # ── Y=0-Harmonielinie auf der Fläche ─────────────────────────────
        fig3d.add_trace(go.Scatter3d(
            x=[-0.6, n - 0.4], y=[0, 0], z=[z_cat, z_cat],
            mode="lines",
            line=dict(color=color_line, width=5),
            showlegend=False, hoverinfo="skip",
        ))

        # ── Kategorie-Label ───────────────────────────────────────────────
        fig3d.add_trace(go.Scatter3d(
            x=[(n - 1) / 2], y=[-y_lim - 0.6], z=[z_cat],
            mode="text",
            text=[f"<b>{kat['name']}</b>"],
            textfont=dict(size=11, color=color_line),
            showlegend=False, hoverinfo="skip",
        ))

        # ── Variablen-Marker, Stäbe, Labels ──────────────────────────────
        for i, var in enumerate(vars_kat):
            rows = df_abw[df_abw["Variable"] == var]
            if rows.empty:
                # Platzhalter-Punkt bei unbekannter Variable
                harmony_x.append(i); harmony_y.append(0); harmony_z.append(z_cat)
                continue

            row = rows.iloc[0]
            dsd     = float(row["Δ in SD"])
            status  = row["Status"]
            pt_col  = STATUS_FARBE[status]
            ideal_v = float(row["Ideal"])
            mess_v  = float(row["Gemessen"])
            einh    = row["Einheit"]

            # Stab von Y=0 zum Patientenpunkt
            fig3d.add_trace(go.Scatter3d(
                x=[i, i], y=[0, dsd], z=[z_cat, z_cat],
                mode="lines",
                line=dict(color=pt_col, width=5),
                showlegend=False, hoverinfo="skip",
            ))

            # Marker
            fig3d.add_trace(go.Scatter3d(
                x=[i], y=[dsd], z=[z_cat],
                mode="markers",
                marker=dict(size=10, color=pt_col,
                            line=dict(color="white", width=1)),
                name=var,
                hovertemplate=(
                    f"<b>{var}</b><br>"
                    f"Ideal:    {ideal_v:.2f} {einh}<br>"
                    f"Gemessen: {mess_v:.2f} {einh}<br>"
                    f"Δ: {dsd:+.2f} SD  [{STATUS_LABEL[status]}]"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

            # Variablenname über dem Marker
            offset = 0.35 if dsd >= 0 else -0.35
            fig3d.add_trace(go.Scatter3d(
                x=[i], y=[dsd + offset], z=[z_cat + 0.6],
                mode="text",
                text=[var],
                textfont=dict(size=8, color=pt_col),
                showlegend=False, hoverinfo="skip",
            ))

            # Pfade für Harmonielinie und Patientenlinie
            harmony_x.append(i); harmony_y.append(0);   harmony_z.append(z_cat)

        # Segment-Trenner
        harmony_x.append(None); harmony_y.append(None); harmony_z.append(None)

    # ── Harmonielinie (Y=0 durch alle Flächen) ───────────────────────────
    fig3d.add_trace(go.Scatter3d(
        x=harmony_x, y=harmony_y, z=harmony_z,
        mode="lines+markers",
        line=dict(color="navy", width=3, dash="dot"),
        marker=dict(size=4, color="navy"),
        name="Harmonielinie (Ideal Y=0)",
    ))

    # ── Verbindungslinien zwischen den Flächen (Y=0-Spine) ───────────────
    for i in range(len(kategorien_3d) - 1):
        k1, k2 = kategorien_3d[i], kategorien_3d[i + 1]
        m1 = (len(k1["vars"]) - 1) / 2
        m2 = (len(k2["vars"]) - 1) / 2
        fig3d.add_trace(go.Scatter3d(
            x=[m1, m2], y=[0, 0], z=[k1["z"], k2["z"]],
            mode="lines",
            line=dict(color="navy", width=2, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))

    # ── ±1 SD und ±2 SD Referenzebenen (halbtransparent) ─────────────────
    for y_ref, col_ref, opacity in [(1, "#2ecc71", 0.04), (2, "#e74c3c", 0.04),
                                     (-1, "#2ecc71", 0.04), (-2, "#e74c3c", 0.04)]:
        max_z = kategorien_3d[-1]["z"]
        xr = np.array([-1, len(kategorien_3d[0]["vars"])])
        zr = np.array([0.0, float(max_z)])
        Xr, Zr = np.meshgrid(xr, zr)
        Yr = np.full_like(Xr, float(y_ref))
        fig3d.add_trace(go.Surface(
            x=Xr, y=Yr, z=Zr,
            colorscale=[[0, col_ref], [1, col_ref]],
            opacity=opacity,
            showscale=False,
            hoverinfo="skip",
            showlegend=False,
        ))

    fig3d.update_layout(
        height=720,
        scene=dict(
            xaxis=dict(title="Variable", showticklabels=False, showgrid=False),
            yaxis=dict(title="Abweichung (SD)", range=[-y_lim, y_lim]),
            zaxis=dict(
                title="Kategorie",
                tickvals=[k["z"] for k in kategorien_3d],
                ticktext=[k["name"] for k in kategorien_3d],
            ),
            camera=dict(eye=dict(x=2.2, y=-2.0, z=1.4)),
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    if pdb_erg:
        st.info(
            f"**Paddenberg:** "
            f"ANB ideal (Modell A) = **{pdb_erg.anb_ideal_A}°** | "
            f"Wits ideal (Järvinen) = **{pdb_erg.wits_ideal_C} mm**"
            + (f" | ANB ideal (Modell B) = {pdb_erg.anb_ideal_B}°" if pdb_erg.anb_ideal_B else "")
        )

# ── Tab 3: Wertetabelle ────────────────────────────────────────────────────
with tab3:
    st.subheader("Wertetabelle – Ideal vs. Gemessen")
    disp = df_abw[["Variable", "Einheit", "Ideal", "Gemessen", "Δ absolut", "Δ in SD", "Status"]].copy()
    disp["Δ in SD"]   = disp["Δ in SD"].map(lambda x: f"{x:+.2f}")
    disp["Δ absolut"] = disp["Δ absolut"].map(lambda x: f"{x:+.2f}")

    def hl(row):
        f = STATUS_FARBE.get(row["Status"], "white")
        return [f"background-color:{f}22" if c == "Status" else "" for c in row.index]

    st.dataframe(disp.style.apply(hl, axis=1), use_container_width=True, hide_index=True)
    st.download_button("⬇ Als CSV herunterladen",
                       data=df_abw.to_csv(index=False).encode("utf-8"),
                       file_name="harmonie_analyse.csv", mime="text/csv")

# ── Tab 4: Formeln ──────────────────────────────────────────────────────────
with tab4:
    col_f1, col_f2 = st.columns(2)
    st.subheader("Segner/Hasund")
    items = list(FORMELN.items())
    for i, (name, formel) in enumerate(items):
        col = col_f1 if i < len(items) // 2 else col_f2
        sd  = STANDARD_ABWEICHUNGEN.get(name, "–")
        col.markdown(f"**{name}** = `{formel}`  \n_SD = ±{sd} {EINHEITEN.get(name, '')}_")

    st.divider()
    st.subheader("Paddenberg et al. 2021 – Floating Norms")
    st.markdown("""
| Modell | Formel | R² |
|--------|--------|----|
| ANB (Modell A) | −45.359 + 0.493·SNA + 0.251·ML-NSL | 0.578 |
| ANB (Modell B) | −41.669 + 0.567·SNA + 0.110·ML-NSL + 0.114·NSBa + 0.132·NL-NSL + 0.062·Index − 0.289·Fazialachse | 0.690 |
| Wits (Modell C ★) | 57.510 + 1.526·ANB − 0.634·SNA − 0.666·SN-Occl | 0.976 |
| Wits (Modell D) | 57.853 + 1.572·ANB − 0.664·SNA − 0.639·SN-Occl − 0.030·ML-NSL + 0.030·Index | 0.984 |
""")
