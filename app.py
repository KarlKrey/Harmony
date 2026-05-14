"""
3D Harmonic Korrelation nach Prof. Krey
========================================
Workflow:
  1. SNA + PgNB → individuelles Ideal berechnen
  2. Patientenmesswerte eingeben
  3. Abweichungen visualisieren:
     - Balkendiagramm (alle Variablen, SD-normiert)
     - Prisma-Harmonietabelle (Tabelle im Raum gefaltet, verschiebbare Linie)
     - 3D Kategorieflächen
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
    ABGELEITET,
    EINHEITEN,
    STANDARD_ABWEICHUNGEN,
    FORMELN,
)
from paddenberg_floating_norms import paddenberg_analyse

# ---------------------------------------------------------------------------
st.set_page_config(page_title="3D Harmonic Korrelation", page_icon="🦷", layout="wide")
st.title("3D Harmonic Korrelation nach Prof. Krey")
st.caption("Individualisierte kephalometrische Strukturanalyse nach Segner/Hasund & Paddenberg")

STATUS_FARBE = {"ok": "#2ecc71", "grenz": "#f39c12", "auffällig": "#e74c3c"}
STATUS_LABEL = {"ok": "≤ 1 SD", "grenz": "1–2 SD", "auffällig": "> 2 SD"}

# Prisma-Fächenstruktur (4 Seiten)
PRISM_FACES = [
    {"name": "FRS (Fernröntgen)",  "y":  0, "color": "rgba(65,105,225,0.18)",  "line": "#6495ED",
     "vars": ["SNB", "ANB", "NL-NSL", "NSBa", "ML-NSL", "ML-NL"]},
    {"name": "Dentale Variablen", "y": 10, "color": "rgba(46,139,87,0.18)",   "line": "#3CB371",
     "vars": ["1-NA_deg", "1-NA_mm", "1-NB_deg", "1-NB_mm", "H-Winkel"]},
    {"name": "Weichteil",          "y": 20, "color": "rgba(204,112,0,0.18)",   "line": "#FFA500",
     "vars": ["Nasolabialwinkel", "Z-Winkel"]},
    {"name": "Modell",             "y": 30, "color": "rgba(106,13,173,0.18)",  "line": "#9370DB",
     "vars": ["HZB", "VZB", "Eckzahn-OK", "Pont-SI-OK", "Pont-SI-UK"]},
]

KATEGORIEN_3D = [
    {"name": "FRS (Fernröntgen)", "z":  0, "color": "rgba(100,149,237,0.18)", "line": "#4169E1",
     "vars": ["SNB", "ANB", "NL-NSL", "NSBa", "ML-NSL", "ML-NL"]},
    {"name": "Dentale Variablen","z":  7, "color": "rgba(60,179,113,0.18)",  "line": "#2E8B57",
     "vars": ["1-NA_deg", "1-NA_mm", "1-NB_deg", "1-NB_mm", "H-Winkel"]},
    {"name": "Weichteil",        "z": 14, "color": "rgba(255,140,0,0.18)",   "line": "#CC7000",
     "vars": ["Nasolabialwinkel", "Z-Winkel"]},
    {"name": "Modell",           "z": 21, "color": "rgba(147,112,219,0.18)", "line": "#6A0DAD",
     "vars": ["HZB", "VZB", "Eckzahn-OK", "Pont-SI-OK", "Pont-SI-UK"]},
]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Schritt 1 – Treibervariablen")
    sna  = st.number_input("SNA (°)", min_value=62.0, max_value=103.0, value=82.0, step=0.5)
    pgnb = st.number_input("PgNB (mm)", min_value=0.0, max_value=10.0, value=2.3, step=0.1)

    with st.expander("Zahnbogen-Konstanten (Bernabe)"):
        st.markdown("**HZB** = a + b·SNA + b·SNB  |  **Eckzahn OK** = 20 + b·SNA")
        a_hzb = st.number_input("HZB – a", 25.0, 40.0, 30.0, 0.5, key="a_hzb")
        b_hzb = st.number_input("b (HZB, VZB, Eckzahn)", 0.05, 0.25, 0.10, 0.01, key="b_hzb", format="%.2f")
        st.markdown("**VZB** = a + b·SNA + b·SNB")
        a_vzb = st.number_input("VZB – a", 15.0, 25.0, 20.0, 0.5, key="a_vzb")

    st.divider()
    with st.expander("Paddenberg Floating Norms"):
        pdb_aktiv = st.checkbox("Paddenberg aktivieren", value=False)
        sn_occl = ml_nsl_pdb = wits_gemessen = idx_hasund = fazialachse = 0.0
        geschlecht = "weiblich"
        if pdb_aktiv:
            sn_occl       = st.number_input("SN-Occl (°)", value=14.5, step=0.5)
            wits_gemessen = st.number_input("Wits gemessen (mm)", value=0.0, step=0.5)
            ml_nsl_pdb    = st.number_input("ML-NSL (°)", value=32.0, step=0.5)
            idx_hasund    = st.number_input("Index Hasund (%, 0=n.v.)", value=0.0, step=1.0)
            fazialachse   = st.number_input("Fazialachse (°, 0=n.v.)", value=0.0, step=0.5)
            geschlecht    = st.radio("Geschlecht", ["weiblich", "männlich"], horizontal=True)

    st.divider()
    for status, farbe in STATUS_FARBE.items():
        st.markdown(
            f'<span style="background:{farbe};padding:2px 8px;border-radius:4px;color:white">'
            f'{STATUS_LABEL[status]}</span>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Ideal & Messwerte
# ---------------------------------------------------------------------------
# b_vzb = b_hzb (gleicher Koeffizient)
b_vzb = b_hzb

ideal = compute_ideal(sna, pgnb_mm=pgnb, a_hzb=a_hzb, b_hzb=b_hzb, a_vzb=a_vzb, b_vzb=b_vzb)

st.subheader("Schritt 2 – Patientenmesswerte")
st.caption("ANB und ML-NL werden automatisch berechnet und müssen nicht eingegeben werden.")

gemessen: dict[str, float] = {}
for col_ui, (gruppe, variablen) in zip(st.columns(len(GRUPPEN)), GRUPPEN.items()):
    with col_ui:
        st.markdown(f"**{gruppe}**")
        for var in variablen:
            gemessen[var] = st.number_input(
                f"{var} ({EINHEITEN.get(var,'')})",
                value=float(ideal[var]), step=0.5, key=f"inp_{var}", format="%.2f",
            )

# Abgeleitete Variablen berechnen
gemessen["ANB"]   = sna - gemessen["SNB"]
gemessen["ML-NL"] = gemessen["ML-NSL"] - gemessen["NL-NSL"]

# Anzeige der berechneten Werte
anb_ideal_val   = ideal["ANB"]
mlnl_ideal_val  = ideal["ML-NL"]
anb_dev   = (gemessen["ANB"]   - anb_ideal_val)  / STANDARD_ABWEICHUNGEN["ANB"]
mlnl_dev  = (gemessen["ML-NL"] - mlnl_ideal_val) / STANDARD_ABWEICHUNGEN["ML-NL"]

col_anb, col_mlnl = st.columns(2)
with col_anb:
    st.info(
        f"**ANB** = SNA ({sna:.1f}°) − SNB ({gemessen['SNB']:.1f}°) "
        f"= **{gemessen['ANB']:.1f}°**  ·  Ideal: {anb_ideal_val:.1f}°  ·  "
        f"Δ {anb_dev:+.2f} SD"
    )
with col_mlnl:
    st.info(
        f"**ML-NL** = ML-NSL ({gemessen['ML-NSL']:.1f}°) − NL-NSL ({gemessen['NL-NSL']:.1f}°) "
        f"= **{gemessen['ML-NL']:.1f}°**  ·  Ideal: {mlnl_ideal_val:.1f}°  ·  "
        f"Δ {mlnl_dev:+.2f} SD"
    )

pdb_erg = None
if pdb_aktiv:
    pdb_erg = paddenberg_analyse(
        sna=sna, anb_gemessen=gemessen.get("ANB", ideal["ANB"]),
        ml_nsl=ml_nsl_pdb, sn_occl=sn_occl,
        nsba=gemessen.get("NSBa", ideal["NSBa"]),
        nl_nsl=gemessen.get("NL-NSL", ideal["NL-NSL"]),
        index_hasund=idx_hasund if idx_hasund > 0 else None,
        fazialachse=fazialachse if fazialachse > 0 else None,
    )

st.divider()

# ---------------------------------------------------------------------------
# Abweichungsanalyse
# ---------------------------------------------------------------------------
abweichungen = compute_abweichungen(gemessen, ideal)
if pdb_erg is not None:
    abweichungen += [
        Abweichung("ANB (Paddenberg)", pdb_erg.anb_ideal_A,
                   gemessen.get("ANB", ideal["ANB"]), 2.0, "°"),
        Abweichung("Wits", pdb_erg.wits_ideal_C, wits_gemessen, 2.0, "mm"),
    ]

df_abw = pd.DataFrame([{
    "Variable": a.variable, "Ideal": a.ideal, "Gemessen": a.gemessen,
    "Δ absolut": round(a.delta, 2), "Δ in SD": round(a.delta_sd, 2),
    "Einheit": a.einheit, "Status": a.status, "Farbe": STATUS_FARBE[a.status],
} for a in abweichungen])

kategorien_3d = [dict(k, vars=list(k["vars"])) for k in KATEGORIEN_3D]
if pdb_erg is not None:
    kategorien_3d[0]["vars"] = ["ANB (Paddenberg)", "Wits"] + kategorien_3d[0]["vars"]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Abweichungsübersicht",
    "📐 Prisma-Harmonietabelle",
    "🧊 3D Kategorieflächen",
    "📋 Wertetabelle",
    "🔬 Formeln",
])

# ── Tab 1: Balkendiagramm ───────────────────────────────────────────────────
with tab1:
    st.subheader("Abweichung vom individuellen Ideal (SD-Einheiten)")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_abw["Δ in SD"], y=df_abw["Variable"], orientation="h",
        marker_color=df_abw["Farbe"].tolist(),
        text=[f"{r['Δ absolut']:+.2f} {r['Einheit']}  ({r['Δ in SD']:+.2f} SD)"
              for _, r in df_abw.iterrows()],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Ideal: %{customdata[0]:.2f}  "
                      "Gemessen: %{customdata[1]:.2f}<br>Δ: %{customdata[2]:+.2f} SD<extra></extra>",
        customdata=df_abw[["Ideal", "Gemessen", "Δ in SD"]].values,
    ))
    for xv, dash, col in [(-2,"dash","red"),(-1,"dot","#aaa"),(1,"dot","#aaa"),(2,"dash","red")]:
        fig_bar.add_vline(x=xv, line_dash=dash, line_color=col, line_width=1.5)
    fig_bar.add_vrect(x0=-1, x1=1, fillcolor="#2ecc71", opacity=0.07, line_width=0)
    fig_bar.add_vrect(x0=-2, x1=-1, fillcolor="#f39c12", opacity=0.07, line_width=0)
    fig_bar.add_vrect(x0= 1, x1= 2, fillcolor="#f39c12", opacity=0.07, line_width=0)
    fig_bar.update_layout(
        height=max(380, len(abweichungen) * 28),
        xaxis=dict(range=[-4, 4], zeroline=True, zerolinewidth=2,
                   zerolinecolor="black", title="Abweichung in SD"),
        plot_bgcolor="white", margin=dict(l=10, r=200, t=30, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Im Normbereich (≤1 SD)", (df_abw["Status"] == "ok").sum())
    c2.metric("Grenzbereich (1–2 SD)",  (df_abw["Status"] == "grenz").sum())
    c3.metric("Auffällig (>2 SD)",      (df_abw["Status"] == "auffällig").sum())

# ── Tab 2: Prisma-Harmonietabelle ───────────────────────────────────────────
with tab2:
    st.subheader("Prisma-Harmonietabelle – Harmoniewerte im Raum")
    st.caption(
        "Jede Fläche des Prismas = eine Messkategorie. "
        "Auf jeder Fläche stehen die **gerundeten Idealwerte** für SNA 62–103. "
        "**Goldener Streifen** = verschiebbare Linie. "
        "**Farbige Zahlen** = Patientenmesswerte (grün/orange/rot nach Abweichung)."
    )

    col_sl, col_info = st.columns([3, 1])
    with col_sl:
        sna_cursor = st.slider(
            "↕ Verschiebbare Linie (SNA)", 62, 103, int(round(sna)), key="prism_slider"
        )
    with col_info:
        st.markdown(f"**SNA = {sna_cursor}°**")
        cursor_ideal = compute_ideal(float(sna_cursor), pgnb_mm=pgnb,
                                      a_hzb=a_hzb, b_hzb=b_hzb, a_vzb=a_vzb, b_vzb=b_vzb)

    # Vollständige Harmonietabelle vorberechnen
    sna_range = list(range(62, 104))
    n_sna = len(sna_range)
    ideals_all = {
        s: compute_ideal(float(s), pgnb_mm=pgnb, a_hzb=a_hzb, b_hzb=b_hzb, a_vzb=a_vzb, b_vzb=b_vzb)
        for s in sna_range
    }
    cursor_idx = sna_cursor - 62   # Z-Index des Cursors
    pat_idx    = int(round(sna)) - 62  # Z-Index des Patienten

    fig_p = go.Figure()

    for face in PRISM_FACES:
        y_f   = float(face["y"])
        vars_f = face["vars"]
        n_v   = len(vars_f)
        col_f = face["color"]
        col_l = face["line"]

        # ── Hintergrundfläche (senkrecht, Z=SNA-Achse) ───────────────────
        xf = np.array([-0.6, n_v - 0.4])
        zf = np.array([-0.5, n_sna - 0.5])
        Xf, Zf = np.meshgrid(xf, zf)
        Yf = np.full_like(Xf, y_f)
        fig_p.add_trace(go.Surface(
            x=Xf, y=Yf, z=Zf,
            colorscale=[[0, col_f], [1, col_f]],
            showscale=False, hoverinfo="skip", showlegend=False,
        ))

        # ── Kategorie-Titel (über der Fläche) ────────────────────────────
        fig_p.add_trace(go.Scatter3d(
            x=[n_v / 2 - 0.5], y=[y_f], z=[n_sna + 0.5],
            mode="text", text=[f"<b>{face['name']}</b>"],
            textfont=dict(size=13, color=col_l),
            showlegend=False, hoverinfo="skip",
        ))

        # ── Variablennamen (Spaltenköpfe) ─────────────────────────────────
        for j, var in enumerate(vars_f):
            short = var.replace("_deg", "°").replace("_mm", "mm").replace("Nasolabialwinkel", "Nasol.")
            fig_p.add_trace(go.Scatter3d(
                x=[j], y=[y_f], z=[n_sna - 0.2],
                mode="text", text=[short],
                textfont=dict(size=9, color=col_l),
                showlegend=False, hoverinfo="skip",
            ))

        # ── SNA-Beschriftung links (jede 5. Zeile) ───────────────────────
        sna_lbl_x, sna_lbl_y, sna_lbl_z, sna_lbl_t = [], [], [], []
        for i, s in enumerate(sna_range):
            if s % 5 == 0 or s == 62 or s == 103:
                sna_lbl_x.append(-1.0)
                sna_lbl_y.append(y_f)
                sna_lbl_z.append(float(i))
                sna_lbl_t.append(str(s))
        fig_p.add_trace(go.Scatter3d(
            x=sna_lbl_x, y=sna_lbl_y, z=sna_lbl_z,
            mode="text", text=sna_lbl_t,
            textfont=dict(size=8, color="#aaa"),
            showlegend=False, hoverinfo="skip",
        ))

        # ── Idealwerte (alle Zeilen außer Cursor und Patient) ─────────────
        norm_x, norm_y, norm_z, norm_t = [], [], [], []
        for i, s in enumerate(sna_range):
            if i == cursor_idx or i == pat_idx:
                continue
            ideal_i = ideals_all[s]
            for j, var in enumerate(vars_f):
                norm_x.append(j)
                norm_y.append(y_f)
                norm_z.append(float(i))
                norm_t.append(str(round(ideal_i.get(var, 0))))
        if norm_t:
            fig_p.add_trace(go.Scatter3d(
                x=norm_x, y=norm_y, z=norm_z,
                mode="text", text=norm_t,
                textfont=dict(size=8, color="white"),
                showlegend=False, hoverinfo="skip",
            ))

        # ── Cursor-Streifen (goldene Linie + Idealwerte in diesem SNA) ───
        # Goldene Linie
        cx = np.linspace(-0.6, n_v - 0.4, n_v * 4)
        fig_p.add_trace(go.Scatter3d(
            x=cx, y=[y_f] * len(cx), z=[float(cursor_idx)] * len(cx),
            mode="lines", line=dict(color="gold", width=5),
            showlegend=False, hoverinfo="skip",
        ))
        # Idealwerte am Cursor
        cur_id = ideals_all[sna_cursor]
        for j, var in enumerate(vars_f):
            fig_p.add_trace(go.Scatter3d(
                x=[j], y=[y_f], z=[float(cursor_idx)],
                mode="text", text=[str(round(cur_id.get(var, 0)))],
                textfont=dict(size=10, color="gold"),
                showlegend=False, hoverinfo="skip",
            ))

        # ── Patient-Zeile (Messwerte farbig, falls pat_idx ≠ cursor_idx) ─
        if pat_idx != cursor_idx:
            pat_ideal = ideals_all[sna_range[min(pat_idx, n_sna - 1)]]
            # Patienten-Streifen (navy)
            fig_p.add_trace(go.Scatter3d(
                x=cx, y=[y_f] * len(cx), z=[float(pat_idx)] * len(cx),
                mode="lines", line=dict(color="#4444ff", width=3, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))
        else:
            pat_ideal = cur_id

        # Patientenmesswerte auf dem Patienten-Z
        for j, var in enumerate(vars_f):
            if var not in gemessen or var not in STANDARD_ABWEICHUNGEN:
                continue
            meas = gemessen[var]
            dev  = (meas - pat_ideal.get(var, meas)) / STANDARD_ABWEICHUNGEN[var]
            status_p = ("auffällig" if abs(dev) > 2 else "grenz" if abs(dev) > 1 else "ok")
            fig_p.add_trace(go.Scatter3d(
                x=[j], y=[y_f], z=[float(pat_idx)],
                mode="text", text=[f"[{round(meas)}]"],
                textfont=dict(size=10, color=STATUS_FARBE[status_p]),
                hovertemplate=(
                    f"<b>{var}</b><br>"
                    f"Ideal: {pat_ideal.get(var, 0):.1f}<br>"
                    f"Gemessen: {meas:.1f}<br>"
                    f"Δ: {dev:+.2f} SD<extra></extra>"
                ),
                showlegend=False,
            ))

    # ── Achsen-Layout ─────────────────────────────────────────────────────
    fig_p.update_layout(
        height=760,
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       showbackground=False, title=""),
            yaxis=dict(
                tickvals=[f["y"] for f in PRISM_FACES],
                ticktext=[f["name"] for f in PRISM_FACES],
                title="", gridcolor="#333", showbackground=False,
            ),
            zaxis=dict(
                tickvals=list(range(0, n_sna, 5)),
                ticktext=[str(s) for s in sna_range[::5]],
                title="SNA (°)", gridcolor="#333", showbackground=False,
            ),
            bgcolor="rgb(12, 12, 20)",
            camera=dict(eye=dict(x=2.2, y=-2.5, z=0.6)),
        ),
        paper_bgcolor="rgb(12, 12, 20)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_p, use_container_width=True)

    # Cursor-Werte als Tabelle unter dem Plot
    st.markdown(f"**Idealwerte bei SNA = {sna_cursor}°** (Cursor-Position):")
    rows_cur = []
    for face in PRISM_FACES:
        for var in face["vars"]:
            val_ideal = cursor_ideal.get(var, 0)
            val_meas  = gemessen.get(var, None)
            dev_str   = ""
            if val_meas is not None and var in STANDARD_ABWEICHUNGEN:
                dev = (val_meas - val_ideal) / STANDARD_ABWEICHUNGEN[var]
                dev_str = f"{dev:+.2f} SD"
            rows_cur.append({
                "Kategorie": face["name"], "Variable": var,
                "Einheit":   EINHEITEN.get(var, ""),
                "Ideal":     round(val_ideal, 1),
                "Gemessen":  round(val_meas, 1) if val_meas is not None else "–",
                "Δ SD":      dev_str,
            })
    st.dataframe(pd.DataFrame(rows_cur), use_container_width=True, hide_index=True)

# ── Tab 3: 3D Kategorieflächen ──────────────────────────────────────────────
with tab3:
    st.subheader("3D Harmonieraum – 4 Messflächen")
    st.caption("Horizontale Ebenen = Kategorien. Y = Abweichung in SD. Harmonielinie bei Y=0.")

    fig3d = go.Figure()
    y_lim = 3.8
    h_x, h_y, h_z = [], [], []

    for kat in kategorien_3d:
        z_cat = float(kat["z"])
        vars_k = kat["vars"]
        n = len(vars_k)
        xp = np.linspace(-0.6, n - 0.4, 10)
        yp = np.linspace(-y_lim, y_lim, 10)
        Xp, Yp = np.meshgrid(xp, yp)
        Zp = np.full_like(Xp, z_cat)
        fig3d.add_trace(go.Surface(x=Xp, y=Yp, z=Zp,
            colorscale=[[0, kat["color"]], [1, kat["color"]]],
            showscale=False, hoverinfo="skip", name=kat["name"]))
        fig3d.add_trace(go.Scatter3d(
            x=[-0.6, n-0.4], y=[0, 0], z=[z_cat, z_cat],
            mode="lines", line=dict(color=kat["line"], width=5),
            showlegend=False, hoverinfo="skip"))
        fig3d.add_trace(go.Scatter3d(
            x=[(n-1)/2], y=[-y_lim-0.8], z=[z_cat],
            mode="text", text=[f"<b>{kat['name']}</b>"],
            textfont=dict(size=11, color=kat["line"]),
            showlegend=False, hoverinfo="skip"))
        for i, var in enumerate(vars_k):
            rows = df_abw[df_abw["Variable"] == var]
            if rows.empty:
                h_x.append(i); h_y.append(0); h_z.append(z_cat)
                continue
            row = rows.iloc[0]
            dsd, status = float(row["Δ in SD"]), row["Status"]
            pt_col = STATUS_FARBE[status]
            fig3d.add_trace(go.Scatter3d(
                x=[i, i], y=[0, dsd], z=[z_cat, z_cat],
                mode="lines", line=dict(color=pt_col, width=5),
                showlegend=False, hoverinfo="skip"))
            fig3d.add_trace(go.Scatter3d(
                x=[i], y=[dsd], z=[z_cat], mode="markers",
                marker=dict(size=10, color=pt_col, line=dict(color="white", width=1)),
                name=var,
                hovertemplate=(f"<b>{var}</b><br>Ideal: {row['Ideal']:.2f} {row['Einheit']}<br>"
                               f"Gemessen: {row['Gemessen']:.2f} {row['Einheit']}<br>"
                               f"Δ: {dsd:+.2f} SD  [{STATUS_LABEL[status]}]<extra></extra>"),
                showlegend=False))
            off = 0.35 if dsd >= 0 else -0.35
            fig3d.add_trace(go.Scatter3d(
                x=[i], y=[dsd + off], z=[z_cat + 0.6],
                mode="text", text=[var], textfont=dict(size=8, color=pt_col),
                showlegend=False, hoverinfo="skip"))
            h_x.append(i); h_y.append(0); h_z.append(z_cat)
        h_x.append(None); h_y.append(None); h_z.append(None)
    fig3d.add_trace(go.Scatter3d(
        x=h_x, y=h_y, z=h_z, mode="lines+markers",
        line=dict(color="navy", width=3, dash="dot"),
        marker=dict(size=4, color="navy"),
        name="Harmonielinie (Y=0)"))
    fig3d.update_layout(
        height=700,
        scene=dict(
            xaxis=dict(title="Variable", showticklabels=False),
            yaxis=dict(title="Abweichung (SD)", range=[-y_lim, y_lim]),
            zaxis=dict(title="Kategorie",
                       tickvals=[k["z"] for k in kategorien_3d],
                       ticktext=[k["name"] for k in kategorien_3d]),
            camera=dict(eye=dict(x=2.2, y=-2.0, z=1.4)),
            bgcolor="white"),
        margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig3d, use_container_width=True)

# ── Tab 4: Wertetabelle ────────────────────────────────────────────────────
with tab4:
    st.subheader("Wertetabelle – Ideal vs. Gemessen")
    disp = df_abw[["Variable","Einheit","Ideal","Gemessen","Δ absolut","Δ in SD","Status"]].copy()
    disp["Δ in SD"]   = disp["Δ in SD"].map(lambda x: f"{x:+.2f}")
    disp["Δ absolut"] = disp["Δ absolut"].map(lambda x: f"{x:+.2f}")
    def hl(row):
        f = STATUS_FARBE.get(row["Status"], "white")
        return [f"background-color:{f}22" if c == "Status" else "" for c in row.index]
    st.dataframe(disp.style.apply(hl, axis=1), use_container_width=True, hide_index=True)
    st.download_button("⬇ CSV", data=df_abw.to_csv(index=False).encode("utf-8"),
                       file_name="harmonie_analyse.csv", mime="text/csv")

# ── Tab 5: Formeln ──────────────────────────────────────────────────────────
with tab5:
    c1, c2 = st.columns(2)
    items = list(FORMELN.items())
    for i, (name, formel) in enumerate(items):
        col = c1 if i < len(items) // 2 else c2
        sd = STANDARD_ABWEICHUNGEN.get(name, "–")
        col.markdown(f"**{name}** = `{formel}`  \n_SD = ±{sd} {EINHEITEN.get(name,'')}_")
    st.divider()
    st.subheader("Paddenberg et al. 2021")
    st.markdown("""
| Modell | Formel | R² |
|--------|--------|----|
| ANB A | −45.359 + 0.493·SNA + 0.251·ML-NSL | 0.578 |
| ANB B | −41.669 + 0.567·SNA + 0.110·ML-NSL + 0.114·NSBa + 0.132·NL-NSL + 0.062·Index − 0.289·Faz. | 0.690 |
| Wits C ★ | 57.510 + 1.526·ANB − 0.634·SNA − 0.666·SN-Occl | 0.976 |
| Wits D | 57.853 + 1.572·ANB − 0.664·SNA − 0.639·SN-Occl − 0.030·ML-NSL + 0.030·Index | 0.984 |
""")
