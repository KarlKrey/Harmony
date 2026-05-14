"""
3D Harmonic Korrelation nach Prof. Krey
========================================
Workflow:
  1. SNA + PgNB + optionale Paddenberg-Eingaben → individuelles Ideal
  2. Alle Patientenmesswerte eingeben
  3. Abweichungen visualisieren:
     - Balkendiagramm
     - Tetraeder-Harmonieraum (NEU)
     - 3D-Flächen-Ansicht
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
from paddenberg_floating_norms import paddenberg_analyse

# ---------------------------------------------------------------------------
st.set_page_config(page_title="3D Harmonic Korrelation", page_icon="🦷", layout="wide")
st.title("3D Harmonic Korrelation nach Prof. Krey")
st.caption("Individualisierte kephalometrische Strukturanalyse nach Segner/Hasund & Paddenberg")

STATUS_FARBE = {"ok": "#2ecc71", "grenz": "#f39c12", "auffällig": "#e74c3c"}
STATUS_LABEL = {"ok": "≤ 1 SD", "grenz": "1–2 SD", "auffällig": "> 2 SD"}

# ── Tetraeder-Geometrie ─────────────────────────────────────────────────────
# Regulärer Tetraeder, zentriert im Ursprung, |Ecke| = 1
_s = 1.0 / np.sqrt(3.0)
TETRA_UNIT = np.array([
    [ _s,  _s,  _s],   # Ecke 0 → FRS
    [ _s, -_s, -_s],   # Ecke 1 → Dental
    [-_s,  _s, -_s],   # Ecke 2 → Weichteil
    [-_s, -_s,  _s],   # Ecke 3 → Modell
])
TETRA_SD3 = TETRA_UNIT * 3.0    # Tetraeder-Kanten bei 3 SD für Zeichnung
TETRA_EDGES = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
TETRA_FACES = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]

CAT_NAMES   = ["FRS (Fernröntgen)", "Dentale Variablen", "Weichteil", "Modell"]
CAT_COLORS  = ["#4169E1", "#2E8B57", "#CC7000", "#6A0DAD"]
CAT_VARS_MAP = {
    "FRS (Fernröntgen)":  ["SNB", "ANB", "NL-NSL", "NSBa", "ML-NSL", "ML-NL"],
    "Dentale Variablen":  ["1-NA_deg", "1-NA_mm", "1-NB_deg", "1-NB_mm", "H-Winkel"],
    "Weichteil":          ["Nasolabialwinkel", "Z-Winkel"],
    "Modell":             ["HZB", "VZB", "Eckzahn-OK", "Pont-SI-OK", "Pont-SI-UK"],
}

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
        a_hzb = st.number_input("HZB – a", 30.0, 40.0, 35.0, 0.5, key="a_hzb")
        b_hzb = st.number_input("HZB – b", 0.10, 0.20, 0.20, 0.01, key="b_hzb", format="%.2f")
        a_vzb = st.number_input("VZB – a", 20.0, 25.0, 22.5, 0.5, key="a_vzb")
        b_vzb = st.number_input("VZB – b", 0.10, 0.20, 0.20, 0.01, key="b_vzb", format="%.2f")

    st.divider()
    with st.expander("Paddenberg Floating Norms"):
        pdb_aktiv = st.checkbox("Paddenberg aktivieren", value=False)
        sn_occl = ml_nsl_pdb = wits_gemessen = idx_hasund = fazialachse = 0.0
        geschlecht = "weiblich"
        if pdb_aktiv:
            sn_occl       = st.number_input("SN-Occl (°)", value=14.5, step=0.5)
            wits_gemessen = st.number_input("Wits gemessen (mm)", value=0.0, step=0.5)
            ml_nsl_pdb    = st.number_input("ML-NSL für Paddenberg (°)", value=32.0, step=0.5)
            idx_hasund    = st.number_input("Index Hasund (%, 0=n.v.)", value=0.0, step=1.0)
            fazialachse   = st.number_input("Fazialachse Ricketts (°, 0=n.v.)", value=0.0, step=0.5)
            geschlecht    = st.radio("Geschlecht", ["weiblich", "männlich"], horizontal=True)

    st.divider()
    for status, farbe in STATUS_FARBE.items():
        st.markdown(
            f'<span style="background:{farbe};padding:2px 8px;border-radius:4px;color:white">'
            f'{STATUS_LABEL[status]}</span>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Ideal & Messwerte
# ---------------------------------------------------------------------------
ideal = compute_ideal(sna, pgnb_mm=pgnb, a_hzb=a_hzb, b_hzb=b_hzb, a_vzb=a_vzb, b_vzb=b_vzb)

st.subheader("Schritt 2 – Patientenmesswerte")
st.caption("Felder mit Idealwerten vorbelegt – bitte tatsächliche Messwerte eintragen.")

gemessen: dict[str, float] = {}
for col_ui, (gruppe, variablen) in zip(st.columns(len(GRUPPEN)), GRUPPEN.items()):
    with col_ui:
        st.markdown(f"**{gruppe}**")
        for var in variablen:
            gemessen[var] = st.number_input(
                f"{var} ({EINHEITEN.get(var,'')})",
                value=float(ideal[var]), step=0.5, key=f"inp_{var}", format="%.2f",
            )

pdb_erg = None
if pdb_aktiv:
    pdb_erg = paddenberg_analyse(
        sna=sna,
        anb_gemessen=gemessen.get("ANB", ideal["ANB"]),
        ml_nsl=ml_nsl_pdb,
        sn_occl=sn_occl,
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

# Kategorie-Flächenplot: Paddenberg-Vars in FRS einhängen
kategorien_3d = [dict(k, vars=list(k["vars"])) for k in KATEGORIEN_3D]
if pdb_erg is not None:
    kategorien_3d[0]["vars"] = ["ANB (Paddenberg)", "Wits"] + kategorien_3d[0]["vars"]

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def cat_score(gemessen_d: dict, ideal_d: dict, var_list: list[str]) -> float:
    """Mittlere SD-normierte Abweichung des Patienten für eine Kategorie."""
    devs = [
        (gemessen_d[v] - ideal_d[v]) / STANDARD_ABWEICHUNGEN[v]
        for v in var_list
        if v in STANDARD_ABWEICHUNGEN and v in gemessen_d and v in ideal_d
    ]
    return float(np.mean(devs)) if devs else 0.0


def sphere_mesh(r: float, n: int = 22):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    return (r * np.outer(np.cos(u), np.sin(v)),
            r * np.outer(np.sin(u), np.sin(v)),
            r * np.outer(np.ones(n), np.cos(v)))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Abweichungsübersicht",
    "🔮 Tetraeder-Harmonieraum",
    "🧊 3D Kategorieflächen",
    "📋 Wertetabelle",
    "📐 Formeln",
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
    for xv, dash, col in [(-2,"dash","red"),(-1,"dot","gray"),(1,"dot","gray"),(2,"dash","red")]:
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

# ── Tab 2: Tetraeder ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Tetraeder-Harmonieraum")
    st.caption(
        "4 Messkategorien an den Ecken eines regulären Tetraeders. "
        "**Harmonielinie** = Patientenposition für SNA 62–103. "
        "**Schieberegler** = verschiebbare Linie durch den Harmonieraum."
    )

    # Trajektorie berechnen (Patient-Position für jeden SNA-Wert)
    sna_range = np.arange(62, 104, 1.0)
    trajectory = np.zeros((len(sna_range), 3))
    cat_scores_by_sna = np.zeros((len(sna_range), 4))

    for idx_s, sna_i in enumerate(sna_range):
        ideal_i = compute_ideal(float(sna_i), pgnb_mm=pgnb,
                                a_hzb=a_hzb, b_hzb=b_hzb, a_vzb=a_vzb, b_vzb=b_vzb)
        # Paddenberg ANB + Wits in FRS-Score einbeziehen falls aktiv
        gemessen_ext = dict(gemessen)
        ideal_ext = dict(ideal_i)
        if pdb_erg is not None:
            pdb_i = paddenberg_analyse(
                sna=float(sna_i), anb_gemessen=gemessen.get("ANB", ideal_i["ANB"]),
                ml_nsl=ml_nsl_pdb, sn_occl=sn_occl,
            )
            gemessen_ext["ANB (Paddenberg)"] = gemessen.get("ANB", ideal_i["ANB"])
            ideal_ext["ANB (Paddenberg)"]    = pdb_i.anb_ideal_A
            gemessen_ext["Wits"]             = wits_gemessen
            ideal_ext["Wits"]                = pdb_i.wits_ideal_C
            STANDARD_ABWEICHUNGEN["ANB (Paddenberg)"] = 2.0
            STANDARD_ABWEICHUNGEN["Wits"] = 2.0

        scores = np.array([
            cat_score(gemessen_ext, ideal_ext,
                      CAT_VARS_MAP[c] + (["ANB (Paddenberg)", "Wits"] if c == "FRS (Fernröntgen)" and pdb_erg else []))
            for c in CAT_NAMES
        ])
        cat_scores_by_sna[idx_s] = scores
        trajectory[idx_s] = TETRA_UNIT.T @ scores  # gewichtete Summe der Ecken

    # Aktuelle Patientenposition
    cur_idx = int(round(sna)) - 62
    cur_idx = max(0, min(len(sna_range) - 1, cur_idx))
    pat_pos = trajectory[cur_idx]
    pat_dist = float(np.linalg.norm(pat_pos))

    # Slider für verschiebbare Linie
    sna_slide = st.slider("↕ Harmonielinie verschieben (SNA)", 62, 103,
                           int(sna), key="sna_tetra")
    slide_idx = sna_slide - 62
    slide_pos = trajectory[slide_idx]
    slide_scores = cat_scores_by_sna[slide_idx]

    fig_t = go.Figure()

    # ── Tetraeder-Flächen (halbtransparent) ──────────────────────────────
    fig_t.add_trace(go.Mesh3d(
        x=TETRA_SD3[:, 0], y=TETRA_SD3[:, 1], z=TETRA_SD3[:, 2],
        i=[f[0] for f in TETRA_FACES],
        j=[f[1] for f in TETRA_FACES],
        k=[f[2] for f in TETRA_FACES],
        opacity=0.06, color="lightgray",
        hoverinfo="skip", showlegend=False,
    ))

    # ── Tetraeder-Kanten ──────────────────────────────────────────────────
    for e in TETRA_EDGES:
        p1, p2 = TETRA_SD3[e[0]], TETRA_SD3[e[1]]
        fig_t.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode="lines", line=dict(color="#555", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Ecken-Labels (Kategorien) ─────────────────────────────────────────
    for i, (name, col) in enumerate(zip(CAT_NAMES, CAT_COLORS)):
        pos_v = TETRA_SD3[i]
        fig_t.add_trace(go.Scatter3d(
            x=[pos_v[0]], y=[pos_v[1]], z=[pos_v[2]],
            mode="markers+text",
            marker=dict(size=10, color=col, line=dict(color="white", width=1)),
            text=[name], textposition="top center",
            textfont=dict(size=11, color=col),
            showlegend=False, hoverinfo="skip",
        ))

    # ── SD-Kugeln: ±1 (grün) und ±2 (rot) ───────────────────────────────
    for r_sd, col_sd, op_sd in [(1.0, "#2ecc71", 0.08), (2.0, "#e74c3c", 0.05)]:
        sx, sy, sz = sphere_mesh(r_sd)
        fig_t.add_trace(go.Surface(
            x=sx, y=sy, z=sz,
            colorscale=[[0, col_sd], [1, col_sd]],
            opacity=op_sd, showscale=False,
            hoverinfo="skip", showlegend=False,
        ))

    # ── Trajektorie (Harmonielinie SNA 62–103) ────────────────────────────
    traj_dist = np.linalg.norm(trajectory, axis=1)
    fig_t.add_trace(go.Scatter3d(
        x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
        mode="lines",
        line=dict(
            color=sna_range.tolist(),
            colorscale="RdYlGn_r",
            width=7,
            colorbar=dict(title="SNA (°)", len=0.5, x=1.02, thickness=14),
        ),
        name="Harmonielinie (SNA 62–103)",
        hovertemplate="SNA=%{text}°<extra></extra>",
        text=[f"{int(s)}" for s in sna_range],
    ))

    # ── Verschiebbare Linie: Schieberegler-Position ───────────────────────
    fig_t.add_trace(go.Scatter3d(
        x=[slide_pos[0]], y=[slide_pos[1]], z=[slide_pos[2]],
        mode="markers",
        marker=dict(size=14, color="white", symbol="circle",
                    line=dict(color="navy", width=3)),
        name=f"Schieberegler SNA={sna_slide}°",
        hovertemplate=f"SNA={sna_slide}°<br>"
                      f"FRS: {slide_scores[0]:+.2f} SD<br>"
                      f"Dental: {slide_scores[1]:+.2f} SD<br>"
                      f"Weichteil: {slide_scores[2]:+.2f} SD<br>"
                      f"Modell: {slide_scores[3]:+.2f} SD"
                      "<extra></extra>",
    ))

    # ── Patient (bei aktuellem SNA) ───────────────────────────────────────
    cur_scores = cat_scores_by_sna[cur_idx]
    fig_t.add_trace(go.Scatter3d(
        x=[pat_pos[0]], y=[pat_pos[1]], z=[pat_pos[2]],
        mode="markers",
        marker=dict(size=16, color="gold", symbol="diamond",
                    line=dict(color="black", width=2)),
        name=f"Patient (SNA={int(sna)}°)",
        hovertemplate=(
            f"<b>Patient</b><br>"
            f"FRS: {cur_scores[0]:+.2f} SD<br>"
            f"Dental: {cur_scores[1]:+.2f} SD<br>"
            f"Weichteil: {cur_scores[2]:+.2f} SD<br>"
            f"Modell: {cur_scores[3]:+.2f} SD<br>"
            f"Abstand vom Zentrum: {pat_dist:.2f}"
            "<extra></extra>"
        ),
    ))

    # ── Zentrum (perfekte Harmonie) ───────────────────────────────────────
    fig_t.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=7, color="navy"),
        name="Harmonie-Zentrum",
    ))

    # ── Linie Zentrum → Patient ───────────────────────────────────────────
    dev_color = (STATUS_FARBE["auffällig"] if pat_dist > 2
                 else STATUS_FARBE["grenz"] if pat_dist > 1
                 else STATUS_FARBE["ok"])
    fig_t.add_trace(go.Scatter3d(
        x=[0, pat_pos[0]], y=[0, pat_pos[1]], z=[0, pat_pos[2]],
        mode="lines", line=dict(color=dev_color, width=4, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))

    fig_t.update_layout(
        height=720,
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       showbackground=False, title=""),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       showbackground=False, title=""),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       showbackground=False, title=""),
            bgcolor="rgb(15, 15, 25)",
            camera=dict(eye=dict(x=2.0, y=1.6, z=1.0)),
        ),
        paper_bgcolor="rgb(15, 15, 25)",
        font=dict(color="white"),
        margin=dict(l=0, r=60, t=30, b=0),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0.4)",
                    font=dict(size=11)),
    )
    st.plotly_chart(fig_t, use_container_width=True)

    # Kategorie-Scores unter dem Plot
    cols_cat = st.columns(4)
    for col_ui, cat_n, score_v, cat_col in zip(
            cols_cat, CAT_NAMES, cur_scores, CAT_COLORS):
        status_c = ("auffällig" if abs(score_v) > 2
                    else "grenz" if abs(score_v) > 1 else "ok")
        col_ui.metric(
            label=cat_n,
            value=f"{score_v:+.2f} SD",
            delta=STATUS_LABEL[status_c],
            delta_color="off",
        )

# ── Tab 3: 3D Kategorieflächen ──────────────────────────────────────────────
with tab3:
    st.subheader("3D Harmonieraum – 4 Messflächen")
    st.caption(
        "Horizontale Ebenen = Messkategorien. Y-Achse = Abweichung in SD. "
        "Harmonielinie Y=0 verbindet alle Flächen."
    )

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
    for i in range(len(kategorien_3d) - 1):
        k1, k2 = kategorien_3d[i], kategorien_3d[i+1]
        fig3d.add_trace(go.Scatter3d(
            x=[(len(k1["vars"])-1)/2, (len(k2["vars"])-1)/2], y=[0, 0],
            z=[k1["z"], k2["z"]], mode="lines",
            line=dict(color="navy", width=2, dash="dash"),
            showlegend=False, hoverinfo="skip"))

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
    st.download_button("⬇ CSV herunterladen",
                       data=df_abw.to_csv(index=False).encode("utf-8"),
                       file_name="harmonie_analyse.csv", mime="text/csv")

# ── Tab 5: Formeln ──────────────────────────────────────────────────────────
with tab5:
    c1, c2 = st.columns(2)
    items = list(FORMELN.items())
    for i, (name, formel) in enumerate(items):
        col = c1 if i < len(items) // 2 else c2
        sd  = STANDARD_ABWEICHUNGEN.get(name, "–")
        col.markdown(f"**{name}** = `{formel}`  \n_SD = ±{sd} {EINHEITEN.get(name,'')}_")
    st.divider()
    st.subheader("Paddenberg et al. 2021")
    st.markdown("""
| Modell | Formel | R² |
|--------|--------|----|
| ANB Modell A | −45.359 + 0.493·SNA + 0.251·ML-NSL | 0.578 |
| ANB Modell B | −41.669 + 0.567·SNA + 0.110·ML-NSL + 0.114·NSBa + 0.132·NL-NSL + 0.062·Index − 0.289·Fazialachse | 0.690 |
| Wits Modell C ★ | 57.510 + 1.526·ANB − 0.634·SNA − 0.666·SN-Occl | 0.976 |
| Wits Modell D | 57.853 + 1.572·ANB − 0.664·SNA − 0.639·SN-Occl − 0.030·ML-NSL + 0.030·Index | 0.984 |
""")
