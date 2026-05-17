# Harmony

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Individualisierte kephalometrische Strukturanalyse nach **Segner/Hasund** – interaktives 3D-Korrelationsdashboard.

## Funktionsumfang

- Berechnung aller Normwerte aus SNA (+ optionale Eingaben PgNB, Jochbogenbreite)
- Interaktives Streamlit-Dashboard mit Patientenwert-Eingabe via Slider
- 3D-Korrelationsplot über den gesamten SNA-Bereich (62°–103°) × PgNB-Bereich
- 2D-Verlaufsgraphen aller Variablen
- Formelübersicht

## Enthaltene Formeln

| Gruppe | Variablen |
|--------|-----------|
| Skelettale Basis | SNB, ANB, NL-NSL, NSBa, ML-NSL, ML-NL |
| Dentale Normwerte | 1-NA (mm/°), 1-NB (mm/°), H-Winkel |
| Weichteil | Nasolabialwinkel, Z-Winkel (Holdaway) |
| Zahnbogen | HZB, VZB, Eckzahn-OK (Bernabe) |
| Indices | Pont SI-OK/SI-UK, Tonn-Ratio, Izard-Index |

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Quellen

- Segner D, Hasund A: *Individualisierte Kephalometrie*
- Bernabe et al.: Eckzahnabstand-Formel
- Pont: Zahnbogenindizes
- Holdaway: H-Winkel, Z-Winkel
