"""
Segner/Hasund Individualisierte Strukturanalyse (Harmonieblatt)
===============================================================
Normwertberechnungen in Abhängigkeit von SNA.

Eingangsgröße : SNA (Grad)
Optionale Eingaben: PgNB_mm (Kinnprominenz), Overjet_mm, Jochbogenbreite_mm
"""

from __future__ import annotations


def compute_harmonie_normwerte(
    sna: float,
    pgnb_mm: float = 2.3,
    overjet_mm: float | None = None,
    jochbogenbreite_mm: float | None = None,
) -> dict:
    """
    Berechnet alle Normwerte der Strukturanalyse nach Segner/Hasund.

    Parameter
    ----------
    sna                : SNA-Winkel (Grad)
    pgnb_mm            : Kinnprominenz PgNB in mm (Standard: 2,3 mm Mittelwert)
    overjet_mm         : Overjet in mm (für Pont-Formeln, optional)
    jochbogenbreite_mm : Jochbogenbreite in mm (für Izard-Index, optional)
    """

    # --- Skelettale Basisvariablen ---
    snb = 0.79 * sna + 15.56
    anb = sna - snb

    # --- Knochenbasisbezogene Winkel (Tab. 3) ---
    nl_nsl = -0.34 * sna + 35.5
    nsba   = -0.49 * sna + 171.17
    ml_nsl = -0.70 * sna + 86.05
    ml_nl  = ml_nsl - nl_nsl

    # --- Dentale Normwerte Inzisivi (Tab. 4) ---
    # Strecken (mm)
    ina_mm  = -0.86 * anb - 0.23 * pgnb_mm + 12.8    # 1-NA_mm
    inb_mm  =  0.51 * anb - 0.30 * pgnb_mm + 10.4    # 1-NB_mm
    # Winkel (Grad)
    ina_deg = -2.19 * anb - 0.61 * pgnb_mm + 27.1    # 1-NA°
    inb_deg =  1.51 * anb - 0.80 * pgnb_mm + 22.4    # 1-NB°

    # --- H-Winkel (Holdaway) ---
    h_winkel = 1.0 * anb - 1.3 * pgnb_mm + 10.5

    # --- Nasolabialwinkel (Anhaltswert aus PPTX) ---
    # Nasolabialwinkel_Foto ≈ 145° − 0,42 × (1-NL in FRS)
    nasolabialwinkel = 145.0 - 0.42 * ina_deg

    # --- Holdaway Z-Winkel (Anhaltswert aus PPTX) ---
    # Z-Winkel = 91° − (ANB × 1,2) − Weichteildicke am Pkt A [mm]
    # Weichteildicke Punkt A wird hier als Näherung nicht separat erfasst;
    # Formel ohne Weichteildicke-Term als Basiswert:
    z_winkel = 91.0 - (anb * 1.2)

    # --- Zahnbogenlängen (aus PPTX) ---
    # HZB = a + 0,2·SNA + 0,2·SNB  (a = 30–40, Mittelwert 35)
    # VZB = a + 0,2·SNA + 0,2·SNB  (a = 20–25, Mittelwert 22.5)
    hzb = 35.0 + 0.2 * sna + 0.2 * snb
    vzb = 22.5 + 0.2 * sna + 0.2 * snb

    # --- Eckzahnabstand OK (Bernabe) ---
    eckzahn_ok = 20.0 + 0.2 * sna

    # --- Pont-Formeln ---
    # SI-OK = Σi_OK × 100 / 80  →  umgestellt: Σi_OK ≈ VZB (Näherung)
    # Hier: Normbreitenindizes aus VZB abgeleitet
    pont_si_ok = vzb * 100.0 / 80.0      # Zahnsumme OK (mm) nach Pont
    pont_si_uk = vzb * 100.0 / 64.0      # Zahnsumme UK (mm) nach Pont

    # --- Tonn-Verhältnis ---
    tonn_ratio = 4.0 / 3.0               # SI-OK : SI-UK = 4:3 (konstant)

    # --- Izard-Index ---
    izard_index: float | None = None
    if jochbogenbreite_mm is not None:
        izard_index = jochbogenbreite_mm / (jochbogenbreite_mm - 10.0) / 2.0

    result = {
        "SNA":              sna,
        "SNB":              round(snb, 2),
        "ANB":              round(anb, 2),
        "NL-NSL":           round(nl_nsl, 2),
        "NSBa":             round(nsba, 2),
        "ML-NSL":           round(ml_nsl, 2),
        "ML-NL":            round(ml_nl, 2),
        "1-NA_mm":          round(ina_mm, 2),
        "1-NA_deg":         round(ina_deg, 2),
        "1-NB_mm":          round(inb_mm, 2),
        "1-NB_deg":         round(inb_deg, 2),
        "H-Winkel":         round(h_winkel, 2),
        "Nasolabialwinkel": round(nasolabialwinkel, 2),
        "Z-Winkel":         round(z_winkel, 2),
        "HZB":              round(hzb, 2),
        "VZB":              round(vzb, 2),
        "Eckzahn-OK":       round(eckzahn_ok, 2),
        "Pont-SI-OK":       round(pont_si_ok, 2),
        "Pont-SI-UK":       round(pont_si_uk, 2),
        "Tonn-Ratio":       round(tonn_ratio, 4),
        "Izard-Index":      round(izard_index, 4) if izard_index is not None else None,
        "PgNB_mm":          pgnb_mm,
    }
    return result


def generate_normwerttabelle(
    sna_min: int = 62,
    sna_max: int = 103,
    pgnb_mm: float = 2.3,
) -> list[dict]:
    """Normwerttabelle für SNA von sna_min bis sna_max (ganzzahlig)."""
    return [compute_harmonie_normwerte(sna, pgnb_mm=pgnb_mm) for sna in range(sna_min, sna_max + 1)]


FORMELN = {
    "SNB":              "0.79 × SNA + 15.56",
    "ANB":              "SNA − SNB",
    "NL-NSL":           "−0.34 × SNA + 35.5",
    "NSBa":             "−0.49 × SNA + 171.17",
    "ML-NSL":           "−0.70 × SNA + 86.05",
    "ML-NL":            "ML-NSL − NL-NSL",
    "1-NA_mm":          "−0.86 × ANB − 0.23 × PgNB + 12.8",
    "1-NA_deg":         "−2.19 × ANB − 0.61 × PgNB + 27.1",
    "1-NB_mm":          "0.51 × ANB − 0.30 × PgNB + 10.4",
    "1-NB_deg":         "1.51 × ANB − 0.80 × PgNB + 22.4",
    "H-Winkel":         "1.0 × ANB − 1.3 × PgNB + 10.5",
    "Nasolabialwinkel": "145 − 0.42 × 1-NA°",
    "Z-Winkel":         "91 − (ANB × 1.2)",
    "HZB":              "35 + 0.2 × SNA + 0.2 × SNB",
    "VZB":              "22.5 + 0.2 × SNA + 0.2 × SNB",
    "Eckzahn-OK":       "20 + 0.2 × SNA  (nach Bernabe)",
    "Pont-SI-OK":       "VZB × 100 / 80",
    "Pont-SI-UK":       "VZB × 100 / 64",
    "Tonn-Ratio":       "4 / 3  (konstant)",
}


if __name__ == "__main__":
    result = compute_harmonie_normwerte(86.0)
    print("Normwerte für SNA = 86°:")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
