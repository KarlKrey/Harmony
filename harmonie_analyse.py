"""
Segner/Hasund Individualisierte Strukturanalyse (Harmonieblatt)
===============================================================
Kernprinzip: SNA als Eingabe → individuelle Normwerte (Ideal) berechnen →
Abweichungen der Patientenmesswerte vom Ideal in SD-Einheiten darstellen.

Quellen SD-Werte: Segner/Hasund, Residualstreuungen der Regressionen (Tab. 3/4).
"""

from __future__ import annotations
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Standardabweichungen der Regressionsresiduen (aus Segner/Hasund + Literatur)
# ---------------------------------------------------------------------------
STANDARD_ABWEICHUNGEN: dict[str, float] = {
    "SNB":              2.0,   # Grad
    "ANB":              2.0,   # Grad
    "NL-NSL":           2.5,   # Grad
    "NSBa":             4.0,   # Grad
    "ML-NSL":           4.0,   # Grad
    "ML-NL":            4.5,   # Grad
    "1-NA_deg":         4.5,   # Grad
    "1-NA_mm":          2.5,   # mm
    "1-NB_deg":         4.5,   # Grad
    "1-NB_mm":          2.5,   # mm
    "H-Winkel":         4.0,   # Grad
    "Nasolabialwinkel": 8.0,   # Grad (höhere Variabilität Weichteil)
    "Z-Winkel":         5.0,   # Grad
    "HZB":              3.0,   # mm
    "VZB":              3.0,   # mm
    "Eckzahn-OK":       2.5,   # mm
    "Pont-SI-OK":       3.5,   # mm
    "Pont-SI-UK":       3.5,   # mm
}

# Einheiten für Anzeige
EINHEITEN: dict[str, str] = {
    "SNB":              "°",
    "ANB":              "°",
    "NL-NSL":           "°",
    "NSBa":             "°",
    "ML-NSL":           "°",
    "ML-NL":            "°",
    "1-NA_deg":         "°",
    "1-NA_mm":          "mm",
    "1-NB_deg":         "°",
    "1-NB_mm":          "mm",
    "H-Winkel":         "°",
    "Nasolabialwinkel": "°",
    "Z-Winkel":         "°",
    "HZB":              "mm",
    "VZB":              "mm",
    "Eckzahn-OK":       "mm",
    "Pont-SI-OK":       "mm",
    "Pont-SI-UK":       "mm",
}

# Eingabevariablen (gemessene Werte – ANB und ML-NL werden berechnet)
GRUPPEN: dict[str, list[str]] = {
    "Skelettale Basis": ["SNB", "NL-NSL", "NSBa", "ML-NSL"],
    "Dentale Variablen": ["1-NA_deg", "1-NA_mm", "1-NB_deg", "1-NB_mm", "H-Winkel"],
    "Weichteil": ["Nasolabialwinkel", "Z-Winkel"],
    "Zahnbogen": ["HZB", "VZB", "Eckzahn-OK", "Pont-SI-OK", "Pont-SI-UK"],
}

# Abgeleitete (berechnete) Variablen
ABGELEITET = {
    "ANB":   "SNA − SNB",
    "ML-NL": "ML-NSL − NL-NSL",
}

FORMELN: dict[str, str] = {
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
    "HZB":              "a + b × SNA + b × SNB  (a: 30–40, b: 0.1–0.2)",
    "VZB":              "a + b × SNA + b × SNB  (a: 20–25, b: 0.1–0.2)",
    "Eckzahn-OK":       "20 + b × SNA  (b=0.1, nach Bernabe)",
    "Pont-SI-OK":       "VZB × 100 / 80",
    "Pont-SI-UK":       "VZB × 100 / 64",
}


@dataclass
class Abweichung:
    variable:   str
    ideal:      float
    gemessen:   float
    sd:         float
    einheit:    str

    @property
    def delta(self) -> float:
        return self.gemessen - self.ideal

    @property
    def delta_sd(self) -> float:
        return self.delta / self.sd if self.sd else 0.0

    @property
    def status(self) -> str:
        """'ok' | 'grenz' | 'auffällig'"""
        a = abs(self.delta_sd)
        if a <= 1.0:
            return "ok"
        if a <= 2.0:
            return "grenz"
        return "auffällig"


# ---------------------------------------------------------------------------
# Kernfunktionen
# ---------------------------------------------------------------------------

def compute_ideal(
    sna: float,
    pgnb_mm: float = 2.3,
    a_hzb: float = 35.0,
    b_hzb: float = 0.2,
    a_vzb: float = 22.5,
    b_vzb: float = 0.2,
) -> dict[str, float]:
    """
    Berechnet den individualisierten Normwert (Ideal) für alle Variablen.

    HZB = a_hzb + b_hzb * SNA + b_hzb * SNB  (a: 30–40, b: 0.1–0.2)
    VZB = a_vzb + b_vzb * SNA + b_vzb * SNB  (a: 20–25, b: 0.1–0.2)
    """
    snb    = 0.79 * sna + 15.56
    anb    = sna - snb
    nl_nsl = -0.34 * sna + 35.5
    nsba   = -0.49 * sna + 171.17
    ml_nsl = -0.70 * sna + 86.05
    ml_nl  = ml_nsl - nl_nsl

    ina_mm  = -0.86 * anb - 0.23 * pgnb_mm + 12.8
    inb_mm  =  0.51 * anb - 0.30 * pgnb_mm + 10.4
    ina_deg = -2.19 * anb - 0.61 * pgnb_mm + 27.1
    inb_deg =  1.51 * anb - 0.80 * pgnb_mm + 22.4
    h_winkel = 1.0 * anb - 1.3 * pgnb_mm + 10.5

    nasolabialwinkel = 145.0 - 0.42 * ina_deg
    z_winkel         = 91.0  - anb * 1.2

    hzb        = a_hzb + b_hzb * sna + b_hzb * snb
    vzb        = a_vzb + b_vzb * sna + b_vzb * snb
    eckzahn_ok = 20.0 + b_hzb * sna
    pont_si_ok = vzb * 100.0 / 80.0
    pont_si_uk = vzb * 100.0 / 64.0

    return {
        "SNB":              round(snb, 2),
        "ANB":              round(anb, 2),
        "NL-NSL":           round(nl_nsl, 2),
        "NSBa":             round(nsba, 2),
        "ML-NSL":           round(ml_nsl, 2),
        "ML-NL":            round(ml_nl, 2),
        "1-NA_deg":         round(ina_deg, 2),
        "1-NA_mm":          round(ina_mm, 2),
        "1-NB_deg":         round(inb_deg, 2),
        "1-NB_mm":          round(inb_mm, 2),
        "H-Winkel":         round(h_winkel, 2),
        "Nasolabialwinkel": round(nasolabialwinkel, 2),
        "Z-Winkel":         round(z_winkel, 2),
        "HZB":              round(hzb, 2),
        "VZB":              round(vzb, 2),
        "Eckzahn-OK":       round(eckzahn_ok, 2),
        "Pont-SI-OK":       round(pont_si_ok, 2),
        "Pont-SI-UK":       round(pont_si_uk, 2),
    }


def compute_abweichungen(
    gemessen: dict[str, float],
    ideal:    dict[str, float],
) -> list[Abweichung]:
    """
    Vergleicht gemessene Patientenwerte mit den individualisierten Idealwerten.
    Gibt für jede Variable ein Abweichungs-Objekt zurück.
    """
    result = []
    for var in STANDARD_ABWEICHUNGEN:
        if var not in gemessen or var not in ideal:
            continue
        result.append(Abweichung(
            variable = var,
            ideal    = ideal[var],
            gemessen = gemessen[var],
            sd       = STANDARD_ABWEICHUNGEN[var],
            einheit  = EINHEITEN.get(var, ""),
        ))
    return result
