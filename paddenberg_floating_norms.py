"""
Paddenberg et al. (2021) – Floating Norms für individualisierten ANB-Winkel und Wits-Appraisal
================================================================================================
Quelle:
    Paddenberg E, Proff P, Kirschneck C.
    Floating norms for individualising the ANB angle and the WITS appraisal
    in orthodontic cephalometric analysis based on guiding variables.
    J Orofac Orthop. 2021 Jul 13;84(1):10–18.
    DOI: 10.1007/s00056-021-00322-1 | PMC9852193 | PMID: 34255093
    Open Access (CC BY 4.0)

Modelle
-------
A) ANB (Panagiotidis/Witt, neu berechnet)     korr. R² = 0.578
   Prädiktoren: SNA, ML-NSL

B) ANB (Paddenberg erweitert)                  korr. R² = 0.690
   Prädiktoren: SNA, ML-NSL, NSBa, NL-NSL, Index (Hasund), Fazialachse (Ricketts)

C) Wits (Järvinen, neu berechnet) ★ empfohlen  korr. R² = 0.976
   Prädiktoren: ANB, SNA, SN-Occl

D) Wits (Paddenberg erweitert)                 korr. R² = 0.984
   Prädiktoren: ANB, SNA, SN-Occl, ML-NSL, Index (Hasund)
"""

from __future__ import annotations
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Modell A: ANB individualisiert nach Panagiotidis/Witt (neu berechnet)
# ---------------------------------------------------------------------------
def anb_panagiotidis_witt(sna: float, ml_nsl: float) -> float:
    """ANB(indiv.) = −45.359 + 0.493·SNA + 0.251·ML-NSL   [R²=0.578]"""
    return -45.359 + 0.493 * sna + 0.251 * ml_nsl


# ---------------------------------------------------------------------------
# Modell B: ANB individualisiert – erweitert (Paddenberg 2021)
# ---------------------------------------------------------------------------
def anb_paddenberg_erweitert(
    sna: float,
    ml_nsl: float,
    nsba: float,
    nl_nsl: float,
    index_hasund: float,
    fazialachse_ricketts: float,
) -> float:
    """ANB(indiv.) = −41.669 + 0.567·SNA + 0.110·ML-NSL + 0.114·NSBa
                     + 0.132·NL-NSL + 0.062·Index − 0.289·Fazialachse  [R²=0.690]"""
    return (
        -41.669
        + 0.567 * sna
        + 0.110 * ml_nsl
        + 0.114 * nsba
        + 0.132 * nl_nsl
        + 0.062 * index_hasund
        - 0.289 * fazialachse_ricketts
    )


# ---------------------------------------------------------------------------
# Modell C: Wits individualisiert nach Järvinen (neu berechnet) ★ empfohlen
# ---------------------------------------------------------------------------
def wits_jaervinen(anb: float, sna: float, sn_occl: float) -> float:
    """Wits(indiv.) = 57.510 + 1.526·ANB − 0.634·SNA − 0.666·SN-Occl  [R²=0.976]"""
    return 57.510 + 1.526 * anb - 0.634 * sna - 0.666 * sn_occl


# ---------------------------------------------------------------------------
# Modell D: Wits individualisiert – erweitert (Paddenberg 2021)
# ---------------------------------------------------------------------------
def wits_paddenberg_erweitert(
    anb: float,
    sna: float,
    sn_occl: float,
    ml_nsl: float,
    index_hasund: float,
) -> float:
    """Wits(indiv.) = 57.853 + 1.572·ANB − 0.664·SNA − 0.639·SN-Occl
                      − 0.030·ML-NSL + 0.030·Index   [R²=0.984]"""
    return (
        57.853
        + 1.572 * anb
        - 0.664 * sna
        - 0.639 * sn_occl
        - 0.030 * ml_nsl
        + 0.030 * index_hasund
    )


# ---------------------------------------------------------------------------
# Empirische Normwerte (Table 1, Paddenberg 2021)
# ---------------------------------------------------------------------------
EMPIRISCHE_NORMEN = {
    "SNA":          {"mean": 81.0,  "sd": 2.0, "unit": "°"},
    "SNB":          {"mean": 79.0,  "sd": 2.0, "unit": "°"},
    "ANB":          {"mean":  2.0,  "sd": 2.0, "unit": "°"},
    "NSBa":         {"mean": 130.0, "sd": 6.0, "unit": "°"},
    "NL-NSL":       {"mean":  8.5,  "sd": 3.0, "unit": "°"},
    "ML-NSL":       {"mean": 32.0,  "sd": 2.0, "unit": "°"},
    "Index_Hasund": {"mean": 80.0,  "sd": 9.0, "unit": "%"},
    "Fazialachse":  {"mean": 90.0,  "sd": 3.0, "unit": "°"},
    "SN_Occl":      {"mean": 14.5,  "sd": 2.0, "unit": "°"},
    "Wits_w":       {"mean":  0.0,  "sd": 2.0, "unit": "mm"},
    "Wits_m":       {"mean": -1.0,  "sd": 2.0, "unit": "mm"},
}


@dataclass
class PaddenbergErgebnis:
    anb_ideal_A:  float          # Modell A
    anb_ideal_B:  float | None   # Modell B (nur wenn alle Prädiktoren vorhanden)
    wits_ideal_C: float          # Modell C ★
    wits_ideal_D: float | None   # Modell D (nur wenn Index vorhanden)


def paddenberg_analyse(
    sna: float,
    anb_gemessen: float,
    ml_nsl: float,
    sn_occl: float,
    nsba: float = 130.0,
    nl_nsl: float = 8.5,
    index_hasund: float | None = None,
    fazialachse: float | None = None,
) -> PaddenbergErgebnis:
    anb_A = anb_panagiotidis_witt(sna, ml_nsl)
    anb_B = (
        anb_paddenberg_erweitert(sna, ml_nsl, nsba, nl_nsl, index_hasund, fazialachse)
        if index_hasund is not None and fazialachse is not None
        else None
    )
    wits_C = wits_jaervinen(anb_gemessen, sna, sn_occl)
    wits_D = (
        wits_paddenberg_erweitert(anb_gemessen, sna, sn_occl, ml_nsl, index_hasund)
        if index_hasund is not None
        else None
    )
    return PaddenbergErgebnis(
        anb_ideal_A=round(anb_A, 2),
        anb_ideal_B=round(anb_B, 2) if anb_B is not None else None,
        wits_ideal_C=round(wits_C, 2),
        wits_ideal_D=round(wits_D, 2) if wits_D is not None else None,
    )


if __name__ == "__main__":
    # Beispiel aus Originalarbeit (ANB=10.4°, Wits=8.3mm)
    erg = paddenberg_analyse(
        sna=80.8, anb_gemessen=10.4, ml_nsl=50.4, sn_occl=14.5,
        nsba=130.0, nl_nsl=8.5, index_hasund=80.0, fazialachse=90.0,
    )
    print(f"ANB ideal (Modell A): {erg.anb_ideal_A}°")
    print(f"ANB ideal (Modell B): {erg.anb_ideal_B}°")
    print(f"Wits ideal (Modell C): {erg.wits_ideal_C} mm")
    print(f"Wits ideal (Modell D): {erg.wits_ideal_D} mm")
