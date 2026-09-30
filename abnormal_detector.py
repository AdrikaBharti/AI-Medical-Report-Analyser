"""
abnormal_detector.py - compare each normalized value with a reference range and label it low / normal / high.

Range priority:  (1) the range PRINTED ON THE REPORT (sex-specific when both are printed)
                 (2) generic adult defaults below (sex-specific where it matters).
These are screening ranges for adults; labs differ by method, age and population, so the report's own range wins.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from unit_normalizer import NormalizedParam
from parameter_extractor import LABEL

# (low, high) or {"male": (lo,hi), "female": (lo,hi), "any": (lo,hi)}   - in the units of unit_normalizer.TARGET
DEFAULT_RANGES: Dict[str, object] = {
    "hemoglobin": {"male": (13.0, 17.0), "female": (12.0, 15.0), "any": (12.0, 17.0)},
    "pcv": {"male": (40, 50), "female": (36, 46), "any": (36, 50)},
    "rbc_count": {"male": (4.5, 5.5), "female": (3.8, 4.8), "any": (3.8, 5.5)},
    "wbc_count": (4000, 11000), "mcv": (80, 100), "mch": (27, 32), "mchc": (32, 36),
    "glucose_fasting": (70, 100), "glucose_pp": (70, 140), "glucose_random": (70, 140),
    "glucose_unspecified": (70, 140), "hba1c": (4.0, 5.6), "insulin": (2, 25),
    "total_cholesterol": (0, 200), "blood_urea": (15, 40), "bun": (7, 20),
    "serum_creatinine": {"male": (0.7, 1.3), "female": (0.6, 1.1), "any": (0.6, 1.3)},
    "sodium": (135, 145), "potassium": (3.5, 5.1),
    "total_bilirubin": (0.1, 1.2), "direct_bilirubin": (0.0, 0.3), "alp": (44, 147),
    "alt": (7, 56), "ast": (10, 40), "total_protein": (6.0, 8.3), "albumin": (3.5, 5.2), "ag_ratio": (1.0, 2.2),
    "urine_sg": (1.005, 1.030), "bmi": (18.5, 24.9), "systolic_bp": (90, 130), "diastolic_bp": (60, 85),
    # qualitative urine: 0 = normal / negative
    "urine_albumin": (0, 0), "urine_sugar": (0, 0), "urine_rbc": (0, 0), "urine_pus_cells": (0, 0),
    "pus_cell_clumps": (0, 0), "bacteria": (0, 0),
}
# values that need urgent medical attention (low, high) - shown as a prominent alert
CRITICAL: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "hemoglobin": (7.0, 20.0), "potassium": (2.5, 6.5), "sodium": (120, 160),
    "glucose_fasting": (50, 400), "glucose_random": (50, 400), "glucose_pp": (50, 400), "glucose_unspecified": (50, 400),
    "serum_creatinine": (None, 10.0), "total_bilirubin": (None, 15.0), "wbc_count": (2000, 30000),
}
QUAL_TEXT = {"urine_albumin": "Protein detected in urine", "urine_sugar": "Sugar detected in urine",
             "urine_rbc": "Red cells in urine above normal", "urine_pus_cells": "Pus cells in urine above normal",
             "pus_cell_clumps": "Pus cell clumps present", "bacteria": "Bacteria present in urine"}


def _pick_range(p: NormalizedParam, sex: Optional[str]):
    if p.ref_by_sex and sex in p.ref_by_sex:
        return p.ref_by_sex[sex], "report"
    if p.ref_low is not None or p.ref_high is not None:
        return (p.ref_low, p.ref_high), "report"
    d = DEFAULT_RANGES.get(p.name)
    if isinstance(d, dict):
        return d.get(sex or "any", d["any"]), "default"
    return d, "default"


def _severity(value: float, lo: Optional[float], hi: Optional[float]) -> str:
    if hi is not None and value > hi:
        ref, dev = hi, value - hi
    elif lo is not None and value < lo:
        ref, dev = lo, lo - value
    else:
        return "none"
    frac = dev / abs(ref) if ref else 1.0
    return "mild" if frac < 0.10 else "moderate" if frac < 0.30 else "marked"


def detect_abnormal(params: Dict[str, NormalizedParam], sex: Optional[str] = None) -> List[dict]:
    out = []
    for name, p in params.items():
        rng, src = _pick_range(p, sex)
        if rng is None:
            continue
        lo, hi = rng
        v = p.value
        if p.kind == "qualitative":
            status = "high" if v > (hi or 0) else "normal"
            sev = "none" if status == "normal" else ("mild" if v <= 1 else "moderate")
            note = QUAL_TEXT.get(name) if status == "high" else None
        else:
            status = "high" if (hi is not None and v > hi) else "low" if (lo is not None and v < lo) else "normal"
            sev = _severity(v, lo, hi)
            note = None
        crit = False
        c = CRITICAL.get(name)
        if c and p.kind != "qualitative":
            crit = (c[0] is not None and v < c[0]) or (c[1] is not None and v > c[1])
        out.append({"parameter": name, "label": LABEL.get(name, name), "value": v, "raw_value": p.raw_value,
                    "unit": p.unit, "status": status, "severity": sev, "critical": bool(crit),
                    "reference_range": [lo, hi], "reference_source": src,
                    "report_flag": p.report_flag, "note": note or p.note})
    order = {"none": 0, "mild": 1, "moderate": 2, "marked": 3}
    out.sort(key=lambda r: (not r["critical"], -order[r["severity"]], r["label"]))
    return out