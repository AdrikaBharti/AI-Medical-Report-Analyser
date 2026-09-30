"""
unit_normalizer.py - convert every extracted value to the units the MODELS WERE TRAINED ON,
and reject values that are physiologically impossible (usually an OCR slip such as a lost decimal point).

Training units (from the datasets):
  glucose/BUN-urea/creatinine/cholesterol/bilirubin  mg/dL      hemoglobin, protein, albumin, MCHC  g/dL
  sodium, potassium                                   mEq/L      insulin                             uIU/mL
  ALT/AST/ALP                                         IU/L       WBC                                 cells/cumm
  RBC count                                           million/uL PCV                                 %
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

from parameter_extractor import ExtractedParam

# target unit, and {source unit key: multiplicative factor}
TARGET = {
    "glucose_fasting": "mg/dl", "glucose_pp": "mg/dl", "glucose_random": "mg/dl", "glucose_unspecified": "mg/dl",
    "insulin": "uiu/ml", "total_cholesterol": "mg/dl", "blood_urea": "mg/dl", "bun": "mg/dl",
    "serum_creatinine": "mg/dl", "sodium": "meq/l", "potassium": "meq/l",
    "total_bilirubin": "mg/dl", "direct_bilirubin": "mg/dl", "alp": "iu/l", "alt": "iu/l", "ast": "iu/l",
    "total_protein": "g/dl", "albumin": "g/dl", "hemoglobin": "g/dl", "mchc": "g/dl",
    "rbc_count": "million/ul", "wbc_count": "cells/cumm", "pcv": "%", "mcv": "fl", "mch": "pg",
    "hba1c": "%", "ag_ratio": "ratio",
}
GLU = {"mmol/l": 18.016}
FACTORS: Dict[str, Dict[str, float]] = {
    "glucose_fasting": GLU, "glucose_pp": GLU, "glucose_random": GLU, "glucose_unspecified": GLU,
    "insulin": {"pmol/l": 1 / 6.945, "miu/l": 1.0, "uiu/ml": 1.0, "u/ml": 1.0},
    "total_cholesterol": {"mmol/l": 38.67},
    "blood_urea": {"mmol/l": 6.006},
    "bun": {"mmol/l": 2.801},
    "serum_creatinine": {"umol/l": 1 / 88.4},
    "sodium": {"mmol/l": 1.0}, "potassium": {"mmol/l": 1.0},
    "total_bilirubin": {"umol/l": 1 / 17.1}, "direct_bilirubin": {"umol/l": 1 / 17.1},
    "alp": {"u/l": 1.0}, "alt": {"u/l": 1.0}, "ast": {"u/l": 1.0},
    "total_protein": {"g/l": 0.1}, "albumin": {"g/l": 0.1},
    "hemoglobin": {"g/l": 0.1}, "mchc": {"g/l": 0.1},
    "rbc_count": {"10^6/ul": 1.0, "/cumm": 1e-6, "/ul": 1e-6, "lakh/cumm": 0.1},
    "wbc_count": {"10^3/ul": 1000.0, "/ul": 1.0, "/cumm": 1.0},
}
# Value must lie inside these (in TARGET units) or it is rejected.
PLAUSIBLE: Dict[str, Tuple[float, float]] = {
    "hemoglobin": (2, 25), "hba1c": (3, 20), "pcv": (8, 70), "rbc_count": (1, 9), "wbc_count": (300, 200000),
    "mcv": (40, 150), "mch": (10, 50), "mchc": (20, 45),
    "glucose_fasting": (20, 800), "glucose_pp": (20, 800), "glucose_random": (20, 900), "glucose_unspecified": (20, 900),
    "insulin": (0.5, 1000), "total_cholesterol": (50, 600), "blood_urea": (3, 400), "bun": (2, 200),
    "serum_creatinine": (0.1, 25), "sodium": (100, 180), "potassium": (1.5, 9),
    "total_bilirubin": (0.1, 60), "direct_bilirubin": (0.0, 40), "alp": (10, 3000), "alt": (2, 5000), "ast": (2, 5000),
    "total_protein": (2, 12), "albumin": (0.8, 7), "ag_ratio": (0.2, 4), "urine_sg": (1.000, 1.060),
    "bmi": (10, 80), "systolic_bp": (60, 260), "diastolic_bp": (30, 160),
}

_SUFFIX_MAP = [                       # (regex on the compact unit string) -> canonical key
    (r"^(?:x)?10(?:\^|\*|e)?[3³]/[uµμ]?l$", "10^3/ul"),
    (r"^(?:x)?10(?:\^|\*|e)?[6⁶]/[uµμ]?l$", "10^6/ul"),
    (r"^(?:thou|thousand|k)/(?:cumm|cu\.?mm|cmm|mm3|mm³|[uµμ]l)$", "10^3/ul"),
    (r"^(?:mill|mil|million|millions)/(?:cumm|cu\.?mm|cmm|mm3|mm³|[uµμ]l)$", "10^6/ul"),
    (r"^lakhs?/(?:cumm|cu\.?mm|cmm|mm3|mm³)$", "lakh/cumm"),
    (r"^(?:cells)?/(?:cumm|cu\.?mm|cmm|mm3|mm³)$", "/cumm"),
    (r"^(?:cells)?/[uµμ]l$", "/ul"),
]


def canon_unit(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    c = re.sub(r"\s+", "", u.lower()).replace("µ", "u").replace("μ", "u")
    for rx, key in _SUFFIX_MAP:
        if re.match(rx, c):
            return key
    return c


# -- heuristics when the report prints no unit (flagged as "assumed") --------------------------------
def _guess_source_unit(name: str, v: float) -> Optional[str]:
    if name.startswith("glucose"):
        return "mmol/l" if v < 30 else None
    if name == "total_cholesterol":
        return "mmol/l" if v < 15 else None
    if name == "hemoglobin":
        return "g/l" if v > 30 else None
    if name == "serum_creatinine":
        return "umol/l" if v > 30 else None
    if name == "wbc_count":
        return "10^3/ul" if v < 200 else None
    if name == "rbc_count":
        return "/ul" if v > 100 else None
    if name in ("albumin", "total_protein"):
        return "g/l" if v > 15 else None
    return None


@dataclass
class NormalizedParam:
    name: str
    value: float
    unit: str                              # unit AFTER conversion
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    ref_by_sex: Optional[Dict[str, Tuple[float, float]]] = None
    report_flag: Optional[str] = None
    raw_value: str = ""
    original_unit: Optional[str] = None
    converted: bool = False
    unit_assumed: bool = False
    kind: str = "numeric"
    needs_review: bool = False
    note: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class NormalizationResult:
    params: Dict[str, NormalizedParam] = field(default_factory=dict)
    rejected: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def normalize(extracted: Dict[str, ExtractedParam], from_ocr: bool = False) -> NormalizationResult:
    res = NormalizationResult()
    for name, p in extracted.items():
        v, lo, hi, sx = float(p.value), p.ref_low, p.ref_high, p.ref_by_sex
        src = canon_unit(p.unit)
        tgt = TARGET.get(name)
        converted = assumed = False
        note = None
        review = from_ocr

        if p.kind == "qualitative" or tgt is None:
            unit_out = p.unit or ""
        else:
            factors = FACTORS.get(name, {})
            if src is None:
                guess = _guess_source_unit(name, v)
                if guess:
                    src, assumed = guess, True
                    note = f"No unit printed; value looks like {guess}, so it was converted."
                    review = True
            if src is not None and src != tgt:
                f = factors.get(src)
                if f is None:
                    res.warnings.append(f"{name}: unrecognised unit '{p.unit}' - value kept as printed; please verify.")
                    review = True
                    note = f"Unrecognised unit '{p.unit}'."
                else:
                    v *= f
                    lo = lo * f if lo is not None else None
                    hi = hi * f if hi is not None else None
                    if sx:
                        sx = {k: (a * f, b * f) for k, (a, b) in sx.items()}
                    converted = f != 1.0
            unit_out = tgt

        lo_hi = PLAUSIBLE.get(name)
        if lo_hi and not (lo_hi[0] <= v <= lo_hi[1]):
            res.rejected.append({"name": name, "raw_value": p.raw_value, "unit": p.unit,
                                 "converted_value": round(v, 3),
                                 "reason": f"Outside plausible range {lo_hi} {unit_out} - possible extraction error."})
            continue

        if p.raw_value and name in ("glucose_unspecified",):
            note = (note + " " if note else "") + "Report does not say fasting/random/post-meal."
            review = True

        res.params[name] = NormalizedParam(
            name=name, value=round(v, 4), unit=unit_out, ref_low=lo, ref_high=hi, ref_by_sex=sx,
            report_flag=p.report_flag, raw_value=p.raw_value, original_unit=p.unit, converted=converted,
            unit_assumed=assumed, kind=p.kind, needs_review=review, note=note)
    return res