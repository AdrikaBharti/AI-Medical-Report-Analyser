"""
report_pipeline.py - glue for the whole extraction side:

  bytes -> report_extractor -> parameter_extractor -> unit_normalizer -> abnormal_detector
        -> build_model_inputs()  (flat dict with the EXACT feature names app.py / the .joblib models expect)

It does NOT call the models - app.py does that with the returned `model_inputs`
(keeps this package free of sklearn/shap imports and easy to unit-test).
"""
from __future__ import annotations
from typing import Dict, Optional

import report_extractor
import parameter_extractor
import unit_normalizer
import abnormal_detector

UREA_PER_BUN = 2.14         # blood urea (mg/dL) = BUN (mg/dL) x 2.14


def build_model_inputs(norm: Dict[str, unit_normalizer.NormalizedParam],
                       age: Optional[float], sex: Optional[str]) -> tuple[dict, list]:
    """Map lab-level names -> the feature names used in training.
    Returns (flat dict, notes). Missing values are simply omitted (app.py imputes optional ones)."""
    v = {k: p.value for k, p in norm.items()}
    notes: list = []
    x: Dict[str, float] = {}

    def put(key, val):
        if val is not None:
            x[key] = float(val)

    sex_code = None if sex is None else (1.0 if sex == "male" else 0.0)

    # ---- demographics (dataset uses different casings per disease; send all) ----
    for k in ("age", "Age"):
        put(k, age)
    for k in ("sex", "Gender"):
        put(k, sex_code)

    # ---- glucose: pick the type that matches each model's training definition ----
    fasting, pp, rnd, unk = (v.get("glucose_fasting"), v.get("glucose_pp"),
                             v.get("glucose_random"), v.get("glucose_unspecified"))
    # Diabetes (Pima) 'Glucose' = 2-hour plasma glucose in an OGTT
    g = next((val for val in (pp, fasting, rnd, unk) if val is not None), None)
    put("Glucose", g)
    if g is not None and pp is None:
        notes.append("Diabetes model was trained on 2-hour (post-load) glucose; a fasting/random value was "
                     "used instead, so treat the diabetes result as approximate.")
    # Kidney 'bgr' = random blood glucose
    put("bgr", next((val for val in (rnd, unk, pp, fasting) if val is not None), None))
    # Heart 'fbs' = fasting glucose > 120 mg/dL (only meaningful if the value really is fasting)
    if fasting is not None:
        put("fbs", 1.0 if fasting > 120 else 0.0)

    put("Insulin", v.get("insulin"))
    put("chol", v.get("total_cholesterol"))
    put("BMI", v.get("bmi"))
    put("trestbps", v.get("systolic_bp"))          # Heart: resting systolic
    put("BloodPressure", v.get("diastolic_bp"))    # Diabetes (Pima): diastolic

    # ---- kidney ----
    urea = v.get("blood_urea")
    if urea is None and v.get("bun") is not None:
        urea = v["bun"] * UREA_PER_BUN
        notes.append("Only BUN was reported; converted to blood urea (x2.14) for the kidney model.")
    put("bu", urea)
    for src, dst in [("serum_creatinine", "sc"), ("sodium", "sod"), ("potassium", "pot"), ("hemoglobin", "hemo"),
                     ("pcv", "pcv"), ("wbc_count", "wbcc"), ("rbc_count", "rbcc"), ("urine_sg", "sg"),
                     ("urine_albumin", "al"), ("urine_sugar", "su"), ("urine_rbc", "rbc"),
                     ("urine_pus_cells", "pc"), ("pus_cell_clumps", "pcc"), ("bacteria", "ba")]:
        put(dst, v.get(src))

    # ---- liver ----
    for src, dst in [("total_bilirubin", "Total_Bilirubin"), ("direct_bilirubin", "Direct_Bilirubin"),
                     ("alp", "Alkaline_Phosphotase"), ("alt", "ALT_SGPT"), ("ast", "AST_SGOT"),
                     ("total_protein", "Total_Proteins"), ("albumin", "Albumin")]:
        put(dst, v.get(src))
    ag = v.get("ag_ratio")
    if ag is None and v.get("albumin") and v.get("total_protein") and v["total_protein"] > v["albumin"]:
        ag = v["albumin"] / (v["total_protein"] - v["albumin"])
        notes.append("A/G ratio was calculated from albumin and total protein.")
    put("AG_Ratio", ag)

    # ---- anemia (CBC) ----
    for src, dst in [("hemoglobin", "Hemoglobin"), ("rbc_count", "RBC"), ("pcv", "PCV_HCT"),
                     ("mcv", "MCV"), ("mch", "MCH"), ("mchc", "MCHC")]:
        put(dst, v.get(src))

    return x, notes


def analyze_report(data: bytes, filename: str, *, age: Optional[float] = None,
                   sex: Optional[str] = None, manual_values: Optional[dict] = None,
                   lang: str = "eng") -> dict:
    """Run extraction -> parsing -> normalization -> abnormal detection -> model-input mapping.
    `age`, `sex` and `manual_values` (e.g. {"BMI": 27, "Pregnancies": 2, "cp": "asymptomatic"}) override/extend
    what the report contains - the report header is often missing them."""
    ext = report_extractor.extract_report(data, filename, lang)
    parsed = parameter_extractor.parse_parameters(ext.text)

    warnings = list(ext.warnings) + list(parsed.warnings)
    if not parsed.params:
        warnings.append("No lab values were recognised in this file. Is it a lab report? "
                        "You can still enter values manually.")

    norm = unit_normalizer.normalize(parsed.params, from_ocr=ext.used_ocr)
    warnings += norm.warnings

    final_age = age if age is not None else parsed.demographics.get("age")
    final_sex = (sex or parsed.demographics.get("sex"))
    final_sex = final_sex.lower() if isinstance(final_sex, str) else None
    if final_sex in ("m", "male"):
        final_sex = "male"
    elif final_sex in ("f", "female"):
        final_sex = "female"
    else:
        final_sex = None
    if final_age is None:
        warnings.append("Age was not found on the report - please enter it.")
    if final_sex is None:
        warnings.append("Sex was not found on the report - please enter it.")

    findings = abnormal_detector.detect_abnormal(norm.params, final_sex)
    model_inputs, notes = build_model_inputs(norm.params, final_age, final_sex)
    warnings += notes

    for k, val in (manual_values or {}).items():           # user-typed values always win
        if val not in (None, ""):
            model_inputs[k] = val

    return {
        "extraction": {"method": ext.method, "pages": ext.pages, "ocr_pages": ext.ocr_pages,
                       "mean_ocr_confidence": ext.mean_ocr_confidence},
        "demographics": {"age": final_age, "sex": final_sex},
        "parameters": {k: p.to_dict() for k, p in norm.params.items()},
        "rejected_values": norm.rejected,
        "abnormal_findings": findings,
        "model_inputs": model_inputs,
        "warnings": warnings,
        "requires_user_review": bool(ext.used_ocr or norm.rejected
                                     or any(p.needs_review for p in norm.params.values())),
    }