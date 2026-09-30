"""
AI Medical Report Analyzer - Prediction API
============================================
Loads the 5 disease .joblib artifacts from train_final.py and serves:

  GET  /diseases                     -> which diseases are available, required/optional fields
  POST /predict/{disease}            -> prediction + calibrated probability + top SHAP factors
  POST /predict/batch                -> run every disease whose required fields are present

Run:  uvicorn app:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""
import os
from typing import Optional, Dict, List, Union

Number = Union[float, str]   # Heart's cp/restecg are categorical strings, not floats
import numpy as np
import pandas as pd
import joblib
import shap
import train_final_best as tfb   # build_explainer() / shap_values_positive_class(): picks the
                                 # right SHAP method per algorithm - see train_final_best.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form   # NEW: UploadFile/File/Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model
import json
import report_pipeline                       # NEW: PDF/OCR -> parameters -> model inputs
from report_extractor import UnsupportedFileType, SUPPORTED_EXT
from pdf_extractor import PDFReadError
from ocr_extractor import OCRError

MODELS_DIR = os.environ.get("MODELS_DIR", "outputs\\final_models_best")   # "final_models" = all-LR version

app = FastAPI(title="AI Medical Report Analyzer", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ARTIFACTS: Dict[str, dict] = {}
EXPLAINERS: Dict[str, object] = {}
SHAP_KIND: Dict[str, str] = {}   # "linear" / "tree" / "generic" - which extraction path to use


@app.on_event("startup")
def load_models():
    if not os.path.isdir(MODELS_DIR):
        raise RuntimeError(f"Models directory not found: {MODELS_DIR}")
    for fn in os.listdir(MODELS_DIR):
        if fn.endswith("_final.joblib"):
            art = joblib.load(os.path.join(MODELS_DIR, fn))
            name = art["disease"]
            ARTIFACTS[name] = art
            # older final_models/ (all-Logistic-Regression) artifacts have no "algorithm" saved
            # the same way - default to Logistic Regression, which build_explainer maps to "linear"
            algo = art.get("algorithm", "Logistic Regression")
            EXPLAINERS[name], SHAP_KIND[name] = tfb.build_explainer(
                algo, art["pipeline"].named_steps["clf"], art["shap_background"])
    if not ARTIFACTS:
        raise RuntimeError(f"No *_final.joblib files found in {MODELS_DIR}")
    print(f"Loaded models for: {list(ARTIFACTS)}")


# ----------------------------------------------------------------------------
# Request/response schemas
# ----------------------------------------------------------------------------
class PredictionResult(BaseModel):
    disease: str
    prediction: int                     # 1 = disease indicated, 0 = not indicated
    label: str                          # human-readable
    probability: float                  # calibrated risk SCORE (isotonic-calibrated) - NOT an individualised clinical probability
    threshold_used: float
    data_completeness: str              # "low" / "borderline" / "high" - SHARE OF FIELDS supplied, NOT model confidence
    fields_used: List[str]
    fields_missing: List[str]
    top_factors: List[dict]             # [{feature, contribution, direction}]
    disclaimer: str = ("This is a screening aid, not a medical diagnosis. "
                       "Please consult a qualified physician.")


class BatchResult(BaseModel):
    results: Dict[str, Optional[PredictionResult]]
    skipped: Dict[str, str]             # disease -> reason skipped (e.g. missing required fields)


DISCLAIMER = PredictionResult.model_fields["disclaimer"].default


# ----------------------------------------------------------------------------
# Core prediction logic (shared by /predict/{disease} and /predict/batch)
# ----------------------------------------------------------------------------
def _build_row(art: dict, values: dict) -> pd.DataFrame:
    row = {f: values.get(f, np.nan) for f in art["feature_names"]}
    return pd.DataFrame([row], columns=art["feature_names"])


def _predict_one(disease: str, values: dict) -> PredictionResult:
    if disease not in ARTIFACTS:
        raise HTTPException(404, f"Unknown disease '{disease}'. Available: {list(ARTIFACTS)}")
    art = ARTIFACTS[disease]

    missing_required = [f for f in art["required_fields"] if values.get(f) in (None, "")]
    if missing_required:
        raise HTTPException(422, f"Missing required field(s) for {disease}: {missing_required}. "
                                 f"Not enough data to predict.")

    fields_present = [f for f in art["feature_names"] if values.get(f) not in (None, "")]
    fields_missing = [f for f in art["feature_names"] if f not in fields_present]

    X = _build_row(art, values)
    pipe = art["pipeline"]
    raw_proba = pipe.predict_proba(X)[0, 1]                 # what SHAP explains (see top_factors)
    cal_proba = art["calibrator"].predict_proba(X)[0, 1]     # display THIS one to users
    pred = int(cal_proba >= art["threshold"])

    Xt = pipe.named_steps["pre"].transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    sv = tfb.shap_values_positive_class(EXPLAINERS[disease], SHAP_KIND[disease], Xt)[0]
    order = np.argsort(-np.abs(sv))[:5]
    top_factors = [dict(feature=art["shap_feature_names"][i],
                        contribution=round(float(sv[i]), 4),
                        direction="increases risk" if sv[i] > 0 else "decreases risk")
                   for i in order]

    completeness = len(fields_present) / len(art["feature_names"])
    completeness_band = "high" if completeness >= 0.8 else "borderline" if completeness >= 0.5 else "low"

    return PredictionResult(
        disease=disease, prediction=pred,
        label="Indicated - please consult a doctor" if pred else "Not indicated",
        probability=round(float(cal_proba), 4), threshold_used=art["threshold"],
        data_completeness=completeness_band, fields_used=fields_present, fields_missing=fields_missing,
        top_factors=top_factors)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/diseases")
def list_diseases():
    return {name: dict(required_fields=art["required_fields"],
                       optional_fields=art["optional_fields"],
                       all_fields=art["feature_names"],
                       source=art["source_description"],
                       oof_roc_auc=art["oof_roc_auc"], threshold=art["threshold"])
           for name, art in ARTIFACTS.items()}


def _run_all(values: dict):
    """Run every disease whose required fields are present; skip the rest."""
    results, skipped = {}, {}
    for name, art in ARTIFACTS.items():
        missing = [f for f in art["required_fields"] if values.get(f) in (None, "")]
        if missing:
            skipped[name] = f"Missing required field(s): {missing}"
            results[name] = None
            continue
        results[name] = _predict_one(name, values)
    return results, skipped


@app.post("/predict/batch", response_model=BatchResult)
def predict_batch(values: Dict[str, Optional[Number]]):
    """Send ONE flat dict of every value extracted from the report / entered manually
    (field names can overlap across diseases, e.g. 'Age' / 'age' - send both casings if
    unsure). Runs every disease whose required fields are present and skips the rest."""
    results, skipped = _run_all(values)
    return BatchResult(results=results, skipped=skipped)


MAX_UPLOAD_MB = 15


@app.post("/analyze-report")
def analyze_report(file: UploadFile = File(...),
                   age: Optional[float] = Form(None),
                   sex: Optional[str] = Form(None),                 # "male" / "female" (overrides the report)
                   manual_values: Optional[str] = Form(None)):      # JSON string, e.g. {"BMI": 27, "Pregnancies": 2}
    """Upload a lab report (PDF / PNG / JPG / TIFF). Returns what was extracted, which values are
    abnormal, and a prediction (with SHAP factors) for every disease that has enough data.
    Plain `def` on purpose: OCR is CPU-bound, and FastAPI runs sync routes in a worker thread."""
    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB.")
    manual = {}
    if manual_values:
        try:
            manual = json.loads(manual_values)
            assert isinstance(manual, dict)
        except Exception:
            raise HTTPException(422, "manual_values must be a JSON object, e.g. {\"BMI\": 27}")
    try:
        analysis = report_pipeline.analyze_report(data, file.filename or "", age=age, sex=sex,
                                                  manual_values=manual)
    except UnsupportedFileType as e:
        raise HTTPException(415, str(e))
    except (PDFReadError, OCRError) as e:
        raise HTTPException(422, str(e))

    results, skipped = _run_all(analysis["model_inputs"])
    analysis["predictions"] = {k: (v.model_dump() if v else None) for k, v in results.items()}
    analysis["skipped_diseases"] = skipped
    analysis["explanation"] = None        # placeholder for the LLM explanation step
    analysis["disclaimer"] = DISCLAIMER
    return analysis


@app.post("/predict/{disease}", response_model=PredictionResult)
def predict(disease: str, values: Dict[str, Optional[Number]]):
    """`values` is a flat dict of {feature_name: number}. Use GET /diseases to see the
    field names (feature_names) expected for that disease. Missing optional fields are
    imputed with the training median; missing required fields return HTTP 422.
    NOTE: this route is matched AFTER /predict/batch, so 'batch' can never be treated
    as a disease name."""
    return _predict_one(disease.capitalize(), values)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(ARTIFACTS)}