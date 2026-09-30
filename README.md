# AI Medical Report Analyzer - ML pipeline & API

## Run order
```
pip install -r requirements.txt          # or: pip install pandas scikit-learn xgboost shap
                                          #     matplotlib seaborn openpyxl joblib fastapi "uvicorn[standard]"

python disease_model_comparison.py       # cleans data, compares 6 models x 5 diseases
python feature_variants.py               # report-only vs full-feature comparison
python train_final.py                    # final Logistic Regression models + thresholds + SHAP
uvicorn app:app --reload --port 8000     # prediction API, docs at /docs
```
Set `UPLOADS=<path to datasets>` and `OUTDIR=<path>` env vars if not using the defaults in the script.

## What each file does
| File | Purpose |
|---|---|
| `disease_model_comparison.py` | Cleans all 5 datasets, nested 5x3-fold CV comparison of 6 models |
| `feature_variants.py` | Same models restricted to report-only / report+manual feature sets |
| `train_final.py` | Final Logistic Regression per disease: recall-targeted threshold, isotonic calibration, SHAP explainer, saved `.joblib` |
| `app.py` | FastAPI service: `/diseases`, `/predict/{disease}`, `/predict/batch` |
| `final_models/*.joblib` | One artifact per disease: pipeline, calibrator, feature list, required/optional fields, threshold, SHAP background |
| `limitations_section.md` | Ready-to-paste limitations section for the project paper |
| `model_comparison_results.xlsx` | Full 6-model x 5-disease comparison |
| `feature_variant_comparison.xlsx` | Report-only vs full-feature comparison |
| `final_model_summary.xlsx` | Final model, threshold, and metrics per disease |
| `plots/` | Metric bars, heatmaps, ROC curves, confusion matrices, timing, SHAP importance |

## Final models
| Disease | Fields required | Optional fields | OOF ROC-AUC | Threshold (target recall) |
|---|---|---|---|---|
| Diabetes | Glucose, Age | Insulin, BMI, BloodPressure, Pregnancies | 0.83 | 0.374 (recall 0.85) |
| Heart | age, sex, chol | cp, trestbps, fbs, restecg, thalch, exang, oldpeak | 0.87 | 0.413 (recall 0.85) |
| Kidney | sc, bu, hemo | age, sg, al, su, rbc, pc, pcc, ba, bgr, sod, pot, pcv, wbcc, rbcc | 1.00 | 0.646 (recall 0.90) |
| Liver | Total_Bilirubin, ALT_SGPT, AST_SGOT | Age, Gender, Direct_Bilirubin, Alkaline_Phosphotase, Total_Proteins, Albumin, AG_Ratio | 0.74 | 0.327 (recall 0.85) |
| Anemia | Hemoglobin, MCV | Gender, Age, RBC, PCV_HCT, MCH, MCHC | 0.99 | 0.824 (recall 0.85) |

Below the required fields, `/predict/{disease}` returns HTTP 422 rather than a prediction.

## API quick reference
```
GET  /diseases                    -> required/optional fields, threshold, AUC per disease
POST /predict/{disease}           -> {"Glucose": 180, "Age": 50, ...}  ->  prediction, probability, top SHAP factors
POST /predict/batch               -> one flat dict of all extracted values; runs every disease with enough data
GET  /health                      -> loaded models
```
`probability` in the response is the **calibrated** value - display this one to users, not a raw model score.

## Still to build (Module 3-7 of the original plan)
- OCR / text extraction from uploaded report PDFs and images, mapped to the field names in `/diseases`
- Reference-range table (age/sex-specific) for flagging abnormal values
- Frontend: upload -> extracted values -> call `/predict/batch` -> show `top_factors` and disclaimer
- LLM explanation layer that turns `top_factors` into plain-language text (should explain only, never diagnose or prescribe)
