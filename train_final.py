"""
Final models for the app (steps 2-6)
=====================================
For each disease:
  1. Fix the feature set to what the app can realistically supply (report-only,
     or report + a short manual-entry form) - chosen from feature_variant_results.csv.
  2. Train Logistic Regression (chosen because every disease tied or nearly tied
     the best model, and LR is by far the easiest to explain to a reviewer/user).
  3. Get out-of-fold probabilities (5-fold CV) and pick a decision threshold that
     reaches a target recall, reporting the precision/specificity cost.
  4. Calibrate probabilities (isotonic) so displayed percentages are meaningful.
  5. Refit on ALL data, build a SHAP explainer, and a per-patient explain function.
  6. Save one self-contained .joblib per disease with: pipeline, calibrator,
     feature list, required/optional fields, threshold, and metrics - everything
     the FastAPI service needs.
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, precision_recall_curve, recall_score,
                             precision_score, f1_score, accuracy_score, confusion_matrix)

import disease_model_comparison as base   # loaders, build_pre()

OUT = base.OUT
SEED = base.SEED
os.makedirs(f"{OUT}/final_models", exist_ok=True)
os.makedirs(f"{OUT}/plots/shap", exist_ok=True)

# ----------------------------------------------------------------------------
# 1. FEATURE SET PER DISEASE  (from feature_variant_results.csv / domain sense)
#    "required"  = must be present on the report or the app should not predict
#    "optional"  = used if available, imputed with training median otherwise
# ----------------------------------------------------------------------------
DISEASE_CONFIG = {
    "Diabetes": dict(
        loader=base.load_diabetes,
        features=["Glucose", "Insulin", "Age", "BMI", "BloodPressure", "Pregnancies"],
        required=["Glucose", "Age"],          # Insulin often not on basic panels
        optional=["Insulin", "BMI", "BloodPressure", "Pregnancies"],
        cat_cols=[], target_recall=0.85,
        source="Report + manual entry (BMI, blood pressure, pregnancies are typed in)."),
    "Heart": dict(
        loader=base.load_heart,
        features=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak"],
        required=["age", "sex", "chol"],
        optional=["cp", "trestbps", "fbs", "restecg", "thalch", "exang", "oldpeak"],
        cat_cols=["cp", "restecg"], target_recall=0.85,
        source="Report (chol, fbs) + short clinical questionnaire (cp, ECG, exercise angina)."),
    "Kidney": dict(
        loader=base.load_kidney,
        features=["age", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc",
                  "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"],
        required=["sc", "bu", "hemo"],        # creatinine, urea, hemoglobin - core CKD markers
        optional=["age", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr",
                  "sod", "pot", "pcv", "wbcc", "rbcc"],
        cat_cols=[], target_recall=0.9,
        source="Report-only (blood chemistry + urine routine + CBC)."),
    "Liver": dict(
        loader=base.load_liver,
        features=["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
                  "ALT_SGPT", "AST_SGOT", "Total_Proteins", "Albumin", "AG_Ratio"],
        required=["Total_Bilirubin", "ALT_SGPT", "AST_SGOT"],
        optional=["Age", "Gender", "Direct_Bilirubin", "Alkaline_Phosphotase", "Total_Proteins",
                  "Albumin", "AG_Ratio"],
        cat_cols=[], target_recall=0.85,
        source="Report-only (liver function panel)."),
    "Anemia": dict(
        loader=base.load_anemia,
        features=["Gender", "Age", "Hemoglobin", "RBC", "PCV_HCT", "MCV", "MCH", "MCHC"],
        required=["Hemoglobin", "MCV"],
        optional=["Gender", "Age", "RBC", "PCV_HCT", "MCH", "MCHC"],
        cat_cols=[], target_recall=0.85,
        source="Report-only (CBC)."),
}


# ----------------------------------------------------------------------------
# 2. THRESHOLD SELECTION FROM OUT-OF-FOLD PROBABILITIES
# ----------------------------------------------------------------------------
def pick_threshold(y, oof_proba, target_recall):
    """Smallest threshold whose recall is still >= target_recall (maximises precision
    subject to the recall floor); falls back to the threshold closest to the target."""
    prec, rec, thr = precision_recall_curve(y, oof_proba)
    prec, rec = prec[:-1], rec[:-1]           # drop the (1, 0) point precision_recall_curve appends
    ok = rec >= target_recall
    if ok.any():
        best_i = np.argmax(prec[ok])          # highest precision among thresholds meeting recall floor
        thr_ok, prec_ok = thr[ok], prec[ok]
        t = thr_ok[best_i]
    else:
        t = thr[np.argmin(np.abs(rec - target_recall))]
    pred = (oof_proba >= t).astype(int)
    return dict(threshold=float(t),
               recall=float(recall_score(y, pred)), precision=float(precision_score(y, pred)),
               f1=float(f1_score(y, pred)), accuracy=float(accuracy_score(y, pred)),
               confusion_matrix=confusion_matrix(y, pred).tolist())


# ----------------------------------------------------------------------------
# 3. PER-DISEASE PIPELINE
# ----------------------------------------------------------------------------
def run_disease(name, cfg):
    X_all, y, _, _ = cfg["loader"]()
    X = X_all[cfg["features"]].copy()
    cat_cols = cfg["cat_cols"]
    pre = base.build_pre(X, cat_cols)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)

    # --- out-of-fold probabilities (5-fold, same protocol as the comparison) ---
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("pre", clone(pre)), ("clf", clf)])
    oof_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    oof_auc = roc_auc_score(y, oof_proba)

    # --- threshold must be chosen on CALIBRATED out-of-fold probabilities: app.py compares the
    #     calibrated probability to it (picking it on raw scores broke the recall target) ---
    cal_oof = cross_val_predict(
        CalibratedClassifierCV(clone(pipe), method="isotonic",
                               cv=StratifiedKFold(5, shuffle=True, random_state=SEED)),
        X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1),
        method="predict_proba", n_jobs=1)[:, 1]
    thr_info = pick_threshold(y, cal_oof, cfg["target_recall"])

    # --- refit final pipeline on ALL data ---
    final_pipe = Pipeline([("pre", clone(pre)), ("clf", clone(clf))]).fit(X, y)

    # --- probability calibration (isotonic, 5-fold, fit on the SAME final pipeline shape) ---
    calibrator = CalibratedClassifierCV(clone(pipe), method="isotonic", cv=cv).fit(X, y)

    # --- SHAP: LinearExplainer on the transformed (scaled/encoded) feature space ---
    Xt = final_pipe.named_steps["pre"].transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    feat_names = list(final_pipe.named_steps["pre"].get_feature_names_out())
    explainer = shap.LinearExplainer(final_pipe.named_steps["clf"], Xt, feature_perturbation="interventional")
    shap_values = explainer.shap_values(Xt)

    # global importance plot
    order = np.argsort(-np.abs(shap_values).mean(axis=0))
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(feat_names) + 1))
    ax.barh([feat_names[i] for i in order][::-1], np.abs(shap_values).mean(axis=0)[order][::-1], color="#1F4E78")
    ax.set_xlabel("Mean |SHAP value| (impact on predicted risk)")
    ax.set_title(f"{name}: feature importance (Logistic Regression)")
    fig.tight_layout(); fig.savefig(f"{OUT}/plots/shap/{name.lower()}_importance.png", dpi=140); plt.close(fig)

    # --- one example per-patient explanation (first test-like row: median patient) ---
    example_idx = int(np.argsort(np.abs(oof_proba - 0.5))[0])   # a borderline case, most informative
    example = explain_one(final_pipe, explainer, feat_names, X.iloc[[example_idx]])

    artifact = dict(
        disease=name, algorithm="Logistic Regression",
        pipeline=final_pipe, calibrator=calibrator,
        feature_names=cfg["features"], categorical_features=cat_cols,
        required_fields=cfg["required"], optional_fields=cfg["optional"],
        source_description=cfg["source"],
        threshold=thr_info["threshold"], target_recall=cfg["target_recall"],
        threshold_metrics=thr_info,
        oof_roc_auc=float(oof_auc),
        shap_feature_names=feat_names,
        shap_background=Xt,             # small dataset -> fine to keep as the SHAP background
        trained_on_n_rows=len(X))
    joblib.dump(artifact, f"{OUT}/final_models/{name.lower()}_final.joblib")

    return dict(Disease=name, Features_used=len(cfg["features"]),
               Required_fields=", ".join(cfg["required"]),
               OOF_ROC_AUC=round(oof_auc, 4), Threshold=round(thr_info["threshold"], 3),
               Target_recall=cfg["target_recall"], Recall_at_threshold=round(thr_info["recall"], 3),
               Precision_at_threshold=round(thr_info["precision"], 3),
               Accuracy_at_threshold=round(thr_info["accuracy"], 3),
               Recall_at_0_5=round(recall_score(y, (cal_oof >= 0.5).astype(int)), 3),
               Precision_at_0_5=round(precision_score(y, (cal_oof >= 0.5).astype(int)), 3),
               Top_3_factors_example=", ".join(f["feature"] for f in example[:3]))


def explain_one(pipe, explainer, feat_names, X_row):
    """Return top contributing factors for ONE patient row (a DataFrame with 1 row of
    raw feature values, same columns the pipeline was trained on)."""
    Xt = pipe.named_steps["pre"].transform(X_row)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    sv = explainer.shap_values(Xt)[0]
    proba = pipe.named_steps["clf"].predict_proba(Xt)[0, 1]
    order = np.argsort(-np.abs(sv))
    out = []
    for i in order:
        out.append(dict(feature=feat_names[i], shap_value=round(float(sv[i]), 4),
                        direction="increases risk" if sv[i] > 0 else "decreases risk"))
    return out  # caller slices top N; predicted_probability = proba available if needed


# ----------------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------------
def main():
    rows = [run_disease(name, cfg) for name, cfg in DISEASE_CONFIG.items()]
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT}/final_model_summary.csv", index=False)

    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    xp = f"{OUT}/final_model_summary.xlsx"
    with pd.ExcelWriter(xp, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Final_Models", index=False)
        pd.DataFrame({"Notes": [
            "All final models are Logistic Regression, chosen because every disease's model "
            "comparison showed LR tied (within 1 SD) or near the best ROC-AUC, and LR is the "
            "most explainable choice for a medical screening tool.",
            "Feature sets are restricted to what the app can obtain: a lab report, plus a short "
            "manual-entry form where noted (see Diabetes/Heart source description in the .joblib).",
            "OOF_ROC_AUC is out-of-fold (5-fold CV), the honest estimate for this feature set - "
            "compare to feature_variant_results.csv, values will be close but not identical (this "
            "run uses cross_val_predict rather than nested grid-search, no hyperparameter tuning).",
            "Threshold: smallest probability cutoff whose recall is still >= target_recall, chosen "
            "for the highest precision at that recall (from CALIBRATED out-of-fold probabilities, the same "
            "quantity app.py compares to the threshold).",
            "Recall_at_0_5 / Precision_at_0_5 show the cost of NOT adjusting the threshold - compare "
            "to the Recall/Precision_at_threshold columns to see what the adjustment bought.",
            "Calibrator (isotonic, 5-fold) is saved alongside the model - use it to convert raw "
            "model scores into probabilities that should be shown to the user.",
            "Kidney/Liver/Anemia are report-only. Diabetes and Heart need extra manual fields; "
            "the app should show those as a short form, not attempt to extract them from text."]}
        ).to_excel(w, sheet_name="Notes", index=False)
    wb = load_workbook(xp)
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                c.font = Font(name="Arial", size=10, bold=(c.row == 1), color="FFFFFF" if c.row == 1 else "000000")
                if c.row == 1: c.fill = PatternFill("solid", fgColor="1F4E78")
                c.alignment = Alignment(wrap_text=(ws.title == "Notes"), vertical="top")
        for col in ws.columns:
            w_ = max(len(str(c.value)) if c.value is not None else 0 for c in col[:20])
            ws.column_dimensions[col[0].column_letter].width = min(max(12, w_ + 2), 90 if ws.title == "Notes" else 30)
        ws.freeze_panes = "A2"
    wb.save(xp)

    pd.set_option("display.width", 220)
    print("\n=== FINAL MODELS ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()