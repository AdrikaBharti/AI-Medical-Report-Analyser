"""
Final models for the app - BEST MODEL PER DISEASE
====================================================
Differs from train_final.py (which always deploys Logistic Regression) by
choosing whichever of the 6 algorithms scores best ON THE DEPLOYMENT FEATURE
SET for each disease - not the full-feature comparison from
disease_model_comparison.py, since the best algorithm can change once features
are restricted to what the app can actually supply (see feature_variants.py).

Steps per disease:
  1. Run all 6 models (same GridSearchCV grids as disease_model_comparison.py)
     via nested cross_val_predict on THIS disease's deployment feature set.
  2. Pick the model with the highest OOF ROC-AUC (F1 as tie-break).
  3. Pick a decision threshold from that model's OOF probabilities, targeting
     the disease's recall floor.
  4. Refit the tuned pipeline on all data; wrap it in isotonic calibration.
  5. Build the RIGHT KIND of SHAP explainer for whichever algorithm won:
       Logistic Regression      -> LinearExplainer (exact, fast)
       Decision Tree / RF / XGB -> TreeExplainer   (exact, fast)
       SVM / MLP                -> Explainer(predict_proba, background)
                                    (model-agnostic permutation method - slower,
                                    but these have no closed-form SHAP solution)
  6. Save one .joblib per disease with everything app.py needs, INCLUDING
     which SHAP path to use at inference time.
"""
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, precision_recall_curve, recall_score,
                             precision_score, f1_score, accuracy_score, confusion_matrix)

import disease_model_comparison as base
from train_final import DISEASE_CONFIG, pick_threshold   # re-use feature sets + threshold logic

OUT = base.OUT
SEED = base.SEED
os.makedirs(f"{OUT}/final_models_best", exist_ok=True)
os.makedirs(f"{OUT}/plots/shap_best", exist_ok=True)


# ----------------------------------------------------------------------------
# 1. COMPARE ALL 6 MODELS ON THE DEPLOYMENT FEATURE SET, PICK THE BEST
# ----------------------------------------------------------------------------
def select_best_model(X, y, cat_cols):
    pre = base.build_pre(X, cat_cols)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, oof = [], {}
    for name in base.MODELS:
        est = base.make_search(name, pre, y)      # GridSearchCV, same grids as the main comparison
        proba = cross_val_predict(est, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        auc = roc_auc_score(y, proba)
        rows.append(dict(Model=name, ROC_AUC=round(auc, 4),
                         F1=round(f1_score(y, (proba >= 0.5).astype(int)), 4)))
        oof[name] = proba
    tab = pd.DataFrame(rows).sort_values(["ROC_AUC", "F1"], ascending=False).reset_index(drop=True)
    best = tab.iloc[0]["Model"]
    return best, tab, oof[best]


# ----------------------------------------------------------------------------
# 2. SHAP: right explainer for the algorithm that won, with a uniform output
# ----------------------------------------------------------------------------
SHAP_KIND = {"Logistic Regression": "linear",
            "Decision Tree": "tree", "Random Forest": "tree", "XGBoost": "tree",
            "SVM": "generic", "MLP Neural Network": "generic"}


def build_explainer(algo, clf, Xt_background):
    kind = SHAP_KIND[algo]
    if kind == "linear":
        return shap.LinearExplainer(clf, Xt_background, feature_perturbation="interventional"), kind
    if kind == "tree":
        return shap.TreeExplainer(clf), kind
    bg = shap.sample(Xt_background, min(50, len(Xt_background)), random_state=SEED)
    return shap.Explainer(clf.predict_proba, bg), kind


def shap_values_positive_class(explainer, kind, Xt):
    sv = explainer(Xt).values
    return sv[:, :, 1] if sv.ndim == 3 else sv     # linear is 2D (single output); tree/generic are 3D


# ----------------------------------------------------------------------------
# 3. PER-DISEASE PIPELINE
# ----------------------------------------------------------------------------
def run_disease(name, cfg):
    X_all, y, _, _ = cfg["loader"]()
    X = X_all[cfg["features"]].copy()
    cat_cols = cfg["cat_cols"]
    pre = base.build_pre(X, cat_cols)

    best_algo, comparison_table, oof_proba = select_best_model(X, y, cat_cols)
    oof_auc = roc_auc_score(y, oof_proba)          # AUC is rank-based, so raw vs calibrated barely matters

    # refit the WINNING algorithm (tuned via its own grid) on all data
    gs = base.make_search(best_algo, pre, y)
    gs.fit(X, y)
    final_pipe = gs.best_estimator_               # already a fitted Pipeline(pre + tuned clf)

    # THRESHOLD: app.py compares the CALIBRATED probability to the threshold, so the threshold must be
    # chosen on calibrated out-of-fold probabilities (choosing it on raw scores broke the recall target).
    cal_oof = cross_val_predict(
        CalibratedClassifierCV(clone(final_pipe), method="isotonic",
                               cv=StratifiedKFold(5, shuffle=True, random_state=SEED)),
        X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1),
        method="predict_proba", n_jobs=1)[:, 1]
    thr_info = pick_threshold(y, cal_oof, cfg["target_recall"])

    # calibrate a fresh copy with the SAME tuned hyperparameters (avoids re-grid-searching
    # inside every calibration fold, which would be very slow for SVM/RF/MLP)
    calibrator = CalibratedClassifierCV(clone(final_pipe), method="isotonic",
                                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED)).fit(X, y)

    # SHAP on the transformed feature space of the FINAL fitted pipeline
    Xt = final_pipe.named_steps["pre"].transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    feat_names = list(final_pipe.named_steps["pre"].get_feature_names_out())
    explainer, shap_kind = build_explainer(best_algo, final_pipe.named_steps["clf"], Xt)
    # For the GLOBAL importance plot only, cap rows for generic (SVM/MLP) explainers - the
    # permutation/exact method re-evaluates the model many times PER ROW, so all 977 anemia
    # rows (say) took 6+ minutes for Liver's SVM. Per-patient calls in app.py explain ONE row,
    # which is fast regardless; this cap only affects this one plot.
    Xt_for_plot = Xt if shap_kind == "tree" or len(Xt) <= 150 else Xt[np.random.RandomState(SEED).choice(len(Xt), 150, replace=False)]
    shap_vals = shap_values_positive_class(explainer, shap_kind, Xt_for_plot)

    order = np.argsort(-np.abs(shap_vals).mean(axis=0))
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(feat_names) + 1))
    ax.barh([feat_names[i] for i in order][::-1], np.abs(shap_vals).mean(axis=0)[order][::-1], color="#1F4E78")
    ax.set_xlabel("Mean |SHAP value| (impact on predicted risk)")
    ax.set_title(f"{name}: feature importance ({best_algo})")
    fig.tight_layout(); fig.savefig(f"{OUT}/plots/shap_best/{name.lower()}_importance.png", dpi=140); plt.close(fig)

    artifact = dict(
        disease=name, algorithm=best_algo, shap_kind=shap_kind,
        pipeline=final_pipe, calibrator=calibrator,
        feature_names=cfg["features"], categorical_features=cat_cols,
        required_fields=cfg["required"], optional_fields=cfg["optional"],
        source_description=cfg["source"],
        threshold=thr_info["threshold"], target_recall=cfg["target_recall"],
        threshold_metrics=thr_info, oof_roc_auc=float(oof_auc),
        model_comparison_on_this_feature_set=comparison_table.to_dict("records"),
        shap_feature_names=feat_names, shap_background=Xt,
        trained_on_n_rows=len(X))
    joblib.dump(artifact, f"{OUT}/final_models_best/{name.lower()}_final.joblib")

    return dict(Disease=name, Best_algorithm=best_algo, Features_used=len(cfg["features"]),
               OOF_ROC_AUC=round(oof_auc, 4), Threshold=round(thr_info["threshold"], 3),
               Recall_at_threshold=round(thr_info["recall"], 3),
               Precision_at_threshold=round(thr_info["precision"], 3),
               Runner_up=comparison_table.iloc[1]["Model"],
               Runner_up_AUC=comparison_table.iloc[1]["ROC_AUC"],
               Gap_to_runner_up=round(comparison_table.iloc[0]["ROC_AUC"] - comparison_table.iloc[1]["ROC_AUC"], 4))


def main():
    rows, all_comparisons = [], []
    for name, cfg in DISEASE_CONFIG.items():
        r = run_disease(name, cfg)
        rows.append(r)
        print(f"{name:9s} -> {r['Best_algorithm']:20s} AUC={r['OOF_ROC_AUC']:.4f}  "
             f"(runner-up {r['Runner_up']}: {r['Runner_up_AUC']:.4f}, gap {r['Gap_to_runner_up']:.4f})",
             flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT}/final_model_best_summary.csv", index=False)

    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    xp = f"{OUT}/final_model_best_summary.xlsx"
    with pd.ExcelWriter(xp, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Best_Model_Per_Disease", index=False)
        pd.DataFrame({"Notes": [
            "Each disease's 6 models were compared on ITS OWN deployment feature set (report-only "
            "or report+manual, per DISEASE_CONFIG in train_final.py) - NOT the full-feature "
            "comparison in disease_model_comparison.py. The winning algorithm can therefore differ "
            "from the earlier full-feature results.",
            "Gap_to_runner_up under about 0.01-0.02 AUC means the 'win' is within normal fold-to-fold "
            "noise - report the runner-up honestly rather than implying a decisive victory.",
            "SHAP method differs by algorithm: LinearExplainer (Logistic Regression) and "
            "TreeExplainer (Decision Tree/Random Forest/XGBoost) are exact. SVM and MLP have no "
            "closed-form SHAP solution, so a model-agnostic permutation explainer is used instead - "
            "slower per prediction and an approximation, not an exact decomposition.",
            "Threshold is chosen on CALIBRATED out-of-fold probabilities (same rule app.py applies at "
            "prediction time), so recall at the threshold matches the target_recall in DISEASE_CONFIG."]}
        ).to_excel(w, sheet_name="Notes", index=False)
    wb = load_workbook(xp)
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                c.font = Font(name="Arial", size=10, bold=(c.row == 1), color="FFFFFF" if c.row == 1 else "000000")
                if c.row == 1: c.fill = PatternFill("solid", fgColor="1F4E78")
                c.alignment = Alignment(wrap_text=(ws.title == "Notes"), vertical="top")
        for col in ws.columns:
            w_ = max(len(str(c.value)) if c.value is not None else 0 for c in col[:10])
            ws.column_dimensions[col[0].column_letter].width = min(max(14, w_ + 2), 90 if ws.title == "Notes" else 30)
    wb.save(xp)

    print("\n=== BEST MODEL PER DISEASE (deployment feature set) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()