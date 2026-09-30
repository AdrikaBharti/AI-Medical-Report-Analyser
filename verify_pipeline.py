"""
verify_pipeline.py - does the whole thing give RIGHT output?   Run from the project root:

    python verify_pipeline.py                 # all checks
    python verify_pipeline.py --skip-models   # extraction checks only (no .joblib needed)

Checks
  A. EXTRACTION ACCURACY  : build reports from KNOWN values (digital PDF + scanned image), extract,
                            compare to the truth (units converted, SI units, qualitative urine).
  B. FEATURE CONTRACT     : every feature each model needs is either produced by the report pipeline
                            or listed as "manual-only" - no silent name mismatches.
  C. SANITY PREDICTIONS   : clearly healthy vs clearly diseased profiles go through the FULL path
                            (report -> extraction -> models -> SHAP) and must rank correctly.
  D. THRESHOLD / CALIBRATION : does the live rule (calibrated proba >= threshold) still reach the
                            recall each model was designed for?
"""
import io, sys, os, tempfile
import pypdfium2 as pdfium
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import report_pipeline as rp

SKIP_MODELS = "--skip-models" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


# ---------------------------------------------------------------- report factory
def make_pdf(lines):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4); c.setFont("Helvetica", 10); y = 810
    for ln in lines:
        c.drawString(40, y, ln); y -= 14
    c.save()
    return buf.getvalue()


def to_scanned_png(pdf_bytes):
    img = pdfium.PdfDocument(pdf_bytes)[0].render(scale=2.0).to_pil().convert("RGB")
    b = io.BytesIO(); img.save(b, "PNG"); return b.getvalue()


HEALTHY = [
    "Patient: Test Healthy      Age/Sex: 35 Y / M",
    "Hemoglobin 14.8 g/dL 13.0 - 17.0", "Total RBC Count 5.0 million/cumm 4.5 - 5.5", "PCV 44 % 40 - 50",
    "MCV 88 fL 80 - 100", "MCH 29.5 pg 27 - 32", "MCHC 33.5 g/dL 32 - 36",
    "Total Leucocyte Count 7,200 /cumm 4000 - 11000",
    "Blood Urea 26 mg/dL 15 - 40", "Serum Creatinine 0.9 mg/dL 0.7 - 1.3",
    "Sodium 140 mmol/L 135 - 145", "Potassium 4.2 mmol/L 3.5 - 5.1",
    "Total Bilirubin 0.7 mg/dL 0.1 - 1.2", "Direct Bilirubin 0.2 mg/dL 0.0 - 0.3",
    "SGOT (AST) 24 U/L 10 - 40", "SGPT (ALT) 26 U/L 7 - 56", "Alkaline Phosphatase 90 IU/L 44 - 147",
    "Total Protein 7.2 g/dL 6.0 - 8.3", "Albumin 4.4 g/dL 3.5 - 5.2", "A/G Ratio 1.6 ratio 1.0 - 2.2",
    "Fasting Blood Sugar 88 mg/dL 70 - 100", "Total Cholesterol 172 mg/dL",
    "URINE EXAMINATION", "Specific Gravity 1.020", "Albumin Nil", "Sugar Nil",
    "Pus Cells 1-2 /hpf", "RBC Nil /hpf", "Bacteria Absent", "BP: 118/76 mmHg"]
HEALTHY_TRUTH = {"age": 35, "sex": 1, "Hemoglobin": 14.8, "RBC": 5.0, "PCV_HCT": 44, "MCV": 88, "wbcc": 7200,
                 "bu": 26, "sc": 0.9, "sod": 140, "pot": 4.2, "Total_Bilirubin": 0.7, "AST_SGOT": 24,
                 "ALT_SGPT": 26, "Albumin": 4.4, "AG_Ratio": 1.6, "Glucose": 88, "fbs": 0, "chol": 172,
                 "sg": 1.020, "al": 0, "su": 0, "pc": 0, "rbc": 0, "ba": 0, "trestbps": 118, "BloodPressure": 76}

# same person in SI units + a CKD/anemia/liver-disease picture
SICK = [
    "Patient: Test Sick      Age/Sex: 58 Y / F",
    "Hemoglobin 82 g/L 120 - 150", "Total RBC Count 3.0 million/cumm 3.8 - 4.8", "PCV 24 % 36 - 46",
    "MCV 72 fL 80 - 100", "MCH 22 pg 27 - 32", "MCHC 30 g/dL 32 - 36",
    "Total Leucocyte Count 9.5 x10^3/uL 4.0 - 11.0",
    "Blood Urea 118 mg/dL 15 - 40", "Serum Creatinine 468 umol/L 53 - 97",
    "Sodium 132 mmol/L 135 - 145", "Potassium 5.8 mmol/L 3.5 - 5.1",
    "Total Bilirubin 68 umol/L 2 - 21", "Direct Bilirubin 30 umol/L 0 - 5",
    "SGOT (AST) 140 U/L 10 - 40", "SGPT (ALT) 160 U/L 7 - 56", "Alkaline Phosphatase 320 IU/L 44 - 147",
    "Total Protein 5.6 g/dL 6.0 - 8.3", "Albumin 2.6 g/dL 3.5 - 5.2", "A/G Ratio 0.8 ratio 1.0 - 2.2",
    "Random Blood Sugar 11.5 mmol/L", "Total Cholesterol 6.4 mmol/L",
    "URINE EXAMINATION", "Specific Gravity 1.010", "Albumin +++", "Sugar ++",
    "Pus Cells 10-15 /hpf", "RBC 6-8 /hpf", "Bacteria Present", "BP: 168/100 mmHg"]
SICK_TRUTH = {"age": 58, "sex": 0, "Hemoglobin": 8.2, "sc": 5.294, "wbcc": 9500, "Total_Bilirubin": 3.977,
              "Direct_Bilirubin": 1.754, "bgr": 207.2, "chol": 247.5, "sg": 1.010, "al": 3, "su": 2,
              "pc": 1, "rbc": 1, "ba": 1, "trestbps": 168, "BloodPressure": 100}


def close(a, b):
    return abs(a - b) <= max(0.02 * abs(b), 0.02)


def section_a():
    print("\nA. EXTRACTION ACCURACY (known values in -> same values out)")
    for label, lines, truth in [("healthy", HEALTHY, HEALTHY_TRUTH), ("sick (SI units)", SICK, SICK_TRUTH)]:
        pdf = make_pdf(lines)
        for mode, data, fname in [("digital PDF", pdf, "r.pdf"), ("scanned PNG (OCR)", to_scanned_png(pdf), "r.png")]:
            out = rp.analyze_report(data, fname)["model_inputs"]
            wrong = {k: (out.get(k), v) for k, v in truth.items() if k not in out or not close(out[k], v)}
            check(f"{label:16s} {mode}", not wrong, f"mismatches (got, expected): {wrong}" if wrong else f"{len(truth)} values correct")


# ---------------------------------------------------------------- model-side checks
def load_app():
    import app
    app.load_models()
    return app


def section_b(app):
    print("\nB. FEATURE CONTRACT (model needs vs what a report can supply)")
    produced = set(rp.analyze_report(make_pdf(HEALTHY + SICK), "r.pdf", age=40, sex="male")["model_inputs"])
    for name, art in app.ARTIFACTS.items():
        feats, req = art["feature_names"], art["required_fields"]
        manual_only = [f for f in feats if f not in produced]
        req_missing = [f for f in req if f not in produced]
        check(f"{name}: required fields obtainable from a report", not req_missing,
              f"NOT produced: {req_missing}" if req_missing else f"required = {req}")
        print(f"         {name}: {len(feats) - len(manual_only)}/{len(feats)} features from report; manual-only: {manual_only}")


def section_c(app):
    print("\nC. SANITY PREDICTIONS (full path: report -> models -> SHAP)")
    for label, lines, expect in [("healthy", HEALTHY, 0), ("sick", SICK, 1)]:
        res, skipped = app._run_all(rp.analyze_report(make_pdf(lines), "r.pdf")["model_inputs"])
        for d, r in res.items():
            if r is None:
                print(f"         {label}/{d}: skipped ({skipped[d]})")
                continue
            if d in ("Diabetes", "Heart"):       # too few report features for a clean expectation - just print
                print(f"         {label}/{d}: p={r.probability} pred={r.prediction} completeness={r.data_completeness}")
                continue
            ok = r.prediction == expect if label == "healthy" else r.prediction == 1
            check(f"{label:8s} {d:8s} prediction={r.prediction} p={r.probability}", ok,
                  "top factors: " + ", ".join(f['feature'] for f in r.top_factors[:3]))


def section_d(app):
    print("\nD. THRESHOLD vs CALIBRATION (live rule: calibrated probability >= threshold)")
    import disease_model_comparison as base
    loaders = {"Diabetes": base.load_diabetes, "Heart": base.load_heart, "Kidney": base.load_kidney,
               "Liver": base.load_liver, "Anemia": base.load_anemia}
    from sklearn.metrics import recall_score, precision_score
    for name, art in app.ARTIFACTS.items():
        X, y, _, _ = loaders[name]()
        X = X[art["feature_names"]]
        thr, target = art["threshold"], art["target_recall"]
        raw = art["pipeline"].predict_proba(X)[:, 1]
        cal = art["calibrator"].predict_proba(X)[:, 1]
        r_raw, r_cal = recall_score(y, raw >= thr), recall_score(y, cal >= thr)
        check(f"{name}: recall with calibrated proba >= {thr:.3f} (target {target})", r_cal >= target - 0.05,
              f"calibrated recall={r_cal:.2f} (precision {precision_score(y, cal >= thr):.2f}); "
              f"raw-score recall={r_raw:.2f}")


if __name__ == "__main__":
    section_a()
    if not SKIP_MODELS:
        try:
            app = load_app()
        except Exception as e:
            print(f"\nCould not load models/app ({type(e).__name__}: {e}). Run from the project root, "
                  f"or use --skip-models.")
            sys.exit(1)
        section_b(app); section_c(app); section_d(app)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)