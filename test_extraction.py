"""Run:  python test_extraction.py      (no models needed; tests the extraction side only)"""
import parameter_extractor as pe, unit_normalizer as un, abnormal_detector as ad, report_pipeline as rp

SAMPLE = """Patient: A B   Age/Sex: 34 Y / F
Hemoglobin (Hb)   9.1 L  g/dL   12.0 - 15.0
Serum Creatinine  159 umol/L
Fasting Blood Sugar  5.9 mmol/L
Total Cholesterol  5.8 mmol/L
Total Leucocyte Count  7.8 x10^3/uL  4.0 - 11.0
Hemoglobin A1c  6.9 %
URINE ROUTINE
Albumin  Trace
Pus cells  2-3 /hpf
"""

def test_all():
    r = pe.parse_parameters(SAMPLE)
    assert r.demographics == {"age": 34, "sex": "female"}
    assert r.params["hemoglobin"].value == 9.1 and r.params["hemoglobin"].report_flag == "L"
    assert r.params["hba1c"].value == 6.9                       # not mistaken for hemoglobin
    assert r.params["urine_albumin"].value == 0.5
    assert r.params["urine_pus_cells"].value == 0
    n = un.normalize(r.params)
    assert abs(n.params["serum_creatinine"].value - 1.8) < 0.01       # umol/L -> mg/dL
    assert abs(n.params["glucose_fasting"].value - 106.3) < 0.2       # mmol/L -> mg/dL
    assert abs(n.params["total_cholesterol"].value - 224.3) < 0.5
    assert n.params["wbc_count"].value == 7800                        # 10^3/uL -> cells/cumm
    f = {x["parameter"]: x for x in ad.detect_abnormal(n.params, "female")}
    assert f["hemoglobin"]["status"] == "low" and f["glucose_fasting"]["status"] == "high"
    x, _ = rp.build_model_inputs(n.params, 34, "female")
    assert x["Hemoglobin"] == 9.1 and x["Gender"] == 0.0 and x["fbs"] == 0.0
    # implausible value (lost decimal) must be rejected, not silently used
    bad = un.normalize(pe.parse_parameters("Hemoglobin 91 g/dL").params)
    assert "hemoglobin" not in bad.params and bad.rejected
    print("all extraction tests passed")

if __name__ == "__main__":
    test_all()