"""
parameter_extractor.py - report text  ->  structured lab parameters (as printed, NOT yet unit-converted).

Handles rows such as
    Hemoglobin (Hb)      12.5  L   g/dL     13.0 - 17.0
    S. Creatinine        1.8       mg/dL    0.7 - 1.3
    Total Leucocyte Count 7,800    /cumm    4000 - 11000
and urine dipstick / microscopy rows such as  "Albumin  ++", "Pus cells  2-3 /hpf".

One parameter per line; the longest matching alias wins (so "Hemoglobin A1c" is
never mistaken for "Hemoglobin"). First occurrence of a parameter is kept.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union

# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class ExtractedParam:
    name: str
    value: Union[float, int]
    raw_value: str                        # text exactly as read (helps the user verify)
    unit: Optional[str] = None
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    ref_by_sex: Optional[Dict[str, Tuple[float, float]]] = None   # {"male": (lo,hi), "female": (lo,hi)}
    report_flag: Optional[str] = None     # "H" / "L" as printed on the report
    line: str = ""
    kind: str = "numeric"                 # numeric | qualitative

    def to_dict(self):
        return asdict(self)


@dataclass
class ParseResult:
    params: Dict[str, ExtractedParam]
    demographics: Dict[str, object]       # age (years), sex ("male"/"female")
    warnings: List[str]


# --------------------------------------------------------------------------
# Alias tables.  (regex fragments, matched case-insensitively)
# --------------------------------------------------------------------------
# name -> (aliases, exclude-regex for the whole line or None)
NUMERIC: Dict[str, Tuple[List[str], Optional[str]]] = {
    "hemoglobin":        ([r"h(?:a)?emoglobin(?:\s*\(?hb\)?)?", r"hb", r"hgb"], r"a1c|electrophoresis"),
    "hba1c":             ([r"hb\s*a1c", r"h(?:a)?emoglobin\s*a1c", r"(?:glycosylated|glycated)\s*h(?:a)?emoglobin"], None),
    "pcv":               ([r"pcv", r"packed\s*cell\s*volume", r"h(?:a)?ematocrit(?:\s*\(?hct\)?)?", r"hct"], None),
    "rbc_count":         ([r"(?:total\s*)?rbc(?:\s*count)?", r"red\s*(?:blood\s*)?cell(?:s)?(?:\s*count)?", r"erythrocyte(?:s)?(?:\s*count)?"], r"urine|hpf|nil|cast|distribution|width|rdw"),
    "wbc_count":         ([r"(?:total\s*)?(?:leuc|leuk)ocyte\s*count(?:\s*\(?tlc\)?)?", r"tlc", r"(?:total\s*)?wbc(?:\s*count)?", r"white\s*(?:blood\s*)?cell(?:s)?(?:\s*count)?"], r"urine|hpf|differential"),
    "mcv":               ([r"mcv", r"mean\s*(?:corpuscular|cell)\s*volume"], None),
    "mch":               ([r"mch", r"mean\s*(?:corpuscular|cell)\s*h(?:a)?emoglobin(?!\s*conc)"], r"mchc"),
    "mchc":              ([r"mchc", r"mean\s*(?:corpuscular|cell)\s*h(?:a)?emoglobin\s*conc\w*"], None),
    "glucose_fasting":   ([r"fasting\s*(?:blood\s*)?(?:sugar|glucose|plasma\s*glucose)(?:\s*\(?(?:fbs|fbg|fpg)\)?)?", r"fbs", r"fbg", r"fpg", r"glucose\s*[-,(]?\s*fasting\)?", r"blood\s*sugar\s*[-,(]?\s*fasting\)?"], None),
    "glucose_pp":        ([r"post\s*-?\s*prandial\s*(?:blood\s*)?(?:sugar|glucose)?(?:\s*\(?(?:ppbs|pp)\)?)?", r"ppbs", r"2\s*h(?:ou)?rs?\s*(?:post|plasma)?\s*(?:glucose|sugar)", r"glucose\s*[-,(]?\s*(?:pp|post\s*prandial)\)?", r"ogtt"], None),
    "glucose_random":    ([r"random\s*(?:blood\s*)?(?:sugar|glucose)(?:\s*\(?(?:rbs|rbg)\)?)?", r"rbs", r"rbg", r"glucose\s*[-,(]?\s*random\)?", r"blood\s*sugar\s*[-,(]?\s*random\)?"], None),
    "glucose_unspecified": ([r"(?:blood|serum|plasma)?\s*glucose", r"blood\s*sugar"], r"urine|fasting|random|post|pp\b|tolerance"),
    "insulin":           ([r"(?:serum\s*|fasting\s*)?insulin"], r"resistance|antibod"),
    "total_cholesterol": ([r"(?:total\s*|serum\s*)?cholesterol(?:\s*[-,(]?\s*total\)?)?"], r"hdl|ldl|vldl|non[\s-]*hdl|ratio"),
    "blood_urea":        ([r"(?:blood\s*|serum\s*)?urea(?!\s*nitrogen)", r"b\.?\s*urea"], r"urine|ratio|uric"),
    "bun":               ([r"bun", r"(?:blood\s*)?urea\s*nitrogen(?:\s*\(?bun\)?)?"], r"ratio"),
    "serum_creatinine":  ([r"(?:s\.?\s*|serum\s*)?creatinine"], r"clearance|egfr|urine|ratio|random|24"),
    "sodium":            ([r"(?:serum\s*)?sodium(?:\s*\(?na\+?\)?)?", r"na\+?"], r"urine"),
    "potassium":         ([r"(?:serum\s*)?potassium(?:\s*\(?k\+?\)?)?", r"k\+"], r"urine"),
    "total_bilirubin":   ([r"(?:serum\s*|s\.?\s*)?(?:total\s*bilirubin|bilirubin\s*[-,(]?\s*total\)?)", r"t\.?\s*bilirubin"], r"indirect"),
    "direct_bilirubin":  ([r"(?:direct|conjugated)\s*bilirubin", r"bilirubin\s*[-,(]?\s*(?:direct|conjugated)\)?", r"d\.?\s*bilirubin"], None),
    "alp":               ([r"alkaline\s*phosph?atase(?:\s*\(?alp\)?)?", r"alk\.?\s*phos\w*", r"alp"], None),
    "alt":               ([r"(?:alanine\s*(?:amino)?transaminase|alanine\s*aminotransferase)(?:\s*\(?(?:alt|sgpt)\)?)?", r"sgpt(?:\s*/?\s*alt)?", r"alt(?:\s*/?\s*sgpt)?"], None),
    "ast":               ([r"(?:aspartate\s*(?:amino)?transaminase|aspartate\s*aminotransferase)(?:\s*\(?(?:ast|sgot)\)?)?", r"sgot(?:\s*/?\s*ast)?", r"ast(?:\s*/?\s*sgot)?"], None),
    "total_protein":     ([r"(?:serum\s*)?total\s*proteins?", r"proteins?\s*[-,(]?\s*total\)?"], r"urine"),
    "albumin":           ([r"(?:serum\s*)?albumin"], r"globulin|ratio|urine|a\s*/\s*g|trace|nil"),
    "ag_ratio":          ([r"a\s*[/:]\s*g\s*ratio", r"albumin\s*[/:]\s*globulin(?:\s*ratio)?", r"alb\s*/\s*glob\w*"], None),
    "urine_sg":          ([r"(?:urine\s*)?specific\s*gravity", r"sp\.?\s*gr\w*"], None),
    "bmi":               ([r"bmi", r"body\s*mass\s*index"], None),
}

# Qualitative urine rows. They only count inside a urine section, except the unambiguous ones (needs_section False).
QUALITATIVE: Dict[str, Tuple[List[str], bool]] = {
    "urine_albumin":   ([r"(?:urine\s*)?(?:albumin|protein)"], True),
    "urine_sugar":     ([r"(?:urine\s*)?(?:sugar|glucose)"], True),
    "urine_rbc":       ([r"(?:urine\s*)?(?:rbc|red\s*(?:blood\s*)?cells?)s?"], True),
    "urine_pus_cells": ([r"pus\s*cells?", r"(?:w\.?b\.?c\.?|leuc?ocytes?)\s*(?:\(pus\s*cells?\))?"], True),
    "pus_cell_clumps": ([r"pus\s*cell\s*clumps?"], False),
    "bacteria":        ([r"bacteria", r"bacilli"], False),
}
# names that should win over the generic urine_pus_cells alias
QUAL_EXCLUDE = {"urine_pus_cells": r"clump"}

LABEL = {
    "hemoglobin": "Hemoglobin", "hba1c": "HbA1c", "pcv": "PCV / Hematocrit", "rbc_count": "RBC count",
    "wbc_count": "WBC count", "mcv": "MCV", "mch": "MCH", "mchc": "MCHC",
    "glucose_fasting": "Fasting glucose", "glucose_pp": "Post-prandial glucose",
    "glucose_random": "Random glucose", "glucose_unspecified": "Blood glucose (type not stated)",
    "insulin": "Insulin", "total_cholesterol": "Total cholesterol", "blood_urea": "Blood urea",
    "bun": "Blood urea nitrogen", "serum_creatinine": "Serum creatinine", "sodium": "Sodium",
    "potassium": "Potassium", "total_bilirubin": "Total bilirubin", "direct_bilirubin": "Direct bilirubin",
    "alp": "Alkaline phosphatase", "alt": "ALT (SGPT)", "ast": "AST (SGOT)", "total_protein": "Total protein",
    "albumin": "Albumin", "ag_ratio": "A/G ratio", "urine_sg": "Urine specific gravity", "bmi": "BMI",
    "urine_albumin": "Urine albumin", "urine_sugar": "Urine sugar", "urine_rbc": "Urine RBC",
    "urine_pus_cells": "Urine pus cells", "pus_cell_clumps": "Pus cell clumps", "bacteria": "Bacteria (urine)",
    "systolic_bp": "Systolic BP", "diastolic_bp": "Diastolic BP",
}

# --------------------------------------------------------------------------
# Regex building blocks
# --------------------------------------------------------------------------
NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_RANGE = rf"({NUM})\s*(?:-|–|—|to)\s*({NUM})"
RANGE_RE = re.compile(_RANGE, re.I)
LABELLED_RANGE_RE = re.compile(rf"(?<![a-z])(female|male|f|m)\s*[:=\-]?\s*{_RANGE}", re.I)
LT_RE = re.compile(rf"(?:<|<=|≤|up\s*to|upto)\s*({NUM})", re.I)
GT_RE = re.compile(rf"(?:>|>=|≥)\s*({NUM})", re.I)
FLAG_RE = re.compile(r"(?<![A-Za-z/])(H|L|HIGH|LOW|HH|LL)(?![A-Za-z/])")

PRE_FLAG_RE = re.compile(r"^\s*(H|L|HIGH|LOW|HH|LL)(?![A-Za-z/])\s*")
UNIT_RE = re.compile(r"""^[\s\[\(]*(?P<u>
    (?:x\s*)?10\s*(?:\^|\*|e)?\s*[³⁶36]\s*/\s*[µμu]?\s*l
  | (?:thou|thousand|k|mill|mil|million|millions|lakhs?|cells)\s*/\s*(?:cu\.?\s*mm|cumm|cmm|mm3|mm³|[µμu]l|l)
  | /\s*(?:cu\.?\s*mm|cumm|cmm|mm3|mm³|[µμu]l|hpf)
  | (?:mg|g|ng|pg|[µμu]g|mmol|[µμu]mol|pmol|meq|miu|[µμu]?iu|iu|u)\s*/\s*(?:d\s*l|m\s*l|l)
  | fl | pg | % | mm/hr | ratio
)(?![A-Za-z])""", re.I | re.X)

_HEADER_SKIP = re.compile(r"^\W*(note|notes|comment|interpretation|reference|method|kindly|criteria|remark|"
                          r"clinical|please|advice|page|printed|lab\s*no)\b", re.I)
_URINE_HDR = re.compile(r"urine|urinalysis", re.I)
_OTHER_HDR = re.compile(r"blood|serum|plasma|haemat|hemat|cbc|liver|kidney|renal|lft|rft|kft|biochem|"
                        r"lipid|thyroid|chemistry|complete\s*count", re.I)


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _sub_ranges(rem: str):
    """Return (ref_low, ref_high, ref_by_sex) from the text that follows a value/unit."""
    by_sex = {}
    for m in LABELLED_RANGE_RE.finditer(rem):
        sx = "female" if m.group(1).lower() in ("f", "female") else "male"
        by_sex[sx] = (_to_float(m.group(2)), _to_float(m.group(3)))
    if len(by_sex) == 2:
        return None, None, by_sex
    m = RANGE_RE.search(rem)
    if m:
        lo, hi = _to_float(m.group(1)), _to_float(m.group(2))
        if lo < hi:
            return lo, hi, None
    m = LT_RE.search(rem)
    if m:
        return None, _to_float(m.group(1)), None
    m = GT_RE.search(rem)
    if m:
        return _to_float(m.group(1)), None, None
    return None, None, None


# ---- qualitative urine interpretation -------------------------------------
_QUAL_WORD = re.compile(
    r"(?P<q>not\s*detected|not\s*seen|nil|absent|negative|neg|trace|traces|present|positive|pos|"
    r"normal|abnormal|occasional|plenty|numerous|loaded|few|many|\d\s*\+|\+{1,4}|"
    r"\d+\s*[-–]\s*\d+|\d+)", re.I)


def _plus_scale(q: str) -> Optional[float]:
    """Dipstick scale used by the UCI CKD data: 0 = negative ... 4/5 = heavy."""
    q = q.lower().replace(" ", "")
    if q in ("nil", "absent", "negative", "neg", "notdetected", "normal", "0"):
        return 0.0
    if q in ("trace", "traces"):
        return 0.5               # midpoint between negative and 1+
    if re.fullmatch(r"\++", q):
        return float(len(q))
    m = re.fullmatch(r"(\d)\+", q)
    if m:
        return float(m.group(1))
    if q in ("present", "positive", "pos"):
        return 1.0
    return None


def _micro_abnormal(kind: str, q: str) -> Optional[int]:
    """0 = normal, 1 = abnormal for urine microscopy (UCI encoding)."""
    ql = q.lower().replace(" ", "")
    if ql in ("nil", "absent", "notseen", "notdetected", "normal", "occasional", "few", "negative"):
        return 0
    if ql in ("abnormal", "plenty", "numerous", "loaded", "many", "present", "positive"):
        return 1
    limit = {"urine_rbc": 2, "urine_pus_cells": 5}.get(kind)
    nums = [float(x) for x in re.findall(r"\d+", ql)]
    if nums and limit is not None:
        return int(max(nums) > limit)
    return None


def _presence(q: str) -> Optional[int]:
    ql = q.lower().replace(" ", "")
    if ql in ("nil", "absent", "notseen", "notdetected", "negative", "neg", "notpresent"):
        return 0
    if ql in ("present", "positive", "pos", "seen", "few", "plenty", "numerous", "occasional") or ql.startswith("+"):
        return 1
    return None


# --------------------------------------------------------------------------
# Compiled patterns
# --------------------------------------------------------------------------
def _compile_numeric():
    out = []
    for name, (aliases, excl) in NUMERIC.items():
        for a in aliases:
            rx = re.compile(rf"(?<![A-Za-z])(?P<alias>{a})(?![A-Za-z])(?P<fill>[^0-9<>\n]{{0,40}}?)"
                            rf"(?P<val>{NUM})", re.I)
            out.append((name, rx, re.compile(excl, re.I) if excl else None))
    return out


def _compile_qual():
    out = []
    for name, (aliases, needs_section) in QUALITATIVE.items():
        for a in aliases:
            rx = re.compile(rf"(?<![A-Za-z])(?P<alias>{a})(?![A-Za-z])[^A-Za-z0-9+\n]{{0,15}}?(?P<val>{_QUAL_WORD.pattern})", re.I)
            out.append((name, rx, needs_section))
    return out


_NUM_PATTERNS = _compile_numeric()
_QUAL_PATTERNS = _compile_qual()
_BP_RE = re.compile(r"(?:\bB\.?\s*P\.?\b|blood\s*pressure)\s*[:\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})", re.I)
_AGE_SEX_RE = re.compile(r"age\s*(?:/\s*(?:sex|gender))?\s*[:\-]?\s*(\d{1,3})\s*(?:y(?:ea)?rs?\.?|y)?\s*[/,\s]*\s*(male|female|m|f)?\b", re.I)
_SEX_RE = re.compile(r"\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f)\b", re.I)
_AGE_ONLY_RE = re.compile(r"\bage\s*[:\-]?\s*(\d{1,3})\b", re.I)
_AGE_UNIT_MONTHS = re.compile(r"age\s*[:\-]?\s*\d{1,3}\s*(?:months?|mths?|days?|d\b)", re.I)


def extract_demographics(text: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    head = text[:2500]                              # demographics live in the report header
    if not _AGE_UNIT_MONTHS.search(head):
        m = _AGE_SEX_RE.search(head)
        if m:
            age = int(m.group(1))
            if 0 < age < 120:
                out["age"] = age
            if m.group(2):
                out["sex"] = "male" if m.group(2).lower() in ("m", "male") else "female"
        elif (m := _AGE_ONLY_RE.search(head)) and 0 < int(m.group(1)) < 120:
            out["age"] = int(m.group(1))
    if "sex" not in out:
        m = _SEX_RE.search(head)
        if m:
            out["sex"] = "male" if m.group(1).lower() in ("m", "male") else "female"
    if "sex" not in out:                            # "45 Y / M", "45Yrs/F"
        m = re.search(r"\b\d{1,3}\s*(?:y|yrs?|years?)\s*[/,]\s*(male|female|m|f)\b", head, re.I)
        if m:
            out["sex"] = "male" if m.group(1).lower() in ("m", "male") else "female"
    return out


def parse_parameters(text: str) -> ParseResult:
    params: Dict[str, ExtractedParam] = {}
    warnings: List[str] = []
    section = "other"

    demo = extract_demographics(text)

    for raw_line in text.splitlines():
        line = re.sub(r"[|]+", " ", raw_line)
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 3:
            continue
        # -- section tracking (header lines carry no numbers) --
        if not re.search(r"\d", line) and len(line) < 70:
            if _URINE_HDR.search(line):
                section = "urine"
            elif _OTHER_HDR.search(line):
                section = "other"
        if _HEADER_SKIP.match(line):
            continue

        # -- blood pressure --
        bp = _BP_RE.search(line)
        if bp:
            s, d = int(bp.group(1)), int(bp.group(2))
            if 60 <= s <= 260 and 30 <= d <= 160 and "systolic_bp" not in params:
                params["systolic_bp"] = ExtractedParam("systolic_bp", s, bp.group(1), "mmHg", line=line)
                params["diastolic_bp"] = ExtractedParam("diastolic_bp", d, bp.group(2), "mmHg", line=line)
            continue

        # -- qualitative urine rows first when we are inside a urine section --
        qual = _match_qual(line, section)
        cand0 = _numeric_candidate(line)
        if qual and cand0 and _has_blood_unit(line, cand0[1]):
            qual = None                      # e.g. "RBC 4.5 million/cumm" under a missed urine header
        if qual and (section == "urine" or not cand0):
            name, q_text = qual
            if name not in params:
                p = _make_qual(name, q_text, line)
                if p:
                    params[name] = p
            continue

        cand = cand0
        if cand:
            name, m, _ = cand
            if name in params:
                continue
            val = _to_float(m.group("val"))
            rest = line[m.end("val"):]
            flag = None
            pm = PRE_FLAG_RE.match(rest)               # "9.1 L g/dL": flag sits between value and unit
            if pm:
                flag, rest = pm.group(1).upper()[0], rest[pm.end():]
            um = UNIT_RE.match(rest)
            unit = um.group("u").strip() if um else None
            rem = rest[um.end():] if um else rest
            if flag is None:
                fm = FLAG_RE.search(rem[:12] if um else rest[:12])
                flag = fm.group(1).upper()[0] if fm else None
            lo, hi, by_sex = _sub_ranges(rem)
            params[name] = ExtractedParam(name, val, m.group("val"), unit, lo, hi, by_sex, flag, line)

    # A single 'glucose' with no type is kept, but flagged later by the pipeline.
    return ParseResult(params, demo, warnings)


def _has_blood_unit(line: str, m) -> bool:
    um = UNIT_RE.match(line[m.end("val"):])
    return bool(um) and "hpf" not in um.group("u").lower()


def _numeric_candidate(line: str):
    best = None
    for name, rx, excl in _NUM_PATTERNS:
        m = rx.search(line)
        if not m or m.start("alias") > 30:              # name must be near the start of the row
            continue
        if excl and excl.search(line):
            continue
        L = len(m.group("alias"))
        if best is None or L > best[2]:
            best = (name, m, L)
    return best


def _match_qual(line: str, section: str):
    best = None
    for name, rx, needs_section in _QUAL_PATTERNS:
        if needs_section and section != "urine":
            continue
        if name in QUAL_EXCLUDE and re.search(QUAL_EXCLUDE[name], line, re.I):
            continue
        m = rx.search(line)
        if not m or m.start("alias") > 30:
            continue
        L = len(m.group("alias"))
        if best is None or L > best[2]:
            best = (name, m.group("val"), L)
    return (best[0], best[1]) if best else None


def _make_qual(name: str, q: str, line: str) -> Optional[ExtractedParam]:
    if name in ("urine_albumin", "urine_sugar"):
        v = _plus_scale(q)
    elif name in ("urine_rbc", "urine_pus_cells"):
        v = _micro_abnormal(name, q)
    else:
        v = _presence(q)
    if v is None:
        return None
    return ExtractedParam(name, v, q.strip(), None, line=line, kind="qualitative")