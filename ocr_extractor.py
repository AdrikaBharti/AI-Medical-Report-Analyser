"""
ocr_extractor.py - OCR for scanned PDF pages and photos/screenshots of reports.

Engine : Tesseract via pytesseract (free, runs locally - no report data leaves the machine).
PDF -> image : pypdfium2 (pure pip wheel, so NO poppler install needed on Windows).

Windows setup (one time):
  1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
  2. Either add it to PATH, or set env var  TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""
from __future__ import annotations
import io
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

from PIL import Image, ImageFilter, ImageOps

TARGET_DPI = 300
MAX_PIXELS = 30_000_000
MAX_SIDE_PX = 4500          # guard against decompression bombs / huge photos
LOW_CONFIDENCE = 60.0            # mean word confidence (0-100) below this => warn user


class OCRError(Exception):
    pass


@dataclass
class OCRPage:
    page: int
    text: str
    mean_confidence: float       # 0-100, mean Tesseract word confidence


def _tesseract():
    try:
        import pytesseract
    except ImportError as e:
        raise OCRError("pytesseract is not installed (pip install pytesseract).") from e
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise OCRError("Tesseract executable not found. Install it and add it to PATH "
                       "or set TESSERACT_CMD.") from e
    return pytesseract


def preprocess(img: Image.Image) -> Image.Image:
    """Grayscale -> upscale small images -> autocontrast -> light sharpen.
    (No hard binarisation: Tesseract's own Otsu step is better on shadows/phone photos.)"""
    img = ImageOps.exif_transpose(img)            # phone photos carry a rotation flag
    img = img.convert("L")
    w, h = img.size
    if w * h > MAX_PIXELS:
        raise OCRError("Image is too large.")
    if w < 1600:                                  # ~ below 150 dpi on an A4 page
        scale = 2000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    return img.filter(ImageFilter.SHARPEN)


def _ocr_image(img: Image.Image, page_no: int, lang: str) -> OCRPage:
    pt = _tesseract()
    img = preprocess(img)
    # psm 6 = "single uniform block of text": keeps each table row (name | value | unit | range) on ONE line
    d = pt.image_to_data(img, lang=lang, config="--oem 3 --psm 6", output_type=pt.Output.DICT)
    lines, confs, key, cur = [], [], None, []
    for i, word in enumerate(d["text"]):
        if not word.strip():
            continue
        k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if key is not None and k != key:
            lines.append(" ".join(cur)); cur = []
        key = k
        cur.append(word)
        c = float(d["conf"][i])
        if c >= 0:
            confs.append(c)
    if cur:
        lines.append(" ".join(cur))
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return OCRPage(page_no, "\n".join(lines), round(mean_conf, 1))


def ocr_image_bytes(data: bytes, lang: str = "eng") -> List[OCRPage]:
    """PNG / JPG / TIFF (multi-frame TIFF = multi-page) / BMP / WEBP."""
    try:
        img = Image.open(io.BytesIO(data))
        n = getattr(img, "n_frames", 1)
        out = []
        for f in range(n):
            img.seek(f)
            out.append(_ocr_image(img.copy(), f + 1, lang))
        return out
    except OCRError:
        raise
    except Exception as e:
        raise OCRError(f"Could not open image: {e}") from e


def ocr_pdf_pages(data: bytes, page_numbers: Optional[Iterable[int]] = None,
                  lang: str = "eng") -> List[OCRPage]:
    """OCR selected (1-based) pages of a PDF, or all pages when page_numbers is None."""
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise OCRError("pypdfium2 is not installed (pip install pypdfium2).") from e
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as e:
        raise OCRError(f"Could not open PDF for OCR: {e}") from e
    wanted = list(page_numbers) if page_numbers is not None else list(range(1, len(pdf) + 1))
    out = []
    for n in wanted:
        page = pdf[n - 1]
        w_pt, h_pt = page.get_size()
        scale = min(TARGET_DPI / 72, MAX_SIDE_PX / max(w_pt, h_pt))   # odd page sizes: cap the raster size
        bitmap = page.render(scale=scale)
        out.append(_ocr_image(bitmap.to_pil(), n, lang))
    return out