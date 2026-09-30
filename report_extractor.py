"""
report_extractor.py - one entry point: bytes of a PDF / image  ->  plain text.

Routing
  PDF   : read text layer page by page; pages with no usable text are OCR'd
          (so a PDF that is half digital, half scanned still works).
  Image : OCR.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List

import pdf_extractor
import ocr_extractor

PDF_EXT = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXT = PDF_EXT | IMAGE_EXT


class UnsupportedFileType(Exception):
    pass


@dataclass
class ExtractionResult:
    text: str
    method: str                       # "pdf_text" | "ocr" | "pdf_text+ocr"
    pages: int
    ocr_pages: List[int] = field(default_factory=list)
    mean_ocr_confidence: float | None = None
    warnings: List[str] = field(default_factory=list)

    @property
    def used_ocr(self) -> bool:
        return bool(self.ocr_pages)


def extract_report(data: bytes, filename: str, lang: str = "eng") -> ExtractionResult:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in SUPPORTED_EXT:
        raise UnsupportedFileType(f"Unsupported file type '{ext}'. Use: {sorted(SUPPORTED_EXT)}")
    warnings: List[str] = []

    if ext in IMAGE_EXT:
        ocr = ocr_extractor.ocr_image_bytes(data, lang)
        return _finish(ocr_only=ocr, text_pages=[], warnings=warnings, total=len(ocr))

    pages = pdf_extractor.extract_pdf_pages(data)
    if not pages:
        raise pdf_extractor.PDFReadError("PDF has no pages.")
    need_ocr = [p.page for p in pages if not p.has_text]
    ocr_pages = ocr_extractor.ocr_pdf_pages(data, need_ocr, lang) if need_ocr else []
    text_pages = [p for p in pages if p.has_text]
    return _finish(ocr_only=ocr_pages, text_pages=text_pages, warnings=warnings, total=len(pages))


def _finish(ocr_only, text_pages, warnings, total) -> ExtractionResult:
    by_page = {p.page: p.text for p in text_pages}
    by_page.update({p.page: p.text for p in ocr_only})
    text = "\n".join(by_page[k] for k in sorted(by_page))
    conf = None
    if ocr_only:
        conf = round(sum(p.mean_confidence for p in ocr_only) / len(ocr_only), 1)
        if conf < ocr_extractor.LOW_CONFIDENCE:
            warnings.append(f"OCR confidence is low ({conf}%). Please check every extracted value "
                            f"against the original report, or upload a sharper scan/PDF.")
    if not text.strip():
        warnings.append("No text could be read from this file.")
    method = ("ocr" if not text_pages else "pdf_text" if not ocr_only else "pdf_text+ocr")
    return ExtractionResult(text=text, method=method, pages=total,
                            ocr_pages=sorted(p.page for p in ocr_only),
                            mean_ocr_confidence=conf, warnings=warnings)