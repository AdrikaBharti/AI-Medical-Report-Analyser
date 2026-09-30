"""
pdf_extractor.py - read the TEXT LAYER of a digital (non-scanned) PDF.

Returns one PageText per page. A page with (almost) no characters is reported
as `has_text=False` so report_extractor.py can send just that page to OCR.
"""
from __future__ import annotations
import io
from dataclasses import dataclass
from typing import List

MIN_CHARS_FOR_TEXT_PAGE = 40      # below this a page is treated as "scanned / image only"


class PDFReadError(Exception):
    pass


@dataclass
class PageText:
    page: int                     # 1-based
    text: str
    has_text: bool


def extract_pdf_pages(data: bytes) -> List[PageText]:
    """pdfplumber first (keeps table rows on one line); pypdf as fallback."""
    pages: List[PageText] = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if getattr(pdf, "doc", None) is not None and pdf.doc.is_extractable is False:
                raise PDFReadError("PDF does not allow text extraction (protected).")
            for i, page in enumerate(pdf.pages, 1):
                txt = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                pages.append(PageText(i, txt, len(txt.strip()) >= MIN_CHARS_FOR_TEXT_PAGE))
        return pages
    except PDFReadError:
        raise
    except Exception as e_plumber:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")          # many "encrypted" PDFs just have an empty user password
                except Exception:
                    raise PDFReadError("PDF is password protected.")
            for i, page in enumerate(reader.pages, 1):
                txt = page.extract_text() or ""
                pages.append(PageText(i, txt, len(txt.strip()) >= MIN_CHARS_FOR_TEXT_PAGE))
            return pages
        except PDFReadError:
            raise
        except Exception as e_pypdf:
            raise PDFReadError(f"Could not read PDF (pdfplumber: {e_plumber}; pypdf: {e_pypdf})")