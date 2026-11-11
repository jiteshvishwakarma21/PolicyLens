
import pymupdf  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF page by page.

    Returns:
        List of dictionaries containing:
        page, text, needs_ocr, character_count
    """

    extracted_pages = []

    try:
        with pymupdf.open(pdf_path) as document:

            for page_number, page in enumerate(
                document,
                start=1
            ):

                text = page.get_text("text").strip()

                # Detect pages that may require OCR.
                needs_ocr = len(text) < 30

                extracted_pages.append({
                    "page": page_number,
                    "text": text,
                    "needs_ocr": needs_ocr,
                    "character_count": len(text)
                })

        return extracted_pages

    except Exception as error:
        raise RuntimeError(
            f"Could not extract PDF text: {error}"
        ) from error


if __name__ == "__main__":
    print("PDF Extractor module is ready.")
