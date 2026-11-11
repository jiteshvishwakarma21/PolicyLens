
import pymupdf
import pytesseract

from PIL import Image, ImageOps, ImageEnhance
from io import BytesIO


def preprocess_image(image):
    """Prepare a scanned page for OCR."""

    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.5)

    return image


def detect_orientation(image):
    """
    Detect page orientation using Tesseract OSD.

    Returns the rotation required to correct the page.
    """

    try:
        result = pytesseract.image_to_osd(
            image,
            config="--psm 0"
        )

        for line in result.splitlines():
            if line.startswith("Rotate:"):
                return int(line.split(":")[1].strip())

    except Exception:
        pass

    return 0


def extract_text_using_ocr(pdf_path, page_number, dpi=300):
    """
    Extract text from a scanned PDF page using Tesseract OCR.
    Page numbering starts from 1.
    """

    try:
        with pymupdf.open(pdf_path) as document:

            if page_number < 1 or page_number > len(document):
                raise ValueError("Invalid page number.")

            page = document[page_number - 1]

            zoom = dpi / 72
            matrix = pymupdf.Matrix(zoom, zoom)

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image = Image.open(
                BytesIO(pixmap.tobytes("png"))
            )

            image = preprocess_image(image)

            # Detect and correct page orientation.
            rotation = detect_orientation(image)

            if rotation:
                image = image.rotate(
                    rotation,
                    expand=True
                )

            # Perform OCR.
            extracted_text = pytesseract.image_to_string(
                image,
                lang="eng",
                config="--oem 3 --psm 3"
            )

            return extracted_text.strip()

    except Exception as error:
        raise RuntimeError(
            f"OCR extraction failed: {error}"
        ) from error


if __name__ == "__main__":
    print("OCR Processor module is ready.")
