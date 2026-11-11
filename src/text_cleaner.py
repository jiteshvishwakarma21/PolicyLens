
import re


def clean_text(text):
    """
    Clean extracted PDF or OCR text.

    Operations:
    - Normalize line breaks.
    - Remove unwanted control characters.
    - Repair words split across lines.
    - Normalize spaces and tabs.
    - Remove excessive blank lines.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Normalize line breaks.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove unwanted control characters.
    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        " ",
        text
    )

    # Repair hyphenated words split across lines.
    text = re.sub(
        r"(?<=\w)-\n(?=\w)",
        "",
        text
    )

    # Normalize spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


if __name__ == "__main__":

    sample = (
        "Financial inclu-\n"
        "sion is important.\n\n\n"
        "Banking   Regulation"
    )

    print("Original text:")
    print(sample)

    print("\nCleaned text:")
    print(clean_text(sample))
