
import re
import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    return len(TOKENIZER.encode(text))


def fixed_size_chunking(text, chunk_size=500, overlap=100):
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid overlap.")

    tokens = TOKENIZER.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        chunks.append({
            "chunk_id": len(chunks) + 1,
            "text": TOKENIZER.decode(chunk_tokens),
            "token_count": len(chunk_tokens),
            "start_token": start,
            "end_token": end
        })

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks


def is_heading(line):
    """Detect likely chapter and section headings."""

    line = line.strip()

    if not line or len(line) > 150:
        return False

    # Ignore table-of-contents lines.
    if re.search(r"\.{3,}|…{2,}", line):
        return False

    ignored = [
        "RESERVE BANK OF INDIA",
        "INSERTED WITH EFFECT",
        "AMENDED WITH EFFECT",
        "SUBSTITUTED WITH EFFECT",
        "WWW.RBI.ORG.IN",
        "UPDATED AS ON",
        "TABLE OF CONTENTS"
    ]

    if any(item in line.upper() for item in ignored):
        return False

    # Chapter headings.
    chapter = re.match(
        r"^Chapter\s+[IVXLCDM0-9]+\s*[-–:]\s*\S",
        line,
        re.IGNORECASE
    )

    # Lettered headings such as A. Introduction.
    lettered = re.match(
        r"^[A-Z]\.\s+[A-Z][A-Za-z]",
        line
    )

    # Short uppercase headings.
    uppercase = (
        len(line.split()) <= 10
        and line.isupper()
        and len(line) >= 5
    )

    return bool(chapter or lettered or uppercase)


def detect_section_title(text):
    if not isinstance(text, str) or not text.strip():
        return "Not detected"

    for line in text.splitlines():
        if is_heading(line):
            return line.strip()

    return "Not detected"


def structure_aware_chunking(text, max_tokens=500):
    """Group short sections and split long sections."""

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    sections = []
    current_title = "General"
    current_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if is_heading(line):
            if current_lines:
                sections.append({
                    "title": current_title,
                    "text": "\n".join(current_lines)
                })

            current_title = line
            current_lines = [line]

        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "title": current_title,
            "text": "\n".join(current_lines)
        })

    chunks = []
    buffer_tokens = []
    buffer_titles = []

    def save_buffer():
        nonlocal buffer_tokens, buffer_titles

        if not buffer_tokens:
            return

        chunks.append({
            "chunk_id": len(chunks) + 1,
            "text": TOKENIZER.decode(buffer_tokens),
            "section": " | ".join(dict.fromkeys(buffer_titles)),
            "token_count": len(buffer_tokens),
            "start_token": 0,
            "end_token": len(buffer_tokens)
        })

        buffer_tokens = []
        buffer_titles = []

    for section in sections:
        tokens = TOKENIZER.encode(section["text"])
        position = 0

        while position < len(tokens):
            available = max_tokens - len(buffer_tokens)

            if available == 0:
                save_buffer()
                available = max_tokens

            take = min(
                available,
                len(tokens) - position
            )

            buffer_tokens.extend(
                tokens[position:position + take]
            )

            if section["title"] not in buffer_titles:
                buffer_titles.append(section["title"])

            position += take

            if len(buffer_tokens) >= max_tokens:
                save_buffer()

    # Keep the final remainder, even if it is shorter than 300.
    save_buffer()

    return chunks


if __name__ == "__main__":
    sample = """
    Chapter I - Preliminary

    A. Introduction

    These Directions provide regulatory guidance.

    B. Applicability

    These instructions apply to regulated entities.
    """

    print("Fixed chunks:", len(fixed_size_chunking(sample)))
    print("Structure chunks:", len(structure_aware_chunking(sample)))
