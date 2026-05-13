from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}


def load_documents(path: Path, pattern: str) -> list[tuple[str, str]]:
    if path.is_file():
        return [(str(path), path.read_text(encoding="utf-8"))]

    documents: list[tuple[str, str]] = []
    for file_path in sorted(path.glob(pattern)):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append((str(file_path), file_path.read_text(encoding="utf-8")))
    return documents
