from pathlib import Path

from .config import DATA_DIR


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """
    Load every .txt file in data_dir.

    Args:
        data_dir (Path): Directory containing .txt files. Default is config.DATA_DIR.

    Returns:
        list[dict]: One {"doc_id", "text"} dict per non-empty file, sorted by filename.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({"doc_id": path.stem, "text": text})
    return documents
