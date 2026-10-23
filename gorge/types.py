"""Content-type detection: magic bytes, extension, entropy, printable ratio."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

MAJORS = [
    "text", "code", "data", "document", "image", "audio", "video",
    "archive", "database", "executable", "binary", "other",
]
MAJOR_ID = {m: i for i, m in enumerate(MAJORS)}

# magics that can look like text get checked BEFORE the text heuristic
_TEXTISH_MAGICS = (
    (b"%PDF", "document/pdf", "document"),
    (b"<?xml", "data/xml", "data"),
)

_MAGICS = (
    (b"\x89PNG\r\n\x1a\n", "image/png", "image"),
    (b"\xff\xd8\xff", "image/jpeg", "image"),
    (b"GIF87a", "image/gif", "image"),
    (b"GIF89a", "image/gif", "image"),
    (b"\x1f\x8b", "archive/gzip", "archive"),
    (b"PK\x03\x04", "archive/zip", "archive"),
    (b"PK\x05\x06", "archive/zip", "archive"),
    (b"\xfd7zXZ\x00", "archive/xz", "archive"),
    (b"BZh", "archive/bzip2", "archive"),
    (b"\x28\xb5\x2f\xfd", "archive/zstd", "archive"),
    (b"7z\xbc\xaf\x27\x1c", "archive/7z", "archive"),
    (b"Rar!\x1a\x07", "archive/rar", "archive"),
    (b"SQLite format 3\x00", "database/sqlite", "database"),
    (b"\x7fELF", "executable/elf", "executable"),
    (b"MZ", "executable/pe", "executable"),
    (b"\x00\x61\x73\x6d", "executable/wasm", "executable"),
    (b"\xca\xfe\xba\xbe", "executable/java-class", "executable"),
    (b"\xca\xfe\xba\xbf", "executable/java-class", "executable"),
    (b"OggS", "audio/ogg", "audio"),
    (b"fLaC", "audio/flac", "audio"),
    (b"ID3", "audio/mpeg", "audio"),
)

_CODE_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".hpp", ".cc",
    ".hh", ".cs", ".go", ".rs", ".java", ".rb", ".php", ".swift", ".kt",
    ".scala", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".lua", ".pl",
    ".r", ".m", ".sql", ".html", ".htm", ".css", ".scss", ".vue", ".svelte",
}
_DATA_EXT = {
    ".json", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".xml", ".ini",
    ".cfg", ".conf", ".ndjson", ".properties",
}
_TEXT_EXT = {".txt", ".md", ".rst", ".log", ".tex", ".srt", ".nfo"}
_OFFICE_ZIP = {
    ".docx": "document/docx", ".pptx": "document/pptx",
    ".xlsx": "document/xlsx", ".epub": "document/epub",
}


def _entropy(data: bytes) -> float:
    counts = np.bincount(np.frombuffer(bytes(data), dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(data)
    return float(-(p * np.log2(p)).sum())


def _printable_ratio(head: bytes) -> float:
    s = head.decode("latin-1", errors="replace")
    good = sum(1 for ch in s if ch.isprintable() or ch in "\n\r\t")
    return good / max(1, len(s))


def detect_type(head: bytes, path: str = "",
                entropy: Optional[float] = None) -> Tuple[str, int]:
    """Return (label, major_id) for a bundle's first bytes.

    Order matters: textish magics (PDF/XML) first, then the printable-text
    heuristic (so prose starting with 'MZ' isn't called an executable), then
    binary magics, then containers, then entropy fallbacks.
    """
    head = bytes(head[:512])
    if not head:
        return ("empty", MAJOR_ID["other"])

    ext = os.path.splitext(path)[1].lower()

    for magic, label, major in _TEXTISH_MAGICS:
        if head.startswith(magic):
            return (label, MAJOR_ID[major])

    if _printable_ratio(head) >= 0.85:
        if ext in _CODE_EXT:
            return (f"code/{ext.lstrip('.')}", MAJOR_ID["code"])
        if ext in _DATA_EXT:
            return (f"data/{ext.lstrip('.')}", MAJOR_ID["data"])
        if ext in _TEXT_EXT:
            return (f"text/{ext.lstrip('.')}", MAJOR_ID["text"])
        return ("text/plain", MAJOR_ID["text"])

    for magic, label, major in _MAGICS:
        if head.startswith(magic):
            if label == "archive/zip":
                if ext in _OFFICE_ZIP:
                    return (_OFFICE_ZIP[ext], MAJOR_ID["document"])
                if ext == ".jar":
                    return ("executable/jar", MAJOR_ID["executable"])
                if ext == ".apk":
                    return ("executable/apk", MAJOR_ID["executable"])
            return (label, MAJOR_ID[major])

    if head[:4] == b"RIFF" and len(head) >= 12:
        sub = head[8:12]
        if sub == b"WEBP":
            return ("image/webp", MAJOR_ID["image"])
        if sub == b"WAVE":
            return ("audio/wav", MAJOR_ID["audio"])
        if sub == b"AVI ":
            return ("video/avi", MAJOR_ID["video"])
        return ("container/riff", MAJOR_ID["binary"])

    if head[4:8] == b"ftyp":
        if ext == ".m4a":
            return ("audio/m4a", MAJOR_ID["audio"])
        return ("video/mp4", MAJOR_ID["video"])

    if entropy is None:
        entropy = _entropy(head)
    printable = _printable_ratio(head)
    if entropy >= 7.4 and printable < 0.4:
        return ("binary/high-entropy", MAJOR_ID["binary"])
    return ("binary/unknown", MAJOR_ID["binary"])