from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass


_WHITESPACE = re.compile(r"\s+")
_SYLLABLE = re.compile(r"^[A-Za-zÀ-ỹĐđ]+$")


def normalize_text(s: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", s)).strip()


def strip_accents(s: str) -> str:
    normalized = unicodedata.normalize("NFD", s)
    without_marks = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return without_marks.replace("Đ", "D").replace("đ", "d")


def fingerprint_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc_id_from_bytes(data: bytes) -> str:
    return fingerprint_bytes(data)[:16]


@dataclass(frozen=True)
class DiacriticReport:
    valid_syllable_ratio: float
    nfd_ratio: float
    text_chars: int
    passed: bool


def check_diacritics(
    text: str, min_ratio: float = 0.9, min_text_chars: int = 1_000
) -> DiacriticReport:
    normalized = unicodedata.normalize("NFC", text)
    visible_text = re.sub(r"<!--.*?-->", "", normalized)
    text_chars = len(visible_text.strip())
    syllables = re.findall(r"\b[^\s\d.,;:!?()\[\]{}\"']+\b", visible_text)
    if not syllables:
        return DiacriticReport(0.0, 0.0, text_chars, False)

    valid = sum(bool(_SYLLABLE.fullmatch(token)) for token in syllables)
    nfd_chars = sum(unicodedata.normalize("NFD", c) != c for c in visible_text)
    nfd_ratio = nfd_chars / max(text_chars, 1)
    ratio = valid / len(syllables)
    return DiacriticReport(ratio, nfd_ratio, text_chars, ratio >= min_ratio and text_chars >= min_text_chars)
