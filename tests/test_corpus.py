import unicodedata

from docsqa.corpus import (
    check_diacritics,
    doc_id_from_bytes,
    normalize_text,
    strip_accents,
)


def test_nfd_and_nfc_normalize_equal() -> None:
    nfc = "tiếng Việt"
    nfd = unicodedata.normalize("NFD", nfc)

    assert normalize_text(nfd) == normalize_text(nfc) == nfc


def test_strip_accents_maps_vietnamese() -> None:
    assert strip_accents("Đại học") == "Dai hoc"


def test_same_bytes_same_doc_id_regardless_of_name() -> None:
    source = b"same pdf bytes"

    assert doc_id_from_bytes(source) == doc_id_from_bytes(source)
    assert len(doc_id_from_bytes(source)) == 16


def test_check_diacritics_fails_when_text_layer_is_nearly_empty() -> None:
    scanned = "<!-- image --> " * 123 + "Toán 9"

    report = check_diacritics(scanned)

    assert not report.passed
    assert report.text_chars < 200


def test_check_diacritics_passes_clean_vietnamese_and_fails_mojibake() -> None:
    clean = check_diacritics("Tiếng Việt có nhiều dấu và âm tiết rõ ràng.", min_text_chars=1)
    mojibake = check_diacritics("Tiõng ViÖt cã nhiÒu dáu vµ ©m tiÕt.", min_text_chars=1)

    assert clean.passed
    assert clean.valid_syllable_ratio >= 0.9
    assert not mojibake.passed
    assert mojibake.valid_syllable_ratio < 0.9
