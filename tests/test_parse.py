import multiprocessing as mp
from pathlib import Path

import pytest

from docsqa.parse import (
    IngestError,
    _collect_worker_result,
    parse_pdf,
    pipeline_options,
)


def _put_large_payload(queue) -> None:
    queue.put(("ok", [("text", 1, None)] * 20_000, 1))


def _write_minimal_pdf(path: Path, pages: int = 3) -> None:
    objects = ["1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"]
    kids = " ".join(f"{i + 3} 0 R" for i in range(pages))
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {pages} >>\nendobj\n")
    for i in range(pages):
        objects.append(
            f"{i + 3} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
            f"/Contents {pages + 3 + i} 0 R /Resources << /Font << /F1 {2 * pages + 3} 0 R >> >> >>\nendobj\n"
        )
    font_id = 2 * pages + 3
    objects.append(f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    for i in range(pages):
        stream = f"BT /F1 18 Tf 40 200 Td (Page {i + 1}) Tj ET"
        objects.append(f"{pages + 3 + i} 0 obj\n<< /Length {len(stream)} >>\nstream\n{stream}\nendstream\nendobj\n")
    body = "%PDF-1.4\n" + "".join(objects) + "%%EOF\n"
    path.write_bytes(body.encode("ascii"))


def test_collect_worker_result_drains_large_payload_before_join() -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_put_large_payload, args=(queue,))
    process.start()

    message = _collect_worker_result(process, queue, timeout_s=10)
    process.join(timeout=2)

    assert message[0] == "ok"
    assert len(message[1]) == 20_000
    assert not process.is_alive()


def test_pipeline_options_use_cuda_when_available() -> None:
    options = pipeline_options()

    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert options.accelerator_options.device == expected
    assert options.do_ocr is False


def test_parse_pdf_keeps_page_numbers(tmp_path: Path) -> None:
    pdf = tmp_path / "three-pages.pdf"
    _write_minimal_pdf(pdf, pages=3)

    blocks = parse_pdf(pdf)

    assert {block.page for block in blocks} == {1, 2, 3}
    assert any("Page 1" in block.text for block in blocks)


def test_empty_or_image_only_pdf_raises_ingest_error(tmp_path: Path) -> None:
    pdf = tmp_path / "empty.pdf"
    _write_minimal_pdf(pdf, pages=1)

    with pytest.raises(IngestError, match="text"):
        parse_pdf(pdf)


def test_encrypted_pdf_raises_ingest_error(tmp_path: Path) -> None:
    pdf = tmp_path / "encrypted.pdf"
    pdf.write_bytes(b"%PDF-1.4\n/Encrypt fake\n%%EOF")

    with pytest.raises(IngestError):
        parse_pdf(pdf)


def test_page_cap_and_timeout_raise_ingest_error(tmp_path: Path) -> None:
    pdf = tmp_path / "too-many-pages.pdf"
    _write_minimal_pdf(pdf, pages=3)

    with pytest.raises(IngestError, match="page"):
        parse_pdf(pdf, max_pages=2)
    with pytest.raises(IngestError, match="timed out"):
        parse_pdf(pdf, timeout_s=0)
