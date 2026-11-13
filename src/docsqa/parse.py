from __future__ import annotations

import multiprocessing as mp
from queue import Empty
from pathlib import Path
from typing import Any

from .chunking import Block
from .corpus import normalize_text


class IngestError(ValueError):
    pass


def pipeline_options() -> Any:
    import torch
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PdfPipelineOptions(
        do_ocr=False,
        accelerator_options=AcceleratorOptions(device=device, num_threads=4),
    )


def _parse_worker(path_str: str, queue: Any) -> None:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = pipeline_options()
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        result = converter.convert(Path(path_str))
        document = result.document
        blocks: list[tuple[str, int, str | None]] = []
        for item, _level in document.iterate_items():
            text = getattr(item, "text", None)
            if not text or not str(text).strip():
                continue
            provenance = getattr(item, "prov", None) or []
            pages = [getattr(prov, "page_no", None) for prov in provenance]
            pages = [page for page in pages if isinstance(page, int)]
            page = pages[0] if pages else None
            if page is None:
                continue
            label = str(getattr(item, "label", ""))
            heading = normalize_text(str(text)) if "heading" in label.lower() else None
            blocks.append((normalize_text(str(text)), page, heading))
        queue.put(("ok", blocks, len(getattr(document, "pages", {}))))
    except Exception as exc:  # child process boundary: send a safe message only
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _collect_worker_result(process: mp.Process, queue: Any, timeout_s: int) -> Any:
    try:
        message = queue.get(timeout=timeout_s)
    except Empty as exc:
        if process.is_alive():
            process.terminate()
            process.join()
        raise IngestError("parser exited without text") from exc
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join()
        raise IngestError("parse timed out")
    return message


def parse_pdf(path: Path, max_pages: int = 300, timeout_s: int = 300) -> list[Block]:
    if timeout_s <= 0:
        raise IngestError("parse timed out")
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise IngestError("input must be an existing PDF")
    data = path.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise IngestError("input is not a PDF")
    if b"/Encrypt" in data[:100_000]:
        raise IngestError("encrypted PDFs are not supported")

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_parse_worker, args=(str(path), queue))
    process.start()
    message = _collect_worker_result(process, queue, timeout_s)
    if message[0] == "error":
        raise IngestError(message[1])
    _, raw_blocks, page_count = message
    if page_count > max_pages:
        raise IngestError(f"PDF exceeds page limit of {max_pages}")
    blocks = [Block(text, page, heading) for text, page, heading in raw_blocks]
    if not blocks or len(" ".join(block.text for block in blocks)) < 20:
        raise IngestError("PDF contains no usable text")
    return blocks
