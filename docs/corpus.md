# Stage 3 corpus

The project uses two local PDF documents from the existing `doc-ocr` workspace. The PDFs are not copied into this repository.

## Approved documents

| Document | Path | SHA-256 prefix (`doc_id`) | PDF pages | Docling extracted chars | Diacritic check | Status |
|---|---|---:|---:|---:|---|---|
| SGK Toán 9, tập một-mau.pdf | `D:\Thien\Code\doc-ocr\workspace\SGK Toán 9, tập một-mau.pdf` | `485d9e52eeaddfd6` | 123 | 3,758 | passed on extracted text | approved, but text layer is sparse |
| English Vocabulary in Use Pre-Intermediate.pdf | `D:\Thien\Code\doc-ocr\workspace\English-Vocabulary-in-Use-Pre-Intermediate.pdf` | `3a357be041df101c` | 265 | 686,012 | passed (`0.9887` valid-syllable ratio) | approved baseline candidate |

Checks were run with Docling 2.150.0 in WSL Ubuntu, with PDF OCR disabled (`do_ocr=False`). The extracted Markdown was used only for this probe; the production parser must use Docling document items so page provenance is retained.

## Licensing and data egress

The documents are user-provided local assets. Before running the API judge, the user must confirm that the license/permission for each selected document permits sending question, answer, ground-truth and retrieved excerpts to the configured third-party-compatible gateway. The API judge is never used on arbitrary web uploads.

## Corpus selection ruling

The English book is the initial V1/V2/V3 baseline because it has a substantial native text layer. The Math book remains an approved second document for metadata-filtering and multi-document tests, but its current PDF extraction is sparse and dominated by image placeholders. It must not be used as the only evaluation corpus until a page-aware OCR/text-quality pass produces enough reliable text. This keeps both user-approved books in the project without allowing a low-quality extraction to distort the baseline eval.

## Chunk validation (Docling, OCR disabled, CUDA enabled)

The English PDF produced 8,720 page-aware blocks across PDF pages 3–265. With whitespace token counting and overlap 64, the chunk sweep produced:

| Chunk size | Chunks | Heading-tagged chunks |
|---:|---:|---:|
| 256 | 472 | 0 |
| 512 | 203 | 0 |
| 1024 | 95 | 0 |

The PDF has no usable Docling heading labels in this configuration, so the chunker uses fixed-size windows and preserves page ranges. The Math PDF was intentionally rejected by the parser's minimum usable-text check (`PDF contains no usable text`) after Docling returned image placeholders; it remains eligible only after an OCR/text-quality pass.

The long validation run also emitted Docling table-structure warnings for several complex tables. Docling retained their text as cells where it discarded an overlong predicted table structure; this is recorded as a known ingestion limitation rather than hidden.

The English PDF's page 3 start is the PDF page index, not necessarily the printed page number. Citations therefore use PDF page ranges until a printed-page offset is explicitly verified.

**Ruling:** use the 512-token configuration as the initial V1 candidate because it provides 203 chunks between the 256 and 1024 extremes; choose the final size only after the frozen dev eval. Cost if wrong: retrieval quality may be lower than another chunk size, but the dev sweep preserves a measured correction path.

**Ruling:** keep `max_pages=300` as the parser default and use the explicit `timeout_s=900` corpus-ingestion command for this 265-page baseline; cost if wrong: smaller deployments may need a shorter cap, but a 300-second default would reject the approved baseline during cold model startup.

**Ruling:** treat the Math PDF's failed native-text check as an expected quality gate, not a Task 3 failure; cost if wrong: multi-document filtering tests need a future OCR-derived fixture instead of this source PDF.


To reproduce the probe in WSL:

```bash
cd /mnt/d/Thien/Code/docs-qa
.venv/bin/python /mnt/d/Thien/Code/_spikes/docsqa_probe_corpus.py \
  "/mnt/d/Thien/Code/doc-ocr/workspace/SGK Toán 9, tập một-mau.pdf" \
  "/mnt/d/Thien/Code/doc-ocr/workspace/English-Vocabulary-in-Use-Pre-Intermediate.pdf"
```

To reproduce chunk counts, use the WSL script in `/mnt/d/Thien/Code/_spikes/docsqa_validate_chunks.sh` with the parser timeout set to 900 seconds.

To reproduce the probe in WSL:

```bash
cd /mnt/d/Thien/Code/docs-qa
.venv/bin/python /mnt/d/Thien/Code/_spikes/docsqa_probe_corpus.py \
  "/mnt/d/Thien/Code/doc-ocr/workspace/SGK Toán 9, tập một-mau.pdf" \
  "/mnt/d/Thien/Code/doc-ocr/workspace/English-Vocabulary-in-Use-Pre-Intermediate.pdf"
```
