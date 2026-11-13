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

To reproduce the probe in WSL:

```bash
cd /mnt/d/Thien/Code/docs-qa
.venv/bin/python /mnt/d/Thien/Code/_spikes/docsqa_probe_corpus.py \
  "/mnt/d/Thien/Code/doc-ocr/workspace/SGK Toán 9, tập một-mau.pdf" \
  "/mnt/d/Thien/Code/doc-ocr/workspace/English-Vocabulary-in-Use-Pre-Intermediate.pdf"
```
