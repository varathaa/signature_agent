# Signature Verification Agent

## What this agent does

The agent compares handwritten signatures on customer letters with reference signatures in a master register PDF. It extracts the customer identifier (CIF), name, and signature image, then assigns a result for each letter.

It is a document-processing aid, not a legal or biometric identity decision. Treat results as triage and have a person inspect uncertain or consequential cases.

## What you get

For each letter, the agent can return one of these results:

| Result | Meaning |
| --- | --- |
| `MATCH` | A signature was extracted, the CIF exists in the master register, and the similarity score is at or above the configured match threshold. |
| `MISMATCH` | A signature was extracted and the score is below the review band. This is a visual dissimilarity flag, not proof of fraud. |
| `REVIEW` | The CIF or signature could not be extracted, no ink was detected, or the score is close to the match threshold and needs a person to inspect it. |
| `NOT_FOUND` | A CIF was extracted from the letter, but the master register has no signature for that CIF. |

Each result row contains the CIF, customer name, letter filename, similarity score, result, and remarks. The score is a visual similarity measure between 0 and 1, not a calibrated probability that the signature is genuine. A score is blank when comparison could not be made.


## How it works internally

1. **Load the master register.** The pipeline detects whether the PDF has a text layer. Text PDFs use PDF text and page geometry to locate CIFs, names, and signature crops. Image-only PDFs are rendered and sent to the configured Ollama vision model to identify those fields and return signature bounding boxes.
2. **Build the reference store.** Records with both a CIF and signature image are kept in memory and their signature crops are written under `work/master_store/`. Records without a usable signature are skipped.
3. **Find letters.** The CLI agent lists PDF files in the requested folder and excludes filenames containing `master`. The batch runner similarly ignores master PDFs. In the Streamlit app, the user uploads one or more letters and chooses whether to check all of them or just one.
4. **Extract each letter.** Text-based and scanned letters use their corresponding extraction paths. The current letter extractors read the first page. CIFs are expected to be eight digits.
5. **Compare signatures.** OpenCV converts each crop to a binary ink mask, trims empty space, and normalizes it to a fixed canvas. The score combines a symmetric stroke-distance score (60%) and a tolerant stroke-overlap score (40%).
6. **Assign a result and report it.** Python applies the threshold rules above. The LLM in the CLI agent coordinates tool calls and summarizes tool results; it does not calculate scores or decide the comparison itself. The summary is instructed to report tool output rather than invent scores.

## Ways to use it

### Browser app

Run `streamlit run streamlit_app.py`, upload the master register and customer letters, choose `All letters` or `One customer`, and select **Run verification**. The app shows counts by result, a per-letter table, and extracted master/letter signature crops. **Download CSV** downloads the current results as `signature_results.csv`.

### Batch run

Run `python run.py [master.pdf] [letters_dir] [out.csv]`. Arguments are optional; defaults use `samples/master_signature.pdf`, the `samples/` directory, and `output/cif_list.csv`. The batch runner prints one line per letter and checks agreement with `samples/expected_results.csv` when that file exists.

### LLM-driven CLI agent

Run `python agent.py "Verify all letters in samples against samples/master_signature.pdf"`. It uses Ollama by default, loads the master, lists letters, verifies each letter, saves `output/cif_list.csv`, and prints a summary. The model and Ollama server URL are configurable as described below.

## Files and configuration

- `output/cif_list.csv`: default CSV output from the CLI agent and batch runner. The browser app instead offers a download and does not write that CSV by default.
- `work/master_store/`: extracted master signature crops.
- `work/letter_crops/`: signature crops from letters processed by the pipeline.
- `OLLAMA_BASE_URL`: Ollama endpoint; defaults to `http://localhost:11434`.
- `OLLAMA_VISION_MODEL`: model for scanned-PDF extraction; defaults to `qwen2.5vl:3b`.
- `OLLAMA_AGENT_MODEL`: model for CLI tool orchestration and summary; defaults to `qwen2.5vl:3b`.
- `RENDER_DPI`: PDF rendering resolution; defaults to `250` for standard rendering. The scanned-PDF vision path renders at `150` DPI.
- `MATCH_THRESHOLD` and `REVIEW_BAND`: defaults are `0.60` and `0.05`.

The Ollama endpoint can be changed through configuration. With the default local endpoint, model requests go to the local Ollama server; scanned pages are included in those vision requests.

## Limitations and practical checks

- Poor scans, unusual layouts, overlapping marks, low contrast, or inaccurate bounding boxes can produce bad crops or missed CIFs.
- The text extraction path relies on expected document wording and geometry. The scanned path relies on the vision model's extracted JSON and bounding boxes.
- Letters are currently read from page one only. Master registers may contain multiple pages.
- Similarity is based on normalized image shape, not signer identity, writing dynamics, or a trained forensic model. A genuine signature can vary, and a forgery can look similar.
- `MATCH` and `MISMATCH` are threshold-based flags. Inspect the displayed crops and route `REVIEW`, missing data, and high-impact decisions to a human.
- The browser app reports extraction/processing exceptions as a verification failure. The CLI and batch runner also require their configured Python packages and PDF/image dependencies; the CLI agent and scanned-PDF path require a reachable Ollama service and the corresponding models.