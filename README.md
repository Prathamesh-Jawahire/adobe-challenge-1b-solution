# approach\_explanation.md

## Overview

This system is designed to extract structured insights from a set of PDF documents, optimized for user personas and specific task contexts. The solution is composed of two major stages, orchestrated by `combined.py`: 

    (1) document outline extraction (`round_1A.py`).
    (2) context-aware summarization (`round_1B.py`).

---

## Stage 1: PDF Structure Extraction (`round_1A.py`)

This stage parses each PDF to generate a structured JSON containing a document title and a hierarchical outline of headings.

**Key Steps:**

* **PDF Parsing**: Using the `PyMuPDF` (`fitz`) library, each PDF page is parsed to extract text blocks, font sizes, and boldness flags.
* **Noise Filtering**: Heuristics eliminate noisy content like dates, form fields, tables, or pagination.
* **Heading Detection**: Candidate headings are selected based on formatting consistency, font size (relative to the median), boldness, spacing (vertical gaps), and regular expression patterns (e.g., numbering).
* **Table Avoidance**: Table rows are detected using serial-number patterns and layout heuristics, and excluded from headings.
* **Title Extraction**: The most prominent text near the top 60% of the first page is chosen as the title, using a scoring function based on size, boldness, centrality, and position.
* **Clustering**: `AgglomerativeClustering` (from `scikit-learn`) groups headings into hierarchical levels (H1–H3) using features like font size, bold flag, X-coordinate, and vertical spacing.
* **Output**: A JSON file per PDF is generated, containing the title and ordered outline with heading levels and page numbers.

---

## Stage 2: Persona-Driven Summarization (`round_1B.py`)

This stage matches sections of the document outlines to a task defined by the user persona and generates tailored summaries.

**Key Steps:**

* **Input Parsing**: Reads a user-defined JSON with the persona, task, and a list of PDF filenames.
* **Context Embedding**: Converts both the task context and outline headings to vector embeddings using a local SentenceTransformer (`all-mpnet-base-v2`) and ranks headings by cosine similarity.
* **Relevant Section Selection**: Top-scoring headings are filtered (similarity threshold ≥ 0.25), emphasizing diversity across documents and limiting to 5 pages.
* **Parallel Summarization**:

  * Text is extracted using `pdfplumber`.
  * Summarization is performed using a fine-tuned local FLAN-T5 model (`flan-t5-small-openai-feedback`).
  * The summarization step is executed concurrently via a `ThreadPoolExecutor` with up to 5 threads.
* **Post-Processing**: Summaries are capped at \~150 words, and duplicate jobs are deduplicated based on document and page number.
* **Output**: A final JSON report is produced with metadata, selected sections, and detailed subsection summaries.

---

## Technical Highlights

* **Multithreading**: The summarization step in `round_1B.py` uses multithreading to speed up processing, making use of `ThreadPoolExecutor`.
* **ML Models**:

  * **Sentence Embeddings**: For semantic matching of headings to the task.
  * **Seq2Seq Summarization**: Custom fine-tuned FLAN-T5 model for lightweight, persona-aware summaries.
* **Robustness**: The system handles missing files, malformed PDFs, and enforces execution time limits for scalability and real-time performance.

---

This modular architecture enables scalable, semantic document understanding and summarization tailored to specific roles and tasks.
