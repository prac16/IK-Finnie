# Financial knowledge base

- `documents/`: curated financial reference PDFs used for RAG.
- `index/`: generated persistent Chroma database; excluded from Git.

Sample holdings, workbooks, and goal-plan inputs belong in `src/data/`.
Generated analysis reports belong in `reports/`.

From the project root, build or update the index with:

```bash
.venv/bin/finnie-index-pdfs
```

Existing chunk IDs use source filenames, pages, and chunk positions. Moving
this knowledge base within the project did not require re-embedding documents.
The default paths are defined in `src/rag/ingest.py`; CLI options can override
the document or index locations.
