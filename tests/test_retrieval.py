import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from rag.categories import infer_category
from rag.ingest import iter_pdf_chunks, update_categories
from rag.retrieval import retrieve


class RetrievalTests(unittest.TestCase):
    def test_filters_before_search_ranks_and_selects_top_k(self):
        collection = Mock()
        collection.query.return_value = {
            "documents": [["far", "closest", "middle", "invalid"]],
            "metadatas": [[{"source": f"{i}.pdf", "page": 1, "category": "taxes"} for i in range(4)]],
            "distances": [[0.8, 0.1, 0.4, float("nan")]],
        }
        result = retrieve(collection, "taxes?", category="taxes", top_k=2, candidate_k=4)
        self.assertEqual(collection.query.call_args.kwargs["where"], {"category": "taxes"})
        self.assertEqual([c.text for c in result.chunks], ["closest", "middle"])
        self.assertEqual(result.metrics.selected_distances, (0.1, 0.4))
        self.assertEqual(result.metrics.candidates_returned, 4)
        self.assertEqual(result.metrics.invalid_candidates, 1)
        self.assertEqual(result.metrics.selected_count, 2)
        self.assertEqual(result.metrics.unique_sources, 2)
        self.assertAlmostEqual(result.metrics.mean_similarity, 0.75)
        self.assertGreaterEqual(result.metrics.retrieval_ms, 0)

    def test_empty_results_have_metrics_and_no_similarity(self):
        collection = Mock()
        collection.query.return_value = {"documents": [[]]}
        result = retrieve(collection, "question")
        self.assertEqual(result.chunks, ())
        self.assertIsNone(result.metrics.mean_similarity)
        self.assertNotIn("where", collection.query.call_args.kwargs)

    def test_rejects_invalid_configuration(self):
        for kwargs in ({"top_k": 0}, {"top_k": 5, "candidate_k": 2}, {"category": "unknown"}):
            with self.assertRaises(ValueError):
                retrieve(Mock(), "question", **kwargs)

    def test_drops_missing_scores_and_malformed_metadata(self):
        collection = Mock()
        collection.query.return_value = {
            "documents": [["missing score", "bad page"]],
            "metadatas": [[{"source": "a.pdf", "page": 1}, {"source": "b.pdf", "page": -1}]],
            "distances": [[None, 0.1]],
        }
        self.assertEqual(retrieve(collection, "question").metrics.invalid_candidates, 2)

    def test_document_category_is_written_to_each_chunk(self):
        with patch("rag.ingest.PdfReader") as reader:
            reader.return_value.pages = [Mock()]
            reader.return_value.pages[0].extract_text.return_value = "sample text"
            chunks = list(iter_pdf_chunks(Path("irs p550.pdf"), 100, 10))
            self.assertEqual(chunks[0][1]["category"], "taxes")
            chunks = list(iter_pdf_chunks(Path("irs p550.pdf"), 100, 10, "investing"))
            self.assertEqual(chunks[0][1]["category"], "investing")

    def test_filename_category_rules(self):
        self.assertEqual(infer_category("Module 14_Personal Finance Insurance.pdf"), "insurance")
        self.assertEqual(infer_category("Module 5_Options.pdf"), "trading")
        self.assertEqual(infer_category("Asset allocation - Bogleheads.pdf"), "investing")

    def test_backfill_preserves_metadata_and_does_not_send_embeddings(self):
        with patch("rag.ingest.chromadb.PersistentClient") as client:
            collection = client.return_value.get_collection.return_value
            collection.count.return_value = 2
            collection.get.return_value = {
                "ids": ["a", "b"], "metadatas": [
                    {"source": "irs p550.pdf", "page": 1},
                    {"source": "other.pdf", "page": 2, "category": "insurance"},
                ],
            }
            self.assertEqual(update_categories(Path("unused")), 2)
            collection.update.assert_called_once_with(
                ids=["a", "b"], metadatas=[
                    {"source": "irs p550.pdf", "page": 1, "category": "taxes"},
                    {"source": "other.pdf", "page": 2, "category": "insurance"},
                ],
            )
