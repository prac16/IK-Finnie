import unittest

from rag.ingest import split_text


class SplitTextTests(unittest.TestCase):
    def test_returns_overlapping_chunks_with_normalized_whitespace(self) -> None:
        chunks = split_text("one two   three four five", chunk_size=12, chunk_overlap=4)

        self.assertEqual(chunks, ["one two", "two three", "three four", "four five"])

    def test_keeps_long_words_and_emits_all_text(self) -> None:
        chunks = split_text("abcdefghijklmno", chunk_size=5, chunk_overlap=1)

        self.assertEqual(chunks, ["abcde", "efghi", "ijklm", "mno"])

    def test_rejects_invalid_chunking_configuration(self) -> None:
        with self.assertRaises(ValueError):
            split_text("text", chunk_size=10, chunk_overlap=10)


if __name__ == "__main__":
    unittest.main()
