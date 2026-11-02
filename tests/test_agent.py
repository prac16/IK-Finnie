import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from agents.chat.agent import (
    ABSTENTION,
    Answer,
    OpenAIGenerator,
    QnAAgent,
    Reference,
    TextGenerator,
)


class FakeCollection:
    def __init__(
        self, documents: list[str | None], metadatas: list[dict[str, object] | None]
    ) -> None:
        self.documents = documents
        self.metadatas = metadatas
        self.last_query: str | None = None

    def query(
        self, *, query_texts: list[str], n_results: int, include: list[str], where: dict | None = None
    ) -> dict[str, list[list[object]]]:
        self.last_query = query_texts[0]
        self.asserted_n_results = n_results
        self.asserted_include = include
        return {
            "documents": [cast(list[object], self.documents)],
            "metadatas": [cast(list[object], self.metadatas)],
            "distances": [[0.2] * len(self.documents)],
        }


class FakeGenerator:
    def __init__(self, response: str = "Diversification may reduce risk [1].") -> None:
        self.response = response
        self.prompt: str | None = None
        self.system: str | None = None

    def generate(self, *, system: str, prompt: str) -> str:
        self.system = system
        self.prompt = prompt
        return self.response


class QnAAgentTests(unittest.TestCase):
    def test_answers_from_retrieved_context_and_links_page_references(self) -> None:
        collection = FakeCollection(
            ["Diversification can reduce portfolio risk."],
            [{"source": "Module 3.pdf", "page": 8}],
        )
        generator = FakeGenerator()
        agent = QnAAgent(
            collection=collection,
            generator=cast(TextGenerator, generator),
            top_k=3,
        )

        answer = agent.answer("How can I reduce investment risk?")

        self.assertIn("[1] Diversification can reduce portfolio risk.", generator.prompt)
        self.assertIn("How can I reduce investment risk?", generator.prompt)
        self.assertIn("untrusted reference material", generator.system)
        self.assertIn(
            "This is for educational purposes only and should not be considered financial advice.",
            generator.system,
        )
        self.assertEqual(collection.last_query, "How can I reduce investment risk?")
        self.assertEqual(collection.asserted_n_results, 20)
        self.assertEqual(
            answer.references,
            (
                Reference(
                    source="Module 3.pdf",
                    page=8,
                    url="knowledge_base/documents/Module%203.pdf#page=8",
                ),
            ),
        )
        self.assertIn(
            "[1] [Module 3.pdf, p. 8](knowledge_base/documents/Module%203.pdf#page=8)",
            answer.to_markdown(),
        )
        self.assertIn(
            "Disclaimer: This is for educational purposes only and should not be considered financial advice.",
            answer.text,
        )

    def test_does_not_duplicate_disclaimer_returned_by_generator(self) -> None:
        disclaimer = (
            "This is for educational purposes only and should not be considered "
            "financial advice."
        )
        collection = FakeCollection(
            ["Diversification can reduce portfolio risk."],
            [{"source": "Module 3.pdf", "page": 8}],
        )
        generator = FakeGenerator(f"Answer [1].\n\nDisclaimer: {disclaimer}")
        agent = QnAAgent(collection, cast(TextGenerator, generator))

        answer = agent.answer("How can I reduce investment risk?")

        self.assertEqual(answer.text.count(disclaimer), 1)

    def test_abstains_when_no_usable_passages_are_retrieved(self) -> None:
        collection = FakeCollection([None], [None])
        generator = FakeGenerator()
        agent = QnAAgent(collection, cast(TextGenerator, generator))

        answer = agent.answer("What is not in the index?")

        self.assertEqual(answer.text, ABSTENTION)
        self.assertEqual(answer.references, ())
        self.assertIsNone(generator.prompt)

    def test_rejects_empty_question(self) -> None:
        agent = QnAAgent(FakeCollection([], []), cast(TextGenerator, FakeGenerator()))

        with self.assertRaises(ValueError):
            agent.answer("  ")

class AnswerTests(unittest.TestCase):
    def test_markdown_without_references_contains_only_answer(self) -> None:
        answer = Answer(text="No evidence found.", references=())

        self.assertEqual(answer.to_markdown(), "No evidence found.")


class OpenAIGeneratorTests(unittest.TestCase):
    @patch("tools.openai_generation.OpenAI")
    def test_sends_system_and_user_messages_to_configured_model(
        self, openai_constructor: object
    ) -> None:
        client = openai_constructor.return_value
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Answer grounded in evidence."
                    )
                )
            ]
        )
        generator = OpenAIGenerator(model="gpt-test")

        result = generator.generate(system="Use sources.", prompt="Question?")

        self.assertEqual(result, "Answer grounded in evidence.")
        client.chat.completions.create.assert_called_once_with(
            model="gpt-test",
            store=False,
            messages=[
                {"role": "system", "content": "Use sources."},
                {"role": "user", "content": "Question?"},
            ],
        )

    @patch("tools.openai_generation.OpenAI")
    def test_rejects_empty_response(self, openai_constructor: object) -> None:
        client = openai_constructor.return_value
        client.chat.completions.create.return_value = SimpleNamespace(choices=[])
        generator = OpenAIGenerator()

        with self.assertRaisesRegex(RuntimeError, "OpenAI returned an empty"):
            generator.generate(system="System", prompt="Question")

if __name__ == "__main__":
    unittest.main()
