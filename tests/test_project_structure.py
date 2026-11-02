"""Keep API transports and vector infrastructure out of agent implementations."""
import ast
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from rag.store import local_collection

SRC = Path(__file__).resolve().parents[1] / 'src'


def imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


class ProjectStructureTests(unittest.TestCase):
    def test_agents_use_tools_and_rag_instead_of_api_and_database_clients(self):
        forbidden = {'openai', 'chromadb', 'requests', 'httpx', 'urllib.request'}
        for path in (SRC / 'agents').rglob('*.py'):
            for module in imported_modules(path):
                with self.subTest(file=path.name, module=module):
                    self.assertFalse(any(module == name or module.startswith(name + '.') for name in forbidden))

    def test_api_tools_do_not_depend_on_agent_modules(self):
        for path in (SRC / 'tools').glob('*.py'):
            self.assertFalse(any(module == 'agents' or module.startswith('agents.') for module in imported_modules(path)))

    def test_new_command_entry_points_are_importable(self):
        for name in ('agents.chat.agent', 'agents.portfolio.analysis', 'agents.portfolio.upload', 'agents.market.agent', 'agents.goals.agent'):
            self.assertTrue(callable(importlib.import_module(name).main))

    @patch('rag.store.chromadb.PersistentClient')
    def test_missing_index_is_not_silently_created(self, client):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                local_collection(Path(directory) / 'missing')
        client.assert_not_called()

    @patch('rag.store.chromadb.PersistentClient')
    def test_collection_connection_accepts_named_collection_objects(self, client):
        client.return_value.list_collections.return_value = [Mock(name='irrelevant')]
        client.return_value.list_collections.return_value[0].name = 'finnie_pdfs'
        with TemporaryDirectory() as directory:
            collection = local_collection(Path(directory))
        self.assertIs(collection, client.return_value.get_collection.return_value)
        client.return_value.get_collection.assert_called_once_with(name='finnie_pdfs')
