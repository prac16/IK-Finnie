"""Cited knowledge PDFs have real browser URLs without publishing user inputs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ui.sources import link_source_pdfs, source_pdf_url


class SourceLinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pdfs = self.root / 'knowledge'
        self.pdfs.mkdir()
        self.content = b'%PDF-1.7\nSource document test\n'
        (self.pdfs / 'Risk Guide.pdf').write_bytes(self.content)
        self.destinations = (self.root / 'wrapper/static/references', self.root / 'ui/static/references')

    def url(self, name, page):
        return source_pdf_url(name, page, pdf_dir=self.pdfs, destinations=self.destinations)

    def test_cited_pdf_is_published_for_both_launchers(self):
        url = self.url('Risk Guide.pdf', 3)
        self.assertTrue(url.startswith('app/static/references/'))
        self.assertTrue(url.endswith('.pdf#page=3'))
        for destination in self.destinations:
            self.assertEqual(next(destination.glob('*.pdf')).read_bytes(), self.content)

    def test_report_filesystem_link_becomes_browser_link(self):
        text = '[Risk Guide.pdf, p. 3](knowledge_base/documents/Risk%20Guide.pdf#page=3)'
        rendered = link_source_pdfs(text, {'Risk Guide.pdf': {3}}, pdf_dir=self.pdfs, destinations=self.destinations)
        self.assertNotIn('knowledge_base/documents', rendered)
        self.assertIn('app/static/references/', rendered)
        self.assertIn('#page=3', rendered)

    def test_traversal_symlinks_missing_and_non_pdf_files_are_not_served(self):
        (self.root / 'private.pdf').write_bytes(self.content)
        (self.pdfs / 'linked.pdf').symlink_to(self.root / 'private.pdf')
        (self.pdfs / 'fake.pdf').write_text('private text')
        for name in ['../private.pdf', 'linked.pdf', 'missing.pdf', 'fake.pdf']:
            self.assertIsNone(self.url(name, 1), name)
        for page in [0, -1, True]:
            self.assertIsNone(self.url('Risk Guide.pdf', page))

    def test_unregistered_pdf_links_are_plain_text(self):
        text = '[Private](file:///Users/user/private.pdf#page=1)'
        self.assertEqual(link_source_pdfs(text, {}, pdf_dir=self.pdfs, destinations=self.destinations), 'Private')
        self.assertFalse(self.destinations[0].exists())
