"""Serve only cited knowledge-base PDFs through Streamlit's static-file route."""
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

from rag.ingest import DEFAULT_PDF_DIR, PROJECT_ROOT

# Streamlit resolves static/ relative to the running entry script. Support both
# the repository wrapper and the installed finnie-ui launcher.
STATIC_DIRS = (PROJECT_ROOT / 'static' / 'references', Path(__file__).parent / 'static' / 'references')


@lru_cache(maxsize=256)
def _publish(path: Path, size: int, modified_ns: int, destinations: tuple[Path, ...]):
    with path.open('rb') as source:
        if source.read(5) != b'%PDF-':
            return None
        source.seek(0)
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    filename = digest + '.pdf'
    for directory in destinations:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / filename
        if not target.exists():
            shutil.copyfile(path, target)
    return 'app/static/references/' + filename


def source_pdf_url(name, page, *, pdf_dir=None, destinations=None):
    root = Path(pdf_dir) if pdf_dir is not None else DEFAULT_PDF_DIR
    path = root / name
    if (path.resolve().parent != root.resolve() or path.suffix.lower() != '.pdf'
            or not path.is_file() or type(page) is not int or page < 1):
        return None
    stat = path.stat()
    try:
        url = _publish(path.resolve(), stat.st_size, stat.st_mtime_ns,
                       tuple(Path(p) for p in (destinations if destinations is not None else STATIC_DIRS)))
    except OSError:
        return None
    return f'{url}#page={page}' if url else None


def link_source_pdfs(markdown, sources, *, pdf_dir=None, destinations=None):
    """Replace filesystem links with browser URLs; never publish uploaded files."""
    def replace(match):
        label, original = match.groups()
        parsed = urlsplit(original)
        name = Path(unquote(parsed.path)).name
        if not name.lower().endswith('.pdf'):
            return match[0]
        page_match = re.fullmatch(r'page=(\d+)', parsed.fragment)
        page = int(page_match[1]) if page_match else 1
        if name not in sources or page not in sources[name]:
            return label
        url = source_pdf_url(name, page, pdf_dir=pdf_dir, destinations=destinations)
        return f'[{label}]({url})' if url else label
    return re.sub(r'\[([^\]\n]*)\]\(([^)\n]*)\)', replace, markdown)
