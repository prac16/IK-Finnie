"""Repository entry point: streamlit run streamlit_app.py."""
from pathlib import Path
import runpy

# Execute the page afresh on reruns rather than caching its module-level code.
runpy.run_path(str(Path(__file__).parent / 'src/ui/app.py'), run_name='__main__')
