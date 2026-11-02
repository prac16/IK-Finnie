"""Installed local-only UI launcher."""
from pathlib import Path
import sys
from streamlit.web.cli import main as streamlit_main


def main():
    app = Path(__file__).with_name('app.py')
    sys.argv = ['streamlit', 'run', str(app), '--server.address=127.0.0.1',
                '--server.maxUploadSize=10', '--server.enableStaticServing=true',
                '--browser.gatherUsageStats=false', *sys.argv[1:]]
    streamlit_main()
