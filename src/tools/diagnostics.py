"""Bounded local diagnostics without prompts, financial inputs or exception text."""
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
from threading import Lock

_LOCK = Lock()
_LOGGER = None


def log_event(event, **fields):
    global _LOGGER
    entry = {'time_utc': datetime.now(timezone.utc).isoformat(), 'event': event}
    for key in ('agent', 'stage', 'attempt', 'error_type', 'elapsed_ms'):
        if key in fields:
            value = fields[key]
            if isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', value)):
                entry[key] = value
    if 'model' in fields:
        model = str(fields['model'])
        entry['model'] = model if re.fullmatch(r'(?:gpt-|o[0-9])[A-Za-z0-9_.-]{0,70}', model) else 'configured_model'
    try:
        with _LOCK:
            if _LOGGER is None:
                directory = Path(__file__).resolve().parents[2] / 'runtime' / 'logs'
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                path = directory / 'finnie.jsonl'
                path.touch(mode=0o600, exist_ok=True)
                path.chmod(0o600)
                _LOGGER = logging.getLogger('finnie.private_diagnostics')
                _LOGGER.setLevel(logging.INFO)
                _LOGGER.propagate = False
                handler = RotatingFileHandler(path, maxBytes=1024 * 1024, backupCount=3)
                handler.setFormatter(logging.Formatter('%(message)s'))
                _LOGGER.addHandler(handler)
            _LOGGER.info(json.dumps(entry))
    except OSError:
        pass  # Logging failure must never interrupt user work or print secrets.
