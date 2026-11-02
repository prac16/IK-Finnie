"""Bounded local artifacts, scoped to one conversation; no raw files in state."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from rag.ingest import PROJECT_ROOT


class ArtifactStore:
    def __init__(self, root: Path | None = None, *, max_bytes=16 * 1024 * 1024, max_per_thread=128, persist=True):
        self.root = Path(root) if root is not None else PROJECT_ROOT / 'runtime' / 'artifacts'
        self.max_bytes, self.max_per_thread = max_bytes, max_per_thread
        self.persist = persist
        self._payloads: dict[tuple[str, str], bytes] = {}
        self._metadata: dict[str, dict[str, dict]] = {}
        self._objects: dict[tuple[str, str], object] = {}

    def list(self, thread_id: str) -> list[dict]:
        return deepcopy(list(self._metadata.get(thread_id, {}).values()))

    def put(self, thread_id: str, kind: str, payload: dict, *, name='', native=None, parent_id=None) -> dict:
        items = self._metadata.setdefault(thread_id, {})
        if len(items) >= self.max_per_thread:
            raise ValueError('Conversation artifact limit reached; start a new conversation')
        data = json.dumps(payload, allow_nan=False, sort_keys=True).encode()
        if len(data) > self.max_bytes:
            raise ValueError('Artifact exceeds the configured size limit')
        artifact_id = uuid4().hex
        parent = self.metadata(thread_id, parent_id) if parent_id else None
        if self.persist:
            directory = self.root / hashlib.sha256(thread_id.encode()).hexdigest()
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = directory / (artifact_id + '.json')
            with path.open('xb') as output:
                path.chmod(0o600)
                output.write(data)
        else:
            self._payloads[thread_id, artifact_id] = data
        metadata = dict(id=artifact_id, kind=kind, name=name, sha256=hashlib.sha256(data).hexdigest(),
                        version=parent['version'] + 1 if parent else 1, parent_id=parent_id)
        items[artifact_id] = metadata
        if native is not None:
            self._objects[thread_id, artifact_id] = native
        return deepcopy(metadata)

    def metadata(self, thread_id: str, artifact_id: str) -> dict:
        try:
            return deepcopy(self._metadata[thread_id][artifact_id])
        except KeyError:
            raise ValueError('Artifact is unavailable in this conversation') from None

    def get(self, thread_id: str, artifact_id: str, *, kind: str | None = None) -> dict:
        meta = self.metadata(thread_id, artifact_id)
        if kind and meta['kind'] != kind:
            raise ValueError('Artifact has the wrong input type')
        if self.persist:
            path = self.root / hashlib.sha256(thread_id.encode()).hexdigest() / (meta['id'] + '.json')
            if path.stat().st_size > self.max_bytes:
                raise ValueError('Artifact exceeds the configured size limit')
            data = path.read_bytes()
        else:
            data = self._payloads[thread_id, artifact_id]
        if hashlib.sha256(data).hexdigest() != meta['sha256']:
            raise ValueError('Artifact integrity check failed')
        return json.loads(data)

    def native(self, thread_id: str, artifact_id: str):
        self.metadata(thread_id, artifact_id)
        try:
            return self._objects[thread_id, artifact_id]
        except KeyError:
            raise ValueError('Report is no longer available; run the analysis again') from None

    def add_holdings(self, thread_id: str, content: bytes, filename: str) -> list[dict]:
        from agents.portfolio.holdings import load_holdings
        portfolios = load_holdings(content, filename)
        if len(portfolios) + len(self.list(thread_id)) > self.max_per_thread:
            raise ValueError('Upload exceeds the conversation artifact limit')
        return [self.put(thread_id, 'portfolio', asdict(p), name=p.name, native=p) for p in portfolios]

    def add_plan(self, thread_id: str, inputs: dict, *, parent_id=None) -> dict:
        from agents.goals.models import GoalPlan
        plan = GoalPlan.model_validate(inputs)
        return self.put(thread_id, 'goal_plan', plan.model_dump(mode='json'), name='Goals plan', parent_id=parent_id)
