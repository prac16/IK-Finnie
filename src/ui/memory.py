"""Opt-in local user memory. Conversation history is never persisted here."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4
import os
import tempfile

from pydantic import BaseModel, ConfigDict, Field
from agents.goals.models import GoalPlan
from rag.ingest import PROJECT_ROOT


class UserProfile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(default_factory=lambda: uuid4().hex, pattern=r'^[a-f0-9]{32}$')
    name: str = Field(min_length=1, max_length=80)
    currency: Literal['USD'] = 'USD'
    risk_preference: Literal['low', 'moderate', 'high'] = 'moderate'
    use_llm: bool = True
    use_rag: bool = True
    saved_plan: GoalPlan | None = None
    saved_holdings_csv: str | None = Field(default=None, max_length=10 * 1024 * 1024)
    saved_holdings_name: str | None = Field(default=None, max_length=160)
    updated_at: str | None = None


class ProfileStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else PROJECT_ROOT / 'runtime' / 'users'

    def _path(self, profile_id):
        if not isinstance(profile_id, str) or len(profile_id) != 32 or any(c not in '0123456789abcdef' for c in profile_id):
            raise ValueError('Invalid local profile ID')
        return self.root / (profile_id + '.json')

    def load(self, profile_id):
        path = self._path(profile_id)
        if path.stat().st_size > 12 * 1024 * 1024:
            raise ValueError('Local profile is too large')
        profile = UserProfile.model_validate_json(path.read_text())
        if profile.id != profile_id:
            raise ValueError('Local profile ID does not match its file')
        return profile

    def list(self):
        profiles = []
        for path in sorted(self.root.glob('*.json')):
            try:
                profiles.append(self.load(path.stem))
            except (OSError, ValueError):
                continue
        return sorted(profiles, key=lambda p: p.name.casefold())

    def save(self, profile: UserProfile):
        profile = UserProfile.model_validate(profile.model_dump())
        if not profile.name.strip():
            raise ValueError('Profile name cannot be blank')
        profile.name = profile.name.strip()
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        serialized = json.dumps(profile.model_dump(mode='json'), allow_nan=False, ensure_ascii=False)
        if len(serialized.encode()) > 12 * 1024 * 1024:
            raise ValueError('Saved profile exceeds 12 MB')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self._path(profile.id)
        # Write atomically so a reload cannot read half of a saved financial input.
        descriptor, temporary = tempfile.mkstemp(dir=self.root, suffix='.tmp')
        try:
            with os.fdopen(descriptor, 'w') as output:
                output.write(serialized)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return profile
