"""Local data minimization at the external LLM boundary; no model-based redaction."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import re

_IDENTIFIERS = ContextVar('llm_private_identifiers', default=())
_DROP = {'email', 'phone', 'phone_number', 'address', 'street_address', 'date_of_birth', 'dob',
         'ssn', 'social_security_number', 'account_number', 'account_id', 'customer_id', 'user_id',
         'profile_id', 'portfolio_id', 'plan_id', 'age', 'owner', 'owner_name', 'full_name',
         'first_name', 'last_name', 'filename', 'file_name', 'input_source', 'sector_source'}
_DROP |= {'api_key', 'access_token', 'password', 'credentials'}


@contextmanager
def privacy_scope(identifiers):
    """Known local names are scoped to this call, never shared across sessions."""
    token = _IDENTIFIERS.set(tuple(str(v) for v in identifiers if v and len(str(v).strip()) >= 3))
    try:
        yield
    finally:
        _IDENTIFIERS.reset(token)


def redact_text(text: str, identifiers=()) -> str:
    for value in sorted(set((*_IDENTIFIERS.get(), *identifiers)), key=len, reverse=True):
        if value and len(value) >= 3:
            text = re.sub(r'(?<!\w)' + re.escape(value) + r'(?!\w)', '[private]', text, flags=re.I)
    patterns = [
        (r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', '[email]'),
        (r'(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)', '[government ID]'),
        (r'(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\d)', '[phone]'),
        (r'(?<!\d)\+?\d{10,12}(?!\d)', '[phone/identifier]'),
        (r'(?<!\d)(?:\d[ -]?){13,19}(?!\d)', '[account/card number]'),
        (r'\b(?:account|acct|routing|passport|tax id|national id|ssn)\s*(?:number|no\.?|#|:)?\s*[:#]?\s*[A-Z0-9-]{5,}\b', '[account/ID]'),
        (r'\b(?:dob|date of birth|born on)\s*[:=]?\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b', '[birth date]'),
        (r'\b\d{1,6}\s+(?:[\w.-]+\s+){0,5}(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd)\b(?:\s+(?:apt|unit|#)\s*[\w-]+)?', '[address]'),
        (r'https?://\S+|(?:/Users/|/home/|[A-Z]:\\)[^\s\"<>]+', '[link/path]'),
        (r'\b[a-f0-9]{32}\b', '[identifier]'),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.I)
    # Additional best-effort coverage for names in chat/excerpts; structured
    # report names are removed deterministically below, not by this heuristic.
    text = re.sub(r'\b[A-Z][a-z]+(?:[-\x27][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+(?:[-\x27][A-Z]?[a-z]+)?){1,3}\b', '[name]', text)
    return text


def goal_purpose(text):
    lower = str(text).casefold()
    purposes = [('retire', 'retirement'), ('house', 'home purchase'), ('home', 'home purchase'),
                ('educat', 'education'), ('college', 'education'), ('emergency', 'emergency reserve'),
                ('travel', 'travel'), ('growth', 'growth'), ('income', 'income'),
                ('capital preservation', 'capital preservation'), ('crypto', 'crypto investment')]
    return next((label for term, label in purposes if term in lower), 'financial goal')


def question_topics(text):
    topics = ['drawdown', 'volatility', 'concentration', 'allocation', 'sector', 'shortfall', 'budget',
              'contribution', 'fees', 'inflation', 'deadline', 'risk', 'return', 'sources']
    selected = [topic for topic in topics if topic in str(text).casefold()]
    return 'Explain the saved report' + (', focusing on: ' + ', '.join(selected) if selected else ' in plain language') + '.'


class PrivatePayload:
    """Request-local aliases are reversed only after the provider returns."""
    def __init__(self, context: dict, kind: str):
        data = deepcopy(context)
        self.reverse = {}
        replacements = {}
        allowed = {
            'portfolio': {'goal', 'holdings', 'valuation_date', 'report', 'retrieved_chunks', 'knowledge_warnings', 'user_question'},
            'goals': {'as_of_date', 'currency', 'funding', 'results', 'limitations', 'warnings', 'retrieved_chunks', 'allowed_field_paths', 'user_question'},
            'market': {'request', 'metrics', 'price_basis', 'data_sources', 'limitations', 'warnings', 'retrieved_chunks', 'user_question'},
            'router': {'message', 'active_context', 'artifacts'},
        }[kind]
        data = {k: v for k, v in data.items() if k in allowed}
        if 'user_question' in data:
            data['user_question'] = question_topics(data['user_question'])
        for index, chunk in enumerate(data.get('retrieved_chunks', [])):
            if 'source' in chunk:
                chunk['source'] = 'reference_' + chr(97 + index)
        if kind == 'portfolio':
            if data.get('goal'):
                replacements[data['goal']] = goal_purpose(data['goal'])
                data['goal'] = goal_purpose(data['goal'])
            data['holdings'] = [{k: v for k, v in h.items() if k in {
                'ticker', 'asset_type', 'asset_class_override', 'weight_pct', 'market_value_usd', 'sector'}}
                for h in data.get('holdings', [])]
            # Only measured fields and generic calculation limitations, not
            # arbitrary report prose or profile metadata.
            fields = {'calculated_value_usd', 'asset_allocation_pct', 'top_holding_pct', 'top_three_pct', 'hhi',
                      'effective_holdings', 'known_technology_stock_pct', 'crypto_pct', 'sector_allocation_pct',
                      'sector_coverage_pct', 'data_limitations'}
            data['report'] = {k: v for k, v in data.get('report', {}).items() if k in fields}
        if kind == 'goals':
            results = {}
            original_ids = set(data.get('results', {}))
            for index, (goal_id, result) in enumerate(data.get('results', {}).items()):
                alias = 'goal_' + chr(97 + index)
                label = 'Goal ' + chr(65 + index)
                name = result.get('goal', {}).get('name', '')
                replacements[goal_id] = alias
                if name:
                    if name not in original_ids:
                        replacements[name] = label
                    self.reverse[label] = name
                self.reverse[alias] = goal_id
                results[alias] = result
                result['goal']['id'] = alias
                result['goal']['name'] = label + ' (' + goal_purpose(name) + ')'
            data['results'] = results
        if kind == 'router':
            active = data.get('active_context', {})
            for index, goal_id in enumerate(active.get('goal_ids', [])):
                alias = 'goal_' + chr(97 + index)
                replacements[goal_id] = alias
                self.reverse[alias] = goal_id
            data['active_context'] = {k: v for k, v in active.items() if k in {'last_agent', 'market', 'goal_ids'}}
            data['active_context']['has_portfolio'] = bool(active.get('portfolio_id'))
            data['active_context']['has_goal_plan'] = bool(active.get('plan_id'))
            data['artifacts'] = [{'kind': a['kind']} for a in data.get('artifacts', [])]
            # Entire prior reports and conversations are unnecessary for routing.
            # The active structured context supplies the relevant referents.
        self.data = self._clean(data, replacements)

    def _clean(self, value, replacements):
        if isinstance(value, dict):
            return {self._replace(k, replacements): self._clean(v, replacements)
                    for k, v in value.items() if k.casefold() not in _DROP and
                    (k.casefold() != 'name' or re.fullmatch(r'Goal [A-T](?: \([a-z ]+\))?', str(v)))}
        if isinstance(value, (list, tuple)):
            return [self._clean(v, replacements) for v in value]
        if isinstance(value, str):
            return redact_text(self._replace(value, replacements))
        return value

    @staticmethod
    def _replace(text, mapping):
        # One simultaneous substitution avoids cascading aliases.
        mapping = {k: v for k, v in mapping.items() if k}
        if not mapping:
            return text
        pattern = r'(?<!\w)(?:' + '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r')(?!\w)'
        return re.sub(pattern, lambda m: mapping[m[0]], text)

    def restore(self, value):
        if isinstance(value, dict):
            return {k: self.restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.restore(v) for v in value]
        return self._replace(value, self.reverse) if isinstance(value, str) else value
