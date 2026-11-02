"""Conversational CLI; one process owns the graph and its conversation memory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4
from dotenv import load_dotenv
from rag.ingest import PROJECT_ROOT
from workflow.schemas import TurnRequest
from workflow.service import WorkflowService


def _read_json(path):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('JSON input exceeds 1 MB')
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description='Finnie LangGraph financial assistant')
    parser.add_argument('message', nargs='?', help='Omit for an interactive conversation')
    parser.add_argument('--route', choices=['chat', 'portfolio', 'market', 'goals'])
    parser.add_argument('--arguments', type=Path, help='JSON object containing typed agent arguments')
    parser.add_argument('--holdings', type=Path, help='CSV/XLSX holdings upload')
    parser.add_argument('--plan', type=Path, help='GoalPlan JSON with explicit assumptions')
    parser.add_argument('--thread-id', default=uuid4().hex, help='Conversation ID within this running process')
    parser.add_argument('--no-llm', action='store_true', help='Disable both routing and explanation LLM calls')
    parser.add_argument('--no-rag', action='store_true', help='Skip the local knowledge index (Q&A then unavailable)')
    parser.add_argument('--json', action='store_true', help='Print structured workflow responses')
    parser.add_argument('--progress', action='store_true', help='Print graph progress stages to stderr')
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / '.env', override=False)
    service = WorkflowService(on_event=(lambda e: print(e['stage'], file=sys.stderr)) if args.progress else None)
    base = {}
    try:
        if args.arguments:
            base = _read_json(args.arguments)
            if not isinstance(base, dict):
                raise ValueError('--arguments must contain a JSON object')
        if args.holdings:
            from agents.portfolio.holdings import MAX_FILE_BYTES
            if args.holdings.stat().st_size > MAX_FILE_BYTES:
                raise ValueError('Holdings file exceeds 10 MB')
            portfolios = service.add_holdings(args.thread_id, args.holdings.read_bytes(), args.holdings.name)
            if len(portfolios) == 1:
                base['portfolio_id'] = portfolios[0]['id']
            else:
                print(json.dumps({'portfolios': portfolios}, indent=2), file=sys.stderr)
        if args.plan:
            base['plan_id'] = service.add_plan(args.thread_id, _read_json(args.plan))['id']
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    def display(response):
        print(response.model_dump_json(indent=2) if args.json else response.message)
        if response.clarification and response.clarification.get('options') and not args.json:
            print(json.dumps(response.clarification['options'], indent=2))

    def send(text, initial=False):
        # JSON replies are typed fields, never executable commands or model-selected paths.
        if text.lstrip().startswith('{'):
            update = json.loads(text)
            if not isinstance(update, dict):
                raise ValueError('Input must be a JSON object')
            request = {'message': 'Updated inputs', 'arguments': update}
        else:
            request = {'message': text, 'arguments': base if initial else {}}
        request.update(use_llm=not args.no_llm, use_rag=not args.no_rag)
        if args.route:
            request['route'] = args.route
        if service.get_state(args.thread_id).get('pending_request'):
            return service.resume(args.thread_id, request)
        return service.submit(args.thread_id, TurnRequest.model_validate(request))

    if args.message:
        try:
            response = send(args.message, initial=True)
            display(response)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if response.status != 'complete':
            raise SystemExit(2 if response.status == 'needs_input' else 1)
        return
    print('Finnie: /quit, /cancel, /upload PATH, /plan PATH, /artifacts. Memory lasts for this process.')
    first = True
    while True:
        try:
            text = input('\nYou: ').strip()
            if text in {'/quit', 'quit', 'exit'}:
                break
            if not text:
                continue
            if text == '/cancel':
                service.cancel(args.thread_id)
                print('Pending request cancelled.')
                continue
            if text == '/artifacts':
                print(json.dumps(service.store.list(args.thread_id), indent=2))
                continue
            if text.startswith('/upload '):
                path = Path(text[8:].strip()).expanduser()
                from agents.portfolio.holdings import MAX_FILE_BYTES
                if path.stat().st_size > MAX_FILE_BYTES:
                    raise ValueError('Holdings file exceeds 10 MB')
                print(json.dumps(service.add_holdings(args.thread_id, path.read_bytes(), path.name), indent=2))
                continue
            if text.startswith('/plan '):
                print(json.dumps(service.add_plan(args.thread_id, _read_json(Path(text[6:].strip()).expanduser())), indent=2))
                continue
            display(send(text, initial=first))
            first = False
        except (EOFError, KeyboardInterrupt):
            break
        except (OSError, ValueError) as exc:
            print('Input error: ' + str(exc), file=sys.stderr)


if __name__ == '__main__':
    main()
