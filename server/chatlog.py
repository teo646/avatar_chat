"""Chat log: one JSON line per greeting / message / failed message in logs/YYYY-MM-DD.jsonl (outside web/, so it is never served).

A page that opens calls /api/greet right away, so the 'greet' line of a session is the time and the IP of the visit.
"""
import json
import threading
from datetime import datetime
from pathlib import Path

from fastapi import Request

DIR = Path(__file__).resolve().parent.parent / 'logs'
_lock = threading.Lock()


def client_ip(request: Request) -> str:
    """Behind the Cloudflare tunnel the peer of the server is cloudflared on 127.0.0.1; the visitor is in these headers."""
    h = request.headers
    return h.get('cf-connecting-ip') or (h.get('x-forwarded-for') or '').split(',')[0].strip() or (request.client.host if request.client else '')


def write(event: str, request: Request, session: str, **fields):
    now = datetime.now().astimezone()
    row = {'ts': now.isoformat(timespec='seconds'), 'event': event, 'session': session, 'ip': client_ip(request), **fields}
    try:
        with _lock:
            DIR.mkdir(exist_ok=True)
            with open(DIR / f'{now:%Y-%m-%d}.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    except OSError as e:   # a full disk must not stop the chat
        print('chat log failed:', e)
