"""
Backend of the avatar chat demo (FastAPI).

    cd /home/ty/Codes/avatar_chat && /home/ty/anaconda3/envs/uika/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

    GET  /api/meta                        avatar and chat settings
    GET  /api/pose?emotion=happy          expression offsets + joint transforms of a preset (intensity from the config)
    POST /api/pose                        the same for arbitrary FLAME values (debug sliders)
    GET  /api/blink                       expression offsets of a full blink (scale 0..1 in the browser)
    POST /api/chat                        {message, session} -> {reply, emotion, pose}
    POST /api/greet                       {session} -> the avatar's first words, same answer as /api/chat
The static page is served from web/.
"""
import base64
import hmac
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .flame_np import FlameNp
from . import chatlog
from .chat import EMOTIONS, ROLES, ChatEngine

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / 'web'
AVATAR = os.environ.get('AVATAR_ASSET', 'custom')   # folder of web/assets with the exported avatar
ASSET = WEB / 'assets' / AVATAR

CONFIG = {
    'intensity': float(os.environ.get('AVATAR_INTENSITY', 0.8)),   # strength of the emotion presets, 0..1
    'llm_model': os.environ.get('AVATAR_LLM', 'exaone3.5:7.8b'),
    'role': os.environ.get('AVATAR_ROLE', 'dating_match'),
    'ollama_url': os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434'),
}

meta = json.loads((ASSET / 'meta.json').read_text())
# An RGBAvatar avatar (export_rgbavatar_web.py) carries its own emotions and is deformed in the browser: no FLAME on the server.
RGBA = meta.get('type') == 'rgbavatar'
if not RGBA:
    flame = FlameNp(str(ASSET / 'flame.npz'))
    presets = json.loads((Path(__file__).parent / 'emotion_presets.json').read_text())
engine = ChatEngine(CONFIG['ollama_url'], CONFIG['llm_model'], CONFIG['role'])

app = FastAPI(title='avatar chat')
app.add_middleware(GZipMiddleware, minimum_size=2048)

# Optional password for a public deployment (AVATAR_PASSWORD): HTTP Basic auth on everything, any user name.
PASSWORD = os.environ.get('AVATAR_PASSWORD', '')


@app.middleware('http')
async def password_gate(request: Request, call_next):
    if PASSWORD:
        ok = False
        auth = request.headers.get('authorization', '')
        if auth.lower().startswith('basic '):
            try:
                given = base64.b64decode(auth[6:]).decode('utf-8', 'ignore').split(':', 1)[-1]
                ok = hmac.compare_digest(given.encode(), PASSWORD.encode())
            except Exception:
                ok = False
        if not ok:
            return Response('password required', status_code=401, headers={'WWW-Authenticate': 'Basic realm="avatar chat"'})
    return await call_next(request)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype='<f4').tobytes()).decode()


def pose_payload(expr, jaw=(0, 0, 0), neck=(0, 0, 0), eyes=(0,) * 6) -> dict:
    """What the browser needs to move the Gaussians (see flame_np.py): d (V x 3, float32, base64) and A (5 x 4 x 4)."""
    expr = np.asarray(expr, dtype=np.float64)
    expr = np.pad(expr, (0, max(0, flame.n_expr - len(expr))))[:flame.n_expr]
    d = flame.expression_offsets(expr)
    A, J = flame.joint_transforms(expr, np.zeros(3), np.asarray(neck, float), np.asarray(jaw, float), np.asarray(eyes, float))
    return {'d': b64(d), 'A': np.round(A, 6).reshape(-1).tolist(), 'pivot': np.round(J[1], 6).tolist()}


def preset_pose(emotion: str, intensity: float) -> dict:
    p = presets['presets'].get(emotion) or presets['presets']['neutral']
    k = float(np.clip(intensity, 0.0, 1.5))
    return pose_payload(np.asarray(p['expr']) * k, np.asarray(p['jaw']) * k, np.asarray(p['neck']) * k, np.asarray(p['eyes']) * k)


class PoseIn(BaseModel):
    expr: list[float] = []
    jaw: list[float] = [0, 0, 0]
    neck: list[float] = [0, 0, 0]


class ChatIn(BaseModel):
    message: str
    session: str = 'default'
    role: str | None = None


def emotion_pose(emotion: str, intensity: float) -> dict:
    """RGBAvatar: the browser holds the emotions, so the 'pose' is only which one and how strong."""
    return {'emotion': emotion if emotion in meta['emotions'] else 'neutral', 'intensity': intensity}


def pose_for(emotion: str, intensity: float) -> dict:
    return emotion_pose(emotion, intensity) if RGBA else preset_pose(emotion, intensity)


@app.get('/api/meta')
def get_meta():
    out = {'emotions': EMOTIONS, 'intensity': CONFIG['intensity'], 'llm_model': CONFIG['llm_model'],
           'role': engine.default_role, 'roles': {k: v['name'] for k, v in ROLES.items()}, 'count': meta['count'],
           'fov_y_deg': meta['fov_y_deg'], 'camera_distance': meta['camera_distance'], 'avatar': AVATAR,
           'type': 'rgbavatar' if RGBA else 'flame'}
    if RGBA:
        out['neck_pivot'] = meta['neck_pivot']
    else:
        out.update(n_expr=flame.n_expr, n_vertices=int(flame.template.shape[0]))
    return out


@app.get('/api/pose')
def get_pose(emotion: str = 'neutral', intensity: float | None = None):
    return {'emotion': emotion, **pose_for(emotion, CONFIG['intensity'] if intensity is None else intensity)}


@app.post('/api/pose')
def post_pose(p: PoseIn):
    if RGBA:
        raise HTTPException(404, 'free FLAME poses are not available for an RGBAvatar avatar')
    return pose_payload(p.expr, p.jaw, p.neck)


@app.get('/api/blink')
def get_blink():
    if RGBA:
        raise HTTPException(404, 'the blink is part of the RGBAvatar emotions')
    return {'d': b64(flame.expression_offsets(np.asarray(presets['blink'])))}


def answer(out: dict, session: str, t0: float) -> dict:
    """What the browser gets for a reply or a greeting: the text, the emotion scores and the pose of the avatar."""
    out['role'] = engine.role_of(session)
    if RGBA:
        # what the face shows: the scores times the configured intensity, the non-neutral ones scaled down to a sum of 1 at most
        w = {e: out['emotions'][e] * CONFIG['intensity'] for e in EMOTIONS if e != 'neutral'}
        total = sum(w.values())
        if total > 1.0:
            w = {e: v / total for e, v in w.items()}
        out['pose'] = {'weights': {e: round(v, 3) for e, v in w.items()}}
    else:
        out['pose'] = pose_for(out['emotion'], CONFIG['intensity'])
    out['latency_s'] = round(time.time() - t0, 2)
    return out


def log_answer(event: str, request: Request, session: str, user: str, out: dict):
    chatlog.write(event, request, session, user=user, reply=out['reply'], emotion=out['emotion'], emotions=out.get('emotions'), latency_s=out['latency_s'])


@app.post('/api/chat')
async def post_chat(c: ChatIn, request: Request):
    t0 = time.time()
    c.message = c.message.strip()[:500]   # keep a public deployment cheap
    try:
        # an RGBAvatar avatar shows the weighted sum of the emotion scores of the model, never a single emotion
        out = answer(await engine.reply(c.session, c.message, c.role, mix=RGBA), c.session, t0)
    except Exception as e:
        chatlog.write('error', request, c.session, user=c.message, error=repr(e)[:300])
        raise
    log_answer('chat', request, c.session, c.message, out)
    return out


@app.post('/api/greet')
def post_greet(c: ChatIn, request: Request):
    """The avatar speaks first. Starts a new conversation (like /api/reset) whose first turn is the greeting."""
    out = answer(engine.greet(c.session, c.role, mix=RGBA), c.session, time.time())
    log_answer('greet', request, c.session, '', out)
    return out


@app.post('/api/reset')
def reset(c: ChatIn):
    engine.reset(c.session)
    return {'ok': True}


@app.get('/')
def index():
    return FileResponse(WEB / 'index.html')


app.mount('/', StaticFiles(directory=str(WEB)), name='web')
