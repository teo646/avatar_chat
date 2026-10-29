"""Chatbot: a local LLM served by Ollama answers in Korean and names the emotion of its answer."""
from __future__ import annotations

import json
from collections import defaultdict

import httpx

EMOTIONS = ['happy', 'sad', 'angry', 'surprised', 'neutral']

_EXAMPLES_POLITE = (
    '{"reply": "어머, 사진 칭찬 감사해요! 취미가 뭐예요?", "emotion": "happy"}\n'
    '{"reply": "헉, 진짜요? 전혀 몰랐어요!", "emotion": "surprised"}\n'
    '{"reply": "그런 말은 좀 상처받아요. 사과해 주세요.", "emotion": "angry"}\n'
    '{"reply": "아, 많이 힘드셨겠어요. 괜찮으세요?", "emotion": "sad"}\n'
    '{"reply": "저는 서울에 살아요. 주말엔 보통 뭐 하세요?", "emotion": "neutral"}'
)
_EXAMPLES_CASUAL = (
    '{"reply": "오 진짜? 대박이다, 나 완전 신나!", "emotion": "happy"}\n'
    '{"reply": "헐, 그걸 지금 말해? 몰랐잖아!", "emotion": "surprised"}\n'
    '{"reply": "아 진짜 그건 좀 아니지 않냐.", "emotion": "angry"}\n'
    '{"reply": "하... 요즘 좀 힘 빠진다.", "emotion": "sad"}\n'
    '{"reply": "오늘은 그냥 집에서 쉬고 있어.", "emotion": "neutral"}'
)

_FORMAT_RULES = (
    '\n\n반드시 아래 JSON 한 개만 출력하세요. 다른 글자, 설명, 마크다운은 절대 쓰지 마세요.\n'
    '{"reply": "<한국어 1~2문장의 답변>", "emotion": "<happy|sad|angry|surprised|neutral 중 하나>"}\n'
    'reply는 반드시 2문장 이하, 전체 100자 이내로 짧게 쓰세요.\n'
    '예시:\n{examples}\n'
    'emotion은 이 답변을 말할 때 상대방의 얼굴에 어울리는 감정입니다: '
    'happy(기쁨, 즐거움, 설렘, 칭찬을 받았을 때), sad(서운함, 아쉬움, 걱정), angry(불쾌함, 짜증, 황당함), '
    'surprised(놀람, 의외의 말을 들었을 때), neutral(평범한 대화).'
)

# Mixed emotions (?mix=1 in the browser): the model scores every emotion from 0 to 1 instead of naming one
_FORMAT_RULES_MIX = (
    '\n\n반드시 아래 JSON 한 개만 출력하세요. 다른 글자, 설명, 마크다운은 절대 쓰지 마세요.\n'
    '{"reply": "<한국어 1~2문장의 답변>", "emotions": {"happy": 0.0, "sad": 0.0, "angry": 0.0, "surprised": 0.0, "neutral": 0.0}}\n'
    'reply는 반드시 2문장 이하, 전체 100자 이내로 짧게 쓰세요.\n'
    'emotions는 이 답변을 말할 때 상대방의 얼굴에 드러나는 감정 하나하나의 세기이고 0.0(전혀 없음)부터 1.0(매우 강함) 사이의 숫자입니다. '
    '감정은 섞일 수 있어서 여러 개가 동시에 0보다 클 수 있습니다. '
    'happy(기쁨, 즐거움, 설렘, 칭찬을 받았을 때), sad(서운함, 아쉬움, 걱정), angry(불쾌함, 짜증, 황당함), '
    'surprised(놀람, 의외의 말을 들었을 때), neutral(평범한 대화).\n'
    '예시:\n'
    '{"reply": "어머, 사진 칭찬 감사해요! 취미가 뭐예요?", "emotions": {"happy": 0.8, "sad": 0.0, "angry": 0.0, "surprised": 0.2, "neutral": 0.0}}\n'
    '{"reply": "헉, 진짜요? 그런데 조금 서운하네요...", "emotions": {"happy": 0.0, "sad": 0.4, "angry": 0.0, "surprised": 0.6, "neutral": 0.0}}\n'
    '{"reply": "저는 서울에 살아요. 주말엔 보통 뭐 하세요?", "emotions": {"happy": 0.1, "sad": 0.0, "angry": 0.0, "surprised": 0.0, "neutral": 0.8}}'
)

# The avatar talks as this role (more can be added here and picked with AVATAR_ROLE).
ROLES = {
    'dating_match': {
        'name': '데이트앱 매칭 상대',
        'system': (
            '당신은 데이트앱에서 방금 매칭된 상대입니다. 사용자와 처음 대화를 시작하는 상황이고, '
            '친근하고 자연스러운 말투(존댓말, 가끔 가벼운 표현)로 한국어 1~2문장만 말합니다. '
            '감정이 얼굴에 잘 드러나는 사람이라서 기쁘거나 놀라거나 서운한 반응을 솔직하게 보입니다. '
            '상대에게 가끔 가벼운 질문을 되묻습니다. 너무 길게 말하지 않습니다.'
        ),
    },
}

# The avatar speaks first (POST /api/greet): this text with these emotion scores
GREETING = {
    'reply': '안녕하세요! 처음 뵙겠습니다. 외모가 정말 아름다우세요.',
    'emotions': {'happy': 0.7, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0, 'neutral': 0.3},
}


def build_system(role_key: str) -> str:
    return ROLES[role_key]['system']


class ChatEngine:
    def __init__(self, ollama_url: str, model: str, role: str, max_turns: int = 10):
        self.url = ollama_url.rstrip('/')
        self.model = model
        self.default_role = role if role in ROLES else 'dating_match'
        self.max_turns = max_turns
        self.history: dict[str, list[dict]] = defaultdict(list)
        self.session_role: dict[str, str] = {}

    def role_of(self, session: str) -> str:
        return self.session_role.get(session, self.default_role)

    def set_role(self, session: str, role: str | None):
        """A session keeps one role; changing it starts a new conversation."""
        if role and role in ROLES and role != self.role_of(session):
            self.session_role[session] = role
            self.history[session] = []

    def reset(self, session: str):
        self.history[session] = []

    def _messages(self, session: str, user_text: str, mix: bool = False):
        hist = self.history[session][-2 * (self.max_turns - 1):]  # the new turn makes it max_turns
        system = build_system(self.role_of(session))
        examples = _EXAMPLES_CASUAL if '반말' in system else _EXAMPLES_POLITE
        note = '\n(예시의 말투는 형식을 보여 주기 위한 것입니다. 말투와 성격은 위의 역할 설정을 따르세요.)'
        rules = _FORMAT_RULES_MIX if mix else _FORMAT_RULES.replace('{examples}', examples)
        return [{'role': 'system', 'content': system + rules + note}, *hist, {'role': 'user', 'content': user_text}]

    @staticmethod
    def parse(raw: str) -> dict:
        """JSON -> {"reply", "emotion"}. Anything unparsable becomes the raw text with the neutral emotion."""
        try:
            obj = json.loads(raw)
            reply = str(obj['reply']).strip()
            emotion = str(obj.get('emotion', 'neutral')).strip().lower()
            if not reply:
                raise ValueError('empty reply')
            return {'reply': reply, 'emotion': emotion if emotion in EMOTIONS else 'neutral'}
        except Exception:
            return {'reply': raw.strip(), 'emotion': 'neutral'}

    @staticmethod
    def parse_mix(raw: str) -> dict:
        """JSON -> {"reply", "emotions": {emotion: 0..1}, "emotion": the strongest one}. Unparsable -> the raw text, neutral 1."""
        try:
            obj = json.loads(raw)
            reply = str(obj['reply']).strip()
            if not reply:
                raise ValueError('empty reply')
            em = {e: min(1.0, max(0.0, float((obj.get('emotions') or {}).get(e, 0.0)))) for e in EMOTIONS}
        except Exception:
            return {'reply': raw.strip(), 'emotions': {e: float(e == 'neutral') for e in EMOTIONS}, 'emotion': 'neutral'}
        return {'reply': reply, 'emotions': em, 'emotion': max(EMOTIONS, key=lambda e: em[e])}

    async def raw_reply(self, messages: list[dict], model: str | None = None, temperature: float = 0.7, mix: bool = False) -> str:
        schema = {'type': 'object',
                  'properties': {'reply': {'type': 'string'},
                                 'emotions': {'type': 'object', 'properties': {e: {'type': 'number'} for e in EMOTIONS},
                                              'required': EMOTIONS}},
                  'required': ['reply', 'emotions']} if mix else {'type': 'object',
                  'properties': {'reply': {'type': 'string'}, 'emotion': {'type': 'string', 'enum': EMOTIONS}},
                  'required': ['reply', 'emotion']}
        body = {'model': model or self.model, 'messages': messages, 'stream': False, 'format': schema,
                'options': {'temperature': temperature, 'num_predict': 200}, 'keep_alive': '30m', 'think': False}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f'{self.url}/api/chat', json=body)
            if r.status_code == 400 and 'think' in r.text:  # model without a thinking mode
                body.pop('think')
                r = await client.post(f'{self.url}/api/chat', json=body)
            r.raise_for_status()
            return r.json()['message']['content']

    def greet(self, session: str, role: str | None = None, mix: bool = True) -> dict:
        """The avatar's first words; a new conversation that starts with them, so the model knows what it has said."""
        self.set_role(session, role)
        out = {'reply': GREETING['reply'], 'emotion': 'happy'}
        if mix:
            out['emotions'] = dict(GREETING['emotions'])
        self.history[session] = [{'role': 'assistant', 'content': json.dumps(out, ensure_ascii=False)}]
        return out

    async def reply(self, session: str, user_text: str, role: str | None = None, mix: bool = False) -> dict:
        self.set_role(session, role)
        messages = self._messages(session, user_text, mix)
        raw = await self.raw_reply(messages, mix=mix)
        out = self.parse_mix(raw) if mix else self.parse(raw)
        h = self.history[session]
        h += [{'role': 'user', 'content': user_text},
              {'role': 'assistant', 'content': json.dumps(out, ensure_ascii=False)}]
        del h[:-2 * self.max_turns]
        return out
