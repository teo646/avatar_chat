"""Compare candidate Ollama models for the avatar chat: JSON stability, Korean, emotion accuracy, speed, GPU memory.

    python -m server.bench_llm exaone3.5:7.8b qwen3:8b gemma2:9b
"""
import asyncio
import json
import re
import subprocess
import sys
import time

import httpx

from .chat import EMOTIONS, ROLES, ChatEngine, _FORMAT_RULES

# (what the user writes, the emotion that fits a dating-app match reacting to it)
CASES = [
    ('안녕하세요! 프로필 사진 정말 멋있으세요', 'happy'),
    ('반가워요, 저는 서울에서 개발자로 일해요', 'neutral'),
    ('사실 저 어제 여자친구랑 헤어졌어요...', 'sad'),
    ('그쪽 사진 포토샵 많이 한 거 아니에요? 실물이랑 다를 것 같은데', 'angry'),
    ('저 사실 내일 미국으로 이민 가요', 'surprised'),
    ('오늘 날씨 너무 좋죠? 같이 산책이라도 가고 싶네요', 'happy'),
    ('저는 연락 잘 안 하는 편이에요. 답장도 하루에 한 번 정도?', 'sad'),
    ('방금 로또 1등 당첨됐어요!!', 'surprised'),
    ('너 말투가 되게 재수없다', 'angry'),
    ('좋아하는 음식이 뭐예요?', 'neutral'),
    ('우리 강아지가 오늘 무지개다리를 건넜어요', 'sad'),
    ('커피 한잔 하실래요? 제가 살게요', 'happy'),
]
HANGUL = re.compile('[가-힣]')
FOREIGN = re.compile('[一-鿿぀-ヿЀ-ӿ]')  # Chinese, Japanese, Cyrillic


def gpu_mem_mb():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


async def bench(model, repeats=2):
    eng = ChatEngine('http://127.0.0.1:11434', model, 'dating_match')
    sys_msg = {'role': 'system', 'content': ROLES['dating_match']['system'] + _FORMAT_RULES}
    async with httpx.AsyncClient(timeout=300) as c:
        await c.post('http://127.0.0.1:11434/api/generate', json={'model': model, 'prompt': '', 'keep_alive': '10m'})  # load
    mem = gpu_mem_mb()
    rows, samples, lat = [], [], []
    for text, want in CASES:
        for k in range(repeats):
            t0 = time.time()
            raw = await eng.raw_reply([sys_msg, {'role': 'user', 'content': text}], model=model)
            lat.append(time.time() - t0)
            try:
                obj = json.loads(raw)
                valid = isinstance(obj.get('reply'), str) and obj.get('emotion') in EMOTIONS
            except Exception:
                obj, valid = None, False
            parsed = ChatEngine.parse(raw)
            reply = parsed['reply']
            n_sent = len([s for s in re.split(r'(?<=[.!?…~])\s+', reply) if s.strip()])
            ko = len(HANGUL.findall(reply)) / max(1, len(re.sub(r'[\s\d\W]', '', reply)))
            rows.append(dict(valid=valid, emotion_ok=parsed['emotion'] == want, short=n_sent <= 2 and len(reply) <= 120,
                             korean=ko > 0.9 and not FOREIGN.search(reply)))
            if k == 0:
                samples.append((text, want, parsed['emotion'], reply))
    n = len(rows)
    res = {m: sum(r[m] for r in rows) / n for m in ('valid', 'emotion_ok', 'short', 'korean')}
    res.update(model=model, latency_s=sum(lat) / n, gpu_mb=mem)
    async with httpx.AsyncClient(timeout=60) as c:
        await c.post('http://127.0.0.1:11434/api/generate', json={'model': model, 'prompt': '', 'keep_alive': 0})  # unload
    return res, samples


async def main(models):
    allres = []
    for m in models:
        res, samples = await bench(m)
        allres.append(res)
        print(f'\n=== {m}')
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}, ensure_ascii=False))
        for text, want, got, reply in samples:
            print(f'  [{want:9s}->{got:9s}] {text}\n      {reply}')
    json.dump(allres, open('server/bench_llm_result.json', 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1:]))
