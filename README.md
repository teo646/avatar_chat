# avatar_chat – a UIKA Gaussian avatar that reacts to a chatbot

Type in the chat box -> a local LLM (Ollama) answers in Korean and names the emotion of the answer -> the avatar
(a Gaussian head built by UIKA from photos) smoothly changes its expression (0.3 s). No voice, no lip sync.

    ./run.sh          # then open http://localhost:8000/   (add ?debug for the expression sliders)

## Layout
    server/app.py            FastAPI: /api/chat, /api/pose, /api/blink, /api/meta, static web
    server/chat.py           Ollama client, system prompts (ROLES), JSON schema, parsing / fallback, 10 turn history
    server/flame_np.py       FLAME 2023 expression + joint transforms in numpy (no torch needed on the server)
    server/emotion_presets.json   happy / sad / angry / surprised / neutral + blink (made by presets/make_presets.py)
    server/bench_llm.py      model comparison used to pick the LLM
    web/                     three.js page, js/splat.js (Gaussian renderer that follows FLAME), js/main.js (UI, animation)
    web/assets/custom/       the exported UIKA avatar (gaussians.bin, meta.json, flame.npz); web/assets/user/ the RGBAvatar one (below)
    presets/                 preset design + check renders;  tools/  puppeteer scripts used for the tests

## How the avatar moves
UIKA puts one Gaussian on every valid UV texel of the FLAME head. A Gaussian i sits in a mesh triangle (vertex ids v_ik,
barycentric weights b_ik); its position is `x_i = T_i (c_i + sum_k b_ik d[v_ik])` with `T_i = sum_j w_ij A_j`.
The server sends `d` (expression offset of the 5023 vertices) and `A` (5 joint transforms); the browser does the rest
for every Gaussian. Scale and rotation of a Gaussian stay as they are, exactly like UIKA's renderer. The formula was
checked against UIKA's own animation: max difference 4e-8 (see `export_avatar_web.py` output, meta.json).

## RGBAvatar avatar with the person's own emotions (default: `AVATAR_ASSET=user`)
A Gaussian head trained by RGBAvatar on a video of the person making each emotion (segments.json: neutral / happy / sad / angry /
surprised / talk). Unlike the UIKA avatar it blends Gaussian attributes per expression, and the emotions are the person's own.

    server/app.py            asset type rgbavatar: no FLAME on the server, /api/pose and /api/chat only name the emotion + intensity
    web/js/rgbavatar.js      BlendAvatar (G0 + sum psi_k dG_k, activations after the blending, TBN binding) and BlendDriver
    web/assets/user/         rgbavatar.bin (G0, dG x 20, binding, triangles + UVs, psi and mesh of 6 emotions), meta.json, test_ref.json

Sent to the browser: no MLP and no FLAME files. The MLP is evaluated at export time for every emotion (psi, 20 values), the mesh of
every emotion (neutral, happy, sad, angry, surprised, blink) is the posed FLAME mesh. In the browser an emotion is an interpolation of
(psi, mesh) over 0.3 s, a blink adds `blink - neutral`. Free FLAME sliders do not exist in this mode.

    cd /home/ty/Codes/RGBAvatar
    python convert_pixel3dmm.py --video-name user --subject user_gaze0 --fix-gaze     # tracked video -> RGBAvatar data, gaze fixed
    python make_wide_images.py --video-name user --src-subject user_gaze0 --subject user_gaze0_wide   # p3dmm env; room above the head
    python train_offline.py --subject user_gaze0_wide --work_name run1 --config config/custom.yaml --preload
    python make_emotion_presets.py --subject user_gaze0_wide --model output/user_gaze0_wide/run1 --out data/user/emotion_presets_gaze0_wide.json
    python export_rgbavatar_web.py --model output/user_gaze0_wide/run1 --presets data/user/emotion_presets_gaze0_wide.json \
                                   --out /home/ty/Codes/avatar_chat/web/assets/user
    # the trained result (model.ply, shape.npy, config, metrics, preset renders, emotion presets) is kept in models/rgbavatar/tae_young/
    AVATAR_ASSET=custom ./run.sh          # the UIKA avatar again

Why `--fix-gaze`: pixel3dmm does not track the gaze, its eyeball rotations are noise (median 138 degrees from straight ahead). A
model trained on them has never seen eyes straight ahead, and every preset (gaze at zero) got dark holes around the eyes.
`make_emotion_presets.py` only uses frames in which the head is turned by at most 20 degrees (`--max-head-deg`): an emotion acted
with a tilted head looks torn from the front. `web/assets/user_before_gaze0` is the avatar before this change.

Why `make_wide_images.py`: pixel3dmm crops every frame with one static box that cuts the top of the hair (92 % of the frames touch the
top row), so the Gaussians on the crown were never supervised and came out as noise. The script cuts the same box 140 px higher from the
original frames (images 256 x 320, principal point moved by 64 px, same tracking). `web/assets/user_before_wide` is the avatar before.
The browser drag turns the head like FLAMEDataset does (neck joint rotation, MLP inputs updated, `export_rgbavatar_web.py` exports the
MLP and the skinning parts for it).

Self tests: the export compares its numpy version of the browser algorithm with RGBAvatar's CUDA code (max error ~1e-8 in position),
and `window.__selfTest()` in the page compares the JavaScript with the same reference (test_ref.json): ~6e-8 position, ~2e-7 rotation.
Cost: about 24 ms per rebuild of the Gaussians (only while the face moves, a blink or a 0.3 s transition); the download is 35 MB.
The presets of this avatar are taken as they come out of the video (the largest change from neutral, eyes open); see the header of
make_emotion_presets.py. sad and angry are the weak ones (large head movements while the person acted them).

## Re-exporting the avatar (other photos)
    cd /home/ty/Codes/UIKA
    /home/ty/anaconda3/envs/uika/bin/python export_avatar_web.py inputs/my_face_fixed --out /home/ty/Codes/avatar_chat/web/assets/custom
Phone photos need their EXIF rotation baked in first (inputs/my_face_fixed was made like that).

## Settings (env vars of run.sh)
    AVATAR_LLM        ollama model, default exaone3.5:7.8b      AVATAR_ROLE   key of ROLES in server/chat.py
    AVATAR_ASSET      folder in web/assets, default user (RGBAvatar); custom = UIKA avatar
    AVATAR_INTENSITY  strength of the emotion presets, 0.8      PORT          8000
URL parameters of the page: `?max=50000` (use only the most important Gaussians; this is the default on phones),
`?hold=10` (seconds until the face relaxes to neutral, 0 = never), `?debug` (sliders), `?emotion=happy`, `?noidle`.

## Emotion presets: read this
The example motions of UIKA (456 frames) contain talking / neutral faces only, nothing recognizable as the four
emotions (`presets/cand_renders/sheet.png`). The presets are therefore **synthetic**: FACS-style landmark displacements
(mouth corners, brows, lids, lips in mm) fitted to the FLAME 2023 expression space (`presets/make_presets.py`).
Replace `expr` / `jaw` of a preset in `server/emotion_presets.json` with the average of tracked NeRSemble EMO frames when
you have them; the format stays the same. FLAME 2023 has no separate eyelid parameters, eyelids are part of `expr`.

## Mixed emotions (RGBAvatar)

An RGBAvatar avatar always shows a mix: the chatbot scores every emotion from 0 to 1 (`emotions` in the `/api/chat` answer) and the face is
`neutral + sum_i w_i (emotion_i - neutral)` over x, verts, vroot and joint, then the MLP runs on the mixed x. The weights of the non-neutral
emotions are multiplied by the intensity and scaled down to a sum of 1 at most; the panel at the bottom left shows what the face uses
(the neutral bar is the model's own score). There is no single-emotion mode and no emotion buttons for RGBAvatar.
`?w=happy:0.5,surprised:0.5` shows a mix without the chatbot. (Avatars of the other kinds still use one emotion per answer.)

## One role, the avatar speaks first

`server/chat.py` has one role (`ROLES['dating_match']`, 데이트앱 매칭 상대). When the page opens, and again after "대화 초기화", the browser calls
`POST /api/greet`: the avatar says `GREETING['reply']` with `GREETING['emotions']` (a fixed text and scores, no LLM call) and the conversation
history starts with that turn, so the model knows what it has already said. `?nogreet` skips the greeting (for tests that wait for the first answer).

## Chat log

`server/chatlog.py` appends one JSON line per event to `logs/YYYY-MM-DD.jsonl` (next to `server/`, not inside `web/`, so it is never served):
`ts` (local time with offset), `event` (`greet` = the page was opened, `chat`, `error`), `session`, `ip`, `user`, `reply`, `emotion`, `emotions`,
`latency_s`. The IP is `CF-Connecting-IP` (set by the Cloudflare tunnel), else `X-Forwarded-For`, else the peer of the connection (127.0.0.1 locally).
Look at it with `tail -f logs/*.jsonl` or `jq -c '[.ts,.ip,.user,.reply]' logs/*.jsonl`.
