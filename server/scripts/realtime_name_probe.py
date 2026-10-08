"""Does speech-to-speech keep a Japanese name's pronunciation?

The cascade loses it: Deepgram writes "もとき" as "本木", and nothing downstream
can tell whether that is もとき or ほんぎ. A speech-to-speech model has no such
step — the audio goes straight in, and the voice that reads the name back is the
same model that heard it. If that holds, the problem goes away structurally.

So this measures the one claim that decides it: send a spoken name to
``gpt-realtime`` and have it say the name back. Two things come out — the audio
it produced (saved as a WAV, which is the only honest check on pronunciation)
and the katakana it writes when asked for the reading in text, which is directly
comparable to the gpt-audio-1.5 numbers from name_reading_probe.py.

Nothing here is wired into bot_phone.py.

    uv run --env-file .env python scripts/realtime_name_probe.py          # spoken read-back
    uv run --env-file .env python scripts/realtime_name_probe.py --text   # katakana, scored
    uv run --env-file .env python scripts/realtime_name_probe.py --runs 3 --text
"""

import argparse
import asyncio
import base64
import io
import json
import os
import time
import wave
from pathlib import Path

from pipecat.audio.utils import create_file_resampler
from pipecat.evals.speech import EvalSpeech
from websockets.asyncio.client import connect as websocket_connect

from name_reading_probe import NAMES, VOICE_CONFIG, normalized, to_wav

DEFAULT_MODEL = "gpt-realtime-2.1"
BASE_URL = "wss://api.openai.com/v1/realtime"

# The Realtime API takes and returns 24 kHz PCM, so the 8 kHz clip is upsampled
# on the way in — which is what pipecat's own realtime services do with a
# telephony pipeline. The band-limiting of the phone line is still in the audio;
# only the sample rate changes.
API_SAMPLE_RATE = 24000

VOICE = "marin"

SPEAK_INSTRUCTIONS = (
    "あなたは電話の受付です。相手が名乗ったお名前を「○○様ですね。」と一度だけ復唱してください。"
    "相手が言ったとおりの読みで発音してください。読み方を変えてはいけません。"
    "それ以外のことは何も言わないでください。"
)

TEXT_INSTRUCTIONS = (
    "この音声で話されている人名の読みを、カタカナだけで答えてください。"
    "姓と名の間に半角スペースを1つ入れてください。"
    "カタカナとそのスペース以外は、説明も句読点も何も書かないでください。"
)

# Asked cold, straight off the caller's audio, the katakana came back 2/5. But
# the model's own read-back is pronounced correctly 5/5 — so this asks it after
# it has spoken, with its own unambiguous audio in the session. The question is
# whether writing down what it just said is easier than writing down what it
# just heard.
WRITE_BACK_PROMPT = (
    "いま復唱したお名前の読みを、カタカナだけで書いてください。"
    "姓と名の間に半角スペースを1つ入れてください。"
    "カタカナとそのスペース以外は、説明も句読点も何も書かないでください。"
)


async def collect(ws) -> tuple[str, bytes]:
    """Read one response off the socket: its text and its audio."""
    text_parts: list[str] = []
    audio = bytearray()
    async for message in ws:
        event = json.loads(message)
        kind = event.get("type", "")
        if kind in ("response.output_text.delta", "response.output_audio_transcript.delta"):
            text_parts.append(event.get("delta", ""))
        elif kind == "response.output_audio.delta":
            audio.extend(base64.b64decode(event["delta"]))
        elif kind == "error":
            text_parts.append(f"<error: {event.get('error', {}).get('message', '')}>")
            break
        elif kind == "response.done":
            break
    return "".join(text_parts).strip(), bytes(audio)


async def ask_realtime(
    model: str, pcm24: bytes, *, want_audio: bool, write_back: bool = False
) -> tuple[str, bytes, float, str]:
    """Send one clip to a fresh session and return ``(text, audio, seconds, written)``.

    A session per clip, so nothing the model said about the previous name is in
    context when it hears the next one. With *write_back*, a second turn asks it
    to write down the reading it just spoke.
    """
    started = time.monotonic()

    async with websocket_connect(
        uri=f"{BASE_URL}?model={model}",
        additional_headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    ) as ws:
        audio_cfg = {
            "input": {
                "format": {"type": "audio/pcm", "rate": API_SAMPLE_RATE},
                # One clip in, one response out: turns are driven from here, so
                # the server's own VAD must not also be proposing them.
                "turn_detection": None,
            }
        }
        if want_audio:
            audio_cfg["output"] = {
                "format": {"type": "audio/pcm", "rate": API_SAMPLE_RATE},
                "voice": VOICE,
            }
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "output_modalities": ["audio"] if want_audio else ["text"],
                        "instructions": SPEAK_INSTRUCTIONS if want_audio else TEXT_INSTRUCTIONS,
                        "audio": audio_cfg,
                    },
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm24).decode(),
                }
            )
        )
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))
        text, audio = await collect(ws)

        written = ""
        if write_back:
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": WRITE_BACK_PROMPT}],
                        },
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {"output_modalities": ["text"]},
                    }
                )
            )
            written, _ = await collect(ws)

    return text, audio, time.monotonic() - started, written


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--runs", type=int, default=1, help="requests per name (default: 1)")
    parser.add_argument(
        "--text",
        action="store_true",
        help="ask for the reading as katakana text and score it, instead of a spoken read-back",
    )
    parser.add_argument(
        "--write-back",
        action="store_true",
        help="after the spoken read-back, ask it to write that reading as katakana",
    )
    parser.add_argument("--out-dir", default="eval-runs/realtime-names", help="where WAVs go")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Run with: uv run --env-file .env python ...")
        return 1

    out_dir = Path(args.out_dir)
    resampler = create_file_resampler()
    rows = []

    async with EvalSpeech.from_config(VOICE_CONFIG) as speech:
        for spoken, expected in NAMES:
            pcm8, rate = await speech.generate(spoken)
            pcm24 = await resampler.resample(pcm8, rate, API_SAMPLE_RATE)
            for run in range(args.runs):
                text, audio, elapsed, written = await ask_realtime(
                    args.model,
                    pcm24,
                    want_audio=not args.text,
                    write_back=args.write_back and not args.text,
                )
                path = ""
                if audio:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    name = f"{expected.replace(' ', '')}_{run + 1}.wav"
                    (out_dir / name).write_bytes(to_wav(audio, API_SAMPLE_RATE))
                    path = str(out_dir / name)
                rows.append((expected, text, elapsed, path, written))

    width = max(len(r[0]) for r in rows) + 2
    if args.text:
        mode = "聞いてすぐカタカナ（採点あり）"
    elif args.write_back:
        mode = "復唱 → 自分の発話をカタカナで書く（採点あり）"
    else:
        mode = "復唱（音声を保存）"
    print(f"\nmodel: {args.model}   {mode}   入力: 8kHz → {API_SAMPLE_RATE}Hz\n")
    scored = args.text or args.write_back
    hits = 0
    for expected, text, elapsed, path, written in rows:
        answer = written if args.write_back and not args.text else text
        ok = normalized(answer) == normalized(expected) if scored else None
        hits += bool(ok)
        mark = ("○" if ok else "×") if scored else " "
        shown = f"{text[:22]} ｜ {written[:20]}" if args.write_back else text[:46]
        print(f"{expected.ljust(width)}{shown.ljust(48)}{mark:3}{elapsed:6.2f}s  {path}")
    print()
    if scored:
        print(f"一致 {hits}/{len(rows)}", end="   ")
    print(f"応答の平均 {sum(r[2] for r in rows) / len(rows):.2f}s\n")
    if not args.text:
        print("音声を聞いて、読みが保たれているかをご確認ください。\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
