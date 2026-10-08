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


async def ask_realtime(
    model: str, pcm24: bytes, *, want_audio: bool
) -> tuple[str, bytes, float]:
    """Send one clip to a fresh session and return ``(text, audio, seconds)``.

    A session per clip, so nothing the model said about the previous name is in
    context when it hears the next one.
    """
    started = time.monotonic()
    text_parts: list[str] = []
    audio = bytearray()

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

    return "".join(text_parts).strip(), bytes(audio), time.monotonic() - started


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--runs", type=int, default=1, help="requests per name (default: 1)")
    parser.add_argument(
        "--text",
        action="store_true",
        help="ask for the reading as katakana text and score it, instead of a spoken read-back",
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
                text, audio, elapsed = await ask_realtime(
                    args.model, pcm24, want_audio=not args.text
                )
                path = ""
                if audio:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    name = f"{expected.replace(' ', '')}_{run + 1}.wav"
                    (out_dir / name).write_bytes(to_wav(audio, API_SAMPLE_RATE))
                    path = str(out_dir / name)
                rows.append((expected, text, elapsed, path))

    width = max(len(r[0]) for r in rows) + 2
    mode = "カタカナの返答（採点あり）" if args.text else "復唱（音声を保存）"
    print(f"\nmodel: {args.model}   {mode}   入力: 8kHz → {API_SAMPLE_RATE}Hz\n")
    hits = 0
    for expected, text, elapsed, path in rows:
        ok = normalized(text) == normalized(expected) if args.text else None
        hits += bool(ok)
        mark = ("○" if ok else "×") if args.text else " "
        print(f"{expected.ljust(width)}{text[:46].ljust(48)}{mark:3}{elapsed:6.2f}s  {path}")
    print()
    if args.text:
        print(f"一致 {hits}/{len(rows)}", end="   ")
    print(f"応答の平均 {sum(r[2] for r in rows) / len(rows):.2f}s\n")
    if not args.text:
        print("音声を聞いて、読みが保たれているかをご確認ください。\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
