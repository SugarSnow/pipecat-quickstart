"""Ask an OpenAI audio model to read a Japanese name back as katakana.

Why this exists: Deepgram returns Japanese names as kanji, and the reading is
gone by the time anything downstream sees it — "もとき" comes back as "本木",
which is also readable as "ほんぎ". No prompt further down the pipeline can
recover it, so the reading has to come out of the speech model itself.

This probe is deliberately outside the bot. It synthesizes a name with the same
voice and sample rate the audio-mode evals use for the caller (Cartesia at
8 kHz, matching the Twilio pipeline), sends the audio to an OpenAI model that
takes audio input, and asks for the reading in katakana. Nothing here is wired
into bot_phone.py.

    uv run --env-file .env python scripts/name_reading_probe.py
    uv run --env-file .env python scripts/name_reading_probe.py --model gpt-audio-mini
    uv run --env-file .env python scripts/name_reading_probe.py --runs 3
"""

import argparse
import asyncio
import base64
import io
import os
import time
import wave

from openai import AsyncOpenAI
from pipecat.evals.speech import EvalSpeech

# The caller's voice from the audio-mode scenarios, so this measures the same
# audio the bot would hear: Cartesia, Japanese, 8 kHz.
VOICE_CONFIG = {
    "service": "cartesia",
    "voice": "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
    "language": "ja",
    "sample_rate": 8000,
}

# Spoken as hiragana so the synthesis says the reading we mean, with no kanji
# anywhere in the loop. The expected answer is that same reading in katakana.
#
# Chosen for the ways a reading goes wrong on a phone line: 小林本木 was the
# name a real call test failed on five times over; さとう ゆうき and
# なかじま あつし have several spellings each; しんいち and りょうすけ carry the
# mora that 8 kHz blurs — ん before a vowel, a long vowel, a palatalised mora.
NAMES = [
    ("こばやし もとき", "コバヤシ モトキ"),
    ("さとう ゆうき", "サトウ ユウキ"),
    ("なかじま あつし", "ナカジマ アツシ"),
    ("いとう しんいち", "イトウ シンイチ"),
    ("おおの りょうすけ", "オオノ リョウスケ"),
]

INSTRUCTION = (
    "この音声で話されている人名の読みを、カタカナだけで答えてください。"
    "姓と名の間に半角スペースを1つ入れてください。"
    "カタカナとそのスペース以外は、説明も句読点も何も書かないでください。"
)

DEFAULT_MODEL = "gpt-audio-1.5"

# Plain transcription of the same clip, as a control: it separates "the model
# cannot read the name" from "the clip does not carry the name". Whatever this
# returns is the ceiling — no instruction can recover a name the audio lost.
CONTROL_MODEL = "gpt-transcribe"

# Transcription takes a `prompt` that biases the output. Written in hiragana, it
# is a known way to pull the orthography away from kanji — which is the whole
# problem here, since kanji is where the reading is lost.
#
# The names in it are deliberately not the ones under test: putting the answers
# in the prompt would steer the content, not just the spelling, and the result
# would mean nothing.
KANA_PROMPT = "おなまえのよみです。やまだ たろう。すずき はなこ。たなか いちろう。"


def to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


async def ask(client: AsyncOpenAI, model: str, wav: bytes) -> tuple[str, float]:
    """Send one clip and return ``(answer, seconds)``.

    A failed request is reported as a row rather than ending the run: the model
    answers some clips with a 500 ("the model produced invalid content"), and
    how often that happens is part of what is being measured.
    """
    started = time.monotonic()
    try:
        response = await client.chat.completions.create(
            model=model,
            modalities=["text"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": INSTRUCTION},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(wav).decode(),
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        return f"<{type(e).__name__}>", time.monotonic() - started
    elapsed = time.monotonic() - started
    return (response.choices[0].message.content or "").strip(), elapsed


async def transcribe(
    client: AsyncOpenAI, model: str, wav: bytes, prompt: str | None = None
) -> str:
    """Transcribe one clip, optionally biasing the spelling with *prompt*."""
    extra = {"prompt": prompt} if prompt else {}
    try:
        result = await client.audio.transcriptions.create(
            model=model, language="ja", file=("name.wav", wav, "audio/wav"), **extra
        )
    except Exception as e:
        return f"<{type(e).__name__}>"
    return (result.text or "").strip()


def normalized(text: str) -> str:
    """Fold to one kana script and drop spacing before comparing.

    What is being measured is whether the reading came back, so an answer in
    hiragana counts: 「おおの りょうすけ」 is the right reading, and turning it
    into katakana is a formatting job, not a listening one. Spacing is likewise
    not the point.
    """
    folded = "".join(
        chr(ord(c) + 0x60) if "\u3041" <= c <= "\u3096" else c for c in text
    )
    return "".join(folded.split())


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--runs", type=int, default=1, help="requests per name (default: 1)")
    # The first run put an h in front of every vowel-initial surname — オオノ came
    # back as ホウノ, イトウ as ヒトウ — which is what a clipped onset sounds
    # like. Padding the front separates "the model mis-hears" from "our clip
    # starts too abruptly".
    parser.add_argument(
        "--lead-silence-ms", type=int, default=0, help="silence to prepend (default: 0)"
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help=f"also transcribe each clip with {CONTROL_MODEL}, with no katakana instruction",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Run with: uv run --env-file .env python ...")
        return 1

    client = AsyncOpenAI()
    rows = []

    async with EvalSpeech.from_config(VOICE_CONFIG) as speech:
        for spoken, expected in NAMES:
            pcm, sample_rate = await speech.generate(spoken)
            if args.lead_silence_ms:
                pcm = b"\x00\x00" * (sample_rate * args.lead_silence_ms // 1000) + pcm
            wav = to_wav(pcm, sample_rate)
            seconds = len(pcm) / 2 / sample_rate
            control = kana = ""
            if args.control:
                control = await transcribe(client, CONTROL_MODEL, wav)
                kana = await transcribe(client, CONTROL_MODEL, wav, KANA_PROMPT)
            for _ in range(args.runs):
                answer, elapsed = await ask(client, args.model, wav)
                rows.append((spoken, expected, answer, elapsed, seconds, control, kana))

    width = max(len(r[1]) for r in rows) + 2
    lead = f"   lead silence: {args.lead_silence_ms}ms" if args.lead_silence_ms else ""
    print(f"\nmodel: {args.model}   voice: cartesia {VOICE_CONFIG['voice']} @ 8 kHz{lead}\n")
    head = f"{'正解'.ljust(width)}{'返答'.ljust(width)}{'':3}{'応答':>7}  {'音声長':>7}"
    print(head + ("  文字起こし｜かなの prompt 付き" if args.control else ""))
    print("-" * (width * 2 + 22))
    hits = 0
    for spoken, expected, answer, elapsed, seconds, control, kana in rows:
        ok = normalized(answer) == normalized(expected)
        hits += ok
        print(
            f"{expected.ljust(width)}{answer.ljust(width)}"
            f"{'○' if ok else '×':3}{elapsed:6.2f}s  {seconds:6.2f}s"
            + (f"  {control} ｜ {kana}" if control else "")
        )
    print("-" * (width * 2 + 22))
    average = sum(r[3] for r in rows) / len(rows)
    print(f"一致 {hits}/{len(rows)}   応答の平均 {average:.2f}s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
