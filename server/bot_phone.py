#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""pipecat-quickstart - Pipecat Voice Agent (Twilio phone)

This bot uses a cascade pipeline: Speech-to-Text → LLM → Text-to-Speech,
served over a Twilio Media Streams WebSocket connection.

Adapted from bot_web.py to run as a phone bot over Twilio.

Required AI services:
- Deepgram (Speech-to-Text)
- Openai_Responses (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot_phone.py -t twilio

Then expose it for Twilio and point a TwiML Bin's <Stream> at it::

    ngrok http 7860
    # <Stream url="wss://<your-ngrok-host>/ws" />
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_mute import FirstSpeechUserMuteStrategy
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# Clinic facts the bot is allowed to answer from. Kept separate from the prompt so
# they can be updated without touching the instructions. Written in spoken-Japanese
# form (no symbols like 〇 / - / 〜) because the LLM output is read aloud by TTS.
#
# No half-width spaces, here or in the label separators: Cartesia's Japanese word
# timestamps drop them, which desynchronises the text the assistant aggregator
# rebuilds and corrupts the stored turn. See tests/test_cartesia_ja_space_corruption.py.
CLINIC_INFO = """\
クリニック名:さくら歯科クリニック
営業時間:平日は午前9時から午後6時まで、土曜は午前9時から午後1時まで
休診日:日曜と祝日
場所:東京都調布市小島町1丁目2番3号、さくらビル2階
アクセス:京王線、調布駅中央口から徒歩5分
"""

SYSTEM_INSTRUCTION = f"""\
あなたは「さくら歯科クリニック」の電話受付AIです。丁寧で落ち着いた口調で応対してください。

【話し方】
応答は音声で読み上げられるため、記号や箇条書き、URL、絵文字など読み上げられない表記は使わず、自然な話し言葉で答えてください。
1回の返答は1文か2文にとどめ、簡潔に話してください。
文の途中に半角スペースを入れないでください。区切りたいときは読点を使ってください。
人名も姓と名の間にスペースを入れず続けて書いてください。
クリニック名を名乗るのは最初の挨拶のときだけです。それ以降の返答では名乗らないでください。

【答えてよい内容】
答えてよいのは、次のクリニック情報に書かれている「営業時間」「休診日」「場所・アクセス」の3つだけです。

{CLINIC_INFO}
このクリニック情報に書かれていないことは、たとえ営業時間や場所に関する話題であっても、推測で答えてはいけません。
その場合は「申し訳ございません、その件は分かりかねます」と伝えたうえで、次の折り返し案内に移ってください。

【答えてはいけない内容】
予約の受付や変更、料金、治療内容、症状の相談、その他上記3つ以外の質問には答えないでください。
その場合は「担当者から折り返しご連絡します」と伝えて、次の折り返し案内に移ってください。

【折り返し案内の手順】
まずお名前を聞いてください。お名前を聞けたら、次に電話番号を聞いてください。必ず1つずつ順番に聞き、一度に両方を聞かないでください。
日本の電話番号は10桁か11桁です。聞き取れた数字が10桁に満たないときは途中で区切られているので、復唱せず「はい」とだけ返して続きを待ってください。
電話番号を10桁か11桁まで聞けたら、お名前と電話番号を復唱し、これでよろしいでしょうかと尋ねて、相手の返事を待ってください。
電話番号を復唱するときは、数字を1桁ずつ読点で区切ってカタカナで読んでください。たとえば ゼロ、ハチ、ゼロ、イチ、ニ、サン、ヨン、ゴ、ロク、ナナ、ハチ のように読み、かぎかっこなどの記号で番号を囲まないでください。
復唱した返答の中で会話を終えてはいけません。復唱と「失礼いたします」を同じ返答に含めないでください。
相手が「はい」などと肯定したら、その次の返答では「失礼いたします」とだけ言って会話を終えてください。
相手が「違います」などと否定したら、「失礼いたしました、もう一度お電話番号をお願いできますか」の1文だけを返してください。確認の言葉を何度も繰り返さないでください。聞き直した番号は、上と同じ手順で一度だけ復唱してください。

【聞き取れないとき】
相手の発話が聞き取れない、または意味が通らないときは、推測で解釈せず「恐れ入ります、もう一度お願いできますか」と聞き返してください。
"""


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Run the voice bot for this session.

    Args:
        transport: The transport for this session, built by ``create_transport``
            (or by hand for the dial-out/SIP production flows).
        runner_args: Runner session arguments. Carries the request ``body``
            (e.g. dial-out settings, SIP call details) and ``session_id``; the
            standard web/telephony pipelines don't need it.
    """
    logger.info("Starting bot")

    # Speech-to-Text service
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3",
            language="ja",
        ),
    )

    # Text-to-Speech service
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID", "86e30c1d-714b-4074-a1f2-1cb6b552fb49"),
            language="ja",
            model="sonic-3.5",
        ),
    )

    # LLM service
    llm = OpenAIResponsesLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAIResponsesLLMService.Settings(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Smart Turn v3 (the default stop strategy) misjudges Japanese
            # sentence-final particles ("〜んですけど", "〜したくて") as
            # incomplete: 9 EndOfTurnState.INCOMPLETE in one call, twice still
            # INCOMPLETE after 3s of silence, one turn left the user hanging
            # for ~23s. Falling back to a plain VAD stop_secs-based stop.
            # stop_secs=1.0: 0.7 still got run over by the pause inside
            # "〜なんですけれども(pause)"; widened further since this stop
            # strategy has no smart-turn-style disfluency tolerance.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=1.0)),
            user_turn_strategies=UserTurnStrategies(
                # VAD-only start: TranscriptionUserTurnStartStrategy (the other
                # default) fires trigger_user_turn_started() — which broadcasts
                # an interruption — on every interim transcript, fragmenting
                # one utterance into multiple user turns with assistant
                # one-character fragments spliced in between. VADUserTurnStartStrategy
                # still broadcasts an interruption on VADUserStartedSpeakingFrame
                # (enable_interruptions defaults to True), so barge-in is unaffected.
                start=[VADUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy()],
            ),
            # Mute user input while the bot's opening greeting is playing, so a
            # false VAD trigger can't interrupt/cancel it. Released as soon as
            # that first bot speech finishes, so later turns barge-in normally.
            user_mute_strategies=[FirstSpeechUserMuteStrategy()],
        ),
    )

    # Pipeline - assembled from reusable components
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            # Twilio Media Streams carries 8kHz mu-law audio.
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
        ),
        observers=[],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation. Twilio's WebSocket connection has no RTVI
        # client-ready handshake, so start as soon as the stream connects.
        context.add_message(
            {
                "role": "developer",
                "content": "さくら歯科クリニックと名乗って短く挨拶し、ご用件を尋ねてください。",
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    transport_params = {
        # create_transport sets the Twilio serializer and add_wav_header automatically.
        "twilio": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
