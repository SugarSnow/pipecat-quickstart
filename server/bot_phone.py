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
            system_instruction="あなたは電話応対のアシスタントです。応答は音声で読み上げられるため、記号や箇条書きを使わず、短く自然な話し言葉で答えてください。",
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
            {"role": "developer", "content": "Start by concisely introducing yourself."}
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
