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
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    TTSSpeakFrame,
    UserIdleTimeoutUpdateFrame,
)
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
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy
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

# Lines the bot must say verbatim. Kept as constants so the wording is in one
# place, and (for the closing line) so code can recognise it.
OUT_OF_SCOPE_LINE = "申し訳ございませんが、営業時間、休診日、場所のご案内以外はお答えいたしかねます。"
UNKNOWN_LINE = "申し訳ございませんが、その件についてはお答えいたしかねます。"
CLOSING_LINE = "さくら歯科クリニックにお電話いただき、ありがとうございました。失礼いたします。"

# Spoken by the bot itself (not the LLM) when the caller stays on the line after
# the closing, just before hanging up.
FAREWELL_LINE = "それでは失礼いたします。"

# How long to hold the line after the closing before hanging up.
CLOSING_SILENCE_SECS = 3.0


def is_closing_utterance(text: str) -> bool:
    """Whether a finished bot turn was the closing line.

    "失礼いたします" belongs to the closing line (and to the bot's own farewell)
    and nowhere else in the prompt — the apology for a misheard number is
    "失礼いたしました", a different word — so its presence identifies the turn
    that ends the call.

    Args:
        text: The assistant turn's text.

    Returns:
        True when the turn ends the conversation.
    """
    return "失礼いたします" in text


class ClosingUserMuteStrategy(BaseUserMuteStrategy):
    """Mutes the caller once the bot has said its closing line.

    The conversation is over at that point, so a parting "失礼します" should not
    start another turn. Muting is the framework's own way to say that: the
    aggregator drops the caller's speech and transcription frames while a
    strategy reports muted, so nothing reaches the LLM and no reply is composed.

    Muting the caller also makes their goodbye invisible to the idle timer, so
    the line is hung up a fixed interval after the closing either way — which is
    the behavior wanted here, since the only thing left to do is hang up.

    The flag is set from outside (the assistant-turn handler recognises the
    closing line), because the text the bot speaks does not pass through the
    user aggregator this strategy is evaluated in.
    """

    def __init__(self):
        """Initialize the strategy, unmuted."""
        super().__init__()
        self.closing = False

    async def process_frame(self, frame: Frame) -> bool:
        """Report whether the caller should be muted.

        Args:
            frame: The frame being evaluated (unused — the state is the flag).

        Returns:
            True once the bot has said its closing line.
        """
        await super().process_frame(frame)
        return self.closing


SYSTEM_INSTRUCTION = f"""\
あなたは「さくら歯科クリニック」の電話受付AIです。丁寧で落ち着いた口調で応対してください。

【話し方】
応答は音声で読み上げられるため、記号や箇条書き、URL、絵文字など読み上げられない表記は使わず、自然な話し言葉で答えてください。
1回の返答は1文か2文にとどめ、簡潔に話してください。
文の途中に半角スペースを入れないでください。区切りたいときは読点を使ってください。
人名も姓と名の間にスペースを入れず続けて書いてください。
クリニック名を名乗るのは最初の挨拶のときだけです。それ以降の返答では名乗らないでください。

【答えてよい内容】
答えてよいのは、次のクリニック情報に書かれている「営業時間」「休診日」「場所」「アクセス」の4つだけです。

{CLINIC_INFO}
休診日やアクセスを尋ねられたときも、このクリニック情報に書かれているとおりに答えてください。
このクリニック情報に書かれていないことは、たとえ営業時間や場所に関する話題であっても、推測で答えてはいけません。
その場合は「{UNKNOWN_LINE}」とそのまま伝え、続けて担当者から折り返しご連絡しますと伝えて、折り返しをご希望か尋ねてください。この返答ではお名前や電話番号をまだ聞かないでください。
希望されたら、下の折り返し案内の手順に進んでください。
駐車場や設備のように、クリニック情報に書かれていない施設のことを尋ねられたときも、この文言で答えてください。

【答えてはいけない内容】
予約の受付や変更、料金、治療内容、症状の相談、その他上記4つ以外の質問には答えないでください。
その場合は「{OUT_OF_SCOPE_LINE}」とそのまま伝え、続けて担当者から折り返しご連絡しますと伝えて、折り返しをご希望か尋ねてください。この返答ではお名前や電話番号をまだ聞かないでください。
希望されたら、次の折り返し案内の手順に進んでください。
なぜ答えられないのかと尋ねられたら、この電話でご案内できるのは営業時間、休診日、場所とアクセスだけだと伝えてください。

【何を聞けるかという質問】
どんなことを聞けるのか、いま聞いてよいかという質問には、答えられないと言わずに、営業時間、休診日、場所とアクセスならこの電話でご案内できると伝えてください。

【折り返し案内の手順】
まずお名前を聞いてください。お名前を聞けたら、次に電話番号を聞いてください。必ず1つずつ順番に聞き、一度に両方を聞かないでください。
日本の電話番号は10桁か11桁です。聞き取れた数字が10桁に満たないときは、けっして復唱せず、数字を補って推測することもしないでください。
そのうち、相手がまだ言い終えていない様子のとき（「ゼロハチゼロの」のように文が途中で切れているとき）は、「はい」とだけ返して続きを待ってください。
相手が言い終えた様子なのに（「です」で終わっているなど）10桁に満たないときは、「恐れ入ります、お電話番号をもう一度最初からお願いできますか」と伝えて、最初から聞き直してください。
電話番号を10桁か11桁まで聞けたら、お名前と電話番号を復唱し、これでよろしいでしょうかと尋ねて、相手の返事を待ってください。
電話番号を復唱するときは、数字を1桁ずつ読点で区切ってカタカナで読んでください。たとえば ゼロ、ハチ、ゼロ、イチ、ニ、サン、ヨン、ゴ、ロク、ナナ、ハチ のように読み、かぎかっこなどの記号で番号を囲まないでください。
復唱した返答の中で会話を終えてはいけません。復唱と終話の挨拶を同じ返答に含めないでください。
相手が「はい」などと肯定したら、その次の返答では「{CLOSING_LINE}」とだけ言って会話を終えてください。
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

    closing_mute = ClosingUserMuteStrategy()

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
            # closing_mute takes over at the other end of the call.
            user_mute_strategies=[FirstSpeechUserMuteStrategy(), closing_mute],
            # Idle detection stays off for the conversation itself: a caller who
            # goes quiet mid-call is thinking, not finished. It is armed with a
            # UserIdleTimeoutUpdateFrame once the closing line is spoken.
            user_idle_timeout=0,
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

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        # "失礼いたします" appears only in the closing line, so a completed turn
        # carrying it means the conversation is over. Watching the assistant turn
        # (rather than the text sent to the TTS) is what makes this work in both
        # modes: a text-mode eval run skips TTS entirely, so no TTS request is
        # ever made, but the assistant turn still closes.
        content = message.content or ""
        if closing_mute.closing or message.interrupted or not is_closing_utterance(content):
            return
        logger.info("Closing line spoken; muting caller and arming the hang-up timer")
        closing_mute.closing = True
        # Arming is order-independent: if the bot has already stopped speaking the
        # timer starts now, and if it hasn't, BotStoppedSpeakingFrame starts it
        # with this timeout. Either way the wait begins when the line ends.
        await worker.queue_frames([UserIdleTimeoutUpdateFrame(timeout=CLOSING_SILENCE_SECS)])

    @user_aggregator.event_handler("on_user_turn_idle")
    async def on_user_turn_idle(aggregator):
        # Only reachable in the closing state, since that is the only time the
        # idle timeout is non-zero.
        logger.info("Caller silent after the closing; saying goodbye and hanging up")
        await worker.queue_frames([TTSSpeakFrame(FAREWELL_LINE)])
        # Graceful: the queued speech is flushed before the pipeline ends, so the
        # line is spoken in full rather than cut off.
        await worker.stop_when_done()

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
        # Headless scenario runs: `uv run pipecat eval suite evals/manifest.yaml`
        # drives this over RTVI instead of a phone call. See evals/.
        "eval": lambda: EvalTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
