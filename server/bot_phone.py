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
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.evals.transport import EvalTransportParams
from pipecat.flows import (
    ConsolidatedFunctionResult,
    FlowManager,
    NodeConfig,
    flows_tool_options,
)
from pipecat.frames.frames import Frame, TTSSpeakFrame, UserIdleTimeoutUpdateFrame
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

from call_record_store import save_call_record
from callback_store import save_callback_request

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
OUT_OF_SCOPE_LINE = (
    "申し訳ございませんが、営業時間、休診日、場所のご案内以外はお答えいたしかねます。"
)
UNKNOWN_LINE = "申し訳ございませんが、その件についてはお答えいたしかねます。"
CALLBACK_OFFER_LINE = "担当者から折り返しご連絡することもできますが、いかがなさいますか。"
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


# The system instruction, set once and kept for the whole call: who the bot is
# and how it speaks. Flows pushes this when the first node is set and leaves it
# in place until a node names a new one.
ROLE_MESSAGE = """\
あなたは「さくら歯科クリニック」の電話受付AIです。丁寧で落ち着いた口調で応対してください。

【話し方】
応答は音声で読み上げられるため、記号や箇条書き、URL、絵文字など読み上げられない表記は使わず、自然な話し言葉で答えてください。
1回の返答は1文か2文にとどめ、簡潔に話してください。
文の途中に半角スペースを入れないでください。区切りたいときは読点を使ってください。
人名も姓と名の間にスペースを入れず続けて書いてください。
クリニック名を名乗るのは最初の挨拶のときだけです。それ以降の返答では名乗らないでください。
相手が「なるほど」「そうですか」のように受け答えだけをしたときは、「はい」「かしこまりました」のようにひとことで短く受けてください。直前に答えた内容を言い直したり、ご案内できる話題を並べ直したり、ほかにご用はないかと尋ねたりしないでください。

【聞き取れないとき】
相手の発話が聞き取れない、または意味が通らないときは、推測で解釈せず「恐れ入ります、もう一度お願いできますか」と聞き返してください。
"""

# What the bot is doing while it takes the call: hearing the caller out, answering
# the four topics it knows, turning everything else down. 案内 and 断り live in the
# same node on purpose — they are the two branches of one per-turn decision, and
# a node apiece would mean a routing function call before every answer.
RECEPTION_TASK = f"""\
【ご用件を伺う】
はじめの挨拶では、さくら歯科クリニックと名乗って短く挨拶し、ご用件を尋ねてください。

【答えてよい内容】
答えてよいのは、次のクリニック情報に書かれている「営業時間」「休診日」「場所」「アクセス」の4つだけです。

{CLINIC_INFO}
休診日やアクセスを尋ねられたときも、このクリニック情報に書かれているとおりに答えてください。
このクリニック情報に書かれていないことは、たとえ営業時間や場所に関する話題であっても、推測で答えてはいけません。
その場合は「{UNKNOWN_LINE}」とそのまま伝えてください。この電話でまだ折り返しをご案内していなければ、続けて「{CALLBACK_OFFER_LINE}」とそのまま伝えてください。この返答ではお名前や電話番号をまだ聞かないでください。
駐車場や設備のように、クリニック情報に書かれていない施設のことを尋ねられたときも、この文言で答えてください。
年末年始やお盆のように、クリニック情報に書かれていない日付や期間について尋ねられたときは、営業時間や休診日の情報で代わりに答えてはいけません。この文言で答えてください。

【答えてはいけない内容】
予約の受付や変更、料金、治療内容、症状の相談、その他上記4つ以外の質問には答えないでください。
その場合は「{OUT_OF_SCOPE_LINE}」とそのまま伝えてください。この電話でまだ折り返しをご案内していなければ、続けて「{CALLBACK_OFFER_LINE}」とそのまま伝えてください。この返答ではお名前や電話番号をまだ聞かないでください。
なぜ答えられないのかと尋ねられたら、この電話でご案内できるのは営業時間、休診日、場所とアクセスだけだと伝えてください。

【折り返しのご案内は一度だけ】
折り返しのご案内は、1回の電話の中で一度だけです。一度ご案内したあとは、答えられないことが続いても、断りの文言だけで返答を終えてください。
2回目以降は「{CALLBACK_OFFER_LINE}」と言ってはいけません。「いかがなさいますか」と尋ね直すこともしないでください。
相手が折り返しを希望すると言ったときだけ、start_callback を呼んでください。
start_callback を呼ぶときは、その返答では何も言わないでください。相槌も、お名前や電話番号を尋ねる言葉も入れず、関数を呼ぶだけにしてください。お名前は関数を呼んだあとの返答で、ひとつずつ伺います。

【何を聞けるかという質問】
どんなことを聞けるのか、いま聞いてよいかという質問には、答えられないと言わずに、営業時間、休診日、場所とアクセスならこの電話でご案内できると伝えてください。
"""

# What the bot is doing once the caller has asked for a callback: collecting the
# two details, one at a time, and reading them back.
CALLBACK_TASK = f"""\
【折り返しのご依頼を受け付ける】
まずお名前を聞いてください。お名前を聞けたら、次に電話番号を聞いてください。必ず1つずつ順番に聞き、一度に両方を聞かないでください。

【電話番号の聞き取り】
日本の電話番号は10桁か11桁です。復唱する前に、これまでに聞き取れた数字の桁数を必ず数えてください。
10桁に満たないときは、けっして復唱してはいけません。足りない桁を補ったり、同じ数字を繰り返して桁を埋めたりするのも禁止です。
そのうち、相手がまだ言い終えていない様子のとき（「ゼロハチゼロの」のように文が途中で切れているとき）は、「はい」とだけ返して続きを待ってください。
相手が言い終えた様子なのに（「です」で終わっているなど）10桁に満たないときは、「恐れ入ります、お電話番号をもう一度最初からお願いできますか」と伝えて、最初から聞き直してください。

【復唱して確認する返答】
電話番号を10桁か11桁まで聞けたら、お名前と電話番号を復唱し、これでよろしいでしょうかと尋ねてください。
復唱は「佐藤様、お電話番号はゼロ、ハチ、ゼロ、イチ、ニ、サン、ヨン、ゴ、ロク、ナナ、ハチ、でよろしいでしょうか」のように、お名前から始めて電話番号を続ける形にしてください。電話番号だけの復唱にしないでください。
電話番号を復唱するときは、数字を1桁ずつ読点で区切ってカタカナで読んでください。たとえば ゼロ、ハチ、ゼロ、イチ、ニ、サン、ヨン、ゴ、ロク、ナナ、ハチ のように読み、かぎかっこなどの記号で番号を囲まないでください。
この返答はここで終わりです。record_callback は呼ばず、終話の挨拶も言わず、相手の返事を待ってください。

【相手が肯定したとき】
復唱に対して相手が「はい」などと肯定したら、その次の返答で record_callback を呼んでください。
渡すお名前と電話番号は、いちばん最後に復唱した内容にしてください。途中で聞き直したときは、古い方ではなく新しい方を渡してください。電話番号は数字だけにしてください。
record_callback を呼ばずに会話を終えてはいけません。

【相手が否定したとき】
「違います」などと否定されたら、どちらが違うのかを聞き分けて、違うと言われた方だけを聞き直してください。
お名前が違うと言われたら「失礼いたしました、もう一度お名前をお願いできますか」の1文だけを返してください。
電話番号が違うと言われたときや、どちらが違うのか分からないときは「失礼いたしました、もう一度お電話番号をお願いできますか」の1文だけを返してください。
確認の言葉を何度も繰り返さないでください。
聞き直さなかった方はすでに聞けているので、もう一度尋ねないでください。
聞き直した方を聞けたら、お名前と電話番号の両方をもう一度復唱して、これでよろしいでしょうかと尋ねてください。この返答でも record_callback は呼ばず、終話の挨拶も言わず、相手の返事を待ってください。2回目、3回目の復唱でも同じです。
肯定が返ってくるまで record_callback を呼んではいけません。復唱は何回でも、肯定は必ず1回必要です。
"""

# The last thing on the line.
CLOSING_TASK = f"""\
【終話】
「{CLOSING_LINE}」とだけ言ってください。
この文言の前に相槌やお礼を付けないでください。「はい」「ありがとうございます」などを先に言ってはいけません。この文言の後にも何も言わないでください。新しい質問も確認もしないでください。
"""


# Each node's procedure goes in its role_message — the system instruction —
# rather than in task_messages. As a developer message inside the context it was
# followed loosely: the bot would read a number back and record it in the same
# turn, however plainly the text said to wait. It also stacked up, since the
# default APPEND strategy leaves the previous node's task messages in the
# context; a role_message is replaced on each transition instead.
def reception_node() -> NodeConfig:
    """The node the call starts in: 用件確認 and, in the same breath, 案内 and 断り."""
    return {
        "name": "reception",
        "role_message": ROLE_MESSAGE + RECEPTION_TASK,
        "task_messages": [{"role": "developer", "content": "ご用件を伺ってください。"}],
        "functions": [start_callback],
    }


def callback_node() -> NodeConfig:
    """折り返し受付: the caller has asked to be called back."""
    return {
        "name": "callback",
        "role_message": ROLE_MESSAGE + CALLBACK_TASK,
        "task_messages": [
            {"role": "developer", "content": "折り返しのご依頼を受け付けてください。"}
        ],
        "functions": [record_callback],
    }


def closing_node() -> NodeConfig:
    """終話: say the closing line, and nothing else."""
    return {
        "name": "closing",
        "role_message": ROLE_MESSAGE + CLOSING_TASK,
        "task_messages": [{"role": "developer", "content": "終話の挨拶をしてください。"}],
        "functions": [],
    }


# Flows registers its functions with cancel_on_interruption=False, which makes
# them async tools: the aggregators then write the async-tool protocol into the
# context — an ASYNC TOOLS instruction block plus a started and a final message
# per call — and it is all re-sent on every turn afterwards. Both of these
# functions return immediately (one swaps a node, the other appends a line to a
# file), so there is nothing to keep running across an interruption.
@flows_tool_options(cancel_on_interruption=True)
async def start_callback(flow_manager: FlowManager) -> ConsolidatedFunctionResult:
    """折り返しのご依頼を受け付けます。相手が折り返しを希望したときに呼んでください。"""
    logger.info("Caller asked for a callback; moving to the callback node")
    return None, callback_node()


@flows_tool_options(cancel_on_interruption=True)
async def record_callback(
    flow_manager: FlowManager, name: str, phone_number: str
) -> ConsolidatedFunctionResult:
    """折り返しのご依頼を記録します。復唱して確認が取れたあとに呼んでください。

    Args:
        name: 相手のお名前。復唱して確認が取れたもの。
        phone_number: 相手の電話番号。数字だけで渡してください。例: 08012345678
    """
    record = save_callback_request(name, phone_number)
    logger.info(f"Callback request saved: {record}")
    return {"saved": True}, closing_node()


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
    started_at = datetime.now()

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
            # No system_instruction here: the flow's first node sets it from
            # ROLE_MESSAGE and it stays for the rest of the call.
        ),
    )

    closing_mute = ClosingUserMuteStrategy()
    hanging_up = False

    # No tools here: each node brings its own, and the flow manager swaps them on
    # every transition.
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Smart Turn v3 (the default stop strategy) misjudges Japanese
            # sentence-final particles ("〜んですけど", "〜したくて") as
            # incomplete: 9 EndOfTurnState.INCOMPLETE in one call, twice still
            # INCOMPLETE after 3s of silence, one turn left the user hanging
            # for ~23s. Falling back to a plain VAD stop_secs-based stop.
            #
            # stop_secs + user_speech_timeout (below) is both how long the bot
            # waits before answering *and* how long a mid-sentence pause may run
            # before the turn is cut: the two timers are in series, and a caller
            # who resumes inside the second one has all progress discarded. It
            # was 1.0 + 0.6, which measured ~2.2s from speech end to the bot's
            # first audio — 1.6s of it this pair. Now 0.2 + 0.8, for 1.0s.
            #
            # 0.2 for the VAD specifically, rather than keeping the wait here:
            # the strategy's third timer waits out the STT's final transcript
            # for max(0, p99 - stop_secs), and Deepgram's p99 is 0.35s, so any
            # stop_secs at or above that collapses the wait to zero and lets a
            # turn end on a transcript that is still arriving. Calls at 1.0 show
            # exactly that — "何時まで?", "駐車場っあります?", "今日っ" — and it
            # costs nothing to fix, since 0.15s runs inside the 0.8s below.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_turn_strategies=UserTurnStrategies(
                # VAD-only start: TranscriptionUserTurnStartStrategy (the other
                # default) fires trigger_user_turn_started() — which broadcasts
                # an interruption — on every interim transcript, fragmenting
                # one utterance into multiple user turns with assistant
                # one-character fragments spliced in between. VADUserTurnStartStrategy
                # still broadcasts an interruption on VADUserStartedSpeakingFrame
                # (enable_interruptions defaults to True), so barge-in is unaffected.
                start=[VADUserTurnStartStrategy()],
                # 0.8 rather than the default 0.6: this is the half of the wait
                # a caller can still interrupt, so the 0.8s the VAD gave up above
                # is better spent here. A 1.4s pause mid-sentence was measured in
                # one call out of nine turns, which 1.0s does not cover — the
                # turn is cut and the rest of the sentence arrives as a new one.
                # Acceptable now that the prompt answers a half-heard phone
                # number with "はい" and waits for the rest.
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.8)],
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
    # The flow manager wants the pair (it reaches for .user() and .assistant());
    # the pipeline wants the two processors.
    user_aggregator, assistant_aggregator = aggregators

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

    flow_manager = FlowManager(llm=llm, context_aggregator=aggregators, worker=worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation. Twilio's WebSocket connection has no RTVI
        # client-ready handshake, so start as soon as the stream connects.
        # Setting the first node is what speaks the greeting: the node's task
        # messages go into the context and the flow runs the LLM straight away.
        await flow_manager.initialize(reception_node())

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
        #
        # Once, though: the goodbye below is itself bot speech, so the caller goes
        # quiet after it too and the timer arms again. Whether that second firing
        # gets as far as speaking depends on how long the pipeline takes to end —
        # a race that would have the bot say goodbye twice.
        nonlocal hanging_up
        if hanging_up:
            return
        hanging_up = True
        logger.info("Caller silent after the closing; saying goodbye and hanging up")
        # Turn the timer off as well, so nothing is left armed while the pipeline
        # drains.
        await worker.queue_frames(
            [UserIdleTimeoutUpdateFrame(timeout=0), TTSSpeakFrame(FAREWELL_LINE)]
        )
        # Graceful: the queued speech is flushed before the pipeline ends, so the
        # line is spoken in full rather than cut off.
        await worker.stop_when_done()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    @worker.event_handler("on_pipeline_finished")
    async def on_pipeline_finished(worker, frame):
        # Whichever way the call ended — the caller hanging up (a CancelFrame
        # through on_client_disconnected above) or the bot hanging up after its
        # goodbye (an EndFrame) — this runs once, at the end of both.
        record = save_call_record(
            context.get_messages(),
            started_at=started_at,
            session_id=getattr(runner_args, "session_id", None),
        )
        logger.info(
            f"Call record saved: {len(record['messages'])} message(s), "
            f"{record['started_at']} to {record['ended_at']}"
        )

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
