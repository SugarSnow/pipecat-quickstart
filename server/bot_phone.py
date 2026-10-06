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
import time
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
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserIdleTimeoutUpdateFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
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
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import (
    BaseUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from call_record_store import save_call_record
from callback_store import (
    has_expected_digit_count,
    is_partial_phone_number,
    merge_phone_number,
    save_callback_request,
)
from silent_tts import SilentTTSService
from tenant_config import TENANT, clinic_info_text, keyterms
from text_cleanup import CollapseRepeatedPunctuation

load_dotenv(override=True)

# Clinic facts the bot is allowed to answer from. Kept separate from the prompt so
# they can be updated without touching the instructions. Written in spoken-Japanese
# form (no symbols like 〇 / - / 〜) because the LLM output is read aloud by TTS.
#
# No half-width spaces, here or in the label separators: Cartesia's Japanese word
# timestamps drop them, which desynchronises the text the assistant aggregator
# rebuilds and corrupts the stored turn. See tests/test_cartesia_ja_space_corruption.py.
CLINIC_INFO = clinic_info_text(TENANT)

# The words this clinic's callers actually say, boosted in the speech-to-text.
KEYTERMS = keyterms(TENANT)

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

# Spoken when the digits do not add up, in place of a read-back.
RE_ASK_NUMBER_LINE = "恐れ入ります、お電話番号をもう一度最初からお願いできますか。"

# Spoken while the caller is still reading their number out.
ACKNOWLEDGE_LINE = "はい。"

# The first thing the caller hears. Spoken by the bot rather than composed by
# the LLM: a greeting has one right wording and nothing to decide, and leaving
# it to the LLM cost a call its opening — the model was still generating when
# the VAD heard the caller, the turn was cancelled, and the bot never gave its
# name. A fixed line also starts speaking a second sooner, since there is no
# LLM round trip in front of it.
GREETING_LINE = (
    "お電話ありがとうございます。さくら歯科クリニックです。ご用件をお聞かせいただけますか。"
)

# How long to hold the line after the closing before hanging up.
CLOSING_SILENCE_SECS = 3.0

# Spoken once when the caller says nothing to the read-back. A phone test sat
# through 33 seconds of silence there: the caller had not realised a reply was
# wanted, and the bot had nothing that would make it ask.
READBACK_NUDGE_LINE = "よろしいでしょうか。"

# How long to wait for that reply. Long enough to be reading a number back off a
# screen or thinking, short enough that the pause does not become a dead line.
READBACK_SILENCE_SECS = 8.0


# How each digit is read out loud. 0 and 4 and 7 have a second reading that is
# easy to mishear on a phone (レイ, シ, シチ), so the clearer one is spelled here.
_DIGIT_READINGS = {
    "0": "ゼロ",
    "1": "イチ",
    "2": "ニ",
    "3": "サン",
    "4": "ヨン",
    "5": "ゴ",
    "6": "ロク",
    "7": "ナナ",
    "8": "ハチ",
    "9": "キュウ",
}


def spell_out_digits(phone_number: str) -> str:
    """Return *phone_number* as digits read one at a time.

    Args:
        phone_number: The number, as the caller gave it.

    Returns:
        The digits separated by 読点, e.g. ``"ゼロ、ハチ、ゼロ"``. Anything that is
        not a digit is dropped: the string is going straight to the TTS, and the
        point of reading digit by digit is lost if a stray character rides along.
    """
    return "、".join(_DIGIT_READINGS[c] for c in phone_number if c in _DIGIT_READINGS)


def read_back_line(name: str, phone_number: str) -> str:
    """Return the sentence the bot speaks to confirm a callback request.

    Spoken by the bot itself rather than written by the LLM, so the wording is
    the same every time and the digits are always read one at a time.

    Args:
        name: The caller's name, as heard.
        phone_number: The caller's number, as heard.

    Returns:
        The read-back, ending in a question.
    """
    return f"{name}様、お電話番号は{spell_out_digits(phone_number)}、でよろしいでしょうか。"


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


def is_callback_offer(text: str) -> bool:
    """Whether *text* is a turn in which the callback was offered in full.

    "いかがなさいますか" is the offer line's last clause and appears in nothing
    else the bot says, so finding it means the caller heard the offer through
    to its question. An offer cut off before it — the context keeps only what
    was actually played — does not match, which is the wanted answer: a caller
    who never heard the question was never really offered anything.

    Args:
        text: The assistant turn to examine.

    Returns:
        True when the offer was made in full.
    """
    return "いかがなさいますか" in text


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


# Long enough that no greeting reaches it, short enough that a caller is never
# stranded. Only a bot that never speaks at all gets this far — a TTS outage,
# say — and by then the call is lost anyway; unmuting at least lets the caller
# hear themselves be heard rather than talk into a line that answers nothing.
OPENING_MUTE_TIMEOUT_SECS = 15.0


class OpeningUserMuteStrategy(BaseUserMuteStrategy):
    """Mutes the caller from the moment the line connects until the greeting ends.

    FirstSpeechUserMuteStrategy, the framework strategy this replaces, is
    explicit that it "allows user input before the bot starts speaking" — it
    only covers the span between the bot's first BotStartedSpeakingFrame and the
    matching stop. The window it leaves open is the one that actually broke a
    call: the caller spoke into the silence before the greeting began, the VAD
    broadcast an interruption, and the greeting was cancelled before the bot had
    named the clinic.

    Starting muted closes that window. Everything else matches the framework
    strategy: the mute lifts when the first bot speech ends, and later turns
    barge in normally.
    """

    def __init__(self, timeout: float = OPENING_MUTE_TIMEOUT_SECS):
        """Initialize the strategy, muted.

        Args:
            timeout: Seconds to stay muted if the bot never speaks at all.
        """
        super().__init__()
        self._timeout = timeout
        self._released = False
        self._first_frame_at: float | None = None

    async def process_frame(self, frame: Frame) -> bool:
        """Report whether the caller should be muted.

        Args:
            frame: The frame being evaluated.

        Returns:
            True until the bot has finished its opening greeting.
        """
        await super().process_frame(frame)

        if self._released:
            return False

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._released = True
            return False

        # The clock starts at the first frame rather than at construction: the
        # strategy is built while the pipeline is still being assembled, which
        # is well before the line connects.
        if self._first_frame_at is None:
            self._first_frame_at = time.monotonic()
        elif time.monotonic() - self._first_frame_at > self._timeout:
            logger.warning("No opening greeting within the timeout; unmuting the caller")
            self._released = True
            return False

        return True


# Below this, a caller talking over the bot is taken for a backchannel rather
# than an interruption. Counted in characters, not words: Japanese transcripts
# come back with few or no spaces ("何時までやっていますか。" is one "word" to
# str.split), so the framework's MinWordsUserTurnStartStrategy can never reach
# a threshold above 1 here.
#
# Six keeps the common acknowledgements out — はい, ええ, うん, なるほど,
# そうですか — while letting anything with a request in it through
# ("ちょっと待って", "もう一度お願いします"). A real interruption shorter than
# this ("すみません") waits for the bot to finish its sentence instead, which is
# the lesser of the two mistakes on a phone call.
BACKCHANNEL_MAX_CHARS = 6


class BackchannelToleranceUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """Starts a user turn on VAD, except over the bot's own speech.

    Replaces VADUserTurnStartStrategy, which starts a turn — and so broadcasts
    an interruption — the moment the VAD hears anything. On a phone line that
    is too eager: a 0.2s noise cut the bot off mid-sentence and the caller never
    heard the end of the callback offer.

    While the bot is speaking, the VAD alone is no longer enough; a transcript
    of at least ``min_chars`` has to arrive first. While the bot is silent,
    nothing changes — the VAD starts the turn as immediately as before, which
    matters for the one-word answers ("はい") the read-back depends on.

    The framework's own MinWordsUserTurnStartStrategy is the same idea, but it
    counts ``str.split()`` words and has no VAD path, so it fits neither
    Japanese nor the latency this bot needs when the line is quiet.
    """

    def __init__(self, *, min_chars: int = BACKCHANNEL_MAX_CHARS, **kwargs):
        """Initialize the strategy.

        Args:
            min_chars: Characters a caller must be heard saying, while the bot
                is speaking, before it counts as an interruption.
            **kwargs: Passed to the base strategy.
        """
        super().__init__(**kwargs)
        self._min_chars = min_chars
        self._bot_speaking = False
        self._user_speaking = False
        self._turn_active = False

    async def handle_user_turn_started(self):
        """Stand down: the turn this strategy was watching for has begun."""
        self._turn_active = True

    async def handle_user_turn_stopped(self):
        """Start watching again, from the next turn's first frame."""
        self._turn_active = False
        self._user_speaking = False

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Decide whether this frame starts a user turn.

        Args:
            frame: The frame to be analyzed.

        Returns:
            STOP when a turn was started, CONTINUE otherwise.
        """
        # Mid-turn, the only thing worth tracking is whether the bot is
        # speaking. Judging transcripts here would mean calling
        # trigger_reset_aggregation() on a turn that has already started —
        # discarding the words it is made of.
        if self._turn_active:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._bot_speaking = True
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._bot_speaking = False
            return ProcessFrameResult.CONTINUE

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            # A caller still talking when the bot finishes is taking their turn,
            # however briefly they have been at it. Without this, speech that
            # began as a backchannel over the bot would have no turn to belong
            # to once the bot fell silent.
            if self._user_speaking:
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            if not self._bot_speaking:
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if self._bot_speaking:
                return await self._handle_transcription(frame)

        return ProcessFrameResult.CONTINUE

    async def _handle_transcription(
        self, frame: TranscriptionFrame | InterimTranscriptionFrame
    ) -> ProcessFrameResult:
        """Weigh a transcript heard over the bot against the backchannel threshold."""
        # Whitespace only: Deepgram puts a space between a surname and a given
        # name but not between words, so stripping it is the whole of what
        # "characters actually said" means here.
        spoken = "".join(frame.text.split())
        if len(spoken) >= self._min_chars:
            logger.debug(f"Interrupting on {len(spoken)} chars over the bot: {spoken!r}")
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP

        # Discard it, the way the framework's min-words strategy does: an
        # acknowledgement should not be waiting in the aggregator to be answered
        # as a turn of its own once the bot stops.
        await self.trigger_reset_aggregation()
        return ProcessFrameResult.CONTINUE


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
相手が「なるほど」「そうですか」のように受け答えだけをしたときは、「はい」「かしこまりました」のようにひとことで短く受けてください。直前に答えた内容を言い直したり、ご案内できる話題を並べ直したり、ほかにご用はないかと尋ねたりしないでください。ただし次の【言い直し】にあてはまるときは、この決まりより【言い直し】を優先してください。

【言い直し】
相手が「途中で切れました」「何て言いました」「もう一度お願いします」「聞こえませんでした」のように聞き返したときは、直前のあなたの発言を最初から言い直してください。「はい」のような短い受け答えで済ませてはいけません。
直前の発言が途中で切れていたときも、切れたところからではなく、最初から言い直してください。

【聞き取れないとき】
相手の発話が聞き取れない、または意味が通らないときは、推測で解釈せず「恐れ入ります、もう一度お願いできますか」と聞き返してください。
"""

# What the bot is doing while it takes the call: hearing the caller out, answering
# the four topics it knows, turning everything else down. 案内 and 断り live in the
# same node on purpose — they are the two branches of one per-turn decision, and
# a node apiece would mean a routing function call before every answer.
def reception_task(*, offered: bool) -> str:
    """The 応対 node's procedure, in its before- and after-the-offer forms.

    The callback offer may be made once per call. That used to be a rule in the
    prompt — "一度だけ", with the line itself quoted right there — and the model
    broke it about one turn in five. Here the offered form simply never mentions
    the line, so there is nothing to repeat; the two forms are otherwise the
    same text.

    Args:
        offered: Whether the callback has already been offered on this call.

    Returns:
        The task text for the matching node.
    """
    # The one clause that differs, appended to each refusal.
    offer_clause = (
        ""
        if offered
        else f"続けて「{CALLBACK_OFFER_LINE}」とそのまま伝えてください。"
    )
    # In the offered form the line survives in one sentence only, and that
    # sentence is about repeating it — the caller asked what was said, and the
    # same offer reaching them twice is not two offers. What is gone is the
    # "and then offer" clause on each refusal above, which is what actually
    # drove the bot to offer again.
    offer_rule = (
        "折り返しのご案内は、この電話でもう済んでいます。こちらから持ちかけてはいけません。答えられないことが続いても、断りの文言だけで返答を終えてください。\n"
        f"例外は聞き返されたときだけです。直前の発言を言い直すよう求められたら、「{CALLBACK_OFFER_LINE}」の部分も省かずにそのまま言い直してください。"
        if offered
        else "折り返しのご案内は、1回の電話の中で一度だけです。"
    )

    return f"""\
【ご用件を伺う】
はじめの挨拶はこちらで読み上げ済みです。あなたが挨拶や名乗りから始めることはありません。相手の用件に答えてください。

【答えてよい内容】
答えてよいのは、次のクリニック情報に書かれている「営業時間」「休診日」「場所」「アクセス」の4つだけです。

{CLINIC_INFO}
休診日やアクセスを尋ねられたときも、このクリニック情報に書かれているとおりに答えてください。
このクリニック情報に書かれていないことは、たとえ営業時間や場所に関する話題であっても、推測で答えてはいけません。
その場合は「{UNKNOWN_LINE}」とそのまま伝えてください。{offer_clause}この返答ではお名前や電話番号をまだ聞かないでください。
駐車場や設備のように、クリニック情報に書かれていない施設のことを尋ねられたときも、この文言で答えてください。
年末年始やお盆のように、クリニック情報に書かれていない日付や期間について尋ねられたときは、営業時間や休診日の情報で代わりに答えてはいけません。この文言で答えてください。

【答えてはいけない内容】
予約の受付や変更、料金、治療内容、症状の相談、その他上記4つ以外の質問には答えないでください。
その場合は「{OUT_OF_SCOPE_LINE}」とそのまま伝えてください。{offer_clause}この返答ではお名前や電話番号をまだ聞かないでください。
なぜ答えられないのかと尋ねられたら、この電話でご案内できるのは営業時間、休診日、場所とアクセスだけだと伝えてください。

【折り返しのご案内】
{offer_rule}
相手が折り返しを希望すると言ったときだけ、start_callback を呼んでください。
相手が自分から折り返しを申し出たとき（「名前と電話番号を伝えてもいいですか」「折り返してもらえますか」「電話がほしいです」など）は、断りの文言を言わずに、すぐ start_callback を呼んでください。もう希望していると分かっているので、希望するかどうかを尋ね直す必要もありません。
start_callback を呼ぶときは、その返答では何も言わないでください。相槌も、お名前や電話番号を尋ねる言葉も入れず、関数を呼ぶだけにしてください。お名前は関数を呼んだあとの返答で、ひとつずつ伺います。

【何を聞けるかという質問】
どんなことを聞けるのか、いま聞いてよいかという質問には、答えられないと言わずに、営業時間、休診日、場所とアクセスならこの電話でご案内できると伝えてください。
"""

# What the bot is doing once the caller has asked for a callback: collecting the
# two details, one at a time, and reading them back.
CALLBACK_TASK = """\
【折り返しのご依頼を受け付ける】
まずお名前を聞いてください。お名前を聞けたら、次に電話番号を聞いてください。必ず1つずつ順番に聞き、一度に両方を聞かないでください。
お名前は名字だけでもかまいません。下のお名前を尋ね直さないでください。

【電話番号の聞き取り】
相手がまだ言い終えていない様子のとき（「ゼロハチゼロの」のように文が途中で切れているとき）は、「はい」とだけ返して続きを待ってください。
相手が言い終えた様子なら、聞き取れた数字をそのまま渡してください。桁数の確認はこちらで行うので、あなたは数えなくてよいです。
桁が足りないと思っても、自分で止めずにそのまま渡してください。足りているかどうかはこちらで判定し、足りなければこちらから聞き直します。
まだ数字をひとつも聞き取れていないときは、関数を呼ばずにお電話番号を尋ねてください。
聞き取れなかった桁を補ったり、同じ数字を繰り返して桁を埋めたりしてはいけません。聞き取れた数字だけを渡してください。

【両方そろったら】
お名前と電話番号の両方が聞けたら、confirm_callback にその2つを渡してください。電話番号は数字だけにしてください。
復唱はこちらで読み上げます。あなたは復唱しないでください。復唱の言葉を返答に含めないでください。
"""

CONFIRM_TASK = """\
【復唱したあとの確認】
お名前と電話番号の復唱はすでに読み上げられています。あなたから復唱し直さないでください。
相手が「はい」「大丈夫です」などと肯定したら、record_callback を呼んでください。
お名前が違うと言われたら correct_name を、電話番号が違うと言われたら correct_phone_number を呼んでください。どちらが違うのか分からないときは correct_phone_number を呼んでください。
肯定も訂正もない返答（聞き返しなど）のときは、どちらの関数も呼ばず、ひとこと短く答えてください。
関数を呼ばずに、終話の挨拶やお礼、折り返しの約束を言ってはいけません。肯定されたのに record_callback を呼ばないまま会話を終えるのは誤りです。
"""

CORRECTION_TASK = """\
【聞き直し】
違うと言われた方だけを、ひとことで聞き直してください。もう片方はすでに聞けているので尋ねないでください。
電話番号を聞き直しているときも、最初に伺うときと同じです。相手がまだ言い終えていない様子のとき（「ゼロハチゼロ」「六一一の」のように文が途中で切れているとき）は、「はい」とだけ返して続きを待ってください。「その後の番号をお願いします」のように言葉を足さないでください。相手の続きとかぶります。
聞き直した方を聞き取れたら、お名前なら confirm_name に、電話番号なら confirm_phone_number に、その1つだけを渡してください。
渡すのは、いま聞き直して新しく聞き取れた方です。前に聞いていた古い値を渡してはいけません。相手が「佐藤祐希です」と言い直したなら、渡すのは「佐藤祐希」です。
電話番号は数字だけにしてください。復唱はこちらで読み上げるので、あなたは復唱しないでください。
"""

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
    """The node the call starts in: 用件確認 and, in the same breath, 案内 and 断り.

    The greeting is a ``tts_say`` pre-action and the node then waits, for the
    same reason the read-back does: the wording is fixed, so there is nothing
    for the LLM to decide, and speaking it directly removes the generation gap
    the caller used to talk into.
    """
    return {
        "name": "reception",
        "role_message": ROLE_MESSAGE + reception_task(offered=False),
        "task_messages": [{"role": "developer", "content": "ご用件を伺ってください。"}],
        "pre_actions": [{"type": "tts_say", "text": GREETING_LINE}],
        "respond_immediately": False,
        "functions": [start_callback],
    }


def reception_offered_node() -> NodeConfig:
    """同じ応対を続ける。ただし折り返しのご案内はもう持ちかけない。

    The same node as above with one thing taken away: its instructions no
    longer contain the offer line, so there is no wording left to repeat. The
    bot is moved here the moment it finishes saying the offer, which is why
    nothing is spoken on arrival — it has just spoken.
    """
    return {
        "name": "reception_offered",
        "role_message": ROLE_MESSAGE + reception_task(offered=True),
        "task_messages": [
            {"role": "developer", "content": "折り返しのご案内は済んでいます。"}
        ],
        "respond_immediately": False,
        "functions": [start_callback],
    }


def callback_collect_node() -> NodeConfig:
    """折り返し受付: collect the name and the number, one at a time."""
    return {
        "name": "callback_collect",
        "role_message": ROLE_MESSAGE + CALLBACK_TASK,
        "task_messages": [
            {"role": "developer", "content": "折り返しのご依頼を受け付けてください。"}
        ],
        "functions": [confirm_callback],
    }


def callback_confirm_node(name: str, phone_number: str) -> NodeConfig:
    """復唱と確認: read the details back, then wait for the caller to answer.

    The read-back is a ``tts_say`` pre-action rather than something the LLM
    writes, and the node does not respond on entry. That is what makes the wait
    reliable: there is no turn in which the model could both ask "is this right?"
    and answer its own question. Saying it ourselves fixes the wording too — the
    digits come from :func:`read_back_line`, not from the model.
    """
    return {
        "name": "callback_confirm",
        "role_message": ROLE_MESSAGE + CONFIRM_TASK,
        "task_messages": [
            {"role": "developer", "content": "復唱に対する相手の返事を待ってください。"}
        ],
        "pre_actions": [{"type": "tts_say", "text": read_back_line(name, phone_number)}],
        "respond_immediately": False,
        "functions": [record_callback, correct_name, correct_phone_number],
    }


def name_correction_node() -> NodeConfig:
    """お名前だけを聞き直す。"""
    return {
        "name": "name_correction",
        "role_message": ROLE_MESSAGE + CORRECTION_TASK,
        "task_messages": [{"role": "developer", "content": "お名前をもう一度伺ってください。"}],
        "functions": [confirm_name],
    }


def phone_correction_node() -> NodeConfig:
    """電話番号だけを聞き直す。"""
    return {
        "name": "phone_correction",
        "role_message": ROLE_MESSAGE + CORRECTION_TASK,
        "task_messages": [{"role": "developer", "content": "お電話番号をもう一度伺ってください。"}],
        "functions": [confirm_phone_number],
    }


def number_fragment_node() -> NodeConfig:
    """まだ読み上げ途中の番号を、相槌だけ返して待つ。

    The caller is mid-number, so the only right answer is "はい" and silence.
    Said as a fixed line for the same reason as the read-back: the model, left
    to word this itself, answered "ありがとうございます、その後の番号をお願い
    します" and talked over the next few digits.
    """
    return {
        "name": "number_fragment",
        "role_message": ROLE_MESSAGE + CORRECTION_TASK,
        "task_messages": [{"role": "developer", "content": "番号の続きを待ってください。"}],
        "pre_actions": [{"type": "tts_say", "text": ACKNOWLEDGE_LINE}],
        "respond_immediately": False,
        "functions": [confirm_phone_number],
    }


def number_retry_node() -> NodeConfig:
    """桁数が合わない番号を、復唱せずに聞き直す。

    Same shape as the read-back node — a fixed line, then silence until the
    caller speaks — because the number must not be read back at all: repeating
    a number that cannot be right invites a "yes" to something wrong.
    """
    return {
        "name": "number_retry",
        "role_message": ROLE_MESSAGE + CORRECTION_TASK,
        "task_messages": [{"role": "developer", "content": "お電話番号をもう一度伺ってください。"}],
        "pre_actions": [{"type": "tts_say", "text": RE_ASK_NUMBER_LINE}],
        "respond_immediately": False,
        "functions": [confirm_phone_number],
    }


def closing_node() -> NodeConfig:
    """終話: say the closing line, and nothing else."""
    return {
        "name": "closing",
        "role_message": ROLE_MESSAGE + CLOSING_TASK,
        "task_messages": [{"role": "developer", "content": "終話の挨拶をしてください。"}],
        "functions": [],
    }


def _confirm(flow_manager: FlowManager, **details: str) -> NodeConfig:
    """Remember whichever detail was just heard and move to the read-back.

    Unless the number cannot be right: a mobile prefix needs 11 digits and
    everything else 10, and a number that does not add up is asked for again
    instead of being read back.
    """
    # A number read out in pieces is assembled here rather than left to the
    # model, which passes sometimes the new piece and sometimes the whole number
    # so far. Merging makes the two indistinguishable.
    if "phone_number" in details:
        details["phone_number"] = merge_phone_number(
            flow_manager.state.get("phone_number", ""), details["phone_number"]
        )
    flow_manager.state.update(details)
    name = flow_manager.state.get("name", "")
    phone_number = flow_manager.state.get("phone_number", "")

    if is_partial_phone_number(phone_number):
        logger.info(f"Phone number is still coming ({phone_number}); waiting for the rest")
        return number_fragment_node()

    if not has_expected_digit_count(phone_number):
        logger.info(f"Phone number has the wrong digit count ({phone_number}); asking again")
        # Cleared so the next attempt is assembled from scratch: these digits
        # are known to be wrong, and merging onto them would carry the mistake.
        flow_manager.state["phone_number"] = ""
        return number_retry_node()

    logger.info(f"Reading back: {name} / {phone_number}")
    # Each read-back gets its own nudge: a corrected number is read back again,
    # and the caller can just as easily miss their cue the second time.
    flow_manager.state["readback_nudged"] = False
    return callback_confirm_node(name, phone_number)


@flows_tool_options(cancel_on_interruption=True)
async def start_callback(flow_manager: FlowManager) -> ConsolidatedFunctionResult:
    """折り返しのご依頼を受け付けます。相手が折り返しを希望したときに呼んでください。"""
    logger.info("Caller asked for a callback; collecting their details")
    return None, callback_collect_node()


@flows_tool_options(cancel_on_interruption=True)
async def confirm_callback(
    flow_manager: FlowManager, name: str, phone_number: str
) -> ConsolidatedFunctionResult:
    """聞き取ったお名前と電話番号を復唱して確認します。両方そろったら呼んでください。

    Args:
        name: 相手のお名前。
        phone_number: 相手の電話番号。数字だけで渡してください。例: 08012345678
    """
    return None, _confirm(flow_manager, name=name, phone_number=phone_number)


@flows_tool_options(cancel_on_interruption=True)
async def confirm_name(flow_manager: FlowManager, name: str) -> ConsolidatedFunctionResult:
    """聞き直したお名前で、もう一度復唱して確認します。

    Args:
        name: 聞き直したお名前。
    """
    return None, _confirm(flow_manager, name=name)


@flows_tool_options(cancel_on_interruption=True)
async def confirm_phone_number(
    flow_manager: FlowManager, phone_number: str
) -> ConsolidatedFunctionResult:
    """聞き直した電話番号で、もう一度復唱して確認します。

    Args:
        phone_number: 聞き直した電話番号。数字だけで渡してください。例: 08012345678
    """
    return None, _confirm(flow_manager, phone_number=phone_number)


@flows_tool_options(cancel_on_interruption=True)
async def correct_name(flow_manager: FlowManager) -> ConsolidatedFunctionResult:
    """お名前が違うと言われたときに呼んでください。お名前だけを聞き直します。"""
    logger.info("Caller says the name is wrong; asking for it again")
    return None, name_correction_node()


@flows_tool_options(cancel_on_interruption=True)
async def correct_phone_number(flow_manager: FlowManager) -> ConsolidatedFunctionResult:
    """電話番号が違うと言われたときに呼んでください。電話番号だけを聞き直します。"""
    logger.info("Caller says the number is wrong; asking for it again")
    # The number just rejected must not be merged into what comes next.
    flow_manager.state["phone_number"] = ""
    return None, phone_correction_node()


@flows_tool_options(cancel_on_interruption=True)
async def record_callback(flow_manager: FlowManager) -> ConsolidatedFunctionResult:
    """復唱に対して相手が肯定したときに呼んでください。折り返しのご依頼を記録します。"""
    # The details come from the flow's state, not from arguments: these are the
    # ones that were read back, so a correction cannot leave a stale name or
    # number in the record — which is what happened when the model passed them.
    record = save_callback_request(
        flow_manager.state.get("name", ""), flow_manager.state.get("phone_number", "")
    )
    logger.info(f"Callback request saved: {record}")
    return {"saved": True}, closing_node()


def _wants_silent_tts(runner_args: RunnerArguments) -> bool:
    """Whether this run should skip synthesis entirely.

    Read from the runner body, which ``pipecat eval suite`` fills from the
    manifest entry's ``runner_body``. That is the only per-scenario channel the
    suite has — its ``spawn`` template is shared by every run — so it is how the
    text-mode scenarios ask for a silent bot while the audio ones keep Cartesia.

    Args:
        runner_args: The session arguments the runner handed this bot.

    Returns:
        True when the run wants no audio synthesized.
    """
    body = runner_args.body
    return bool(isinstance(body, dict) and body.get("silent_tts"))


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
            # The clinic's own name and the handful of words the call turns on.
            # Read from config/tenant.json rather than written here, so a second
            # clinic is a second file. Deepgram weights these without forcing
            # them, so a caller who says something else is still heard.
            keyterm=KEYTERMS,
        ),
    )

    # Text-to-Speech service. A text-mode eval scenario never listens to the
    # bot, so it asks (through the manifest's runner_body) for the silent
    # service instead and the run costs no Cartesia at all. The decision has to
    # be made here rather than from the eval transport's connect-time skip_tts
    # flag, because CartesiaTTSService opens its websocket as the pipeline
    # starts — long before a client connects.
    if _wants_silent_tts(runner_args):
        logger.info("Silent TTS: this run synthesizes nothing")
        tts = SilentTTSService()
    else:
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
                # VAD-driven, like the plain VADUserTurnStartStrategy this
                # started as, and for the same reason:
                # TranscriptionUserTurnStartStrategy (the other default) fires
                # on every interim transcript, fragmenting one utterance into
                # several turns with assistant one-character fragments spliced
                # in between. What this adds is a threshold over the bot's own
                # speech, so a cough or a "はい" no longer cuts it off — see the
                # strategy's own docstring.
                start=[BackchannelToleranceUserTurnStartStrategy()],
                # 0.8 rather than the default 0.6: this is the half of the wait
                # a caller can still interrupt, so the 0.8s the VAD gave up above
                # is better spent here. A 1.4s pause mid-sentence was measured in
                # one call out of nine turns, which 1.0s does not cover — the
                # turn is cut and the rest of the sentence arrives as a new one.
                # Acceptable now that the prompt answers a half-heard phone
                # number with "はい" and waits for the rest.
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.8)],
            ),
            # Mute user input from the moment the line connects until the
            # opening greeting has been spoken in full, so neither a false VAD
            # trigger nor a caller who starts talking early can cancel it.
            # Released as soon as that first bot speech finishes, so later turns
            # barge-in normally. closing_mute takes over at the other end.
            user_mute_strategies=[OpeningUserMuteStrategy(), closing_mute],
            # A caller who goes quiet mid-call is usually thinking, not
            # finished, so most of the conversation ignores this entirely —
            # on_user_turn_idle returns without doing anything. The one place
            # it acts is the read-back, where silence means the caller is
            # waiting on the bot rather than the other way round. The closing
            # replaces this with its own, shorter timeout.
            user_idle_timeout=READBACK_SILENCE_SECS,
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
            # Between the LLM and the TTS on purpose: the TTS splits text into
            # sentences before its own text_transforms run, by which point a
            # run of 。 has already become separate chunks to speak.
            CollapseRepeatedPunctuation(),
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
        # Setting the first node is what speaks the greeting: its tts_say
        # pre-action runs on the transition. The node does not respond
        # immediately, so nothing else is said until the caller speaks.
        await flow_manager.initialize(reception_node())

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        # "失礼いたします" appears only in the closing line, so a completed turn
        # carrying it means the conversation is over. Watching the assistant turn
        # (rather than the text sent to the TTS) is what makes this work in both
        # modes: a text-mode eval run skips TTS entirely, so no TTS request is
        # ever made, but the assistant turn still closes.
        content = message.content or ""

        # The offer is made once per call, and this is where "once" is decided.
        # It is read off the turn the bot actually produced rather than left to
        # the prompt, which quoted the line while forbidding it and lost about
        # one turn in five. Moving to a node whose instructions never mention
        # the line leaves nothing to repeat.
        #
        # Only while 応対 is the node in play: the read-back and the closing are
        # elsewhere in the flow, and a stray match there would swap the
        # instructions out from under them.
        if (
            flow_manager.current_node == "reception"
            and not flow_manager.state.get("callback_offered")
            and is_callback_offer(content)
        ):
            logger.info("Callback offered; it will not be offered again on this call")
            flow_manager.state["callback_offered"] = True
            await flow_manager.set_node_from_config(reception_offered_node())
            return

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
        # Fires wherever the caller falls quiet, which is most of the call and
        # almost always fine — they are thinking. Two places it means something.
        nonlocal hanging_up

        if not closing_mute.closing:
            # Waiting on an answer to the read-back. Silence here is the caller
            # not realising it was their turn, so say so — once per read-back,
            # which is what the flag counts.
            if flow_manager.current_node == "callback_confirm" and not flow_manager.state.get(
                "readback_nudged"
            ):
                logger.info("No reply to the read-back; asking once")
                flow_manager.state["readback_nudged"] = True
                await worker.queue_frames([TTSSpeakFrame(READBACK_NUDGE_LINE)])
            # Anywhere else, let the caller think.
            return

        # Past the closing line: the call is over bar the hang-up.
        #
        # Once, though: the goodbye below is itself bot speech, so the caller goes
        # quiet after it too and the timer arms again. Whether that second firing
        # gets as far as speaking depends on how long the pipeline takes to end —
        # a race that would have the bot say goodbye twice.
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
