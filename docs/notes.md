# 知見

作業の中で分かったこと。CLAUDE.md から読み込まれる。

## eval の運用

- `pipecat eval suite` の `-s` は複数指定すると最後の1つしか効かない。複数のシナリオを
  指定したいときは1本ずつ実行する
- 全件1回で通ったからといって安定しているとは限らない。
  `callback_offer_not_repeated` は8回ずつ回して 6/8 / 7/8 だったのに、
  その前の全件実行2回は連続で通っていた。揺れを疑うシナリオは -r で回す
- 全件を1回流すだけでも OpenAI の TPM 上限（gpt-4.1 で30k）に当たるようになった
  （シナリオ11本・約3分20秒の時点）。毎回ちがうシナリオが犠牲になるので、
  落ちた1本を `-s` で単独実行して判定の失敗と切り分ける
- 全件の `-r 3` はさらに上限に当たる。ボットが応答を返せず
  「no response text yet」でタイムアウトするため判定の失敗と見分けづらい。ボット側の
  ログに「Rate limit reached」が出ていればこれ
- 判定に迷いの要らないこと（定型文言、固有名詞、別れの挨拶）は judge ではなく
  `text_contains` で見る。gpt-4o でも誤判定する。「佐藤様、お電話番号は…」で
  始まる返答を「電話番号しか復唱していない」と判定した例が何度もある
- 判定文に出てくる言葉が、ボットの定型文言の中にも入っていると誤判定する。
  終話文言そのもの（「…お電話いただき、ありがとうございました」）を
  「お礼を前置きしていないか」と尋ねたら「前置きしている」と答えた。
  「返答が『さくら歯科クリニックに』で始まっている」のように、何で始まるかを問う
- 「○○と言っていない」という否定の判定文は judge に渡さない。返答単体ではなく
  会話全体に対して判定されるため、以前のターンで言っていると落ちる。
  言うべきことを `text_contains` で見るか、後続のターンの挙動で担保する
- 割り込みは eval では再現できない。テキストモードには音声がなく、audio モードでも
  ボットの発話にかぶせる発話は作れない。strategy にフレームを直接流す pytest で見る
- 期待は「待ち始めたあとに届いたイベント」しか拾わない。切電前後のようにイベントが
  数秒遅れて出る場面では、ターンを分けずに同じターンで順に検証する
- 関数呼び出しだけで満たされるターンの次は `send_after: {delay_ms: 3000}` を置く。
  期待が満たされた瞬間に次のターンが送られるため、遷移や pre_action の読み上げに
  かぶってしまう
- `tts_response` は audio モード専用。`tts_say` で読み上げた台詞はテキストモードの
  eval では観測できないので、文面は pytest で、流れは function_call の連鎖で見る
- テキストモードの発話は、文字起こしではなく `LLMMessagesAppendFrame` として
  RTVI から直接コンテキストに入る（`rtvi/processor.py` の `_handle_send_text`）。
  そのため user mute strategy では止まらず、`run_immediately` で `interrupt_bot()` も
  呼ばれる。ミュートの挙動を eval で見たいときは audio モードを使う

## TTS に渡す前のテキスト整形

- TTS サービスの `text_transforms` は、文に分割された**あと**に1チャンクずつ
  走る。「はい。。。。」はこの時点で「はい。」「。」「。」に割れているので、
  重複の圧縮には使えない。LLM と TTS の間に FrameProcessor を挟む
- LLM の出力はトークンごとに届くので、連続する記号がフレームをまたぐ。
  直前に出した末尾の文字を覚えておいて、次のフレームの先頭がそれと同じなら削る
- `InterruptionFrame` は SystemFrame で順序キューを飛び越す。`run_test` に
  他のフレームと並べて渡しても送った順には届かない。順序に依存する検証は
  ControlFrame（`LLMFullResponseStartFrame` など）で書く

## eval での TTS

- テキストモードでも `tts_say` は実際の TTS を通る。ハーネスが接続時に送る
  `skip_tts` は LLM の出力にしか効かない。固定文言を tts_say に移すたび
  Cartesia の消費が増える
- 偽の TTS に差し替えるときは、**無音の音声フレームを1つ返す**。何も返さないと
  `TTSService` が「リクエストを受けて何も喋らないプロバイダ」とみなし、
  毎回エラーを push したうえ3回で service を unusable にして以降を黙らせる
  （`max_consecutive_zero_audio_contexts`）。`TTSSettings(model=None,
  voice=None, language=None)` も渡す。未設定だと起動時にフィールドごとに
  ERROR が出る
- シナリオごとにボットの作り方を変えたいときは manifest の `runner_body`。
  `spawn` は manifest 全体で1つだが、`bot` と `runner_body` は suite の
  エントリごとに指定でき、JSON が `runner_args.body` に入る。サービスの
  組み立ては接続前なので、eval transport の接続時フラグでは間に合わない

## audio モードの eval

- 全件実行に含まれる。1本あたり約70秒、Cartesia を約205文字消費する
  （テキストモードだけ回したいときは `-s` でシナリオを指定する）
- 発信者音声は `sample_rate: 8000` の指定が必須。ボットのパイプラインが Twilio 向けに
  8kHz 固定のため、既定の16kHzのままだとボットの STT が何も返さない。
  µ-law は再現されない
- 判定文は文字起こしの表記に依存させない。8kHz では固有名詞が崩れ（「さくら歯科」が
  「桜市科」）、1桁ずつ読んだ数字も数字列として書き戻される。数字の正確さも判定させない
- Whisper は `base` を使う。`large-v3-turbo` は1発話の文字起こしに約4秒かかって
  ハーネスを塞ぎ、発信者の発話を切電（3秒）前に差し込めない
- `tts_say` の読み上げは、テキストモードの eval でも Cartesia を通る。復唱のある
  シナリオは TTS の障害に影響される（読み上げに失敗しても流れは止まらないが、
  再試行の遅延で全件実行がタイムアウトに連鎖することがある）
- `server rejected WebSocket connection: HTTP 402` は Cartesia のクレジット切れ。
  ネットワーク不調と紛らわしいので、ログで 402 を確認する。全件実行1回で
  TTS を約1000文字使う（復唱と終話の読み上げ、audio シナリオのボット音声）
- Cartesia の接続が不安定なことがある。ボットが起動時に再接続を繰り返して Bot ready に
  到達しない場合は、残っている `bot_phone.py` と `pipecat eval` のプロセスを落とし、
  数分おいてから実行し直す

## tts_say と eval の観測

- `tts_say` で読み上げた台詞は、テキストモードの eval からは一切観測できない。
  ハーネスが `bot-tts-text` を `bot_audio` が False のとき捨てるため
  （`pipecat/evals/harness.py` の `_segment_event` 周辺）。`response` は
  テキストモードでは `llm_response` に読み替えられるので、LLM が書いていない文は
  どのイベントにも出てこない。文面は pytest、流れは audio モードで見る
- そのため、定型文言を `tts_say` に移すと、その文を見ているテキストシナリオが
  すべて落ちる。挨拶を固定文にしたときは14本の冒頭の判定を外した

## Pipecat Flows

- Flows は pipecat 1.8.1 に同梱されている（`pipecat.flows`）。別パッケージの
  `pipecat-ai-flows` は不要で、入れると pipecat 自身が警告する。しかも最新の
  1.4.0 は `pipecat-ai<1.5.0` を要求するので入れてはいけない
- プロバイダ別のアダプタはない（universal context 前提）。OpenAI Responses でそのまま動く
- `FlowManager(context_aggregator=...)` には `LLMContextAggregatorPair`
  そのものを渡す（`.user()` と `.assistant()` を呼ぶため）。パイプラインには
  分解した2つを渡す
- 直接関数は第1引数が `flow_manager`、戻り値は `(結果, 次のノード)`。
  `result_callback` は使わない
- ノードの `role_message` が system instruction になる。次のノードが指定しない限り
  そのまま残るので、LLM サービス側の `system_instruction` は空にしておく
- ノード遷移を伴う関数呼び出しの返答は、関数呼び出しの前後で2つに分かれる。
  eval の `eval:` 判定は最初の断片で「いいえ」が出た時点で失敗するため、
  遷移時は何も喋らせない（プロンプトで「関数を呼ぶだけ」と明示する）
- 既定の context strategy は APPEND なので、遷移しても前のノードの task messages は
  コンテキストに残る。ノードごとの手順は task_messages ではなく role_message に
  入れる。developer メッセージとしてコンテキストに積むと指示が守られにくく
  （復唱と同じターンで記録してしまう）、積み重なってトークンも増える。
  role_message は遷移ごとに置き換わるので、task_messages は1行の短い指示だけにする
- Flows の関数は既定で `cancel_on_interruption=False`、つまり非同期ツール扱いになる。
  ASYNC TOOLS の説明文と started/final のメッセージがコンテキストに毎ターン残るので、
  すぐ返る関数には `@flows_tool_options(cancel_on_interruption=True)` を付ける
- 「言ったあとで待つ」をプロンプトで守らせるのは無理だった（復唱と同じターンで
  記録してしまう。待たせる指示を強めると別のターンが崩れる）。決め打ちの台詞は
  `pre_actions` の `tts_say` で読み上げ、そのノードを `respond_immediately: False`
  にすると、相手が話すまで LLM が動かないので構造的に待つ。台詞の文面も固定できる
- 次の行動が関数呼び出しだけのノードでは、値を引数で受けずに `flow_manager.state`
  に持たせる。訂正のあとにモデルが古い値を渡してくる余地がなくなる
- モデルは「関数を呼ばずに自分で答える」方に逃げる。判定をコードに移したときは
  「自分で判断して止めず、聞き取れたものをそのまま渡す」「関数を呼ばずに
  終話の挨拶やお礼を言ってはいけない」まで書く。聞き直しのときは
  「渡すのは新しく聞き取れた方」と例つきで書く（古い値を渡してくる）
- 番号を分けて言われたとき、モデルは「新しい断片」と「ここまでの全部」の
  どちらを関数に渡すか一定しない。プロンプトで片方に寄せても長さが変わると
  崩れる。`merge_phone_number` のように、どちらを渡されても同じ結果になる形で
  コードで組み立てる。訂正・桁数やり直しのときは state を空にしてから始める
- 「一度しか言ってはいけない文言」をプロンプトで縛るのは無理（2割ほど破る）。
  言い終わった時点で `set_node_from_config` で別ノードに移し、繰り返しを誘う
  指示そのものを消す。効いたのは文言を消すことではなく、断り文言に付いていた
  「続けて○○と伝えてください」という節を消すこと。文言まで全部消すと、
  聞き返されたときに言い直せなくなった（eval 0/3）。文言は「聞き返された
  ときだけ言い直す」の一文にだけ残す。遷移先は `respond_immediately: False`
  （直前に喋り終えたばかりなので）
- ノードを手で移すときは `flow_manager.current_node` で今いるノードを確かめる。
  文言の検出だけで移ると、復唱や終話など別のノードにいるときに
  instruction を差し替えてしまう
- 同じ入力に対してモデルが「関数に渡す」か「自分で答える」かを選ぶ場面では、
  どちらでも発信者の体験が同じになるように作る。eval はどちらか一方を
  前提にせず、最終状態（最後に関数に渡された値など）だけを見る。
  経路ごとの振り分けはコードに寄せて pytest で見る
- LLM がテキストも関数呼び出しもない空の応答を返すことがある。1回の失敗では
  挙動と判断せず、同じシナリオを単独で回して切り分ける
- ContextStrategy.RESET は LLMMessagesUpdateFrame でコンテキストを丸ごと置き換える。
  会話履歴も消えるため、通話記録（終了時に context から保存）が前半を失う。使わない

## 通話が終わったときの後始末

- `on_pipeline_finished` は終わり方によらず1回だけ走る。発信者が切った場合
  （CancelFrame）とボットが切った場合（EndFrame）の両方を1箇所で拾える
- テキストモードの eval ではボットが流用のため生かされたままで、パイプラインは
  終了しない。終了時の処理を確かめたいときは audio モードのシナリオを使う

## 最初の挨拶

- `FirstSpeechUserMuteStrategy` は、ボットが話し**始める前**はミュートしない
  （docstring に明記）。接続から読み上げ開始までの隙間で相手が話すと挨拶が
  取り消される。接続時点からミュートする自前の strategy が要る
- ミュートを解除する条件がボットの発話だけだと、TTS が落ちたときに相手が
  永久にミュートされる。時間の上限を併せて入れる
- `respond_immediately: False` は最初のノード（`FlowManager.initialize`）でも効く。
  `_set_node` が `LLMRunFrame` を送らなくなるだけなので、初期ノードも同じ

## 割り込みの判定

- `MinWordsUserTurnStartStrategy` は日本語では使えない。語数を
  `str.split()` で数えるため、「何時までやっていますか。」が1語になる。
  閾値を1より大きくすると割り込みが一切できなくなる。文字数で数える
  strategy を自前で書く
- start strategy は**リストの先頭から順に**試され、最初に STOP を返したものが
  フレームを消費する（`user_turn_controller.py`）。`VADUserTurnStartStrategy` を
  並べたままにすると VAD が先に発火するので、条件付きにしたいときは
  1つの strategy にまとめる
- 同じターン内で複数回 `trigger_user_turn_started()` を呼んでも害はない。
  コントローラが「Prevent two consecutive user turn starts」で弾く
- VAD の `start_secs` を上げて短い物音を消す手もあるが、ボットが黙っている
  ときの「はい」まで拾えなくなる（復唱の確認が通らない）。ボットが
  喋っている間だけ閾値を上げる

## ターン検出

- 待ち時間と、文中の間に耐えられる時間は同じ値（`stop_secs` + `user_speech_timeout`）。
  2つは直列で、猶予の途中で話し始めると進行が破棄されるため、片方だけ縮める設定はない
- `stop_secs` は STT の p99（Deepgram 0.35秒）より小さく保つ。以上だと最終の文字起こしを
  待つ保険が0秒に潰れ、末尾が欠けたままターンが切れる

## 通話テスト

- 安定したネットワークと静かな場所で、スピーカーを使わずに行う。スピーカーだと
  周りの会話を STT が拾い、会話が崩れる
- 次が同時に出ているときは、ボットの不具合ではなくサーバー側のネットワーク不調を疑う
  - `no audio received while speaking`
  - Deepgram の `Keepalive failed` / websockets の `keepalive ping failed`
  - `TTS context ... completed with no audio`、TTS や LLM の TTFB が数秒から十数秒

## Pipecat の既知の不具合

- 文末の句点が履歴で二重になる。音声そのものには出ないが、LLM が次のターンで
  それを真似るので放っておくと増える（「はい。。。。」まで育ち、TTS に「。。」
  だけの読み上げ依頼が飛んでいた）。LLM と TTS の間で詰めている
  （`text_cleanup.py`）。履歴の二重化自体は直していない

## 設計上の割り切り

- 断り方は検証用に厳しめにしている。実用化時は「確認して折り返す」方向に見直す
