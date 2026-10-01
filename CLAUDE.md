@AGENTS.md
## 作業ルール
- VAD・ターン検出・STT/TTS/LLMモデルの設定には触れない
- 変更方針を説明してから編集し、差分を表示
- 変更後は server/ で uv run pytest tests と uv run --env-file .env pipecat eval suite evals/manifest.yaml を実行
- eval は通常は全件を1回ずつ実行し、すべて合格すること
- -r 3 は今回追加・修正したシナリオだけに対して実行する（-s で指定）。全件の -r 3 は OpenAI の TPM 上限（30k）に当たるため実行しない
- eval の失敗は原因で分けて報告する。判定（judge や text_contains）による失敗と、TPM 上限による失敗（応答が返らず no response text yet になる。ボットのログに Rate limit reached が出る）は区別し、後者は挙動の失敗として扱わない

## Git ルール
- main ブランチには直接コミットしない。作業は feature/fix ブランチで行う
- pytest と evals が全件合格したらコミットしてよい（1修正 = 1コミット）
- テストが通らない状態ではコミットしない
- プッシュは私の承認を得てから行う
- マージはしない（私が通話テスト後に PR 上で行う）