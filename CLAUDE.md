@AGENTS.md
## 作業ルール
- VAD・ターン検出・STT/TTS/LLMモデルの設定には触れない
- 変更方針を説明してから編集し、差分を表示
- 変更後は server/ で uv run pytest tests と uv run --env-file .env pipecat eval suite evals/manifest.yaml を実行

## Git ルール
- main ブランチには直接コミットしない。作業は feature/fix ブランチで行う
- pytest と evals が全件合格したらコミットしてよい（1修正 = 1コミット）
- テストが通らない状態ではコミットしない
- プッシュは私の承認を得てから行う
- マージはしない（私が通話テスト後に PR 上で行う）