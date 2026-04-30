# Codexモデルエラー修正レポート

## 問題

Codexで`gpt-5.2-codex`モデルを使用しようとすると、以下のエラーが発生していました：

```
[codex] message: {"type":"error","status":400,"error":{"type":"invalid_request_error","message":"The 'gpt-5.2-codex' model is not supported when using Codex with a ChatGPT account."}}
```

## 原因

ChatGPTアカウントでは`gpt-5.2-codex`モデルがサポートされていません。

## サポートされているモデル（2026年4月28日現在）

- `gpt-5.5` (最新・推奨)
- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.3-codex`
- `gpt-5.3-codex-spark`
- `gpt-5.2`

## 修正内容

### 1. セッション設定ファイル (`data/sessions.json`)

すべての`gpt-5.2-codex`を`gpt-5.5`に変更：
- グローバルスコープ
- `__admin__:1465943297657798789`スコープ
- `__admin__:admin:standard:1465943297657798789`スコープ
- `__bot_maintenance__:admin:1465943297657798789`スコープ

### 2. CodexWorkerクラス (`bot/agents/codex_worker.py`)

`_FALLBACK_MODELS`リストからサポートされていないモデルを削除：
- ❌ 削除: `gpt-5.2-codex`, `gpt-5.1-codex-max`, `gpt-5.1-codex`, `gpt-5.1`, `gpt-5-codex`, `gpt-5-codex-mini`
- ✅ 保持: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.2`

### 3. テストファイル (`tests/test_session_manager.py`)

テストコード内の`gpt-5.2-codex`を`gpt-5.5`に変更（4箇所）

## テスト結果

✅ すべてのsession_managerテストがパス（18 passed）
✅ すべてのcodex関連テストがパス（21 passed）

## 確認コマンド

```bash
# 利用可能なモデル一覧を確認
python -c "from bot.agents.codex_worker import CodexWorker; w = CodexWorker(); import asyncio; print('\n'.join(asyncio.run(w.get_models())))"

# sessions.jsonの全モデル設定を確認
cat data/sessions.json | jq -r '.. | .model? | select(. != null)' | sort | uniq -c
```

## 結論

修正により、Codexは以下のモデルのみを使用します：
- デフォルト: `gpt-5.5`
- フォールバック: サポートされているモデルのみ

エラー「The 'gpt-5.2-codex' model is not supported」は解消されるはずです。
