# VRCColliderProbe

VRChat のワールドを OSC で自動歩行し、床コライダーの張り忘れ（床抜け）を見つけて
地図付きで報告するツール。

**作業を始める前に `SPEC.md` を読むこと。** 設計方針・段階ごとの完了条件・未検証の前提がまとまっている。

- 段階 1（OSC 疎通、`tools/osc_probe.py`）と段階 2（反射行動、`tools/walker.py`）は動いた。
  段階 3（地図、`tools/mapper.py`。walker 終了時に自動実行）も動いた。
  段階 4（探索、`tools/explore.py`）も実機でランダムより踏破面積が増えた。
  段階 5（報告、`tools/report.py` → `runs/<日時>/report.html`）まで一通りできた。
  拡張: RelateAnything の注釈・高さ（3D）・リアルタイムのオーバーレイ・どこに何があるか（`SPEC.md` 5 章「拡張」）
- RelateAnything は torch 入りの別環境（`E:\AIProject\RelateAnything\.venv`）で動かす。
  このプロジェクトに torch を入れないこと。`tools/scene_annotate.py` をその venv の python で別プロセス実行する
- 今後の候補: 長時間の走行、床抜けを仕込んだワールドでの確認、VR モードでの確認
  結果は `SPEC.md` 3 章・5 章
- 判定ロジックを VRChat なしで試すときは、偽 VRChat を 19000/19001 で立てて `--send-port 19000 --recv-port 19001`。
  9000 に偽サーバーを立てると本物の VRChat と取り合いになる。偽 VRChat は同じ値を送らない（実機より厳しい）ので、
  「値が変化時しか届かない」ことに依存したバグを見つけやすい
- 偽ワールドでの走行結果（runs/ に出る）は比較が終わったら消す。実機の走行と混ざらないように
- **偽ワールドで walker を動かすときは必ず `--fresh` か `--world-id sim_…` を付ける**（後者は記録を使うテスト用。
  終わったら `worlds/sim_…` と該当の走行を消す）。付けないと、VRChat のログから今いる実際の
  ワールドを読み取り、偽のデータがそのワールドの記録（`worlds/<ワールドID>/knowledge.json`）に混ざる。
  混ざったら `knowledge.json` の `runs` からその走行を消す
- ワールドの記録は `tools/knowledge.py`。VRChat のログからワールド ID と名前だけを取り、インスタンス（ユーザー ID を含む）は保存しない
- 実行は `uv run tools/<script>.py`。対話スクリプトは PowerShell で（Git Bash ではキー入力が取れない）
- 簡易 GUI は `uv run tools/gui.py`（http://127.0.0.1:8790/、標準ライブラリのみ）。walker を子プロセスで起動し、
  停止は停止ファイル（`walker.py --stop-file`）。GUI の「開始」を押すと本物の VRChat が動くので、テストは
  API の `extra`（`--send-port 19000` など許可した引数のみ）で偽ワールドに向けて行う
- `SPEC.md` で「要検証」となっている OSC の仕様は記憶ベース。実機で確かめてから前提にする
- 画面キャプチャ周りは `E:\AIProject\RelateAnything\tools\game_scene.py` から流用できる
- 会話・コメント・ドキュメントは日本語
