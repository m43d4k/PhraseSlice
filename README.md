# PhraseSlice

スピーチを録音したWAVを、発話の意味と間に合わせてフレーズごとに分割する。音声に合わせた切断位置の調整は英語のPCM WAVに対応する。

## セットアップ

文字起こしにはMetal対応のMac、Python 3.12、`uv`、`ffmpeg` が必要。以下のコマンドはリポジトリ直下で実行する。

プロジェクト直下に `.venv` がなければ作成する。

```sh
uv venv --python 3.12
```

文字起こしと英語の音声照合に必要なパッケージを入れ、照合用モデルをダウンロードする。

```sh
uv pip install mlx-whisper jsonschema onnxruntime numpy scipy
model_dir="$(uv run python -c 'import sys; print(sys.prefix)')/share/phraseslice"
mkdir -p "$model_dir"
curl -fL --retry 2 -o "$model_dir/model_quantized.onnx" \
  https://huggingface.co/Xenova/wav2vec2-base-960h/resolve/a19f851b3d42865797e410752b4c570c871e4825/onnx/model_quantized.onnx
```

## 使い方

### 1. 文字起こし

入力WAVを `source/` に置いて実行し、録音名の一覧から番号を選ぶ。サブフォルダのWAVも選べる。初回実行時にMLX Whisperの文字起こしモデルをダウンロードする。

```sh
uv run transcribe.py
```

`source/speech.wav` を選ぶと、各単語のIDと開始・終了時刻が `dist/speech/transcript.json` に保存される。`source/talks/speech.wav` の場合は `dist/talks/speech/` が出力先になる。

### 2. フレーズ選定

このリポジトリを開いたCodexに、次のように依頼する。

> フレーズ選定をしてください。

Codexが `dist/` 以下の `transcript.json` を探し、フォルダ名で候補を示す。番号を答えると、その録音の `phrase-ends.json` が同じフォルダに保存される。

選定は、[専用設定](.codex/agents/phrase_selector.toml) のモデルと推論レベルを使う子エージェント1体に委譲する。会話履歴の引き継ぎと親による選定の重複を省き、親子合計のトークン消費を抑える。初期設定は暫定候補で、録音での品質と消費量は未比較。[モデルの採用・更新規則](docs/phrase-selection.md#モデルの採用と更新) に従って見直す。

### 3. WAVの分割

次のコマンドで、`dist/` 以下に `phrase-ends.json` がある録音をフォルダ名から選ぶ。続けて語尾の余裕を短め（80 ms）、標準（160 ms）、長め（250 ms）から選ぶ。余裕の選択ではEnterで標準を使える。

```sh
uv run slice_phrases.py
```

選んだ録音のフォルダに分割計画 `phrases.json` と `clips/001.wav` 以降の音声が保存される。「要試聴」と表示された番号は、そのクリップの語尾と次のクリップの語頭を確認する。すでに `clips/` がある場合は、次の手順で別の出力先を指定する。

## 語尾の調整

フレーズの区切りが合っていて語尾が短く聞こえる場合は、同じ `phrases.json` を使って再調整できる。Codexへの再依頼は不要。次の例では、語尾に残す余裕の目安を160 msから250 msに増やす。次の語がすぐに始まる箇所では、実際の余裕は短くなる。

```sh
uv run slice_phrases.py dist/speech/phrases.json --transcript dist/speech/transcript.json \
  --tail-ms 250 --plan-output dist/speech/phrases.refined.json --output-dir dist/speech/clips-tail250
```

`dist/speech/clips-tail250/` にある同じ番号のクリップを元の書き出しと聴き比べられる。保存済みの計画どおりに再度書き出す場合は、`uv run split_wav.py dist/speech/phrases.json --output-dir dist/speech/clips-copy` を使う。

## 文字起こしのオプション

WAVのパスを直接渡す従来の使い方（例 `uv run transcribe.py source/speech.wav`）では、既定の出力先はカレントディレクトリの `transcript.json`。保存先は `--output`、手元のMLX Whisperモデルは `--model` で指定できる。`slice_phrases.py` も末尾IDのファイルを直接渡すと対話を省略できる。この場合、文字起こしと出力先の既定値はカレントディレクトリ基準になる。

## ファイル仕様

| ファイル | 作成元 | 内容 |
| --- | --- | --- |
| `transcript.json` | `transcribe.py` | 全文と単語ごとのID・テキスト・開始秒・終了秒。IDは録音全体で1から順に付き、時刻はWAVの先頭から数える。[スキーマと出力例](schemas/transcript.v1.schema.json) |
| `phrase-ends.json` | Codex | 各フレーズの末尾単語IDを並べた配列。Codexはすべての録音で同じ定義を使う。[選定規則](docs/phrase-selection.md#子エージェントの手順)・[出力スキーマ](schemas/phrase-ends.v1.schema.json) |
| `phrases.json` | `slice_phrases.py` | 元のWAVを特定する情報、各フレーズの単語ID範囲・本文・切断位置。各WAVには `start_frame` 以上、`end_frame` 未満の音声が入る。 |

`phrase-ends.json` の要素は1個以上の正の整数で、厳密な昇順に並べる。最後の値は対象の `transcript.json` の最終単語IDと一致する。たとえば全42語を「1〜14語目」「15〜25語目」「26〜42語目」に分けた場合、ファイル全体は次の配列になる。

```json
[14, 25, 42]
```

`transcript.json` の現行形式は `schema_version: 1`。版番号を付ける前に生成したファイルも同じ規則で読み込める。`phrases.refined.json` は再調整した分割計画の保存名で、形式は `phrases.json` と同じ。音声照合で境界を確定できなかった理由は、該当フレーズの `alignment_review` に入る。

## 制作者

[m43d4k](https://github.com/m43d4k)
