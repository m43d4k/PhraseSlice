import argparse
import json
import os
import tempfile
from pathlib import Path

from phraseslice.menu import DIST, SOURCE, choose_source
from phraseslice.transcript_format import VERSION, validate_transcript


def main():
    parser = argparse.ArgumentParser(description="WAVの文字起こしと単語時刻をJSONに保存する")
    parser.add_argument("wav", nargs="?", type=Path, help="省略すると source/ のWAVから選択")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    parser.add_argument("--output", type=Path, help="文字起こしの保存先")
    args = parser.parse_args()
    if args.wav is None:
        try:
            args.wav = choose_source()
        except (OSError, ValueError) as error:
            parser.error(str(error))
        folder = DIST / args.wav.relative_to(SOURCE).with_suffix("")
        args.output = args.output or folder / "transcript.json"
    else:
        args.output = args.output or Path("transcript.json")
    if not args.wav.is_file():
        parser.error(f"WAVが見つかりません: {args.wav}")
    if args.output.resolve() == args.wav.resolve():
        parser.error("入力と出力に同じファイルは指定できません")

    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(args.wav), path_or_hf_repo=args.model, word_timestamps=True, verbose=False
    )
    segments = []
    next_id = 1
    for segment in result["segments"]:
        words = []
        for word in segment["words"]:
            entry = {"id": next_id, "word": word["word"], "start": word["start"], "end": word["end"]}
            if "probability" in word:
                entry["probability"] = word["probability"]
            words.append(entry)
            next_id += 1
        segments.append({
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "words": words,
        })
    transcript = {
        "schema_version": VERSION,
        "source": os.path.relpath(args.wav, args.output.parent),
        "language": result["language"],
        "time_unit": "seconds",
        "text": result["text"].strip(),
        "segments": segments,
    }
    validate_transcript(transcript)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as staging:
        path = Path(staging) / "transcript.json"
        with path.open("w", encoding="utf-8") as temporary:
            json.dump(transcript, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write("\n")
        path.replace(args.output)
    print(f"{args.output}: {len(segments)} 区間、{next_id - 1} 語")


if __name__ == "__main__":
    main()
