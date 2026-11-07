import argparse
import wave
from pathlib import Path

from phraseslice.wav import split_wav, validated_ranges


def main():
    parser = argparse.ArgumentParser(description="フレーズの境界に従ってPCM WAVを分割する")
    parser.add_argument("plan", type=Path, nargs="?", default=Path("phrases.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("clips"), help="新規出力ディレクトリ")
    args = parser.parse_args()
    try:
        count = split_wav(args.plan, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError, EOFError, wave.Error) as error:
        parser.exit(1, f"分割できませんでした: {error}\n")
    print(f"{args.output_dir}: {count} 個のWAVを書き出しました")


if __name__ == "__main__":
    main()
