import argparse
import wave
from pathlib import Path

from phraseslice.alignment import build_plan, print_review


def main():
    parser = argparse.ArgumentParser(description="英語の語尾と次の語頭に合わせてWAV分割位置を決める")
    parser.add_argument("groups", type=Path, help="末尾単語IDのJSON配列、または既存の phrases.json")
    parser.add_argument("--transcript", type=Path, default=Path("transcript.json"))
    parser.add_argument("--output", type=Path, default=Path("phrases.refined.json"))
    parser.add_argument("--tail-ms", type=int, default=160, help="語尾に確保する余裕の目安（ミリ秒、既定160）")
    args = parser.parse_args()
    try:
        plan = build_plan(args.transcript, args.groups, args.output, args.tail_ms)
    except (OSError, ValueError, KeyError, TypeError, EOFError, wave.Error) as error:
        parser.exit(1, f"境界を調整できませんでした: {error}\n")
    print(f"{args.output}: {len(plan['phrases'])} フレーズ")
    print_review(plan)


if __name__ == "__main__":
    main()
