import argparse
import wave
from pathlib import Path

from phraseslice.alignment import build_plan, print_review
from phraseslice.menu import choose, choose_phrase_ends
from phraseslice.wav import check_output_dir, write_clips


def main():
    parser = argparse.ArgumentParser(description="フレーズの切断位置を決め、分割計画とWAVを保存する")
    parser.add_argument("groups", nargs="?", type=Path, help="末尾単語IDの配列、または分割計画。省略すると dist/ から選択")
    parser.add_argument("--transcript", type=Path, help="照合する文字起こし")
    parser.add_argument("--plan-output", type=Path, help="保存する分割計画")
    parser.add_argument("--output-dir", type=Path, help="WAVを保存する新規ディレクトリ")
    parser.add_argument("--tail-ms", type=int, help="語尾の余裕の目安（ミリ秒）")
    args = parser.parse_args()
    try:
        folder = Path(".")
        interactive = args.groups is None
        if interactive:
            args.groups = choose_phrase_ends()
            folder = args.groups.parent
        args.transcript = args.transcript or folder / "transcript.json"
        args.plan_output = args.plan_output or folder / "phrases.json"
        args.output_dir = args.output_dir or folder / "clips"
        check_output_dir(args.output_dir)
        if args.plan_output.resolve() == args.output_dir.resolve():
            raise ValueError("分割計画とWAVの出力ディレクトリには別のパスを指定してください")
        if args.tail_ms is None:
            args.tail_ms = 160
            if interactive:
                args.tail_ms = choose("語尾に残す余裕の目安を選んでください", [
                    ("短め（80 ms）", 80), ("標準（160 ms）", 160), ("長め（250 ms）", 250),
                ], default=2)
        plan = build_plan(args.transcript, args.groups, args.plan_output, args.tail_ms)
        print(f"分割計画を保存しました: {args.plan_output}")
        count = write_clips(plan, args.plan_output.parent, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError, EOFError, wave.Error) as error:
        parser.exit(1, f"分割できませんでした: {error}\n")
    print(f"{args.output_dir}: {count} 個のWAVを書き出しました")
    print_review(plan)


if __name__ == "__main__":
    main()
