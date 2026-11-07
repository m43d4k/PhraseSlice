from pathlib import Path


SOURCE = Path("source")
DIST = Path("dist")


def choose(prompt, options, default=None):
    """表示名と値の組を番号で選ぶ。default は1始まりの番号。"""
    print(prompt)
    for number, (label, _) in enumerate(options, 1):
        print(f"  {number}. {label}")
    hint = f" [Enterで{default}]" if default is not None else ""
    while True:
        try:
            answer = input(f"番号を入力{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("選択を中止しました") from None
        try:
            number = default if not answer else int(answer)
        except ValueError:
            number = None
        if number is not None and 1 <= number <= len(options):
            return options[number - 1][1]
        print(f"1〜{len(options)} の番号を入力してください")


def choose_source():
    options = []
    for wav in sorted(SOURCE.rglob("*.[wW][aA][vV]")):
        if wav.is_file():
            if wav.stem in {".", ".."}:
                raise ValueError(f"出力フォルダ名を作れません: {wav}")
            folder = wav.relative_to(SOURCE).with_suffix("")
            options.append((folder, wav))
    if not options:
        raise ValueError(f"{SOURCE}/ にWAVがありません")
    selected = choose("文字起こしする録音を選んでください", options)
    if sum(wav.with_suffix("") == selected.with_suffix("") for _, wav in options) > 1:
        raise ValueError(f"同じ出力フォルダになるWAVが複数あります。録音名を区別してください: {selected}")
    return selected


def choose_phrase_ends():
    options = [(path.parent.relative_to(DIST), path)
               for path in sorted(DIST.rglob("phrase-ends.json")) if path.is_file()]
    if not options:
        raise ValueError(f"{DIST}/ に phrase-ends.json がありません。先にフレーズ選定を行ってください")
    return choose("分割する録音を選んでください", options)
