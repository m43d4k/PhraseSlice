import hashlib
import json
import wave


def validated_ranges(plan, source):
    total = source.getnframes()
    if (plan["sample_rate_hz"], plan["channels"], plan["total_frames"]) != (
        source.getframerate(), source.getnchannels(), total
    ):
        raise ValueError("phrases.json と入力WAVの形式または長さが一致しません")

    ranges = []
    cursor = 0
    for number, phrase in enumerate(plan["phrases"], 1):
        start, end = phrase["start_frame"], phrase["end_frame"]
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"フレーズ {number} の境界には整数のフレーム番号が必要です")
        if phrase["id"] != number or start != cursor or not start < end <= total:
            raise ValueError(f"フレーズ {number} のIDまたはフレーム範囲が不正です")
        ranges.append((start, end))
        cursor = end
    if not ranges or cursor != total:
        raise ValueError("分割範囲が入力WAVの全体をカバーしていません")
    return ranges


def check_output_dir(output):
    if output.exists() or output.is_symlink():
        raise ValueError(f"出力先が既に存在します。新しいディレクトリを指定してください: {output}")


def split_wav(plan_path, output):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return write_clips(plan, plan_path.parent, output)


def write_clips(plan, plan_dir, output):
    check_output_dir(output)
    source_path = plan_dir / plan["source"]
    with source_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != plan["source_sha256"]:
            raise ValueError("入力WAVのSHA-256が phrases.json と一致しません")
        stream.seek(0)
        with wave.open(stream, "rb") as source:
            ranges = validated_ranges(plan, source)
            frame_bytes = source.getnchannels() * source.getsampwidth()
            output.mkdir(parents=True)
            created = []
            try:
                width = max(3, len(str(len(ranges))))
                for number, (start, end) in enumerate(ranges, 1):
                    filename = output / f"{number:0{width}d}.wav"
                    with filename.open("xb") as stream_out, wave.open(stream_out, "wb") as clip:
                        created.append(filename)
                        clip.setparams(source.getparams())
                        remaining = end - start
                        clip.setnframes(remaining)
                        while remaining:
                            count = min(remaining, 65536)
                            data = source.readframes(count)
                            if len(data) != count * frame_bytes:
                                raise ValueError(f"入力WAVの音声データが不足しています: {source_path}")
                            clip.writeframesraw(data)
                            remaining -= count
            except BaseException:
                for filename in created:
                    filename.unlink()
                output.rmdir()
                raise
    return len(ranges)
