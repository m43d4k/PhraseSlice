import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError


VERSION = 1
SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "transcript.v1.schema.json"
VALIDATOR = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))


def validate_interval(item, location):
    start, end = item["start"], item["end"]
    if not (math.isfinite(start) and math.isfinite(end) and start <= end):
        raise ValueError(f"{location}: start と end には開始順の有限な秒数が必要です")


def validate_transcript(data):
    try:
        VALIDATOR.validate(data)
    except ValidationError as error:
        raise ValueError(f"transcript.json {error.json_path}: {error.message}") from None

    next_word_id = 1
    previous_start = 0
    for index, segment in enumerate(data["segments"]):
        location = f"segments[{index}]"
        if segment["id"] != index:
            raise ValueError(f"{location}.id: 期待値は {index} です")
        validate_interval(segment, location)
        for position, word in enumerate(segment["words"]):
            field = f"{location}.words[{position}]"
            if word["id"] != next_word_id:
                raise ValueError(f"{field}.id: 期待値は {next_word_id} です")
            validate_interval(word, field)
            if word["start"] < previous_start:
                raise ValueError(f"{field}.start: 前の単語の開始時刻より前です")
            if "probability" in word and not math.isfinite(word["probability"]):
                raise ValueError(f"{field}.probability: 有限な確率が必要です")
            next_word_id += 1
            previous_start = word["start"]


def load_transcript(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "schema_version" not in data:
        # 版番号の導入前に transcribe.py が出力した形式。
        data["schema_version"] = 1
    validate_transcript(data)
    return data
