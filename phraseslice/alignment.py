"""単語IDで指定した英語フレーズの切断位置をローカルの音声から求める。"""

import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly
from scipy.special import log_softmax

from .wav import validated_ranges
from .transcript_format import load_transcript


MODEL = Path(sys.prefix) / "share/phraseslice/model_quantized.onnx"
MODEL_SHA256 = "cd5040c147381580ed73258143dd8e0c28e800a09e74ee42ee2b3e8cb4d760a3"
# Model vocabulary, including its four special symbols.
VOCAB = "____|ETAONIHSRDLUMWCFGYPBVK'XJQZ"
MODEL_RATE = 16000


def pcm_samples(source, start, stop):
    source.setpos(start)
    width = source.getsampwidth()
    data = source.readframes(stop - start)
    if len(data) != (stop - start) * width * source.getnchannels():
        raise ValueError("入力WAVの音声データが不足しています")
    if width == 3:
        octets = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        values = (values ^ 0x800000) - 0x800000
    elif width == 1:
        values = np.frombuffer(data, np.uint8).astype(np.int16) - 128
    else:
        values = np.frombuffer(data, dtype=f"<i{width}")
    return values.astype(np.float32).reshape(-1, source.getnchannels()) / (1 << (8 * width - 1))


def word_tokens(text):
    text = unicodedata.normalize("NFKD", text.upper().replace("’", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.strip(" \t\n.,!?:;\"“”()[]-")
    if not re.fullmatch("[A-Z']+(?:[- ][A-Z']+)*", text):
        return []
    return [VOCAB.index(c) for c in text.replace("-", "|").replace(" ", "|")]


def ctc_alignment(log_probs, labels):
    """Best CTC path for a short word sequence, allowing unused audio at both ends."""
    states = np.zeros(2 * len(labels) + 1, dtype=int)
    states[1::2] = labels
    scores = np.full(len(states), -np.inf)
    scores[0] = 0
    skips = (states[2:] != 0) & (states[2:] != states[:-2])
    back = np.zeros((len(log_probs), len(states)), dtype=np.uint8)
    best_end, best_score = 0, -np.inf
    for t, emission in enumerate(log_probs):
        advance = np.r_[-np.inf, scores[:-1]]
        skip = np.r_[-np.inf, -np.inf, np.where(skips, scores[:-2], -np.inf)]
        best = np.maximum(scores, advance)
        back[t] = np.where(skip > best, 2, advance > scores)
        scores = np.maximum(best, skip) + emission[states]
        # Compare against emitting blank so that real speech gives evidence
        # for the matching characters, regardless of where the crop starts.
        scores -= emission[0]
        scores[0] = 0
        if scores[-2] > best_score:
            best_end, best_score = t, scores[-2]
    if not np.isfinite(best_score):
        return None
    path = np.zeros(len(log_probs), dtype=int)
    state = len(states) - 2
    for t in range(best_end, -1, -1):
        if state == 0:
            break
        path[t] = state
        state -= int(back[t, state])
    return path


def quiet_frame(audio, rate, low, high, target):
    """Search near target within the permitted interval, in local sample frames."""
    hop = max(1, round(rate * .005))
    radius = max(1, round(rate * .0225))
    centers = np.arange(max(low, target - round(.15 * rate)), min(high, target + round(.20 * rate)) + 1, hop)
    def cost(center):
        window = audio[max(0, center-radius):center+radius+1]
        power = np.mean(window**2, axis=0).max()
        return 10 * np.log10(max(power, 10**-6.5)) + 12 * abs(center-target) / rate
    best = min(centers, key=cost) if len(centers) else target
    frames = np.arange(max(1, low, best-round(.002*rate)), min(high, best+round(.002*rate)) + 1)
    level = np.maximum(np.abs(audio[frames]), np.abs(audio[frames-1])).max(axis=1)
    return int(frames[np.argmin(level)])


def refine_boundary(source, words, end_id, frame, session, tail_seconds):
    rate = source.getframerate()
    left, right = words[end_id-1:end_id+1]
    # Keep context on both sides so repeated short words can be distinguished.
    context = words[max(0, end_id-3):end_id+3]
    first = max(0, math.floor((context[0]["start"] - .65) * rate))
    stop = min(source.getnframes(), math.ceil((context[-1]["end"] + .65) * rate))
    if stop-first > 24*rate or stop-first < .05*rate:
        return frame, "照合に必要な連続した音声区間を取得できない"
    labels, spans = [], {}
    for word in context:
        symbols = word_tokens(word["word"])
        if not symbols:
            if word["id"] in (left["id"], right["id"]):
                return frame, "境界の単語が英語モデルの語彙外"
            continue
        if labels and not word["word"].startswith("-"):
            labels.append(VOCAB.index("|"))
        start = len(labels)
        labels.extend(symbols)
        spans[word["id"]] = (start, len(labels)-1)

    audio = pcm_samples(source, first, stop)
    divisor = math.gcd(rate, MODEL_RATE)
    mono = resample_poly(audio.mean(axis=1), MODEL_RATE // divisor, rate // divisor)
    mono = (mono - mono.mean()) / np.sqrt(mono.var() + 1e-7)
    logits = session.run(["logits"], {"input_values": mono[None].astype(np.float32)})[0][0]
    probs = log_softmax(logits, axis=1)
    path = ctc_alignment(probs, labels)
    if path is None:
        return frame, "単語列を音声に照合できない"
    edges = []
    for word in (left, right):
        a, b = spans[word["id"]]
        times = np.flatnonzero((path >= 2*a+1) & (path <= 2*b+1) & (path % 2 == 1))
        tokens = np.array(labels)[(path[times]-1)//2]
        if not len(times) or np.exp(probs[times, tokens]).mean() < .50:
            return frame, "境界の単語の照合確度が低い"
        # The encoder has a 320-sample stride and a 400-sample receptive field.
        edges.append((times[0]*.02+.0025, times[-1]*.02+.0225))
    end, onset = edges[0][1], edges[1][0]
    gap = onset-end
    if gap <= 0 or max(abs(first/rate+end-left["end"]), abs(first/rate+onset-right["start"])) > 1.25:
        return frame, "前後の単語の位置を確定できない"
    low = max(1, math.ceil((end + min(tail_seconds, .44*gap))*rate))
    high = min(len(audio)-1, math.floor((onset - min(.13, .47*gap))*rate))
    if low > high:
        return frame, "前後の発話に切り出せる間隔がない"
    if low <= frame-first <= high:
        return frame, None
    target = min(high, max(low, frame-first))
    return first + quiet_frame(audio, rate, low, high, target), None


def build_plan(transcript_path, groups_path, output, tail_ms=160):
    if tail_ms < 0:
        raise ValueError("--tail-ms は0以上で指定してください")
    transcript = load_transcript(transcript_path)
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    if transcript["language"] != "en":
        raise ValueError("境界調整には英語の transcript.json が必要です")
    words = [word for segment in transcript["segments"] for word in segment["words"]]

    existing = isinstance(groups, dict)
    source_path = transcript_path.parent / transcript["source"]
    if existing and (groups_path.parent / groups["source"]).resolve() != source_path.resolve():
        raise ValueError("フレーズと文字起こしの入力WAVが一致しません")
    ends = [p["last_word_id"] for p in groups["phrases"]] if existing else groups
    if not isinstance(ends, list) or not ends or any(type(k) is not int for k in ends):
        raise ValueError("フレーズ末尾の単語IDを整数の配列で指定してください")
    if ends != sorted(set(ends)) or ends[0] < 1 or ends[-1] != len(words):
        raise ValueError("末尾IDは昇順で、最後の単語IDまで含めてください")
    if output.resolve() in (source_path.resolve(), transcript_path.resolve(), groups_path.resolve()):
        raise ValueError("入力とは別のJSONファイルを出力先に指定してください")

    with source_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if existing and groups["source_sha256"] != digest:
            raise ValueError("フレーズと入力WAVのSHA-256が一致しません")
        stream.seek(0)
        with wave.open(stream, "rb") as source:
            sr, total = source.getframerate(), source.getnframes()
            if not all(0 <= w["start"] <= w["end"] <= total/sr+.02 for w in words):
                raise ValueError("入力WAVの範囲外の単語時刻があります")
            if existing:
                validated_ranges(groups, source)
                cuts = [row["end_frame"] for row in groups["phrases"]]
            else:
                cuts = [round((words[k-1]["end"]+words[k]["start"])*sr/2) for k in ends[:-1]] + [total]
            if any(a >= b for a, b in zip([0]+cuts, cuts)):
                raise ValueError("単語時刻から昇順の初期境界を作成できません")
            if len(cuts) > 1:
                with MODEL.open("rb") as f:
                    if hashlib.file_digest(f, "sha256").hexdigest() != MODEL_SHA256:
                        raise ValueError("照合モデルのSHA-256が一致しません")
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                ort.disable_telemetry_events()
                session = ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])
            plan = dict(groups) if existing else {}
            plan.update(source=os.path.relpath(source_path, output.parent), source_sha256=digest,
                        sample_rate_hz=sr, channels=source.getnchannels(), total_frames=total,
                        frame_interval="[start_frame, end_frame)", phrases=[])
            start_id, start_frame = 1, 0
            for i, (end_id, frame) in enumerate(zip(ends, cuts)):
                text = "".join(w["word"] for w in words[start_id-1:end_id]).strip()
                row = dict(groups["phrases"][i]) if existing else {}
                if existing and (row["first_word_id"] != start_id or row["text"] != text):
                    raise ValueError(f"フレーズ {i+1} と文字起こしの単語が一致しません")
                note = None
                if i+1 < len(cuts):
                    candidate, note = refine_boundary(source, words, end_id, frame, session, tail_ms / 1000)
                    if start_frame < candidate < cuts[i+1]:
                        frame = candidate
                    else:
                        note = "隣のフレーズと境界が交差する"
                row.update(id=i+1, first_word_id=start_id, last_word_id=end_id, text=text,
                           start_frame=start_frame, end_frame=frame, start_seconds=round(start_frame/sr,6),
                           end_seconds=round(frame/sr,6), first_word_start_seconds=words[start_id-1]["start"],
                           last_word_end_seconds=words[end_id-1]["end"])
                row.pop("alignment_review", None)
                if note:
                    row["alignment_review"] = note
                plan["phrases"].append(row)
                start_id, start_frame = end_id+1, frame
            validated_ranges(plan, source)
    with tempfile.NamedTemporaryFile(dir=output.parent, mode="w", encoding="utf-8", delete=False) as f:
        tmp = Path(f.name)
    try:
        tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return plan


def print_review(plan):
    review = [str(row["id"]) for row in plan["phrases"] if row.get("alignment_review")]
    if review:
        print("境界を維持・要試聴: " + ", ".join(review))
