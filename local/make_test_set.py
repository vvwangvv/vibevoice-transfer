import argparse
import io
import json
import random
from pathlib import Path

import torchaudio


def _load_audio_from_tar(entry):
    tar_path, offset, size = entry.split(":")
    offset, size = int(offset), int(size)

    with open(tar_path, "rb") as f:
        f.seek(offset)
        audio_bytes = f.read(size)
    try:
        audio, sr = torchaudio.load(io.BytesIO(audio_bytes))
    except:
        audio, sr = torchaudio.load(io.BytesIO(audio_bytes), format="mp3")

    return audio, sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample Speaker Audios")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="path to test.jsonl",
    )
    parser.add_argument(
        "-n",
        "--num_audios",
        type=int,
        default=1,
        help="number of audio samples per speaker",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path to output dir",
    )

    args = parser.parse_args()
    data = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()]
    spks = set()

    meta = args.output / "meta.lst"

    desc_dir = args.output / "desc"
    desc_dir.mkdir(parents=True, exist_ok=True)

    wavdir = args.output / "wavs"
    wavdir.mkdir(parents=True, exist_ok=True)

    num_items = 0
    spk2cnt = {}
    with meta.open("w", encoding="utf-8") as fout, args.data.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            speaker = item["speaker"]
            if speaker not in spk2cnt:
                spk2cnt[speaker] = 0
            spk2cnt[speaker] += 1
            if spk2cnt[speaker] > args.num_audios:
                continue
            uid = item["id"]

            text = item["text"].strip()

            audio, sr = _load_audio_from_tar(item["audio"])
            wav_path = wavdir / f"{uid}.wav"
            torchaudio.save(wav_path, audio, sr)

            fout.write(f"{uid}|{text}|{wav_path}|{text}\n")
