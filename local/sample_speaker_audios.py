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
        help="path to train.jsonl",
    )
    parser.add_argument(
        "-n",
        "--n-audios-per-speaker",
        type=int,
        default=1,
        help="number of audio samples per speaker",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="whether to dump wav files",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path to output test.meta",
    )

    args = parser.parse_args()
    data = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()]
    spk2item = {}
    for item in data:
        speaker = item["speaker"]
        audio_path = item["audio"]
        if speaker not in spk2item:
            spk2item[speaker] = []
        spk2item[speaker].append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wavdir = None
    if args.dump:
        wavdir = args.output.parent / "wavs"
        wavdir.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for speaker, items in spk2item.items():
            random.shuffle(items)
            sampled_items = items[: args.n_audios_per_speaker]
            for item in sampled_items:

                audio_path = item["audio"]
                audio = Path(audio_path)
                target_path = wavdir / f"{item['id']}.wav"

                if args.dump:
                    audio, sr = _load_audio_from_tar(audio_path)
                    torchaudio.save(target_path, audio, sr)
                    audio_path = target_path.absolute().as_posix()
                text = item["text"]
                uid = item["id"]
                f.write(f"{uid}|{text}|{audio_path}|{text}\n")
