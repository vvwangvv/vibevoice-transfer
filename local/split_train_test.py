import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

MIN_AUDIO_PER_TRAIN_SPK = 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="split train test sets")
    parser.add_argument(
        "data",
        type=Path,
        help="path to data.jsonl",
    )
    parser.add_argument(
        "-n",
        "--audios-per-test-speaker",
        type=int,
        default=2,
        help="minimum number of audio samples per speaker for test set",
    )

    args = parser.parse_args()
    spk2cnt = {}
    with (
        (args.data.parent / "train.jsonl").open("w", encoding="utf-8") as f_train,
        (args.data.parent / "test.jsonl").open("w", encoding="utf-8") as f_test,
        args.data.open("r", encoding="utf-8") as f,
    ):
        for line in tqdm(f):
            item = json.loads(line)
            speaker = item["text"].strip().split(maxsplit=3)[1][:-1]
            if speaker not in spk2cnt:
                spk2cnt[speaker] = 0
            spk2cnt[speaker] += 1
            del item["task"]
            if spk2cnt[speaker] <= MIN_AUDIO_PER_TRAIN_SPK:
                f_train.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif spk2cnt[speaker] <= MIN_AUDIO_PER_TRAIN_SPK + args.audios_per_test_speaker:
                f_test.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                f_train.write(json.dumps(item, ensure_ascii=False) + "\n")
