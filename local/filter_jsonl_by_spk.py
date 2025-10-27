import argparse
import json
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output filtered JSONL file.")
    parser.add_argument("--spks", type=Path, required=True, help="Speaker ID to filter by.")
    args = parser.parse_args()

    spks = set()
    with args.spks.open("r") as f:
        for line in f:
            spk = line.strip().split()[0]
            spks.add(spk)

    with (
        open(args.input, "r", encoding="utf-8") as infile,
        open(args.output, "w", encoding="utf-8") as outfile,
    ):
        for line in tqdm(infile):
            item = json.loads(line)
            spk = item["text"].split(maxsplit=3)[1][:-1]
            if spk in spks:
                new_item = {"text": item["text"], "speaker": spk, "audio": item["audio"], "generation_task_prob": 1.0}
                outfile.write(json.dumps(new_item, ensure_ascii=False) + "\n")
