import argparse
import json
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to the input META file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output filtered META file.")
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
            if not line.strip():
                continue
            uid = line.split("|", maxsplit=1)[0]
            spk = uid.rsplit("_", maxsplit=1)[0]
            if spk in spks:
                outfile.write(line)
