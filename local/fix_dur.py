import json
import sys
from pathlib import Path

from tqdm import tqdm

source = sys.argv[1]
target = sys.argv[2]

data_roots = [
    Path("/mnt/bn/wangwei-nas-lq-01/mlx/users/wangwei.0/workspace/F5-TTS/dump/Emilia/EN"),
    Path("/mnt/bn/wangwei-nas-lq-01/mlx/users/wangwei.0/workspace/F5-TTS/dump/Emilia/ZH"),
]

utt2dur = {}
for data_root in data_roots:
    for wav_scp in tqdm(list(data_root.rglob("utt2dur"))):
        lines = wav_scp.read_text(encoding="utf-8").splitlines()
        for line in lines:
            uid, dur = line.strip().split(maxsplit=1)
            dur = float(dur)
            utt2dur[uid] = dur


with open(source, "r", encoding="utf-8") as fsrc, open(target, "w", encoding="utf-8") as ftgt:
    for line in tqdm(fsrc):
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)
        item["duration"] = utt2dur[item["id"]]
        ftgt.write(json.dumps(item, ensure_ascii=False) + "\n")
