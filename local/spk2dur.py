import json
from pathlib import Path

from tqdm import tqdm

data_roots = [
    Path("/mnt/bn/wangwei-nas-lq-01/mlx/users/wangwei.0/workspace/F5-TTS/dump/Emilia/EN"),
    Path("/mnt/bn/wangwei-nas-lq-01/mlx/users/wangwei.0/workspace/F5-TTS/dump/Emilia/ZH"),
]

utt2dur, utt2spk = {}, {}
for data_root in data_roots:
    for utt2dur_file in tqdm(data_root.rglob("utt2dur")):
        utt2spk_file = utt2dur_file.parent / "utt2spk"
        lines = utt2dur_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            uid, dur = line.strip().split()
            utt2dur[uid] = float(dur)
        lines = utt2spk_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            uid, spk = line.strip().split()
            utt2spk[uid] = spk

spk2dur = {}
for uid, dur in tqdm(utt2dur.items(), total=len(utt2dur), desc="Calculating spk2dur"):
    spk = utt2spk[uid]
    if spk not in spk2dur:
        spk2dur[spk] = 0.0
    spk2dur[spk] += dur

with open("spk2dur", "w", encoding="utf-8") as f:
    for spk, dur in tqdm(spk2dur.items(), total=len(spk2dur), desc="Writing spk2dur"):
        f.write(f"{spk} {dur:.3f}\n")
