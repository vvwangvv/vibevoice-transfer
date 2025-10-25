import json
from pathlib import Path

from tqdm import tqdm


def load_scp(scp_path):
    utt2value = {}
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            key, path = line.strip().split(maxsplit=1)
            utt2value[key] = path
    return utt2value


data_roots = [
    Path("/mnt/bn/wangwei-nas-lq-01/mlx/users/wangwei.0/workspace/F5-TTS/dump/Emilia/EN"),
    Path("/mnt/bn/wangwei-nas-lq-01/mlx/users/wangwei.0/workspace/F5-TTS/dump/Emilia/ZH"),
]

if __name__ == "__main__":
    output = Path("data/raw/emilia/train.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for data_root in data_roots:
            wav_scps = list(data_root.rglob("**/wav.scp"))
            for wav_scp in tqdm(wav_scps):
                utt2wav = load_scp(wav_scp)
                utt2text = load_scp(wav_scp.parent / "text")
                utt2dur = load_scp(wav_scp.parent / "utt2dur")
                utt2spk = load_scp(wav_scp.parent / "utt2spk")

                for utt in utt2wav:
                    item = {
                        "id": utt,
                        "wav": utt2wav[utt],
                        "text": utt2text[utt],
                        "duration": float(utt2dur[utt]),
                        "speaker": utt2spk[utt],
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
