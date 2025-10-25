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


if __name__ == "__main__":
    raw_path = Path("data/raw/emilia/train.jsonl")
    manifest_path = Path("data/manifest/emilia/train.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fout, raw_path.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin):
            if not line.strip():
                continue
            item = json.loads(line)
            new_item = {
                "text": f"Speaker {item['speaker']}: {item['text']}",
                "audio": item["wav"],
                "task": "generation",
            }
            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
