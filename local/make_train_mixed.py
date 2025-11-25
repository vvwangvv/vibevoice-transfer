import argparse
import json
import random

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mixed training jsonl file for VibeVoice finetuning.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input training jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output mixed training jsonl file.",
    )
    parser.add_argument(
        "-u",
        "--understanding_speakers",
        type=int,
        required=True,
        help="Number of understanding speakers.",
    )
    parser.add_argument(
        "-g",
        "--generation_speakers",
        type=int,
        required=True,
        help="Number of generation speakers.",
    )
    parser.add_argument(
        "--mix_ratio",
        type=float,
        default=0.5,
        help="Ratio of mixing clean and noisy samples.",
    )

    args = parser.parse_args()

    speakers = set()
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            speakers.add(data["speaker"])
    speakers = list(speakers)

    random.seed(1234)
    random.shuffle(speakers)
    understanding_speakers = speakers[: args.understanding_speakers]
    generation_speakers = speakers[::-1][: args.generation_speakers]

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            if not line.strip():
                continue
            data = json.loads(line.strip())
            if data["speaker"] in understanding_speakers and data["speaker"] in generation_speakers:
                data["generation_task_prob"] = args.mix_ratio
            elif data["speaker"] in understanding_speakers:
                data["generation_task_prob"] = 0.0
            elif data["speaker"] in generation_speakers:
                data["generation_task_prob"] = 1.0
            else:
                raise ValueError(f"Speaker {data['speaker']} not in understanding or generation speakers.")
            fout.write(json.dumps(data) + "\n")
