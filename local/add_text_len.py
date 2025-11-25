import json
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

from tqdm import tqdm

from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# Initialize the processor globally so each process can access it.
processor = VibeVoiceProcessor.from_pretrained("vibevoice/VibeVoice-1.5B")

jobs = 10
MAX_ITEMS_IN_QUEUE = 100_000


def _load_desc(entry):
    path, offset, size = entry.split(":")
    offset, size = int(offset), int(size)

    with open(path, "rb") as f:
        f.seek(offset)
        item_bytes = f.read(size)
        item = item_bytes.decode("utf-8")
    item = json.loads(item)["text"]
    return item


def process_item(item):
    """Process a single line: parse JSON, calculate text length, and return the item."""
    text = item["text"]
    item["text_len"] = len(processor.tokenizer.encode(text))
    return item


def producer(input_file, queue):
    """Producer function that reads lines from the input file and puts them in the queue."""
    with input_file.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            while queue.qsize() >= MAX_ITEMS_IN_QUEUE:
                time.sleep(10)
                print("Waiting for producer queue to have space...")
            if line.strip():  # Skip empty lines
                queue.put(json.loads(line))
    # Signal consumers that the producer is done
    for _ in range(jobs):
        queue.put(None)


def consumer(producer_queue, consumer_queue):
    """Consumer function that processes lines from the queue and writes to the output file."""
    while True:
        while consumer_queue.qsize() >= MAX_ITEMS_IN_QUEUE:
            time.sleep(10)
            print("Waiting for consumer queue to have space...")
        item = producer_queue.get()
        if item is None:
            break
        item = process_item(item)
        consumer_queue.put(item)
    consumer_queue.put(None)


def main(source, target):
    """Process the file using the producer-consumer pattern."""
    producer_queue, consumer_queue = Queue(), Queue()

    Process(target=producer, args=(source, producer_queue)).start()

    for _ in range(jobs):
        Process(target=consumer, args=(producer_queue, consumer_queue)).start()

    pbar = tqdm()
    remaining_consumers = jobs
    with target.open("w", encoding="utf-8") as f_out:
        while remaining_consumers > 0:
            item = consumer_queue.get()
            if item is None:
                remaining_consumers -= 1
            else:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    # Input and output file paths
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])

    # Process the file
    main(source, target)
