import os
from typing import List, Tuple

DATASET_FILE = os.path.dirname(os.path.abspath(__file__)) + "/ita.txt"

def clean_sentence(sentence : str) -> str:
    mapping = str.maketrans("", "", ".!'\",?")
    return sentence.translate(mapping)

def read_dataset() -> List[Tuple[str, str]]:
    """
    Reads a parallel dataset from a specified file.

    Assumes the file contains tab-separated English and Italian sentences,
    with English in the first column and Italian in the second.

    Returns:
        A list of tuples, where each tuple contains an (English sentence, Italian sentence) pair.
        Returns an empty list if the file is not found or is empty.
    """
    dataset = []
    if not os.path.exists(DATASET_FILE):
        print(f"Error: Dataset file not found at {DATASET_FILE}")
        return dataset

    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip whitespace and split by tab
                parts = line.strip().split('\t')
                dataset.append((clean_sentence(parts[0]), clean_sentence(parts[1])))
        print(f"Successfully loaded {len(dataset)} sentence pairs from {DATASET_FILE}")
    except Exception as e:
        print(f"Error reading dataset file: {e}")
        return []
    
    return dataset

if __name__ == '__main__':
    print("Dataset reader")