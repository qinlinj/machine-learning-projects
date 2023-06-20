import numpy as np
import sys
from collections import Counter


def read_input_file(file):
    with open(file, 'r') as f:
        next(f)  # skip the first line
        input_data = [line.strip().split('\t')[-1] for line in f]
    return input_data


def calculate_entropy(input_data):
    n = len(input_data)
    y_counts = Counter(input_data)
    y_entropy = sum(- count / n * np.log2(count / n) for count in y_counts.values())
    return y_entropy


def calculate_error_rate(input_data):
    return 1 - max(Counter(input_data).values()) / len(input_data)


if __name__ == "__main__":
    data = read_input_file(sys.argv[1])
    entropy = calculate_entropy(data)
    error_rate = calculate_error_rate(data)
    with open(sys.argv[2], 'w') as f:
        f.write(f"entropy: {entropy}\n")
        f.write(f"error: {error_rate}\n")