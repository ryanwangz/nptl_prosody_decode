
import numpy as np
from collections import Counter

# Load the labels
labels = np.load('/home/groups/henderj/rzwang/processed_data_phonemes/labels.npy')

# Basic information
print("Shape of labels array:", labels.shape)
print("Data type:", labels.dtype)

# If labels are not strings, convert if needed
if labels.dtype.kind in ['i', 'f']:  # if numeric
    print("Unique values:", np.unique(labels))
else:  # if strings
    unique_phonemes = np.unique(labels)
    print("Number of unique phonemes:", len(unique_phonemes))
    print("\nUnique phonemes:")
    print(unique_phonemes)

# Get frequency distribution
phoneme_counts = Counter(labels.flatten())
print("\nPhoneme frequencies:")
for phoneme, count in phoneme_counts.most_common():
    print(f"{phoneme}: {count}")