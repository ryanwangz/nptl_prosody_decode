
import numpy as np
from collections import Counter

#for looking at phoneme data
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



### this was for debugging trial info with the Nans (which ended up being a step indexing error when iterating through train/test trials)
# trial_info_dict = {
#     'block_file': mat_file,
#     'audio_file': audio_file,
#     'trial_idx': trial_idx,
#     'n_bins': len(labels),
#     'trial_start_idx': trial_start_idx,
#     'trial_end_idx': trial_end_idx                
# }
# if del_sil:
#     trial_info_dict.update({
#         'n_bins_original': original_length,
#         'n_bins_non_silence': np.sum(non_zero_mask),
#         'non_silence_mask': non_zero_mask.tolist()  # Store mask for reference
#     })

trial_info_full = np.load("/home/groups/henderj/rzwang/processed_data_hz_no_sil/trial_info.npy", allow_pickle=True)
trial_info = trial_info_full[97] #looking at 255, which had NaNs in results
n_bins_original = trial_info['n_bins_original']
non_sil_mask = trial_info['non_silence_mask']
non_sil_bins = trial_info['n_bins_non_silence']
tr_idx = trial_info['trial_idx']
audio_file = trial_info['audio_file']

print(f"Original num of bins: {n_bins_original}; num of non_silence bins: {non_sil_bins}; trial index {tr_idx}. Audio file {audio_file}.")


# chanSets = {129:192, 193:256, 1:64, 65:128};
# chanSetNames = {'44i', '44s', '6vi', '6vs'};

#this would be for visualizing the different areas or trying to use decoders with just one or the other area
