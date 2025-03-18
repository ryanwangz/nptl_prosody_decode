import numpy as np
import glob
import os


import scipy.io as sio
import numpy as np
import os
def convert_txt_to_npy(input_dir, output_dir):
    """
    Convert pitch and intensity text files to npy files. (from praat script)
    Handles undefined values by converting them to 0. (TODO: double check if we will end up with issues here)
    
    Parameters:
    input_dir (str): Directory containing the text files
    output_dir (str): Directory where npy files will be saved
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get lists of pitch and intensity files
    pitch_files = glob.glob(os.path.join(input_dir, "*_pitch.txt"))
    intensity_files = glob.glob(os.path.join(input_dir, "*_intensity.txt"))
    
    # Process pitch files
    all_pitch_data = []
    pitch_filenames = []
    for file in pitch_files:
        # Read the data, skipping header
        try:
            # data = np.loadtxt(file, skiprows=1, dtype=str)  # Read as strings first. skip first row (header)
            # # Convert to float, replacing '--undefined--' with 0
            # pitch_values = np.where(data[:, 1] == '--undefined--', 0, data[:, 1].astype(float))
            # all_pitch_data.append(pitch_values)

            # Read the raw data as text
            with open(file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
            # Process each lin
            values = []
            for line in lines:
                time, value = line.strip().split()
                values.append(float(0.0) if value == '--undefined--' else float(value))
            pitch_values = np.array(values)            
            # Store filename for reference
            basename = os.path.basename(file).replace("_pitch.txt", "")
            pitch_filenames.append(basename)
            
            # Save individual file
            output_file = os.path.join(output_dir, f"{basename}_pitch.npy")
            np.save(output_file, pitch_values)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    # Process intensity files
    all_intensity_data = []
    intensity_filenames = []
    for file in intensity_files:
        try:
            # data = np.loadtxt(file, skiprows=1, dtype=str)  # Read as strings first
            # # Convert to float, replacing '--undefined--' with 0
            # intensity_values = np.where(data[:, 1] == '--undefined--', 0, data[:, 1].astype(float))
            # all_intensity_data.append(intensity_values)
            # Read the raw data as text
            with open(file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
            # Process each lin
            values = []
            for line in lines:
                time, value = line.strip().split()
                values.append(float(0.0) if value == '--undefined--' else float(value))
            intensity_values = np.array(values)            

            # Store filename for reference
            basename = os.path.basename(file).replace("_intensity.txt", "")
            intensity_filenames.append(basename)
            
            # Save individual file
            output_file = os.path.join(output_dir, f"{basename}_intensity.npy")
            np.save(output_file, intensity_values)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Create combined arrays if we have matching files
    if all_pitch_data and all_intensity_data:
        # Find common length (in case files have different lengths)
        min_length = min(min(len(x) for x in all_pitch_data), 
                        min(len(x) for x in all_intensity_data))
        
        # Trim all arrays to minimum length and stack
        pitch_array = np.vstack([x[:min_length] for x in all_pitch_data])
        intensity_array = np.vstack([x[:min_length] for x in all_intensity_data])
        
        # Save combined arrays
        np.save(os.path.join(output_dir, 'all_pitch.npy'), pitch_array)
        np.save(os.path.join(output_dir, 'all_intensity.npy'), intensity_array)
        
        # Save filename references
        np.save(os.path.join(output_dir, 'pitch_filenames.npy'), np.array(pitch_filenames))
        np.save(os.path.join(output_dir, 'intensity_filenames.npy'), np.array(intensity_filenames))
    print(f"conveted {len(pitch_filenames)} pitch and {len(intensity_filenames)} intensity texts to npys!\n")



#TODO: add the theshold crossings, as this code only has the neural spike band power
def process_all_blocks(exp_dir, label_dir, output_dir, footer, padding=1):
    """
    Process all blocks in directory and save aligned data as .npy files. Should overwrite existing label files in the folder (if any)
    Assumes padding should have 1s (indicating Silence)
    Parameters:
    -----------
    exp_dir : str
        Directory containing .mat files with the neural data
    label_dir : str
        Directory containing label files (should be .npy). should correspond to the 20ms bins of the neural data
    output_dir : str
        Directory to save processed .npy files
    footer : str
        file label footer (vs .wav file)
        for example, if a.wav has labels a_lbl.npy, this string would be "_lbl.npy"
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_neural_data = []
    all_neural_data_tc = []
    all_labels = []
    trial_info = []  # Store metadata about each trial
    
    #Process each .mat file in directory
    mat_files = [f for f in os.listdir(exp_dir) if f.endswith('.mat')]
    for i, mat_file in enumerate(mat_files):
#         print(f"Processing block {i+1}/{len(mat_files)}: {mat_file}")
        print("Processing block {}/{}: {}".format(i + 1, len(mat_files), mat_file))
        # Load mat file
        mat_data = sio.loadmat(os.path.join(exp_dir, mat_file))
        
        # Get block data
        neural_data = mat_data['binned_neural_spike_band_power'] #we should also
        #include the binned_neural_threshold_crossings here?
        neural_data_tc = mat_data['binned_neural_threshold_crossings']
        bin_times = mat_data['binned_neural_redis_clock'][0]
        go_cues = mat_data['go_cue_redis_time'][0]
        trial_ends = mat_data['trial_end_redis_time'][0]
        
        #Process each trial in block
        for trial_idx, (cue_line, go_cue, trial_end) in enumerate(zip(mat_data['cue'], go_cues, trial_ends)):
            #Parse audio filename
            audio_file = cue_line.split(';')[0]
            
            # Load corresponding labels
            label_file = os.path.join(label_dir, audio_file.replace('.wav', footer)) #TODO: change this if doing a different label type
            if not os.path.exists(label_file):
#                 print(f"Warning: Missing label file for {audio_file}")
                print("Warning: Missing label file for {}".format(audio_file))
                continue
            labels = np.load(label_file)

            # normalized_labels = []
            # for labels in all_labels:
            #     labels_norm = (labels - np.mean(labels)) / np.std(labels)
            #     # or
            #     labels_norm = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
            #     normalized_labels.append(labels_norm)
            
            # Find trial boundaries
            trial_start_idx = np.searchsorted(bin_times, go_cue)
            trial_end_idx = np.searchsorted(bin_times, trial_end)
            
            # Extract trial data
            trial_neural = neural_data[trial_start_idx:trial_end_idx]
            trial_neural_tc = neural_data_tc[trial_start_idx:trial_end_idx]
            
            # Align lengths
            n_neural_bins = trial_neural.shape[0] #should be the same for tc and sbp
            n_label_bins = len(labels)
            
            if n_neural_bins > n_label_bins:
                labels = np.pad(labels, (0, n_neural_bins - n_label_bins),
                              mode='constant', constant_values=padding)
            else:
                labels = labels[:n_neural_bins]
                trial_neural = trial_neural[:n_label_bins]
                trial_neural_tc = trial_neural_tc[:n_label_bins]
            
            # Store aligned data
            all_neural_data.append(trial_neural)
            all_neural_data_tc.append(trial_neural_tc)
            all_labels.append(labels)
            
            # Store trial metadata
            trial_info.append({
                'block_file': mat_file,
                'audio_file': audio_file,
                'trial_idx': trial_idx,
                'n_bins': len(labels),
                'trial_start_idx': trial_start_idx,
                'trial_end_idx': trial_end_idx                
            })
    
    # Convert to arrays and save
    neural_data_array = np.concatenate(all_neural_data, axis=0)
    neural_data_array_tc = np.concatenate(all_neural_data_tc, axis=0)
    labels_array = np.concatenate(all_labels)
    labels_array = np.concatenate(all_labels)
    labels_normalized = (labels_array - np.mean(labels_array)) / np.std(labels_array)
    # labels_normalized = (labels_array - np.min(labels_array)) / (np.max(labels_array) - np.min(labels_array))

#     np.save(f"{output_dir}/neural_data.npy", neural_data_array)
#     np.save(f"{output_dir}/labels.npy", labels_array)
#     np.save(f"{output_dir}/trial_info.npy", trial_info)
    
#     print(f"Processed {len(mat_files)} blocks, {len(trial_info)} trials")
#     print(f"Total timepoints: {len(labels_array)}")
    np.save("{}/neural_data_sbp.npy".format(output_dir), neural_data_array) #spike band power 
    np.save("{}/neural_data_tc.npy".format(output_dir), neural_data_array_tc) #spike band power 
    np.save("{}/labels.npy".format(output_dir), labels_array)
    np.save("{}/trial_info.npy".format(output_dir), trial_info)
    np.save("{}/labels_normalized.npy".format(output_dir), labels_normalized)
    
    print("Processed {} blocks, {} trials".format(len(mat_files), len(trial_info)))
    print("Total timepoints: {}".format(len(labels_array)))
    return neural_data_array

def verify_conversion(txt_dir, npy_dir, n_samples=5, n_files=3):
    """
    Verify that the conversion from txt to npy was successful by comparing samples.
    
    Parameters:
    txt_dir (str): Directory containing original text files
    npy_dir (str): Directory containing converted npy files
    n_samples (int): Number of samples to check from each file
    n_files (int): Number of files to check
    """
    # Get lists of files
    pitch_txt_files = sorted(glob.glob(os.path.join(txt_dir, "*_pitch.txt")))
    intensity_txt_files = sorted(glob.glob(os.path.join(txt_dir, "*_intensity.txt")))
    
    # Select random files to check
    if len(pitch_txt_files) > n_files:
        import random
        indices = random.sample(range(len(pitch_txt_files)), n_files)
        pitch_txt_files = [pitch_txt_files[i] for i in indices]
        intensity_txt_files = [intensity_txt_files[i] for i in indices]
    
    print("\nVerifying conversion for {} files, {} samples each:".format(n_files, n_samples))
    print("-" * 80)
    
    for pitch_txt, intensity_txt in zip(pitch_txt_files, intensity_txt_files):
        # Get base filename
        basename = os.path.basename(pitch_txt).replace("_pitch.txt", "")
        print(f"\nChecking file: {basename}")
        print("-" * 40)
        
        # Load original txt data
        with open(pitch_txt, 'r') as f:
            pitch_txt_lines = f.readlines()[1:]  # Skip header
        with open(intensity_txt, 'r') as f:
            intensity_txt_lines = f.readlines()[1:]  # Skip header
            
        # Load npy data
        pitch_npy = np.load(os.path.join(npy_dir, f"{basename}_pitch.npy"))
        intensity_npy = np.load(os.path.join(npy_dir, f"{basename}_intensity.npy"))
        
        # Get sample indices
        if len(pitch_txt_lines) > n_samples:
            sample_indices = sorted(random.sample(range(len(pitch_txt_lines)), n_samples))
        else:
            sample_indices = range(len(pitch_txt_lines))
        
        print("Pitch values:")
        print("Index  |  Original (txt)  |  Converted (npy)")
        print("-" * 40)
        for idx in sample_indices:
            txt_value = pitch_txt_lines[idx].strip().split()[1]
            npy_value = pitch_npy[idx]
            print(f"{idx:5d}  |  {txt_value:13s}  |  {npy_value:14.2f}")
        
        print("\nIntensity values:")
        print("Index  |  Original (txt)  |  Converted (npy)")
        print("-" * 40)
        for idx in sample_indices:
            txt_value = intensity_txt_lines[idx].strip().split()[1]
            npy_value = intensity_npy[idx]
            print(f"{idx:5d}  |  {txt_value:13s}  |  {npy_value:14.2f}")
        
        print("\nArray shapes:")
        print(f"Pitch: {pitch_npy.shape}")
        print(f"Intensity: {intensity_npy.shape}")
        print("-" * 80)



if __name__ == "__main__":

    #### creating the hz and db labels
    # input_dir = "/home/groups/henderj/rzwang/labels_hz_db_txts" #should be the label files from the praat script
    # output_dir = "/home/groups/henderj/rzwang/labels_hz_db_npys"
    # convert_txt_to_npy(input_dir, output_dir)
    # txt_dir = "/home/groups/henderj/rzwang/labels_hz_db_txts"
    # npy_dir = "/home/groups/henderj/rzwang/labels_hz_db_npys"
    # verify_conversion(txt_dir, npy_dir)

    # for making volume labels:
    
    label_dir = "/home/groups/henderj/rzwang/labels_hz_db_npys"
    exp_dir = "/home/groups/henderj/rzwang/blocks/t12-02-20-2025"
    output_dir = "/home/groups/henderj/rzwang/processed_data_hz"

    process_all_blocks(exp_dir, label_dir, output_dir, "_intensity.npy", padding=0)

    #### general template for making categorical labels
    # label_dir = "/home/groups/henderj/rzwang/labels_hz_db_npys"
    # exp_dir = "/home/groups/henderj/rzwang/blocks/t12-02-20-2025"
    # output_dir = "/home/groups/henderj/rzwang/processed_data"

    # process_all_blocks(exp_dir, label_dir, output_dir, "_silence.npy")

    # for making phoneme labels:
    # label_dir = "/home/groups/henderj/rzwang/labels_phonemes"
    # exp_dir = "/home/groups/henderj/rzwang/blocks/t12-02-20-2025"
    # output_dir = "/home/groups/henderj/rzwang/processed_data_phonemes"

    # process_all_blocks(exp_dir, label_dir, output_dir, "_phonemes.npy", padding=b'SIL')
