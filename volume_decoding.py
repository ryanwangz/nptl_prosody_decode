
print("running")

import argparse
import torch
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
import logging
import json
import time
import random
from datetime import datetime
# from torchsummary import summary
from scipy import stats
#some other ones: torchsummary, sklearn metrics, early stopping? seaborn/matplotlib
import seaborn as sns
import torch.multiprocessing as mp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from decoders import *
from train_functions import *

# DECODING_TYPE = 'pitch_no_silence' # For plot titles 'volume' or 'pitch'
# UNITS = '(hz)' # For plots '(db)' or '(hz)'
# FILE_TAG = 'hz_no_sil' #(processed_data_{FILE_TAG}) 'hz' or 'db

# Add after imports
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#TODO: Simplify this?
class CNNVolumeDecoder(nn.Module):
    def __init__(self, n_channels=256, window_size=5):
        super(CNNVolumeDecoder, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * window_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Single output for volume prediction
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze()

def train_volume_decoder(neural_data, labels, trial_info, decoder_class, run_id, decoding_type, units, normalized, masked, window_size=5, stride=1, 
                        batch_size=32, n_epochs=10, learning_rate=0.001, train_split=0.8):
    
    num_workers = mp.cpu_count
    # Create windows (same as before)
    windowed_data, windowed_labels = create_windows_and_labels_mean(
        neural_data, labels, window_size, stride
    )
    
    trial_data = []
    trial_labels = []
    current_idx = 0
    train_losses = []
    test_losses = []
    train_maes = []
    test_maes = []

    for trial in trial_info:
        n_bins = trial['n_bins']
        # Calculate how many windows we'll get from this trial
        n_windows = (n_bins - window_size) // stride + 1
        
        trial_windows = windowed_data[current_idx:current_idx + n_windows]
        trial_window_labels = windowed_labels[current_idx:current_idx + n_windows]
        
        trial_data.append(trial_windows)
        trial_labels.append(trial_window_labels)
        
        current_idx += n_windows

    trial_data = np.array(trial_data)
    trial_labels = np.array(trial_labels)
    
    # Generate rdm indices for train/test split
    n_trials = len(trial_info)
    n_train_trials = int(n_trials * train_split) #rounding here
    trial_indices = np.random.permutation(n_trials)
    train_trial_indices = trial_indices[:n_train_trials]
    test_trial_indices = trial_indices[n_train_trials:]

    np.save(f'/home/groups/henderj/rzwang/results/test_idxs/test_indices_{run_id}.npy', test_trial_indices)
    
    #Split the data
    train_data = np.concatenate(trial_data[train_trial_indices])
    train_labels = np.concatenate(trial_labels[train_trial_indices])

    test_data = np.concatenate(trial_data[test_trial_indices])
    test_labels = np.concatenate(trial_labels[test_trial_indices])
    #ok, so this is where the later error might be occurring-- because they're being split? should it be trial_labels then?

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_data),
        torch.FloatTensor(train_labels)  # Changed from LongTensor
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_data),
        torch.FloatTensor(test_labels)  # Changed from LongTensor
    )
    
    # # Create data loaders (same as before)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=False # Do we want to shuffle data here?
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False
    # )
    # Create data loaders (same as before)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True # Do we want to shuffle data here?
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    # Init
    # model = decoder_class(n_channels=neural_data.shape[1], window_size=window_size)
    model = decoder_class(n_channels=neural_data.shape[1], window_size=window_size).to(device)
    # Use MSE loss instead of CrossEntropyLoss for continuous
    # criterion = nn.MSELoss()
    # NEW: changed to Masked MSE Loss
    if masked:
        criterion = MaskedMSELoss().to(device)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001) # may want to switch back to Adam (remnant of phoneme decoder)
    # scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=40, verbose=True, min_lr=1e-6
    ) #changed factor and patience from 0.5 and 5 originally
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 100
    min_improvement = 0.0001

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_mae = 0  #Mean Absolute Error
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            outputs = model(batch_data)
            # loss = criterion(outputs, batch_labels)
            # NEW: updated for Masked MSE Loss:
            if masked:
                loss = criterion(outputs, batch_labels, normalized=(normalized == "True"))
            else:
                loss = criterion(outputs, batch_labels)
            
            # Backward pass and optimize (syntax from GPT bc it was giving me issues)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - batch_labels)).item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        test_mae = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_data)
                if masked:
                    # NEW: MSE update with mask for silence
                    loss = criterion(outputs, batch_labels, normalized=(normalized == "True"))
                else:
                    loss = criterion(outputs, batch_labels)
                

                test_loss += loss.item()
                #previously:
                # test_mae += torch.mean(torch.abs(outputs - batch_labels)).item()
                # NEW:
                if masked:
                    if normalized == "True":
                        mask = ~torch.isclose(batch_labels, torch.zeros_like(batch_labels), atol=1e-5)
                    else:
                        mask = batch_labels != 0
                    if mask.sum() > 0:  # Only calculate MAE for non-silence portions
                        test_mae += torch.mean(torch.abs(outputs[mask] - batch_labels[mask])).item()
                else:
                    test_mae += torch.mean(torch.abs(outputs - batch_labels)).item()



        avg_test_loss = test_loss / len(test_loader)
        avg_test_mae = test_mae / len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_maes.append(avg_train_mae)
        test_maes.append(avg_test_mae)

        # Update scheduler
        scheduler.step(avg_test_loss)
        
        # Early stopping check
        if avg_test_loss < (best_loss - min_improvement):
            best_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_model_{run_id}.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        #Print epoch statistics
        print(f'Epoch: {epoch+1}/{n_epochs}')
        print(f'Train Loss (MSE): {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f}')
        print(f'Test Loss (MSE): {avg_test_loss:.4f} | Test MAE: {avg_test_mae:.4f}')
        print('--------------------')

        logging.info(f'Epoch: {epoch+1}/{n_epochs}')
        logging.info(f'Train Loss (MSE): {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f}')
        logging.info(f'Test Loss (MSE): {avg_test_loss:.4f} | Test MAE: {avg_test_mae:.4f}')
        logging.info('--------------------')
    
    print(f'Saving test statistics \n')

    # Load best model
    model.load_state_dict(torch.load(f'best_model_{run_id}.pt'))
    # make predictions on test set
    predictions_test = decode_volume(
        model=model,
        test_data=test_data,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size
    )
    
    #Calculate and log metrics
    accuracy = np.mean(np.abs(predictions_test - test_labels)).item()
    logging.info(f"Test MAE: {accuracy:.4f}")

    # Do the same with the training data for curiosity/overfitting's sake
    predictions_train = decode_volume(
        model=model,
        test_data=train_data,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size
    )
    
    #Calculate and log metrics
    accuracy = np.mean(np.abs(predictions_train - train_labels)).item()
    logging.info(f"Train MAE: {accuracy:.4f}")

    #save a handful of trial plots here (maybe 3 best and 3 worst trials?)
    
    # Create directory for saving plots if it doesn't exist
    plot_dir = f'/home/groups/henderj/rzwang/figures/{decoding_type}_plots/{decoding_type}_plot_{run_id}'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Calculate error for each trial
    trial_errors_test = []
    trial_predictions = []
    trial_actual = []
    trial_errors_train = []

    current_idx_test = 0
    current_idx_train = 0
    for i, trial in enumerate(trial_info):
        if i in test_trial_indices:  # Only evaluate test trials
            n_bins = trial['n_bins']
            n_windows = (n_bins - window_size) // stride + 1
            
            trial_pred = predictions_test[current_idx_test:current_idx_test + n_windows]
            trial_true = test_labels[current_idx_test:current_idx_test + n_windows]
            trial_mae = np.mean(np.abs(trial_pred - trial_true))
            
            trial_errors_test.append({
                'index': i,
                'error': trial_mae,
                'predictions': trial_pred,
                'actual': trial_true,
                'audio_file': trial['audio_file']
            })
            
            current_idx_test += n_windows
        else: #in train trials
            n_bins = trial['n_bins']
            n_windows = (n_bins - window_size) // stride + 1
            
            trial_pred = predictions_train[current_idx_train:current_idx_train + n_windows]
            trial_true = train_labels[current_idx_train:current_idx_train + n_windows]
            trial_mae = np.mean(np.abs(trial_pred - trial_true))
            
            trial_errors_train.append({
                'index': i,
                'error': trial_mae,
                'predictions': trial_pred,
                'actual': trial_true,
                'audio_file': trial['audio_file']
            })
            current_idx_train += n_windows            


    # Sort trials by error
    sorted_trials_test = sorted(trial_errors_test, key=lambda x: x['error'])
    sorted_trials_train = sorted(trial_errors_train, key=lambda x: x['error'])

    # Plot best 3 and worst 3 trials
    best_trials_test = sorted_trials_test[:3]
    worst_trials_test = sorted_trials_test[-3:]
    best_trials_train = sorted_trials_train[:3]
    worst_trials_train = sorted_trials_train[-3:]

    # Pearson's R test for correlation
    all_predictions_test = np.concatenate([t['predictions'] for t in trial_errors_test])
    all_actuals_test = np.concatenate([t['actual'] for t in trial_errors_test])
    pearson_r_test, p_value_test = stats.pearsonr(all_actuals_test, all_predictions_test)


    all_predictions_train = np.concatenate([t['predictions'] for t in trial_errors_train])
    all_actuals_train = np.concatenate([t['actual'] for t in trial_errors_train])
    pearson_r_train, p_value_train = stats.pearsonr(all_actuals_train, all_predictions_train)

    plot_learning_curves(train_losses, test_losses, train_maes, test_maes, plot_dir, decoding_type)

    # def plot_trial(trial_data, title, filename, decoding_type, units):
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(trial_data['actual'], label='Actual', color='blue', alpha=0.6)
    #     # plt.plot(trial_data['predictions'], label='Predicted', color='red', alpha=0.6)
    #     if masked: #only plot non-zero predictions
    #         # Create masked predictions
    #         if normalized:
    #             mask = ~np.isclose(trial_data['actual'], np.zeros_like(trial_data['actual']), atol=1e-5)
    #             # Plot predictions only for non-silence portions
    #             x_coords = np.arange(len(trial_data['predictions']))
    #             plt.plot(x_coords[mask], trial_data['predictions'][mask], 
    #                     label='Predicted', color='red', alpha=0.6)
    #         else:
    #             mask = trial_data['actual'] != 0
    #             # Plot predictions only for non-silence portions
    #             x_coords = np.arange(len(trial_data['predictions']))
    #             plt.plot(x_coords[mask], trial_data['predictions'][mask], 
    #                     label='Predicted', color='red', alpha=0.6)
    #     else:
    #         plt.plot(trial_data['predictions'], label='Predicted', color='red', alpha=0.6)
        
        

    #     plt.title(f"{title}\nMAE: {trial_data['error']:.4f}\nFile: {trial_data['audio_file']}")
    #     plt.xlabel('Time (20ms bins)')
    #     plt.ylabel(f'{decoding_type} {units}')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.savefig(os.path.join(plot_dir, filename))
    #     plt.close()
    def plot_trial(trial_data, title, filename, decoding_type, units):
        plt.figure(figsize=(12, 6))
        plt.plot(trial_data['actual'], label='Actual', color='blue', alpha=0.6)
        
        if masked: #only plot non-zero predictions
            # Create masked predictions with NaN values
            masked_predictions = np.full_like(trial_data['predictions'], np.nan, dtype=float)
            
            if normalized:
                mask = ~np.isclose(trial_data['actual'], np.zeros_like(trial_data['actual']), atol=1e-5)
            else:
                mask = trial_data['actual'] != 0
                
            # Fill in only the non-silence portions
            masked_predictions[mask] = trial_data['predictions'][mask]
            
            # Plot predictions with gaps (NaN values will create gaps in the line)
            plt.plot(masked_predictions, label='Predicted', color='red', alpha=0.6)
        else: #plot everything
            plt.plot(trial_data['predictions'], label='Predicted', color='red', alpha=0.6)

        plt.title(f"{title}\nMAE: {trial_data['error']:.4f}\nFile: {trial_data['audio_file']}")
        plt.xlabel('Time (20ms bins)')
        plt.ylabel(f'{decoding_type} {units}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()
    for i, trial in enumerate(best_trials_test):
        plot_trial(
            trial,
            f'Best Trial (test) #{i+1}',
            f'best_trial_test_{i+1}.png',
            decoding_type,
            units
        )
    for i, trial in enumerate(worst_trials_test):
        plot_trial(
            trial,
            f'Worst Trial (test) #{i+1}',
            f'worst_trial_test_{i+1}.png',
            decoding_type,
            units
        )


    for i, trial in enumerate(best_trials_train):
        plot_trial(
            trial,
            f'Best Trial (train) #{i+1}',
            f'best_trial_train_{i+1}.png',
            decoding_type,
            units
        )
    for i, trial in enumerate(worst_trials_train):
        plot_trial(
            trial,
            f'Worst Trial (train) #{i+1}',
            f'worst_trial_train_{i+1}.png',
            decoding_type,
            units
        )

    # Create summary plot
    plt.figure(figsize=(15, 10))
    # figures
    for i, trial in enumerate(best_trials_test):
        plt.subplot(2, 3, i+1)
        plt.plot(trial['actual'], label='Actual', color='blue', alpha=0.6)
        # plt.plot(trial['predictions'], label='Predicted', color='red', alpha=0.6)
        if masked:
            # Create masked predictions with NaN values
            masked_predictions = np.full_like(trial['predictions'], np.nan, dtype=float)
            
            if normalized:
                mask = ~np.isclose(trial['actual'], np.zeros_like(trial['actual']), atol=1e-5)
            else:
                mask = trial['actual'] != 0
                
            # Fill in only the non-silence portions
            masked_predictions[mask] = trial['predictions'][mask]
            
            # Plot predictions with gaps (NaN values will create gaps in the line)
            plt.plot(masked_predictions, label='Predicted', color='red', alpha=0.6)
        else:
            plt.plot(trial['predictions'], label='Predicted', color='red', alpha=0.6)
        
        plt.title(f'Best #{i+1}\nMAE: {trial["error"]:.4f} (test set)')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()

    for i, trial in enumerate(worst_trials_test):
        plt.subplot(2, 3, i+4)
        plt.plot(trial['actual'], label='Actual', color='blue', alpha=0.6)
        # plt.plot(trial['predictions'], label='Predicted', color='red', alpha=0.6)
        if masked:
            # Create masked predictions with NaN values
            masked_predictions = np.full_like(trial['predictions'], np.nan, dtype=float)
            
            if normalized:
                mask = ~np.isclose(trial['actual'], np.zeros_like(trial['actual']), atol=1e-5)
            else:
                mask = trial['actual'] != 0
                
            # Fill in only the non-silence portions
            masked_predictions[mask] = trial['predictions'][mask]
            
            # Plot predictions with gaps (NaN values will create gaps in the line)
            plt.plot(masked_predictions, label='Predicted', color='red', alpha=0.6)
        else:
            plt.plot(trial['predictions'], label='Predicted', color='red', alpha=0.6)
        
        plt.title(f'Worst #{i+1}\nMAE: {trial["error"]:.4f} (test set)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'summary_plot_test.png'))
    plt.close()

    # Create summary plot
    plt.figure(figsize=(15, 10))
    # figures
    for i, trial in enumerate(best_trials_train):
        plt.subplot(2, 3, i+1)
        plt.plot(trial['actual'], label='Actual', color='blue', alpha=0.6)
        # plt.plot(trial['predictions'], label='Predicted', color='red', alpha=0.6)
        if masked:
           # Create masked predictions with NaN values
            masked_predictions = np.full_like(trial['predictions'], np.nan, dtype=float)
            
            if normalized:
                mask = ~np.isclose(trial['actual'], np.zeros_like(trial['actual']), atol=1e-5)
            else:
                mask = trial['actual'] != 0
                
            # Fill in only the non-silence portions
            masked_predictions[mask] = trial['predictions'][mask]
            
            # Plot predictions with gaps (NaN values will create gaps in the line)
            plt.plot(masked_predictions, label='Predicted', color='red', alpha=0.6)
        else:
            plt.plot(trial['predictions'], label='Predicted', color='red', alpha=0.6)
        plt.title(f'Best #{i+1}\nMAE: {trial["error"]:.4f} (train)')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()

    for i, trial in enumerate(worst_trials_train):
        plt.subplot(2, 3, i+4)
        plt.plot(trial['actual'], label='Actual', color='blue', alpha=0.6)
        # plt.plot(trial['predictions'], label='Predicted', color='red', alpha=0.6)
        if masked:
            # Create masked predictions with NaN values
            masked_predictions = np.full_like(trial['predictions'], np.nan, dtype=float)
            
            if normalized:
                mask = ~np.isclose(trial['actual'], np.zeros_like(trial['actual']), atol=1e-5)
            else:
                mask = trial['actual'] != 0
                
            # Fill in only the non-silence portions
            masked_predictions[mask] = trial['predictions'][mask]
            
            # Plot predictions with gaps (NaN values will create gaps in the line)
            plt.plot(masked_predictions, label='Predicted', color='red', alpha=0.6)
        else:
            plt.plot(trial['predictions'], label='Predicted', color='red', alpha=0.6)
        plt.title(f'Worst #{i+1}\nMAE: {trial["error"]:.4f} (train)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'summary_plot_train.png'))
    plt.close()

    # Save trial statistics 
    trial_stats = {
        'best_trials_test': [
            {
                'audio_file': trial['audio_file'],
                'error': trial['error'],
                'index': trial['index']
            } for trial in best_trials_test
        ],
        'worst_trials_test': [
            {
                'audio_file': trial['audio_file'],
                'error': trial['error'],
                'index': trial['index']
            } for trial in worst_trials_test
        ],
        'mean_error_test': np.mean([t['error'] for t in trial_errors_test]),
        'std_error_test': np.std([t['error'] for t in trial_errors_test]),

        'best_trials_train': [
            {
                'audio_file': trial['audio_file'],
                'error': trial['error'],
                'index': trial['index']
            } for trial in best_trials_train
        ],
        'worst_trials_train': [
            {
                'audio_file': trial['audio_file'],
                'error': trial['error'],
                'index': trial['index']
            } for trial in worst_trials_train
        ],
        'mean_error_train': np.mean([t['error'] for t in trial_errors_train]),
        'std_error_train': np.std([t['error'] for t in trial_errors_train]),
    }

    trial_stats['test_trial_indices'] = test_trial_indices.tolist()
    trial_stats.update({
        'pearson_r_test': pearson_r_test,
        'pearson_p_value_test': p_value_test,
        'pearson_r_train': pearson_r_train,
        'pearson_p_value_train': p_value_train
    })
    trial_stats.update({
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_maes': train_maes,
        'test_maes': test_maes
    })

    with open(os.path.join(plot_dir, 'trial_statistics.json'), 'w') as f:
        json.dump(trial_stats, f, indent=4)

    print(f"\nPlots and statistics saved to {plot_dir}/")
    print("Statistics on TEST set:\n")
    print("Best trials:")
    for i, trial in enumerate(best_trials_test):
        print(f"  {i+1}. File: {trial['audio_file']}, MAE: {trial['error']:.4f}")
    print("\nWorst trials:")
    for i, trial in enumerate(worst_trials_test):
        print(f"  {i+1}. File: {trial['audio_file']}, MAE: {trial['error']:.4f}")
    print(f"\nMean trial error: {trial_stats['mean_error_test']:.4f} ± {trial_stats['std_error_test']:.4f}")
    print(f"Test Pearson r: {trial_stats['pearson_r_test']:.4f} (p={trial_stats['pearson_p_value_test']:.4f})")

    logging.info(f"\nPlots and statistics saved to {plot_dir}/")
    logging.info("Best trials:")
    for i, trial in enumerate(best_trials_test):
        logging.info(f"  {i+1}. File: {trial['audio_file']}, MAE: {trial['error']:.4f}")
    logging.info("\nWorst trials:")
    for i, trial in enumerate(worst_trials_test):
        logging.info(f"  {i+1}. File: {trial['audio_file']}, MAE: {trial['error']:.4f}")
    logging.info(f"\nMean trial error: {trial_stats['mean_error_test']:.4f} ± {trial_stats['std_error_test']:.4f}")
    logging.info(f"Test Pearson r: {trial_stats['pearson_r_test']:.4f} (p={trial_stats['pearson_p_value_test']:.4f})")

    print("Statistics on TRAIN set:\n")
    print("Best trials:")
    for i, trial in enumerate(best_trials_train):
        print(f"  {i+1}. File: {trial['audio_file']}, MAE: {trial['error']:.4f}")
    print("\nWorst trials:")
    for i, trial in enumerate(worst_trials_train):
        print(f"  {i+1}. File: {trial['audio_file']}, MAE: {trial['error']:.4f}")
    print(f"\nMean trial error: {trial_stats['mean_error_train']:.4f} ± {trial_stats['std_error_train']:.4f}")
    print(f"Train Pearson r: {trial_stats['pearson_r_train']:.4f} (p={trial_stats['pearson_p_value_train']:.4f})")

    logging.info(f"\nPlots and statistics saved to {plot_dir}/")
    logging.info("Best trials:")
    for i, trial in enumerate(best_trials_train):
        logging.info(f"  {i+1}. File: {trial['audio_file']}, MAE: {trial['error']:.4f}")
    logging.info("\nWorst trials:")
    for i, trial in enumerate(worst_trials_train):
        logging.info(f"  {i+1}. File: {trial['audio_file']}, MAE: {trial['error']:.4f}")
    logging.info(f"\nMean trial error: {trial_stats['mean_error_train']:.4f} ± {trial_stats['std_error_train']:.4f}")
    logging.info(f"Train Pearson r: {trial_stats['pearson_r_train']:.4f} (p={trial_stats['pearson_p_value_train']:.4f})")

    return model, test_data, test_labels


def save_config(config_path, args, run_id, decoding_type, units, file_tag):
    config = {
        'args': vars(args),
        'run_id': run_id,
        'decoding_type': decoding_type,
        'units': units,
        'file_tag': file_tag,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def decode_volume(model, test_data, window_size=5, stride=1, batch_size=32):
    model = model.to(device)
    model.eval()
    windowed_data = test_data  #assuming test_data is already windowed
    
    # Convert to torch tensor
    test_dataset = torch.FloatTensor(windowed_data).to(device)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, 
    #     batch_size=batch_size,
    #     shuffle=False
    # )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:

            batch = batch.to(device)
            # Get model predictions
            outputs = model(batch)
            predictions.extend(outputs.cpu().numpy())
            
    predictions = np.array(predictions)
    
    return predictions

def main():
    args = parse_args()
    # decoding_type = 'volume'
    # units = '(db)'
    # file_tag = 'db'
    data_type = args.type #hz or db
    masked = (args.masked == "True")
    if data_type == 'db':
        decoding_type = 'volume'
        units = '(db)'
        file_tag = 'db'
    elif data_type == 'hz':
        decoding_type = 'pitch' 
        units = '(hz)'
        file_tag = 'hz'
    else:
        # Handle unexpected data_type
        print("Invalid data type:", data_type)
    
    if args.normalized == "False":
        normal_tag = ''
    else:
        normal_tag = '_normalized'
    ## To prevent the parallel conflicts-- generate a unique (ish) ID for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Added microseconds
    random_suffix = f"{random.randint(0, 99999):04d}"
    run_id = f"{timestamp}_{random_suffix}_{file_tag}"

    log_file = os.path.join(args.output_dir, 'logs', f'{decoding_type}_log_{run_id}.txt')
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    time.sleep(random.uniform(0, 1))

    try:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Starting {decoding_type} decoding training with parameters: {vars(args)}")

        os.makedirs(args.output_dir, exist_ok=True)
        neural_data = np.load(f"/home/groups/henderj/rzwang/processed_data_{file_tag}/neural_data_sbp.npy")
        volume_labels = np.load(f"/home/groups/henderj/rzwang/processed_data_{file_tag}/labels{normal_tag}.npy")
        trial_info = np.load(f"/home/groups/henderj/rzwang/processed_data_{file_tag}/trial_info.npy", allow_pickle=True)
        
        #train
        model, test_neural_data, test_labels = train_volume_decoder(
            neural_data=neural_data,
            labels=volume_labels,
            trial_info=trial_info,
            decoder_class=CNNVolumeDecoder,
            run_id=run_id,
            decoding_type=decoding_type,
            units=units,
            normalized=args.normalized,
            masked=masked,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate
        )

        #save
        model_path = os.path.join(args.output_dir, 'models', f'{decoding_type}_model_{run_id}.pt')
        torch.save(model.state_dict(), model_path)
        predictions = decode_volume(
            model=model,
            test_data=test_neural_data,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size
        )
        

        if masked:
            logging.info("We are masked... (for debugging)\n")
            mse, mae, r2, pearson_r, p_value = calculate_masked_metrics(
                predictions, test_labels, normalized=(args.normalized == "True"))
        else:
            logging.info("We are unmasked... (for debug)\n")
            # Calculate and log metrics
            mse = np.mean((predictions - test_labels) ** 2)
            mae = np.mean(np.abs(predictions - test_labels))
            r2 = r2_score(test_labels, predictions)
            pearson_r, p_value = stats.pearsonr(test_labels, predictions)
        
        logging.info(f"Test MSE: {mse:.4f}")
        logging.info(f"Test MAE: {mae:.4f}")
        logging.info(f"Test R2: {r2:.4f}")
        logging.info(f"Test Pearson r: {pearson_r:.4f} (p={p_value:.4f})")
       
        # Create visualization plots
        plt.figure(figsize=(12, 6))
        
        # Plot actual vs predicted values
        plt.subplot(1, 2, 1)
        plt.scatter(test_labels, predictions, alpha=0.5)
        plt.plot([test_labels.min(), test_labels.max()], 
                [test_labels.min(), test_labels.max()], 
                'r--', lw=2)
        plt.xlabel(f'Actual {decoding_type} {units}')
        plt.ylabel(f'Predicted {decoding_type} {units}')
        plt.title(f'Actual vs Predicted {decoding_type} {units}')
        
        # Plot prediction time series
        plt.subplot(1, 2, 2)
        sample_size = min(1000, len(predictions))  # Plot first 1000 points or all if less
        time_points = np.arange(sample_size)
        plt.plot(time_points, test_labels[:sample_size], 'b-', label='Actual', alpha=0.7)
        plt.plot(time_points, predictions[:sample_size], 'r-', label='Predicted', alpha=0.7)
        plt.xlabel('Time Bins')
        plt.ylabel(f'{decoding_type} {units}')
        plt.title(f'{decoding_type} Prediction Time Series {units}')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plots
        plot_path = f'/home/groups/henderj/rzwang/figures/{decoding_type}_prediction_{run_id}.png'
        plt.savefig(plot_path)
        plt.close()
        
        # Save predictions and actual values for further analysis
        results = {
            'predictions': predictions,
            'actual': test_labels,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': p_value
        }
        results_path = os.path.join(args.output_dir, 'npys', f'{decoding_type}_results_{run_id}.npy')
        np.save(results_path, results)
        # In main():
        config_path = os.path.join(args.output_dir, 'configs', f'config_{run_id}.json')
        save_config(config_path, args, run_id, decoding_type=decoding_type, units=units, file_tag=file_tag)

        logging.info("Training and evaluation completed successfully")

        logging.info(f"Model saved to: {model_path}")
        logging.info(f"Results saved to: {results_path}")
        logging.info(f"Plots saved to: {plot_path}")
        logging.info(f"Config saved to: {config_path}")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise e

def parse_args():
    parser = argparse.ArgumentParser(description='Train neural network model for decoding')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--type', type=str, default='hz') #hz or db
    parser.add_argument('--normalized', type=str, default="False")
    parser.add_argument('--masked', type=str, default="True")
    return parser.parse_args()

if __name__ == "__main__":
    main()
