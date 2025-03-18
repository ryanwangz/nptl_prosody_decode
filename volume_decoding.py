
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
from datetime import datetime
# from torchsummary import summary
from scipy import stats
#some other ones: torchsummary, sklearn metrics, early stopping? seaborn/matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from decoders import *
from train_functions import *


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

def train_volume_decoder(neural_data, labels, trial_info, decoder_class, window_size=5, stride=1, 
                        batch_size=32, n_epochs=10, learning_rate=0.001, train_split=0.8):
    
    # Create windows (same as before)
    windowed_data, windowed_labels = create_windows_and_labels(
        neural_data, labels, window_size, stride
    )
    
    trial_data = []
    trial_labels = []
    current_idx = 0
    
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
    
    #Split the data
    train_data = np.concatenate(trial_data[train_trial_indices])
    train_labels = np.concatenate(trial_labels[train_trial_indices])
    test_data = np.concatenate(trial_data[test_trial_indices])
    test_labels = np.concatenate(trial_labels[test_trial_indices])
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_data),
        torch.FloatTensor(train_labels)  # Changed from LongTensor
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_data),
        torch.FloatTensor(test_labels)  # Changed from LongTensor
    )
    
    # Create data loaders (same as before)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False # Do we want to shuffle data here?
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Init
    model = decoder_class(n_channels=neural_data.shape[1], window_size=window_size)
    
    # Use MSE loss instead of CrossEntropyLoss for continuous
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001) # may want to switch back to Adam (remnant of phoneme decoder)
    # scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    min_improvement = 0.001

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_mae = 0  #Mean Absolute Error
        
        for batch_data, batch_labels in train_loader:
            # Forward pass
            outputs = model(batch_data)
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
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()
                test_mae += torch.mean(torch.abs(outputs - batch_labels)).item()
        
        avg_test_loss = test_loss / len(test_loader)
        avg_test_mae = test_mae / len(test_loader)
        
        # Update scheduler
        scheduler.step(avg_test_loss)
        
        # Early stopping check
        if avg_test_loss < (best_loss - min_improvement):
            best_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
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
    
    print(f'Saving test statistics \n')

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    # make predictions on test set
    predictions = decode_volume(
        model=model,
        test_data=test_data,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size
    )
    
    #Calculate and log metrics
    accuracy = np.mean(np.abs(predictions - test_labels)).item()
    logging.info(f"Test MAE: {accuracy:.4f}")

    #save a handful of trial plots here (maybe 3 best and 3 worst trials?)
    
    # Create directory for saving plots if it doesn't exist
    plot_dir = f'/home/groups/henderj/rzwang/figures/volume_plots/vol_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Calculate error for each trial
    trial_errors = []
    trial_predictions = []
    trial_actual = []
    
    current_idx = 0
    for i, trial in enumerate(trial_info):
        if i in test_trial_indices:  # Only evaluate test trials
            n_bins = trial['n_bins']
            n_windows = (n_bins - window_size) // stride + 1
            
            trial_pred = predictions[current_idx:current_idx + n_windows]
            trial_true = test_labels[current_idx:current_idx + n_windows]
            trial_mae = np.mean(np.abs(trial_pred - trial_true))
            
            trial_errors.append({
                'index': i,
                'error': trial_mae,
                'predictions': trial_pred,
                'actual': trial_true,
                'audio_file': trial['audio_file']
            })
            
            current_idx += n_windows

    # Sort trials by error
    sorted_trials = sorted(trial_errors, key=lambda x: x['error'])
    
    # Plot best 3 and worst 3 trials
    best_trials = sorted_trials[:3]
    worst_trials = sorted_trials[-3:]
    
    def plot_trial(trial_data, title, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(trial_data['actual'], label='Actual', color='blue', alpha=0.6)
        plt.plot(trial_data['predictions'], label='Predicted', color='red', alpha=0.6)
        plt.title(f"{title}\nMAE: {trial_data['error']:.4f}\nFile: {trial_data['audio_file']}")
        plt.xlabel('Time (100ms bins?)')
        plt.ylabel('Volume (db)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()

    for i, trial in enumerate(best_trials):
        plot_trial(
            trial,
            f'Best Trial #{i+1}',
            f'best_trial_{i+1}.png'
        )
    for i, trial in enumerate(worst_trials):
        plot_trial(
            trial,
            f'Worst Trial #{i+1}',
            f'worst_trial_{i+1}.png'
        )

    # Create summary plot
    plt.figure(figsize=(15, 10))
    # figures
    for i, trial in enumerate(best_trials):
        plt.subplot(2, 3, i+1)
        plt.plot(trial['actual'], label='Actual', color='blue', alpha=0.6)
        plt.plot(trial['predictions'], label='Predicted', color='red', alpha=0.6)
        plt.title(f'Best #{i+1}\nMAE: {trial["error"]:.4f}')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()

    for i, trial in enumerate(worst_trials):
        plt.subplot(2, 3, i+4)
        plt.plot(trial['actual'], label='Actual', color='blue', alpha=0.6)
        plt.plot(trial['predictions'], label='Predicted', color='red', alpha=0.6)
        plt.title(f'Worst #{i+1}\nMAE: {trial["error"]:.4f}')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'summary_plot.png'))
    plt.close()

    # Save trial statistics 
    trial_stats = {
        'best_trials': [
            {
                'audio_file': trial['audio_file'],
                'error': trial['error'],
                'index': trial['index']
            } for trial in best_trials
        ],
        'worst_trials': [
            {
                'audio_file': trial['audio_file'],
                'error': trial['error'],
                'index': trial['index']
            } for trial in worst_trials
        ],
        'mean_error': np.mean([t['error'] for t in trial_errors]),
        'std_error': np.std([t['error'] for t in trial_errors])
    }

    with open(os.path.join(plot_dir, 'trial_statistics.json'), 'w') as f:
        json.dump(trial_stats, f, indent=4)

    print(f"\nPlots and statistics saved to {plot_dir}/")
    print("Best trials:")
    for i, trial in enumerate(best_trials):
        print(f"  {i+1}. File: {trial['audio_file']}, MAE: {trial['error']:.4f}")
    print("\nWorst trials:")
    for i, trial in enumerate(worst_trials):
        print(f"  {i+1}. File: {trial['audio_file']}, MAE: {trial['error']:.4f}")
    print(f"\nMean trial error: {trial_stats['mean_error']:.4f} ± {trial_stats['std_error']:.4f}")


    return model, test_data, test_labels


def decode_volume(model, test_data, window_size=5, stride=1, batch_size=32):
    
    model.eval()
    windowed_data = test_data  #assuming test_data is already windowed
    
    # Convert to torch tensor
    test_dataset = torch.FloatTensor(windowed_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False
    )
    
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Get model predictions
            outputs = model(batch)
            predictions.extend(outputs.cpu().numpy())
            
    predictions = np.array(predictions)
    
    return predictions

def main():
    args = parse_args()
    log_file = os.path.join(args.output_dir, f'volume_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    try:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Starting volume decoding training with parameters: {vars(args)}")

        os.makedirs(args.output_dir, exist_ok=True)
        neural_data = np.load("/home/groups/henderj/rzwang/processed_data_db/neural_data_sbp.npy")
        volume_labels = np.load("/home/groups/henderj/rzwang/processed_data_db/labels_normalized.npy")
        trial_info = np.load("/home/groups/henderj/rzwang/processed_data_db/trial_info.npy", allow_pickle=True)
        
        #train
        model, test_neural_data, test_labels = train_volume_decoder(
            neural_data=neural_data,
            labels=volume_labels,
            trial_info=trial_info,
            decoder_class=CNNVolumeDecoder,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate
        )

        #save
        model_path = os.path.join(args.output_dir, f'volume_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
        torch.save(model.state_dict(), model_path)
        predictions = decode_volume(
            model=model,
            test_data=test_neural_data,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size
        )
        
        # Calculate and log metrics
        mse = np.mean((predictions - test_labels) ** 2)
        mae = np.mean(np.abs(predictions - test_labels))
        r2 = r2_score(test_labels, predictions)
        
        logging.info(f"Test MSE: {mse:.4f}")
        logging.info(f"Test MAE: {mae:.4f}")
        logging.info(f"Test R2: {r2:.4f}")
       
        # Create visualization plots
        plt.figure(figsize=(12, 6))
        
        # Plot actual vs predicted values
        plt.subplot(1, 2, 1)
        plt.scatter(test_labels, predictions, alpha=0.5)
        plt.plot([test_labels.min(), test_labels.max()], 
                [test_labels.min(), test_labels.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Volume (dB)')
        plt.ylabel('Predicted Volume (dB)')
        plt.title('Actual vs Predicted Volume')
        
        # Plot prediction time series
        plt.subplot(1, 2, 2)
        sample_size = min(1000, len(predictions))  # Plot first 1000 points or all if less
        time_points = np.arange(sample_size)
        plt.plot(time_points, test_labels[:sample_size], 'b-', label='Actual', alpha=0.7)
        plt.plot(time_points, predictions[:sample_size], 'r-', label='Predicted', alpha=0.7)
        plt.xlabel('Time Bins')
        plt.ylabel('Volume (dB)')
        plt.title('Volume Prediction Time Series')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plots
        plot_path = f'/home/groups/henderj/rzwang/figures/volume_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path)
        plt.close()
        
        # Save predictions and actual values for further analysis
        results = {
            'predictions': predictions,
            'actual': test_labels,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        results_path = os.path.join(args.output_dir, f'volume_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npy')
        np.save(results_path, results)
        
        logging.info("Training and evaluation completed successfully")
        logging.info(f"Model saved to: {model_path}")
        logging.info(f"Results saved to: {results_path}")
        logging.info(f"Plots saved to: {plot_path}")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise e

def parse_args():
    parser = argparse.ArgumentParser(description='Train neural network model for volume decoding')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--output_dir', type=str, default='results')
    return parser.parse_args()

if __name__ == "__main__":
    main()
