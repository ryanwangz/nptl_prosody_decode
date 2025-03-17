# testing the running features on this VSCode server 
print("running")

#pilot of the model
import argparse
import torch
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
import logging
from datetime import datetime
# from torchsummary import summary
from scipy import stats
#some other ones: torchsummary, sklearn metrics, early stopping? seaborn/matplotlib
import seaborn as sns

from decoders import *
from train_functions import *

# Function to create windowed data
def create_windows(data, window_size, stride=1):
    """
    Create windowed data for both neural data and labels

    Parameters:
    -----------
    data : np.ndarray
        Neural data (n_timepoints, n_channels)
    window_size : int
        Number of time bins in each window
    stride : int
        Number of bins to stride between windows

    Returns:
    --------
    windowed_data : np.ndarray
        Windowed data (n_windows, n_channels, window_size)
    """

    n_samples = data.shape[0]
    n_windows = ((n_samples - window_size) // stride) + 1

    windows = np.zeros((n_windows, data.shape[1], window_size))

    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        # Transpose to get (n_channels, window_size)
        windows[i] = data[start_idx:end_idx].T

    return windows


def parse_args():

    parser = argparse.ArgumentParser(description='Train neural network model')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--output_dir', type=str, default='results')
    return parser.parse_args()


#ideas: can take structure inspiration from willsey paper, maybe wairgarkar paper as well
class CNNSilenceDecoderSimple(nn.Module):
    '''
    Simple silence CNN decoder. one layer. mostly for testing pipeline of things
    '''
    def __init__(self, n_channels=256, window_size=5):
        super(CNNSilenceDecoderSimple, self).__init__()
        self.window_size = window_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * window_size, 1),
            nn.Sigmoid() #output sigmoid for binary guess
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


'''
Function that trains a decoder, prints progress of test accuracy

'''
def train_decoder_silence(neural_data, labels, decoder_class, window_size=5, stride=1, batch_size=32, n_epochs=10, learning_rate=0.001, train_split=0.8):
    '''
    params:
    neural_data: 
    '''
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create windowed data
    X = create_windows(neural_data, window_size, stride)
    
    #For labels, take the mode within each window
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        start_idx = i * stride
        end_idx = start_idx + window_size
        y[i] = round(np.mean(labels[start_idx:end_idx])) #round should give mode for binary
    
    # Split into train/test sets
    n_samples = len(X)
    n_train = int(train_split * n_samples)
    indices = np.random.permutation(n_samples)
    
    #TODO: add train/test characterization (ie print distribution of labels, etc)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    
    # X_train = torch.FloatTensor(X[train_indices]).to(device)
    # y_train = torch.FloatTensor(y[train_indices]).to(device)
    # X_test = torch.FloatTensor(X[test_indices]).to(device)
    # y_test = torch.FloatTensor(y[test_indices]).to(device)
    X_train = torch.FloatTensor(X[train_indices])
    y_train = torch.FloatTensor(y[train_indices])
    X_test = torch.FloatTensor(X[test_indices])
    y_test = torch.FloatTensor(y[test_indices])
    
    #Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train.unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test.unsqueeze(1))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #Initialize model and optimizer
    model = decoder_class(n_channels=neural_data.shape[1], window_size=window_size)
    print(f"Model is {model.__class__.__name__}\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #TODO: do we want to use Adam
    criterion = nn.BCELoss() #TODO: change this loss function from binary cross entrop
    
    #traibn
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        #TODO: xval
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        #eval
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test.unsqueeze(1))
            test_preds = (test_outputs > 0.5).float()
            accuracy = (test_preds == y_test.unsqueeze(1)).float().mean()
        
        print("Epoch {}/{}: Train Loss = {:.4f}, Test Loss = {:.4f}, Test Accuracy = {:.4f}".format(
            epoch+1, n_epochs, total_loss/len(train_loader), test_loss.item(), accuracy.item()))
    
    print("Done training-- evaluation with test set:\n")

    #confusion matrix stuff below
    model.eval()

    # Initialize lists to store predictions and labels
    all_preds = []
    all_labels = []

    # Get predictions for test set
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the device
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # Get model outputs
            outputs = model(inputs)

            # Convert outputs to binary predictions (assuming a threshold of 0.5)
            predictions = (outputs >= 0.5).float()

            # Append predictions and labels to the lists
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Save confusion matrix as a NumPy array
    np.save('confusion_matrix.npy', conf_matrix)
        # Optionally, plot and save the confusion matrix as an image
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Silence', 'Silence'],
                yticklabels=['No Silence', 'Silence'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'/home/groups/henderj/rzwang/figures/confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()
    return model

def evaluate_model(model, neural_data, labels, window_size=5, stride=1):
    model.eval()
    X = create_windows(neural_data, window_size, stride)
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        start_idx = i * stride
        end_idx = start_idx + window_size
        y[i] = np.mean(labels[start_idx:end_idx])
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    with torch.no_grad():
        outputs = model(X)
        preds = (outputs > 0.5).float()
        accuracy = (preds.squeeze() == y).float().mean()
    
    return accuracy.item()


def main():
    args = parse_args()
    log_file = os.path.join(args.output_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    try:
        logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Starting training with parameters: {vars(args)}")


        os.makedirs(args.output_dir, exist_ok=True)
        neural_data = np.load("/home/groups/henderj/rzwang/processed_data_silence/neural_data_sbp.npy") #TODO: need to add other neural data
        labels = np.load("/home/groups/henderj/rzwang/processed_data_silence/labels.npy")
        model = train_decoder_silence(neural_data, labels, DeepCNNDecoder, window_size = args.window_size, stride = args.stride, batch_size = args.batch_size,
                            n_epochs = args.n_epochs, learning_rate = args.learning_rate, train_split = args.train_split)
        
        model_path = os.path.join(args.output_dir, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
        torch.save(model.state_dict(), model_path)
        logging.info(f'Training completed successfully at time {datetime.now().strftime("%Y%m%d_%H%M%S")}')
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise e



if __name__ == "__main__":
    main()
