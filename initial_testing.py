# testing the running features on this VSCode server 
print("running")

#pilot of the model

import torch
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
# from torchsummary import summary
from scipy import stats
#some other ones: torchsummary, sklearn metrics, early stopping?

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


# # Training script modifications
# def train_decoder(neural_data, labels, window_size=5, stride=1):
#     # Create windowed data
#     X = create_windows(neural_data, window_size, stride)

#     # For labels, take the mode within each window
#     y = np.zeros(X.shape[0])
#     for i in range(X.shape[0]):
#         start_idx = i * stride
#         end_idx = start_idx + window_size
#         y[i] = np.mean(labels[start_idx:end_idx])

#     #Convert to PyTorch tensors
#     X = torch.FloatTensor(X)
#     y = torch.FloatTensor(y)

#     #Create model and train
#     model = CNNSilenceDecoder(n_channels=neural_data.shape[1],
#                              window_size=window_size)

#     #now train the decoder




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
def train_decoder(neural_data, labels, decoder_class, window_size=5, stride=1, batch_size=32, n_epochs=10, learning_rate=0.001, train_split=0.8):
    '''
    params:
    neural_data: 
    '''
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
    
    X_train = torch.FloatTensor(X[train_indices])
    y_train = torch.FloatTensor(y[train_indices])
    X_test = torch.FloatTensor(X[test_indices])
    y_test = torch.FloatTensor(y[test_indices])
    
    #Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train.unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    #Initialize model and optimizer
    model = decoder_class(n_channels=neural_data.shape[1], window_size=window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #TODO: do we want to use Adam
    criterion = nn.BCELoss() #TODO: change this loss function from binary cross entrop
    
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
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test.unsqueeze(1))
            test_preds = (test_outputs > 0.5).float()
            accuracy = (test_preds == y_test.unsqueeze(1)).float().mean()
        
        print("Epoch {}/{}: Train Loss = {:.4f}, Test Loss = {:.4f}, Test Accuracy = {:.4f}".format(
            epoch+1, n_epochs, total_loss/len(train_loader), test_loss.item(), accuracy.item()))
    
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

if __name__ == "__main__":
    neural_data = np.load("/home/groups/henderj/rzwang/processed_data/neural_data_sbp.npy") #TODO: need to add other neural data

    labels = np.load("/home/groups/henderj/rzwang/processed_data/labels.npy")

    # Train the model
    window_size = 5  # 100ms window (assuming 20ms bins)
    stride = 1
    model = train_decoder(neural_data, labels, decoder_class = DeepCNNDecoder, window_size=window_size, stride=stride,batch_size=16,n_epochs=40,learning_rate=0.001)

    # Save the trained model
    torch.save(model.state_dict(), "/home/groups/henderj/rzwang/decoders/silence_decoder2.pt")
