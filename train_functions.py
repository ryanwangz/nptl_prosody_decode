#functions for training decoders
#pilot of the model

import torch
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
# from torchsummary import summary
from scipy import stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#some other ones: torchsummary, sklearn metrics, early stopping?
from datetime import datetime
# from torchsummary import summary
from scipy import stats
#some other ones: torchsummary, sklearn metrics, early stopping? seaborn/matplotlib
import seaborn as sns
from decoders import *

#Function to create windowed data
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

def create_label_windows(labels, window_size, stride=1):
    """Create label windows by taking the rounded mean of labels in each window"""
    n_samples = (len(labels) - window_size) // stride + 1
    windowed_labels = np.zeros(n_samples)
    for i in range(n_samples):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windowed_labels[i] = round(np.mean(labels[start_idx:end_idx]))  # round gives mode for binary
    return windowed_labels
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

def create_windows_and_labels(neural_data, labels, window_size=5, stride=1):
    """Create sliding windows from data and corresponding label modes
    
    Parameters:
    -----------
    neural_data : np.ndarray
        Neural data (n_timepoints, n_channels)
    labels : np.ndarray
        Phoneme labels (n_timepoints,)
    window_size : int
        Size of sliding window
    stride : int
        Stride for sliding window
        
    Returns:
    --------
    windows : np.ndarray
        Windowed neural data (n_windows, n_channels, window_size)
    window_labels : np.ndarray
        Mode of labels for each window
    """
    n_timepoints, n_channels = neural_data.shape
    n_windows = ((n_timepoints - window_size) // stride) + 1
    
    # Create windows for neural data
    windows = np.zeros((n_windows, n_channels, window_size))
    window_labels = np.zeros(n_windows, dtype=labels.dtype)
    
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        # Window the neural data
        windows[i] = neural_data[start_idx:end_idx].T
        
        # Get mode of labels in this window
        try:
            # For newer scipy versions
            window_labels[i] = stats.mode(labels[start_idx:end_idx], keepdims=True)[0][0]
        except TypeError:
            # For older scipy versions
            window_labels[i] = stats.mode(labels[start_idx:end_idx])[0][0]
    return windows, window_labels


# training function for phoneme decoder
def train_phoneme_decoder(neural_data, labels, trial_info, decoder_class, window_size=5, stride=1, 
                         batch_size=32, n_epochs=10, learning_rate=0.001, train_split=0.8):
    """
    Train phoneme decoder
    
    neural_data : np.ndarray
        Neural data (n_timepoints, n_channels)
    labels : np.ndarray
        Phoneme labels (n_timepoints,)
    decoder_class : torch.nn.Module
        Decoder class to use
    """
    
    # Create windows
    # windowed_data = create_windows(neural_data, window_size, stride)
    # windowed_labels = labels[window_size-1::stride]  # align labels with windows
    windowed_data, windowed_labels = create_windows_and_labels(
        neural_data, labels, window_size, stride
    )
    
    #TODO: fix this train and test split, xval
    #extract trial idxs from trial_info

    #previous (take train/test split at index)
    # n_samples = len(windowed_data)
    # n_train = int(n_samples * train_split)

    # train_data = windowed_data[:n_train]
    # train_labels = windowed_labels[:n_train]
    # test_data = windowed_data[n_train:]
    # test_labels = windowed_labels[n_train:]

    #Random train/test split by trial (rounded)
    # First, organize data by trials
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

    # Convert to numpy
    trial_data = np.array(trial_data)
    trial_labels = np.array(trial_labels)
    
    #changed to trial indices here (from random)
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

    #Calculate class weights to handle imbalance
    # unique_labels, label_counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)
    unique_labels = np.unique(labels)  # Use ALL labels, not just training set
    n_classes = len(unique_labels)
    label_counts = np.zeros(n_classes)
    # Count occurrences of each class in training set
    for i, label in enumerate(unique_labels):
        label_counts[i] = np.sum(train_labels == label)
    
    #Add small constant to avoid division by zero
    label_counts = label_counts + 1e-5
    # class_weights = torch.FloatTensor([total_samples / (len(unique_labels) * count) 
    #                                  for count in label_counts])
    class_weights = torch.FloatTensor(total_samples / (n_classes * label_counts))
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_data),
        torch.LongTensor(train_labels)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_data),
        torch.LongTensor(test_labels)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    #nitialize model (not working rn with sherlock rip)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = decoder_class(n_channels=neural_data.shape[1], window_size=window_size)
    # model = model.to(device)
    
    #Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #GPT recommends trying "weighted adam"? for not staying on silent all the time
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001) #TODO: adjust as necessary

    # NEW: Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    # NEW: Early stopping setup
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    min_improvement = 0.001


    #train
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in train_loader:
            # batch_data = batch_data.to(device)
            # batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()            
            # NEW: Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        # NEW: Calculate average loss for scheduler
        avg_train_loss = train_loss / len(train_loader)
        # NEW: Update learning rate based on loss
        scheduler.step(avg_train_loss)
        # NEW: Early stopping check
        if avg_train_loss < (best_loss - min_improvement):
            best_loss = avg_train_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                # batch_data = batch_data.to(device)
                # batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                test_loss += loss.item()
                
                _, predicted = outputs.max(1)
                test_total += batch_labels.size(0)
                test_correct += predicted.eq(batch_labels).sum().item()
        
        # Print epoch statistics
        print(f'Epoch: {epoch+1}/{n_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {100.*correct/total:.2f}%')
        print(f'Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {100.*test_correct/test_total:.2f}%')
        print('--------------------')
    
    return model, test_data, test_labels

'''
Function that trains a decoder, prints progress of test accuracy

'''
def train_decoder_silence(neural_data, labels, trial_info, decoder_class, window_size=5, stride=1, batch_size=32, n_epochs=10, learning_rate=0.001, train_split=0.8):
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
    
    ### Previous split: random assignment
    # # Split into train/test sets
    # n_samples = len(X)
    # n_train = int(train_split * n_samples)
    # indices = np.random.permutation(n_samples)
    
    # #TODO: add train/test characterization (ie print distribution of labels, etc)

    # train_indices = indices[:n_train]
    # test_indices = indices[n_train:]

    ###new split: assign by trial train/test

    trial_data = []
    trial_labels = []
    current_idx = 0
    
    for trial in trial_info:
        n_bins = trial['n_bins']
        # Calculate how many windows we'll get from this trial
        n_windows = (n_bins - window_size) // stride + 1
        
        trial_windows = X[current_idx:current_idx + n_windows]
        trial_window_labels = y[current_idx:current_idx + n_windows]
        
        trial_data.append(trial_windows)
        trial_labels.append(trial_window_labels)
        
        current_idx += n_windows
    trial_data = np.array(trial_data)
    trial_labels = np.array(trial_labels)
    
    #changed to trial indices here (from random)
    n_trials = len(trial_info)
    n_train_trials = int(n_trials * train_split) #rounding here
    trial_indices = np.random.permutation(n_trials)
    train_trial_indices = trial_indices[:n_train_trials]
    test_trial_indices = trial_indices[n_train_trials:]
    

    train_data = np.concatenate(trial_data[train_trial_indices])
    train_labels = np.concatenate(trial_labels[train_trial_indices])
    test_data = np.concatenate(trial_data[test_trial_indices])
    test_labels = np.concatenate(trial_labels[test_trial_indices])
    

    
    
    # X_train = torch.FloatTensor(X[train_indices]).to(device)
    # y_train = torch.FloatTensor(y[train_indices]).to(device)
    # X_test = torch.FloatTensor(X[test_indices]).to(device)
    # y_test = torch.FloatTensor(y[test_indices]).to(device)
    ### previous
    # X_train = torch.FloatTensor(X[train_indices])
    # y_train = torch.FloatTensor(y[train_indices])
    # X_test = torch.FloatTensor(X[test_indices])
    # y_test = torch.FloatTensor(y[test_indices])
    X_train = torch.FloatTensor(train_data)
    y_train = torch.FloatTensor(train_labels)
    X_test = torch.FloatTensor(test_data)
    y_test = torch.FloatTensor(test_labels)
    
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
            # Move inputs and labels to the device (not working with Sherlock at the moment)
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            outputs = model(inputs)

            # Convert outputs to binary predictions (assuming a threshold of 0.5)
            predictions = (outputs >= 0.5).float()

            # Append predictions and labels to the lists
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #calculate confusion matrix
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


'''
Function that trains a decoder, prints progress of test accuracy

'''
def train_decoder(neural_data, labels, decoder_class, window_size=5, stride=1, batch_size=32, n_epochs=10, learning_rate=0.001, train_split=0.8):
    '''
    params:
    neural_data: 
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    X_train = torch.FloatTensor(X[train_indices]).to(device)
    y_train = torch.FloatTensor(y[train_indices]).to(device)
    X_test = torch.FloatTensor(X[test_indices]).to(device)
    y_test = torch.FloatTensor(y[test_indices]).to(device)
    
    #Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train.unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test.unsqueeze(1))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #Initialize model and optimizer
    model = decoder_class(n_channels=neural_data.shape[1], window_size=window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #TODO: do we want to use Adam
    criterion = nn.BCELoss() #TODO: change this loss function from binary cross entrop
    
    # Training loop
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


# '''
# Function that trains a decoder, prints progress of test accuracy

# '''
# def train_decoder_dual(sbp_data, tc_data, labels, decoder_class, window_size=5, stride=1, batch_size=32, n_epochs=10, learning_rate=0.001, train_split=0.8):
#     '''
#     params:
#     neural_data: 
#     '''
#     # Create windowed data
#     X = create_windows(neural_data, window_size, stride)
    
#     #For labels, take the mode within each window
#     y = np.zeros(X.shape[0])
#     for i in range(X.shape[0]):
#         start_idx = i * stride
#         end_idx = start_idx + window_size
#         y[i] = round(np.mean(labels[start_idx:end_idx])) #round should give mode for binary
    
#     # Split into train/test sets
#     n_samples = len(X)
#     n_train = int(train_split * n_samples)
#     indices = np.random.permutation(n_samples)
    
#     #TODO: add train/test characterization (ie print distribution of labels, etc)

#     train_indices = indices[:n_train]
#     test_indices = indices[n_train:]
    
#     X_train = torch.FloatTensor(X[train_indices])
#     y_train = torch.FloatTensor(y[train_indices])
#     X_test = torch.FloatTensor(X[test_indices])
#     y_test = torch.FloatTensor(y[test_indices])
    
#     #Create data loaders
#     train_dataset = torch.utils.data.TensorDataset(X_train, y_train.unsqueeze(1))
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
#     #Initialize model and optimizer
#     model = decoder_class(n_channels=neural_data.shape[1], window_size=window_size)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #TODO: do we want to use Adam
#     criterion = nn.BCELoss() #TODO: change this loss function from binary cross entrop
    
#     # Training loop
#     for epoch in range(n_epochs):
#         model.train()
#         total_loss = 0
        
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         # Evaluate on test set
#         model.eval()
#         with torch.no_grad():
#             test_outputs = model(X_test)
#             test_loss = criterion(test_outputs, y_test.unsqueeze(1))
#             test_preds = (test_outputs > 0.5).float()
#             accuracy = (test_preds == y_test.unsqueeze(1)).float().mean()
        
#         print("Epoch {}/{}: Train Loss = {:.4f}, Test Loss = {:.4f}, Test Accuracy = {:.4f}".format(
#             epoch+1, n_epochs, total_loss/len(train_loader), test_loss.item(), accuracy.item()))
    
#     return model

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




def evaluate_decoder_binary(model, test_loader, device):
    """
    Evaluate decoder and create confusion matrix
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained decoder model
    test_loader : torch.utils.data.DataLoader
        DataLoader containing test data
    device : torch.device
        Device to run evaluation on
    
    Returns:
    --------
    confusion_mat : numpy.ndarray confusion matrix
    accuracy : float
        Test accuracy
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            predictions = (outputs >= 0.5).float()  # Binary threshold at 0.5
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy = (all_preds == all_labels).mean()
    
    return conf_matrix, accuracy
