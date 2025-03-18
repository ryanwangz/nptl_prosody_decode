#for testing decoders in more detail (what types of errors are they making?)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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
#some other ones: torchsummary, sklearn metrics, early stopping?

from decoders import *
from train_functions import *

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_and_evaluate_model(model_path, test_data_in, test_labels_in, window_size=5, batch_size=32):
    #Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    windowed_test_data = create_windows(test_data_in, window_size)
    windowed_test_labels = create_label_windows(test_labels_in, window_size)
    # Prepare test data
    test_data = torch.FloatTensor(windowed_test_data)
    test_labels = torch.FloatTensor(windowed_test_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Load the model
    model = DeepCNNDecoder(n_channels=test_data.shape[1], window_size=window_size)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    #
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            predictions = (outputs >= 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    #Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    all_preds = all_preds.squeeze()
    accuracy = (all_preds == all_labels).mean()
    # Debug prints
    return conf_matrix, accuracy

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Silence', 'Silence'],
                yticklabels=['No Silence', 'Silence'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/home/groups/henderj/rzwang/figures/confusion_matrix.png')
    plt.show()
    # plt.savefig('/figures/confusion_matrix.png')

def plot_confusion_matix_cat(conf_matrix):
        idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}
        labels = list(idx_to_category.values())

        plt.figure(figsize=(12, 12))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=list(idx_to_category.values()) if decode_categories else list(idx_to_phoneme.values()),
                    yticklabels=list(idx_to_category.values()) if decode_categories else list(idx_to_phoneme.values()))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        # Save confusion matrix
        # cm_path = os.path.join(args.output_dir, f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        cm_path = f'/home/groups/henderj/rzwang/figures/confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(cm_path)
        plt.close()

labels = np.array([category_to_idx[phoneme_to_category[label]] for label in phoneme_labels])

model_path = '/home/groups/henderj/rzwang/results/model_20250314_015616.pt' 
test_data = np.load("/home/groups/henderj/rzwang/processed_data_silence/neural_data_sbp.npy") 
test_labels = np.load("/home/groups/henderj/rzwang/processed_data_silence/labels.npy") # Your test labels
# make predictions on test set
predictions, probabilities = decode_phonemes(
    model=model,
    test_data=test_neural_data,
    window_size=args.window_size,
    stride=args.stride,
    batch_size=args.batch_size
)
plot_confusion_matrix(conf_matrix)
print(f"Test Accuracy: {accuracy:.4f}")


def calculate_metrics(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

# Calculate and print metrics
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

