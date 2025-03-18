
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
from datetime import datetime
# from torchsummary import summary
from scipy import stats
#some other ones: torchsummary, sklearn metrics, early stopping? seaborn/matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from decoders import *
from train_functions import *

# DECODE_CATEGORIES = False

#TODO: Check these mappings from gpt
# def create_simple_phoneme_mapping():
#     """Create mapping from MFA phonemes to simple ASCII representations"""
#     return {
#         b'd\xca\x92': b'jh',  # dʒ -> jh
#         b't\xca\x83': b'ch',  # tʃ -> ch
#         b'\xca\x83': b'sh',   # ʃ -> sh
#         b'\xca\x92': b'zh',   # ʒ -> zh
#         b'\xc9\x99': b'ah',   # ə -> ah
#         b'\xc9\x94': b'ao',   # ɔ -> ao
#         b'\xc9\x91': b'aa',   # ɑ -> aa
#         b'\xc9\x9b': b'eh',   # ɛ -> eh
#         b'\xc9\xaa': b'ih',   # ɪ -> ih
#         b'\xca\x8a': b'uh',   # ʊ -> uh
#         b'\xc3\xa6': b'ae',   # æ -> ae
#         b'\xc5\x8b': b'ng',   # ŋ -> ng
#         b'\xc9\xb9': b'r',    # ɹ -> r
#         b'\xce\xb8': b'th',   # θ -> th
#         b'\xc3\xb0': b'dh',   # ð -> dh
#         b'\xc9\x90': b'ax',     # ɐ -> ax
#         b'\xc9\x9c': b'er',     # ɜ -> er
#         b'\xc9\x9d': b'er',     # ɝ -> er
#         b'\xc9\xb2': b'dx',     # ɲ -> dx
#         b'\xcb\x90': b'',       # ː (length mark) - remove it
#     }

def create_simple_phoneme_mapping():
    """Create mapping from MFA phonemes to simple ASCII representations"""
    return {
        # Simple ASCII characters (keep as is)
        b'a': b'a',
        b'b': b'b',
        b'c': b'c',
        b'd': b'd',
        b'e': b'e',
        b'f': b'f',
        b'h': b'h',
        b'i': b'i',
        b'j': b'j',
        b'k': b'k',
        b'l': b'l',
        b'm': b'm',
        b'n': b'n',
        b'o': b'o',
        b'p': b'p',
        b's': b's',
        b't': b't',
        b'v': b'v',
        b'w': b'w',
        b'z': b'z',
        
        # Special tokens
        b'SIL': b'SIL',
        b'spn': b'spn',
        
        # Diphthongs and sequences
        b'aj': b'ay',
        b'aw': b'aw',
        b'ej': b'ey',
        b'ow': b'ow',
        b'\xc9\x94j': b'oy',    # ɔj
        b'\xc9\x99w': b'ow',    # əw
        
        # Aspirated stops
        b'p\xca\xb0': b'p',     # pʰ
        b't\xca\xb0': b't',     # tʰ
        b'k\xca\xb0': b'k',     # kʰ
        b'c\xca\xb0': b'ch',    # cʰ
        
        # Modified consonants
        b'b\xca\xb2': b'b',     # bʲ
        b'd\xca\xb2': b'd',     # dʲ
        b'f\xca\xb2': b'f',     # fʲ
        b'm\xca\xb2': b'm',     # mʲ
        b'p\xca\xb2': b'p',     # pʲ
        b't\xca\xb2': b't',     # tʲ
        b'v\xca\xb2': b'v',     # vʲ
        b'\xca\x88\xca\xb2': b't',  # tʲ
        
        # Affricates
        b't\xca\x83': b'ch',    # tʃ
        b'd\xca\x92': b'jh',    # dʒ
        
        # Fricatives
        b'\xca\x83': b'sh',     # ʃ
        b'\xca\x92': b'zh',     # ʒ
        b'\xce\xb8': b'th',     # θ
        b'\xc3\xb0': b'dh',     # ð
        b'\xc3\xa7': b's',      # ç
        
        # Nasals
        b'\xc5\x8b': b'ng',     # ŋ
        b'\xc9\xb2': b'nx',     # ɲ
        b'm\xcc\xa9': b'm',     # m̩
        
        # Vowels
        b'\xc3\xa6': b'ae',     # æ
        b'\xc9\x90': b'ax',     # ɐ
        b'\xc9\x91': b'aa',     # ɑ
        b'\xc9\x92': b'ao',     # ɒ
        b'\xc9\x94': b'ao',     # ɔ
        b'\xc9\x96': b'ah',     # ɖ
        b'\xc9\x99': b'ah',     # ə
        b'\xc9\x9a': b'er',     # ə˞
        b'\xc9\x9b': b'eh',     # ɛ
        b'\xc9\x9c': b'er',     # ɜ
        b'\xc9\x9d': b'er',     # ɝ
        b'\xc9\x9f': b'g',      # ɟ
        b'\xc9\xa1': b'g',      # ɡ
        b'\xc9\xaa': b'ih',     # ɪ
        b'\xc9\xab': b'uh',     # ʋ
        b'\xc9\xb9': b'r',      # ɹ
        b'\xca\x88': b't',      # ʈ
        b'\xca\x89': b'uh',     # ʉ
        b'\xca\x8a': b'uh',     # ʊ
        b'\xca\x8b': b'uh',     # ʋ
        b'\xca\x8e': b'y',      # ʎ
        
        # Length markers and modifications
        b'a\xcb\x90': b'aa',    # aː
        b'e\xcb\x90': b'ee',    # eː
        b'i\xcb\x90': b'ii',    # iː
        b'o\xcb\x90': b'oo',    # oː
        b'u\xcb\x90': b'uu',    # uː
        b'\xc9\x91\xcb\x90': b'aa',  # ɑː
        b'\xc9\x92\xcb\x90': b'ao',  # ɒː
        b'\xc9\x9b\xcb\x90': b'eh',  # ɛː
        b'\xc9\x9c\xcb\x90': b'er',  # ɜː
        b'\xca\x89\xcb\x90': b'uh',  # ʉː
        
        # Special cases
        b'd\xcc\xaa': b'd',     # d̪
        b't\xcc\xaa': b't',     # t̪
        b'c\xca\xb7': b'ch',    # cˇ
        b'k\xca\xb7': b'k',     # kˇ
    }

def convert_labels_to_ascii(labels):
    """Convert MFA phoneme labels to simple ASCII representations"""
    phoneme_map = create_simple_phoneme_mapping()
    new_labels = []
    
    for label in labels:
        if label in phoneme_map:
            new_labels.append(phoneme_map[label])
        else:
            new_labels.append(label)
    
    return np.array(new_labels)


def create_phoneme_mappings():
    """Create mappings for both full phoneme set and categories"""
    # Define phoneme categories using ASCII representations
    phoneme_categories = {
        'vowels': {
            'monophthongs': [b'a', b'e', b'i', b'o', b'u', b'ae', b'ah', b'ax', b'aa', b'ao', b'eh', 
                           b'er', b'ih', b'uh', b'oo', b'ee', b'ii', b'uu'],
            'diphthongs': [b'aj', b'ay', b'aw', b'ey', b'ow', b'oy'],
        },
        'consonants': {
            'stops': [b'b', b'p', b't', b'd', b'k', b'g', b'c'],
            'fricatives': [b'f', b'v', b's', b'z', b'sh', b'zh', b'th', b'dh', b'h'],
            'affricates': [b'ch', b'jh'],
            'nasals': [b'm', b'n', b'ng', b'nx'],
            'liquids': [b'l', b'r', b'dx'],
            'glides': [b'y', b'w', b'j']  # Added 'j' here as a glide
        },
        'special': [b'SIL', b'spn']
    }
    
    # Create flattened category mapping
    phoneme_to_category = {}
    for main_cat, subcat_dict in phoneme_categories.items():
        if isinstance(subcat_dict, dict):
            for subcat, phones in subcat_dict.items():
                for phone in phones:
                    phoneme_to_category[phone] = f"{main_cat}_{subcat}"
        else:
            for phone in subcat_dict:
                phoneme_to_category[phone] = main_cat
    
    # Create numerical mappings
    unique_categories = sorted(set(phoneme_to_category.values()))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    
    return phoneme_to_category, category_to_idx

def parse_args():

    parser = argparse.ArgumentParser(description='Train neural network model')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--decode_categories', type=str, choices=['True', 'False'], default='True',
                        help='True for category decoding, False for full phoneme decoding')
    return parser.parse_args()

class DeepCNNPhonemeDecoder(nn.Module):
    def __init__(self, n_channels=256, window_size=5, n_classes=85):
        super(DeepCNNPhonemeDecoder, self).__init__()
        
        #more filters and added more layers
        self.conv = nn.Sequential(
            # First conv block
            nn.Conv1d(n_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            # Second conv block
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            # Third conv block
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            # Fourth conv block
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32 * window_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNNPhonemeDecoder(nn.Module):
    """CNN decoder for phoneme classification"""
    def __init__(self, n_channels=256, window_size=5, n_classes=85):
        super(CNNPhonemeDecoder, self).__init__()
        self.window_size = window_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * window_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def decode_phonemes(model, test_data, window_size=5, stride=1, batch_size=32, device='cuda'):
    """
    Decode phonemes using trained model
    
    params:
    model : torch.nn.Module
        Trained neural network model
    test_data : np.ndarray
        Neural data to decode (n_timepoints, n_channels)
    window_size : int
        Size of sliding window
    stride : int
        Stride for sliding window
    batch_size : int
        Batch size for processing
    device : str
        Device to run model on ('cuda' or 'cpu')
        
    output:
    predictions : np.ndarray
        Predicted phoneme indices for each time window
    probabilities : np.ndarray
        Probability distributions over phonemes for each window
    """
    
    # Prepare model for evaluation
    model.eval()
    # model = model.to(device)
    
    # Create windows from test data
    # windowed_data = create_windows(test_data, window_size, stride)
    windowed_data = test_data
    
    # Convert to torch tensor
    test_dataset = torch.FloatTensor(windowed_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False
    )
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            # batch = batch.to(device)
            
            # Get model predictions
            outputs = model(batch)
            
            #softmax
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            probabilities.extend(probs)
            
            #  most likely phoneme
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    return predictions, probabilities



def main():
    args = parse_args()
    decode_categories = args.decode_categories == 'True'  # Convert string to boolean
    log_file = os.path.join(args.output_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    try:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Starting training with parameters: {vars(args)}")

        os.makedirs(args.output_dir, exist_ok=True)
        
        neural_data = np.load("/home/groups/henderj/rzwang/processed_data_phonemes/neural_data_sbp.npy")
        phoneme_labels = np.load("/home/groups/henderj/rzwang/processed_data_phonemes/labels.npy")
        trial_info = np.load("/home/groups/henderj/rzwang/processed_data_phonemes/trial_info.npy")
        phoneme_labels = convert_labels_to_ascii(phoneme_labels)
        phoneme_to_category, category_to_idx = create_phoneme_mappings()
        
        if decode_categories:
            labels = np.array([category_to_idx[phoneme_to_category[label]] for label in phoneme_labels])
            n_classes = len(category_to_idx)
            logging.info(f"Using category decoding with {n_classes} categories")
        else:
            # For full phoneme set
            unique_phonemes = np.unique(phoneme_labels)
            phoneme_to_idx = {p: i for i, p in enumerate(unique_phonemes)}
            labels = np.array([phoneme_to_idx[p] for p in phoneme_labels])
            n_classes = len(unique_phonemes)
            logging.info(f"Using full phoneme decoding with {n_classes} phonemes")
            phoneme_labels = convert_labels_to_ascii(phoneme_labels)

        #debug
        unique_phonemes = np.unique(phoneme_labels)
        logging.info("Unique phonemes in dataset:")
        for p in unique_phonemes:
            logging.info(f"Phoneme: {p}")
            if p in phoneme_to_category:
                logging.info(f"  Category: {phoneme_to_category[p]}")
            else:
                logging.info("  Not found in categories!")

        #traiun
        model, test_neural_data, test_labels = train_phoneme_decoder(
            neural_data,
            labels,
            trial_info,
            decoder_class=lambda n_channels, window_size: DeepCNNPhonemeDecoder(
                n_channels=n_channels,
                window_size=window_size,
                n_classes=n_classes
            ),
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate
        )

        model_path = os.path.join(args.output_dir, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
        torch.save(model.state_dict(), model_path)
        
        # make predictions on test set
        predictions, probabilities = decode_phonemes(
            model=model,
            test_data=test_neural_data,  # You'll need to define this
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size
        )
        
        # for 

        # Calculate and log metrics
        accuracy = np.mean(predictions == test_labels) 
        logging.info(f"Test Accuracy: {accuracy:.4f}")
        
        # Create and save confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        # Create idx_to_category mapping

        if decode_categories:
            idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}
            labels = list(idx_to_category.values())
        else:
            idx_to_phoneme = {idx: phoneme for phoneme, idx in phoneme_to_idx.items()}
            labels = list(idx_to_phoneme.values())

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
        
        logging.info(f'Training completed successfully at time {datetime.now().strftime("%Y%m%d_%H%M%S")}')
        logging.info(f"Model saved to: {model_path}")
        logging.info(f"Confusion matrix saved to: {cm_path}")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()