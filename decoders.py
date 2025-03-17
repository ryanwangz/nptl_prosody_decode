#decoder architectures from some papers: (TODO: insert architectures from Wairgarkar paper here)

#also TODO: add soft-dtw and similar loss functions

import torch
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
#takes both spike band power and threshold crossings channels (requires training funciton tweaks)
class DualStreamCNN(nn.Module):
    def __init__(self, n_channels_sbp, n_channels_tc, n_outputs):
        super(DualStreamCNN, self).__init__()
        
        # Spike Band Power stream
        self.sbp_conv = nn.Sequential(
            nn.Conv1d(n_channels_sbp, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # Threshold Crossings stream
        self.tc_conv = nn.Sequential(
            nn.Conv1d(n_channels_tc, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # Combined layers
        self.combined = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, n_outputs, kernel_size=3, padding=1)
        )
        
    def forward(self, x_sbp, x_tc):
        # Process each stream
        sbp_features = self.sbp_conv(x_sbp)
        tc_features = self.tc_conv(x_tc)
        
        # Combine features
        combined = torch.cat((sbp_features, tc_features), dim=1)
        
        # Final processing
        output = self.combined(combined)
        return output

#a bunch of other decoder classes to try, from an LLM's suggesstions

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


class DeepCNNDecoder(nn.Module):
    def __init__(self, n_channels=256, window_size=5):
        super(DeepCNNDecoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * window_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LSTMDecoder(nn.Module):
    def __init__(self, n_channels=256, window_size=5, hidden_size=128):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=n_channels,
                           hidden_size=hidden_size,
                           num_layers=2,
                           batch_first=True,
                           dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Reshape input: (batch, channels, time) -> (batch, time, channels)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        # Use last time step output
        return self.fc(lstm_out[:, -1, :])


class CNNLSTMDecoder(nn.Module):
    def __init__(self, n_channels=256, window_size=5, hidden_size=128):
        super(CNNLSTMDecoder, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(input_size=64,
                           hidden_size=hidden_size,
                           num_layers=2,
                           batch_first=True,
                           dropout=0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # CNN feature extraction
        x = self.conv(x)
        # Reshape for LSTM: (batch, channels, time) -> (batch, time, channels)
        x = x.permute(0, 2, 1)
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        # Use last time step
        return self.fc(lstm_out[:, -1, :])

# 5. Bidirectional LSTM
class BiLSTMDecoder(nn.Module):
    def __init__(self, n_channels=256, window_size=5, hidden_size=128):
        super(BiLSTMDecoder, self).__init__()
        
        self.lstm = nn.LSTM(input_size=n_channels,
                           hidden_size=hidden_size,
                           num_layers=2,
                           batch_first=True,
                           bidirectional=True,
                           dropout=0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 6. Transformer Decoder (continued)
class TransformerDecoder(nn.Module):
    def __init__(self, n_channels=256, window_size=5, nhead=8):
        super(TransformerDecoder, self).__init__()
        
        # Ensure input dimension is divisible by number of heads
        self.embedding_dim = ((n_channels // nhead) * nhead)
        
        self.embed = nn.Linear(n_channels, self.embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        x = self.embed(x)
        x = self.transformer(x)
        # Use the last token's representation
        return self.fc(x[:, -1, :])
