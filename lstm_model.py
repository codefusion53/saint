"""
LSTM Model Components for SAINT1 v3.28
=============================================================================
PyTorch LSTM implementation matching MATLAB's trainNetwork behavior

This module provides:
- LSTMClassifier: LSTM neural network model
- LSTMMarketDataset: PyTorch dataset for market data
- Training and prediction functions
- MATLAB-style weight initialization

6-1-2024 aii llc
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict
import math


class LSTMClassifier(nn.Module):
    """
    LSTM classifier matching MATLAB's stacked LSTM architecture
    
    MATLAB Architecture:
    - sequenceInputLayer(104)
    - lstmLayer(hidden_size, 'StateActivationFunction', 'tanh', 'OutputMode', 'sequence')
    - dropoutLayer(dropout_rate)
    - lstmLayer(hidden_size, 'StateActivationFunction', 'tanh', 'OutputMode', 'sequence')
    - dropoutLayer(dropout_rate)
    - fullyConnectedLayer(2)
    - softmaxLayer
    - classificationLayer
    
    Python Architecture:
    - First LSTM layer (unidirectional)
    - Dropout
    - Second LSTM layer (unidirectional)
    - Dropout
    - Fully connected layer
    """
    
    def __init__(self, input_size: int, sequence_length: int, hidden_size: int, 
                 dropout_rate: float, num_classes: int = 2):
        super(LSTMClassifier, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # First LSTM layer (matches MATLAB's first lstmLayer)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # Dropout handled separately
            bidirectional=False  # MATLAB lstmLayer is unidirectional
        )
        
        # First dropout layer
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        # Second LSTM layer (matches MATLAB's second lstmLayer)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,  # Takes output from first LSTM
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # Dropout handled separately
            bidirectional=False  # MATLAB lstmLayer is unidirectional
        )
        
        # Second dropout layer
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        # Fully connected layer (unidirectional LSTM outputs hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        Forward pass matching MATLAB's stacked LSTM architecture
        
        Args:
            x: Input tensor of shape (batch, input_size, sequence_length)
               Note: MATLAB uses (features, time_steps) format
        
        Returns:
            logits: Output logits of shape (batch, sequence_length, num_classes)
        """
        # Transpose from (batch, features, time) to (batch, time, features)
        # MATLAB: (104, 20) -> PyTorch: (20, 104)
        x = x.transpose(1, 2)  # (batch, sequence_length, input_size)
        
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)  # (batch, sequence_length, hidden_size)
        
        # First dropout
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)  # (batch, sequence_length, hidden_size)
        
        # Second dropout
        lstm2_out = self.dropout2(lstm2_out)
        
        # Fully connected layer for classification
        logits = self.fc(lstm2_out)  # (batch, sequence_length, num_classes)
        
        return logits


class LSTMMarketDataset(Dataset):
    """
    PyTorch Dataset for market data sequences
    
    Handles variable-length sequences stored as lists
    """
    
    def __init__(self, sequences: List[np.ndarray], labels: List[np.ndarray]):
        """
        Args:
            sequences: List of input sequences, each shape (features, time_steps)
            labels: List of label sequences, each shape (1, time_steps)
        """
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Convert to tensors using float64 (double precision) to match MATLAB
        seq = torch.from_numpy(self.sequences[idx]).double()  # (features, time_steps)
        label = torch.from_numpy(self.labels[idx]).long().squeeze(0)  # (time_steps,)
        
        return seq, label
        

def init_weights_matlab_style(m):
    """
    Initialize weights to match MATLAB's default initialization
    
    MATLAB uses:
    - Glorot (Xavier) initialization for LSTM weights
    - Zero initialization for biases
    - Forget gate bias = 1.0 for better gradient flow
    
    This applies to both lstm1 and lstm2 layers automatically via .apply()
    """
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Glorot uniform
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal initialization
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases: zeros
                param.data.fill_(0.0)
                # LSTM has 4 gates, set forget gate bias to 1.0 (helps training)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
                
    elif isinstance(m, nn.Linear):
        # Glorot uniform for fully connected layers
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def create_dataloader(dataset: LSTMMarketDataset, batch_size: int, 
                     shuffle: bool, num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader for the dataset
    
    Args:
        dataset: LSTMMarketDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # Speed up CPU to GPU transfer
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )


def train_lstm_model(
    model: LSTMClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hyperparams: Dict,
    avnin: List[np.ndarray],
    avnout1: List[np.ndarray],
    device: str = 'cpu'
) -> Tuple[LSTMClassifier, Dict]:
    """
    Train LSTM model matching MATLAB's trainNetwork behavior
    
    Args:
        model: LSTMClassifier instance
        train_loader: Training data loader
        val_loader: Validation data loader
        hyperparams: Dictionary with training hyperparameters
        avnin: Validation input sequences (for evaluation)
        avnout1: Validation output labels (for evaluation)
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        trained_model: Trained model
        info: Training information dictionary
    """
    # Extract hyperparameters
    initial_lr = hyperparams['initial_lr']
    max_epochs = hyperparams['max_epochs']
    l2_regularization = hyperparams['l2_regularization']
    lr_drop_factor = hyperparams['lr_drop_factor']
    lr_drop_period = hyperparams['lr_drop_period']
    validation_patience = hyperparams.get('validation_patience', 5)
    validation_frequency = hyperparams.get('validation_frequency', 7)
    
    # Convert model to double precision to match MATLAB (float64)
    model = model.double()
    
    # Move model to device
    model = model.to(device)
    
    # Use DataParallel for multi-GPU training if available
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Loss function: Cross-entropy (matches MATLAB's classificationLayer)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: ADAM (matches MATLAB's 'adam' solver)
    # MATLAB: SquaredGradientDecayFactor = 0.999 (corresponds to beta2 in PyTorch)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.999),  # beta1=0.9 (default), beta2=0.999 (MATLAB SquaredGradientDecayFactor)
        weight_decay=l2_regularization  # L2 regularization
    )
    
    # Learning rate scheduler: Step decay
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_drop_period,
        gamma=lr_drop_factor
    )
    
    # Training loop
    # MATLAB: OutputNetwork = 'best-validation-loss'
    # We need to track best validation LOSS and save the best model
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(sequences)  # (batch, seq_len, num_classes)
            
            # Reshape for loss calculation
            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.reshape(-1, num_classes)  # (batch * seq_len, num_classes)
            labels_flat = labels.reshape(-1)  # (batch * seq_len,)
            
            # Calculate loss
            loss = criterion(logits_flat, labels_flat)
            
            # Backward pass
            loss.backward()
            
            # NOTE: MATLAB's default GradientThreshold = Inf (no clipping)
            # Only clip if MATLAB configuration explicitly sets GradientThreshold
            # Commented out to match MATLAB default behavior:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(logits_flat, 1)
            train_total += labels_flat.size(0)
            train_correct += (predicted == labels_flat).sum().item()
        
        # Learning rate decay (MATLAB: LearnRateSchedule = 'piecewise')
        scheduler.step()
        
        # Validation phase (every validation_frequency epochs)
        # MATLAB: ValidationFrequency = 7
        if (epoch + 1) % validation_frequency == 0 or epoch == max_epochs - 1:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            num_val_samples = 0
            
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    
                    logits = model(sequences)
                    
                    # Reshape for loss and accuracy calculation
                    batch_size, seq_len, num_classes = logits.shape
                    logits_flat = logits.reshape(-1, num_classes)
                    labels_flat = labels.reshape(-1)
                    
                    # Calculate validation loss
                    # CrossEntropyLoss with reduction='mean' averages over batch
                    # We accumulate weighted by number of samples for true average
                    loss = criterion(logits_flat, labels_flat)
                    val_loss += loss.item() * labels_flat.size(0)
                    num_val_samples += labels_flat.size(0)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(logits_flat, 1)
                    val_total += labels_flat.size(0)
                    val_correct += (predicted == labels_flat).sum().item()
            
            # Average validation loss across all samples (matches MATLAB behavior)
            avg_val_loss = val_loss / num_val_samples if num_val_samples > 0 else float('inf')
            val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
            
            # MATLAB: OutputNetwork = 'best-validation-loss'
            # Save model if this is the best validation loss so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save a deep copy of the model state
                if isinstance(model, nn.DataParallel):
                    best_model_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
                else:
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            # MATLAB: ValidationPatience = 5
            # Stop training if validation loss doesn't improve for 5 consecutive checks
            if patience_counter >= validation_patience:
                break
    
    # CRITICAL: Restore the best model weights (MATLAB: OutputNetwork = 'best-validation-loss')
    if best_model_state is not None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
        # Move back to device
        model = model.to(device)
    
    # Check model parameters for sanity
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Final validation accuracy (re-evaluate with best model)
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            logits = model(sequences)
            
            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.reshape(-1, num_classes)
            labels_flat = labels.reshape(-1)
            
            _, predicted = torch.max(logits_flat, 1)
            val_total += labels_flat.size(0)
            val_correct += (predicted == labels_flat).sum().item()
    
    final_val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
    
    # Return unwrapped model if using DataParallel
    if isinstance(model, nn.DataParallel):
        model = model.module
    
    info = {
        'final_val_acc': final_val_acc,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }
    
    return model, info


def classify_sequences(
    model: LSTMClassifier,
    sequences: List[np.ndarray],
    device: str = 'cpu',
    return_all_logits: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify sequences using trained model
    
    MATLAB Behavior:
    - classify(net, data) returns class labels for each sequence
    - With 'OutputMode' = 'sequence', it outputs for all time steps
    - Typically uses the LAST time step for sequence classification
    
    Args:
        model: Trained LSTMClassifier
        sequences: List of input sequences, each shape (features, time_steps)
        device: Device to run on
        return_all_logits: If True, also return raw logits for ensemble methods
    
    Returns:
        predictions: Predicted class for each sequence (using last time step)
        confidences: Confidence scores for each class, shape (num_sequences, num_classes)
        If return_all_logits=True, also returns logits array
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_confidences = []
    all_logits = [] if return_all_logits else None
    
    with torch.no_grad():
        for seq in sequences:
            # Convert to tensor and add batch dimension (double precision to match model)
            seq_tensor = torch.from_numpy(seq).double().unsqueeze(0).to(device)
            
            # Forward pass
            logits = model(seq_tensor)  # (1, seq_len, num_classes)
            
            # Get predictions for last time step (matches MATLAB behavior)
            last_logits = logits[0, -1, :]  # (num_classes,)
            
            # Softmax to get probabilities
            probs = F.softmax(last_logits, dim=0)
            
            # Predicted class
            pred_class = torch.argmax(last_logits).item()
            
            all_predictions.append(pred_class)
            all_confidences.append(probs.cpu().numpy())
            
            if return_all_logits:
                all_logits.append(last_logits.cpu().numpy())
    
    if return_all_logits:
        return np.array(all_predictions), np.array(all_confidences), np.array(all_logits)
    
    return np.array(all_predictions), np.array(all_confidences)


if __name__ == "__main__":
    # Test the model
    print("Testing LSTM Model Components...")
    
    # Create dummy data
    input_size = 104
    sequence_length = 20
    hidden_size = 128
    batch_size = 16
    num_sequences = 27
    
    # Create model
    model = LSTMClassifier(
        input_size=input_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        dropout_rate=0.5,
        num_classes=2
    )
    
    # Initialize weights
    model.apply(init_weights_matlab_style)
    
    # Convert to double precision
    model = model.double()
    
    # Create dummy dataset (use float64 to match MATLAB)
    sequences = [np.random.randn(input_size, sequence_length).astype(np.float64) for _ in range(num_sequences)]
    labels = [np.random.randint(0, 2, size=(1, sequence_length)).astype(np.int64) for _ in range(num_sequences)]
    
    dataset = LSTMMarketDataset(sequences, labels)
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)
    
    print(f"Model created: {model}")
    print(f"Dataset size: {len(dataset)}")
    print(f"DataLoader batches: {len(dataloader)}")
    
    # Test forward pass (use double precision)
    test_input = torch.randn(2, input_size, sequence_length, dtype=torch.float64)
    output = model(test_input)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape}")
    
    # Test classification
    predictions, confidences = classify_sequences(model, sequences[:3])
    print(f"Predictions shape: {predictions.shape}")
    print(f"Confidences shape: {confidences.shape}")
    print(f"Sample predictions: {predictions}")
    print(f"Sample confidences: {confidences}")
    
    print("\nAll tests passed!")

