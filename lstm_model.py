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
import warnings

# Suppress NCCL warning on Windows (NCCL is not needed for DataParallel)
warnings.filterwarnings('ignore', message='PyTorch is not compiled with NCCL support')


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
        # Convert to tensors using pure float16 for maximum memory efficiency
        seq = torch.from_numpy(self.sequences[idx]).half()  # (features, time_steps)
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
    Adjusted for float16 stability
    """
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Glorot uniform (optimized for pure float16)
                nn.init.xavier_uniform_(param.data, gain=0.3)  # Further reduced for pure float16
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal initialization (optimized for pure float16)
                nn.init.orthogonal_(param.data, gain=0.3)  # Further reduced for pure float16
            elif 'bias' in name:
                # Biases: zeros
                param.data.fill_(0.0)
                # LSTM has 4 gates, set forget gate bias to 1.0 (helps training)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
                
    elif isinstance(m, nn.Linear):
        # Glorot uniform for fully connected layers (optimized for pure float16)
        nn.init.xavier_uniform_(m.weight.data, gain=0.3)  # Further reduced for pure float16
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def create_dataloader(dataset: LSTMMarketDataset, batch_size: int, 
                     shuffle: bool, num_workers: int = 0, drop_last: bool = True) -> DataLoader:
    """
    Create a DataLoader for the dataset optimized for GPU performance
    
    Args:
        dataset: LSTMMarketDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        drop_last: Whether to drop the last incomplete batch (False for validation)
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # Speed up CPU to GPU transfer
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches for GPU
        drop_last=drop_last  # Allow small validation batches
    )


def check_tensor_core_support():
    """Check if the system supports tensor cores"""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    # Check for tensor core capable GPU
    device_name = torch.cuda.get_device_name(0)
    tensor_core_gpus = [
        'RTX', 'V100', 'A100', 'A40', 'A30', 'A10', 'A16', 'A2',
        'Titan RTX', 'Quadro RTX', 'Tesla T4'
    ]
    
    has_tensor_cores = any(gpu in device_name for gpu in tensor_core_gpus)
    
    if has_tensor_cores:
        return True, f"Tensor cores supported on {device_name}"
    else:
        return False, f"No tensor cores detected on {device_name}"

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
    validation_patience = hyperparams.get('validation_patience', 5)  # MATLAB default: 5
    validation_frequency = hyperparams.get('validation_frequency', 7)  # MATLAB default: 7
    
    # Convert model to pure float16 for maximum memory efficiency
    model = model.half()
    
    # Move model to device
    model = model.to(device)
    
    # Use DataParallel for multi-GPU training if available
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Set device to use all available GPUs
        device_ids = list(range(torch.cuda.device_count()))
        print(f"Available GPU IDs: {device_ids}")
    elif device == 'cuda':
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        device_ids = [0]
    
    # Loss function: Cross-entropy with numerical stability for float16
    criterion = nn.CrossEntropyLoss()
    
    # Check tensor core support
    tensor_core_available, tensor_core_msg = check_tensor_core_support()
    
    # Note: Gradient scaling is not compatible with pure float16 models
    # We use mixed precision autocast for tensor cores without gradient scaling
    scaler = None  # Disabled for pure float16 models
    
    # Optimizer: ADAM optimized for GPU tensor cores
    # Adjusted parameters for RTX 4060 Ti performance
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.999),  # beta1=0.9 (default), beta2=0.999 (MATLAB SquaredGradientDecayFactor)
        eps=1e-4,  # Optimized for tensor cores
        weight_decay=l2_regularization,  # L2 regularization
        amsgrad=False  # Standard Adam for MATLAB compatibility
    )
    
    # Learning rate scheduler: Step decay (MATLAB: LearnRateSchedule = 'piecewise')
    # Drop learning rate by lr_drop_factor every lr_drop_period epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=lr_drop_period, 
        gamma=lr_drop_factor
    )
    
    # Training loop with performance monitoring
    # Track both validation loss AND accuracy for better model selection
    # MATLAB: OutputNetwork = 'best-validation-loss'
    # We also track accuracy to ensure we're not just minimizing loss without improving predictions
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    epochs_since_improvement = 0
    
    # Performance monitoring
    import time
    start_time = time.time()
    
    # Print GPU information
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        if torch.cuda.device_count() > 1:
            print(f"Multi-GPU: {torch.cuda.device_count()} GPUs detected")
        else:
            print("Single GPU mode")
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass with automatic mixed precision
            optimizer.zero_grad()
            
            # High-performance training with tensor core optimization
            if device == 'cuda' and tensor_core_available:
                # Use automatic mixed precision for tensor cores (no gradient scaling)
                with torch.amp.autocast('cuda'):
                    logits = model(sequences)  # (batch, seq_len, num_classes)
                    
                    # Reshape for loss calculation
                    batch_size, seq_len, num_classes = logits.shape
                    logits_flat = logits.reshape(-1, num_classes)  # (batch * seq_len, num_classes)
                    labels_flat = labels.reshape(-1)  # (batch * seq_len,)
                    
                    # Calculate loss
                    loss = criterion(logits_flat, labels_flat)
            else:
                # CPU or non-tensor-core GPU training
                logits = model(sequences)  # (batch, seq_len, num_classes)
                
                # Reshape for loss calculation
                batch_size, seq_len, num_classes = logits.shape
                logits_flat = logits.reshape(-1, num_classes)  # (batch * seq_len, num_classes)
                labels_flat = labels.reshape(-1)  # (batch * seq_len,)
                
                # Calculate loss
                loss = criterion(logits_flat, labels_flat)
            
            # Check for NaN loss and handle it
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected, skipping this batch")
                continue
            
            # Backward pass (no gradient scaling for pure float16)
            loss.backward()
            
            # Gradient clipping for stability (reduced for float16)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(logits_flat, 1)
            train_total += labels_flat.size(0)
            train_correct += (predicted == labels_flat).sum().item()
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
                
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
                    
                    # High-performance mixed precision validation
                    if device == 'cuda' and tensor_core_available:
                        with torch.amp.autocast('cuda'):
                            logits = model(sequences)
                    else:
                        logits = model(sequences)
                    
                    # Reshape for loss and accuracy calculation
                    batch_size, seq_len, num_classes = logits.shape
                    logits_flat = logits.reshape(-1, num_classes)
                    labels_flat = labels.reshape(-1)
                    
                    # Calculate validation loss
                    # CrossEntropyLoss with reduction='mean' averages over batch
                    # We accumulate weighted by number of samples for true average
                    loss = criterion(logits_flat, labels_flat)
                    
                    # Skip if loss is NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
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
            # Save model only when validation loss improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                epochs_since_improvement = 0
                # Save a deep copy of the model state
                if isinstance(model, nn.DataParallel):
                    best_model_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
                else:
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                epochs_since_improvement += 1
            
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
            
            # High-performance mixed precision final validation
            if device == 'cuda' and tensor_core_available:
                with torch.amp.autocast('cuda'):
                    logits = model(sequences)
            else:
                logits = model(sequences)
            
            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.reshape(-1, num_classes)
            labels_flat = labels.reshape(-1)
            
            _, predicted = torch.max(logits_flat, 1)
            val_total += labels_flat.size(0)
            val_correct += (predicted == labels_flat).sum().item()
    
    final_val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
    
    # Calculate total training time
    total_time = time.time() - start_time
    
    # Return unwrapped model if using DataParallel
    if isinstance(model, nn.DataParallel):
        model = model.module
    
    info = {
        'final_val_acc': final_val_acc,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'tensor_core_used': tensor_core_available,
        'device_used': device,
        'total_training_time': total_time,
        'avg_epoch_time': total_time / (epoch + 1)
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
            # Convert to tensor and add batch dimension (pure float16)
            seq_tensor = torch.from_numpy(seq).half().unsqueeze(0).to(device)
            
            # High-performance mixed precision forward pass
            if device == 'cuda':
                tensor_core_available, _ = check_tensor_core_support()
                if tensor_core_available:
                    with torch.amp.autocast('cuda'):
                        logits = model(seq_tensor)  # (1, seq_len, num_classes)
                else:
                    logits = model(seq_tensor)  # (1, seq_len, num_classes)
            else:
                logits = model(seq_tensor)  # (1, seq_len, num_classes)
            
            # Get predictions for last time step (matches MATLAB behavior)
            last_logits = logits[0, -1, :]  # (num_classes,)
            
            # Softmax to get probabilities (stable for float16)
            # Clamp logits to prevent overflow/underflow in softmax
            last_logits_clamped = torch.clamp(last_logits, min=-50.0, max=50.0)
            probs = F.softmax(last_logits_clamped, dim=0)
            
            # Ensure no NaN values in probabilities
            if torch.isnan(probs).any():
                # Fallback to uniform distribution if NaN detected
                probs = torch.ones_like(probs) / probs.size(0)
            
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
    
    # Convert to half precision
    model = model.half()
    
    # Create dummy dataset (use float16 for memory efficiency)
    sequences = [np.random.randn(input_size, sequence_length).astype(np.float16) 
                 for _ in range(num_sequences)]
    labels = [np.random.randint(0, 2, size=(1, sequence_length)).astype(np.int64) 
              for _ in range(num_sequences)]
    
    dataset = LSTMMarketDataset(sequences, labels)
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)
    
    print(f"Model created: {model}")
    print(f"Dataset size: {len(dataset)}")
    print(f"DataLoader batches: {len(dataloader)}")
    
    # Test forward pass (use half precision)
    test_input = torch.randn(2, input_size, sequence_length, dtype=torch.float16)
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

