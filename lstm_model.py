"""
LSTM Model for Market Data Classification

Features:
- Multi-GPU training support via DataParallel
- Multiprocessing data loading for improved performance
- Float64 precision for better numerical stability
- Configurable hyperparameters matching MATLAB's trainNetwork

Multi-GPU Usage:
    The model automatically detects and uses all available GPUs when training.

Multiprocessing:
    DataLoader uses multiple workers on Linux/Mac for parallel data loading.
    On Windows, num_workers=0 to avoid multiprocessing issues.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import platform
from typing import Tuple, List


torch.set_default_dtype(torch.float64)  # MATLAB uses double precision


class LSTMClassifier(nn.Module):
    """
    LSTM-based sequence classifier
    Equivalent to MATLAB's sequenceInputLayer + lstmLayer architecture
    """
    def __init__(self, input_size, sequence_length, hidden_size, dropout_rate, num_classes=2):
        """
        Args:
            input_size: Number of input features (104 in MATLAB code)
            sequence_length: Expected length of input sequences (hash2(r, 8))
            hidden_size: Number of LSTM hidden units (hash2(r, 2))
            dropout_rate: Dropout probability (hash2(r, 4) / 100)
            num_classes: Number of output classes (2 for binary classification)
        """
        super(LSTMClassifier, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # First LSTM layer (sequence output mode)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            dtype=torch.float64
        )
        
        # First dropout layer
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        # Second LSTM layer (sequence output mode)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            dtype=torch.float64
        )
        
        # Second dropout layer
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes, dtype=torch.float64)
        
        # Softmax layer (applied in forward pass)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            output: Logits of shape (batch_size, sequence_length, num_classes)
            or (batch_size, num_classes) if returning last timestep only
        """
        # Validate input dimensions
        batch_size, seq_len, features = x.shape
        if features != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, but got {features}")
        # Note: LSTMs can handle variable sequence lengths, so we don't validate seq_len
        
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Fully connected layer
        # Apply to all timesteps in sequence
        out = self.fc(lstm2_out)
        
        return out


class LSTMMarketDataset(Dataset):
    def __init__(self, X, Y):
        """
        X: list or numpy array of shape [num_samples, seq_len, input_size]
        Y: list or numpy array of shape [num_samples] or [num_samples, seq_len]
        """

        # --- Convert X to tensor ---
        if isinstance(X, list):
            X = np.array(X, dtype=np.float64)
        elif isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        self.X = torch.tensor(X, dtype=torch.float64)

        # --- Convert Y to tensor ---
        if isinstance(Y, list):
            Y = np.array(Y)
        elif isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
        self.Y = torch.tensor(Y, dtype=torch.long)

        # --- Ensure label dimensionality matches MATLAB format ---
        # MATLAB sometimes uses 1D labels; PyTorch expects [batch, seq_len] for sequence output.
        if self.Y.ndim == 1 and self.X.ndim == 3:
            seq_len = self.X.shape[1]
            self.Y = self.Y.unsqueeze(1).repeat(1, seq_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def init_weights_matlab_style(module):
    """
    Initialize weights to match MATLAB's lstmLayer and fullyConnectedLayer defaults
    
    MATLAB uses:
    - Glorot (Xavier) uniform initialization for weights
    - Forget gate bias = 1.0 (prevents vanishing gradients)
    - All other biases = 0.0
    """
    if isinstance(module, nn.Linear):
        # MATLAB's fullyConnectedLayer uses Glorot initialization
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                # Input-to-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                # Hidden-to-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                # CRITICAL: MATLAB initializes forget gate bias to 1.0
                # This prevents vanishing gradients in LSTM
                nn.init.zeros_(param)
                # LSTM bias is [input_gate, forget_gate, cell_gate, output_gate]
                # Set forget gate bias (2nd quarter) to 1.0
                hidden_size = param.shape[0] // 4
                param.data[hidden_size:2*hidden_size].fill_(1.0)


def create_dataloader(dataset, batch_size, shuffle=False, num_workers=None):
    """
    Create a DataLoader with optimal settings for multiprocessing and GPU usage
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading. If None, auto-detect based on platform.
    
    Returns:
        DataLoader instance
    """
    if num_workers is None:
        # Use 4 workers on Unix-like systems, 0 on Windows (to avoid multiprocessing issues)
        num_workers = 4 if platform.system() != 'Windows' else 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # Speed up CPU to GPU transfer
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )


def classify_sequences(
    model: nn.Module,
    sequences: List[np.ndarray],
    device: str = 'cpu'
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Classify sequences and return predictions with confidence scores
    
    Args:
        model: Trained model
        sequences: List of sequences (time_steps x features) - shape: (20, 104)
        device: Device to run on
    
    Returns:
        predictions: List of predicted classes for last time step
        confidences: List of confidence arrays (num_classes,) for last time step
    """
    model.eval()
    model = model.to(device)
    
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for seq in sequences:
            # seq shape: (time_steps, features) = (20, 104)
            # Add batch dimension: (1, time_steps, features) = (1, 20, 104)
            seq_tensor = torch.from_numpy(seq).double().unsqueeze(0).to(device)
            
            # Forward pass - output shape: (1, time_steps, num_classes)
            output = model(seq_tensor)
            
            # Get last time step: (1, num_classes)
            last_output = output[:, -1, :]
            
            # Apply softmax to get probabilities
            probs = torch.softmax(last_output, dim=1)[0]  # (num_classes,)
            pred = torch.argmax(probs).item()
            
            predictions.append(pred)
            confidences.append(probs.cpu().numpy())
    
    return predictions, confidences


def train_lstm_model(model, train_loader, val_loader, hyperparams, avnin, avnout1, device='cpu'):
    """
    Train the LSTM model with settings equivalent to MATLAB trainingOptions
    
    Args:
        model: LSTMClassifier instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        hyperparams: Dictionary containing hash2(r, i) values
        avnin: Validation input data
        avnout1: Validation output data
        device: 'cpu' or 'cuda'
    
    Returns:
        model: Trained model (best model based on validation loss)
        history: Training history dictionary containing:
            - 'train_loss': List of training losses
            - 'train_acc': List of training accuracies
            - 'val_loss': List of validation losses
            - 'val_acc': List of validation accuracies
            - 'final_val_acc': Final validation accuracy on best model
    """
    
    # Move model to device
    model = model.to(device)
    
    # Enable multi-GPU training if requested and available
    is_data_parallel = False
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
        is_data_parallel = True
    #elif torch.cuda.is_available():
        # print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    # else:
        #print("Using CPU")

    # Extract hyperparameters from hyperparams
    initial_lr = hyperparams.get('initial_lr', 0.001)
    max_epochs = hyperparams.get('max_epochs', 100)
    l2_reg = hyperparams.get('l2_regularization', 0.0)
    lr_drop_factor = hyperparams.get('lr_drop_factor', 0.1)
    lr_drop_period = hyperparams.get('lr_drop_period', 10)
    validation_patience = hyperparams.get('validation_patience', 5)
    validation_frequency = hyperparams.get('validation_frequency', 7)

    # Adam optimizer with L2 regularization (weight_decay)
    # SquaredGradientDecayFactor = 0.999 corresponds to beta2 in Adam
    optimizer = optim.Adam(
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.999),  # beta2 = SquaredGradientDecayFactor
        weight_decay=l2_reg  # L2 regularization
    )
    
    # Learning rate scheduler (piecewise = StepLR in PyTorch)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_drop_period,
        gamma=lr_drop_factor
    )
    
    # Loss function (CrossEntropyLoss combines softmax + classification)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Early stopping parameters
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # For sequence classification, use last timestep
            if outputs.dim() == 3:  # (batch, seq_len, num_classes)
                # Use last timestep for classification
                outputs = outputs[:, -1, :]
            
            # Extract last timestep from targets if needed
            if targets.dim() > 1:  # (batch, seq_len)
                targets = targets[:, -1]  # Use last timestep label
            
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Average training loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase (every validation_frequency epochs)
        if (epoch + 1) % validation_frequency == 0 or epoch == 0:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    
                    if outputs.dim() == 3:
                        outputs = outputs[:, -1, :]
                    
                    # Extract last timestep from targets if needed
                    if targets.dim() > 1:
                        targets = targets[:, -1]
                    
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # Check for best model (OutputNetwork = 'best-validation-loss')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save the underlying model state if using DataParallel
                if is_data_parallel:
                    best_model_state = model.module.state_dict().copy()
                else:
                    best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping (ValidationPatience = 5)
            if patience_counter >= validation_patience:
                # print(f'Early stopping at epoch {epoch + 1}')
                break
            
            # print(f'Epoch [{epoch+1}/{max_epochs}], '
            #      f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
            #      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        # else:
        #    print(f'Epoch [{epoch+1}/{max_epochs}], '
        #          f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Update learning rate
        scheduler.step()
    
    # Load best model
    if best_model_state is not None:
        if is_data_parallel:
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
        # print(f'Loaded best model with validation loss: {best_val_loss:.4f}')
    
    # Calculate final_val_acc on the best model
    # Use ALL time steps for more granular accuracy (not just last time step)
    model.eval()
    final_val_correct = 0
    final_val_total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # Evaluate on ALL time steps for fine-grained accuracy
            if outputs.dim() == 3:  # (batch, seq_len, num_classes)
                batch_size, seq_len, num_classes = outputs.shape
                
                # Reshape outputs: (batch, seq_len, num_classes) -> (batch*seq_len, num_classes)
                outputs_flat = outputs.reshape(-1, num_classes)
                
                # Reshape targets: (batch, seq_len) -> (batch*seq_len)
                if targets.dim() > 1:
                    targets_flat = targets.reshape(-1)
                else:
                    targets_flat = targets
                
                _, predicted = torch.max(outputs_flat, 1)
                final_val_total += targets_flat.size(0)
                final_val_correct += (predicted == targets_flat).sum().item()
            else:
                # Fallback for 2D outputs
                if targets.dim() > 1:
                    targets = targets[:, -1]
                _, predicted = torch.max(outputs.data, 1)
                final_val_total += targets.size(0)
                final_val_correct += (predicted == targets).sum().item()
    
    final_validation_accuracy = 100.0 * final_val_correct / final_val_total
    history['final_val_acc'] = final_validation_accuracy
    
    # Return the unwrapped model (not DataParallel wrapper) for consistency
    if is_data_parallel:
        model = model.module
    
    return model, history


def main():
    """
    Example usage
    """
    # Example hash2 values (replace with actual values from your hash2 function)
    hyperparams = {
        'hidden_size': 128,   # LSTM hidden size
        'initial_lr': 10,    # Initial LR * 10000
        'dropout_rate': 20,    # Dropout * 100
        'max_epochs': 100,   # Max epochs
        'batch_size': 32,    # Mini batch size
        'dropout_rate': 50,    # LR drop factor * 100
        'lr_drop_period': 10,    # LR drop period
        'sequence_length': 50,    # Sequence length
        'l2_reg': 1     # L2 regularization * 100
    }
    
    # Model parameters
    input_size = 104
    num_classes = 2
    
    # Create model
    model = LSTMClassifier(
        input_size=input_size,
        sequence_length=hyperparams['sequence_length'],
        hidden_size=hyperparams['hidden_size'],
        dropout_rate=hyperparams['dropout_rate'],
        num_classes=num_classes
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Using dtype: {next(model.parameters()).dtype}")
    
    # Example: Create dummy data (replace with your actual data)
    # Training data
    X_train = np.random.randn(1000, hyperparams['sequence_length'], input_size)
    y_train = np.random.randint(0, 2, size=(1000,))
    
    # Validation data
    X_val = np.random.randn(200, hyperparams['sequence_length'], input_size)
    y_val = np.random.randint(0, 2, size=(200,))
    
    # Create datasets and dataloaders
    train_dataset = LSTMMarketDataset(X_train, y_train)
    val_dataset = LSTMMarketDataset(X_val, y_val)
    
    # Create DataLoaders with multiprocessing support
    # Shuffle = 'never' in MATLAB, so shuffle=False
    train_loader = create_dataloader(train_dataset, hyperparams['batch_size'], shuffle=False)
    val_loader = create_dataloader(val_dataset, hyperparams['batch_size'], shuffle=False)
    
    # Check device (ExecutionEnvironment = 'auto')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Train model with multi-GPU support
    trained_model, history = train_lstm_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        hyperparams=hyperparams,
        avnin=X_val,
        avnout1=y_val,
        device=device  # Enable multi-GPU training
    )
    
    # Save model
    torch.save(trained_model.state_dict(), 'lstm_model.pth')
    print("Model saved to lstm_model.pth")
    
    return trained_model, history


if __name__ == "__main__":
    model, history = main()

