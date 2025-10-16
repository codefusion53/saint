# SAINT1 v3.28 - Python/PyTorch Implementation

```
         OOOOOOOOO

           ***
         **    **
         *      *
         **   **
           ***
            *
         *  * *         * **
       *    *   *    *    *
      *     *     *
     *      *
        * * *
       *    *  *
             *     *
               *      *
                 *     *
             *       *
          *       *
        *      *
     *        *
     *****    *****
```

**Stochastic Artificial Intelligence Numerical Transform**

A high-performance LSTM-based neural network for financial time series prediction with multi-GPU acceleration and parallel processing support.

## Overview

SAINT1 is a sophisticated machine learning system designed to predict LONG/SHORT positions in financial markets using deep learning. This Python implementation leverages PyTorch to provide GPU-accelerated training with support for multiple markets and assets.

### Key Features

- 🚀 **Multi-GPU Training**: Automatic detection and utilization of all available GPUs
- ⚡ **Parallel Processing**: Process multiple days simultaneously for faster training
- 🎯 **High Accuracy**: LSTM-based architecture with configurable hyperparameters
- 📊 **Multi-Market Support**: Analyzes 26 different markets simultaneously
- 🔬 **MATLAB Alignment**: Produces results identical to the original MATLAB implementation
- 🛠️ **Flexible Configuration**: Easy asset selection and hyperparameter tuning

## Supported Markets

The system supports predictions for the following financial instruments:

1. **GC** - Gold Futures
2. **PE** - Crude Oil (Petroleum)
3. **RX** - Russell Index
4. **ZB** - Treasury Bonds
5. **GC Extended** - Gold Extended (default)
6. **CL** - Crude Oil
7. **YM** - Mini Dow
8. **ZC** - Corn
9. **ZS** - Soybeans
10. **ZW** - Wheat
11. **BTC** - Bitcoin

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for performance)
- Windows/Linux/macOS

### Dependencies

Install required packages using:

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `torch>=2.0.0` - PyTorch for deep learning
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `openpyxl>=3.1.0` - Excel file reading
- `psutil>=5.9.0` - Process management
- `scikit-learn>=1.3.0` - Additional preprocessing
- `matplotlib>=3.7.0` - Visualization

## Project Structure

```
Z:\Projects\Saint\Python\
├── saint1v328.py        # Main training script
├── lstm_model.py        # LSTM model architecture
├── loaddata28.py        # Single-market data loader
├── loaddata228.py       # Multi-market data loader
├── requirements.txt     # Python dependencies
```

## Usage

### Basic Training

Run the main training script:

```bash
python saint1v328.py
```

### Configuration

Edit `saint1v328.py` to configure:

**Asset Selection** (line 617):
```python
uu = 5  # 1=GC, 2=PE, 3=RX, 4=ZB, 5=GC Extended, etc.
```

**Mode Selection** (line 520):
```python
prod = 0  # 0 = development mode, 1 = production mode
```

**Training Range** (lines 484-485):
```python
sst = 670   # Start day
sfin = 0    # Finish day (0 = use all available data)
```

**Data Source** (line 507):
```python
mypath = "D:/cgc26.xlsx"  # Path to your market data
```

**Processing Mode** (line 720):
```python
use_parallel = True  # True for parallel, False for sequential
```

**GPU Workers** (lines 724-725):
```python
max_workers = 28
max_gpu_workers_override = 6  # Limit GPU workers to avoid OOM
```

## Model Architecture

### LSTM Classifier

The model uses a stacked LSTM architecture:

- **Input Layer**: 104 features (26 market prices + 26 slopes + 52 deltas)
- **LSTM Layer 1**: Configurable hidden units with dropout
- **LSTM Layer 2**: Configurable hidden units with dropout
- **Dense Layer**: Fully connected layer
- **Output**: 2 classes (SHORT=0, LONG=1)

### Feature Engineering

The system processes 26 different markets to create 104 input features:

1. **Price Features** (26): Close prices from 26 markets
2. **Slope Features** (26): Normalized price changes
3. **Delta Features** (52): Advanced delta calculations

### Hyperparameters

Each configuration includes:

- `hidden_size`: Number of LSTM hidden units
- `initial_lr`: Initial learning rate (÷10000)
- `dropout_rate`: Dropout probability (÷100)
- `max_epochs`: Maximum training epochs
- `batch_size`: Mini-batch size
- `lr_drop_factor`: Learning rate decay factor (÷100)
- `lr_drop_period`: Epochs between LR drops
- `sequence_length`: Input sequence length
- `l2_regularization`: L2 regularization strength (÷100)

## Training Process

### Workflow

1. **Data Loading**: Reads Excel file with 26 market sheets
2. **Preprocessing**: Normalizes data and calculates features
3. **Training**: Uses LSTM with Adam optimizer
4. **Validation**: Validates on hold-out set
5. **Prediction**: Makes LONG/SHORT predictions
6. **Evaluation**: Compares predictions with actual outcomes

### Parallel Processing

The system can train multiple days in parallel:

- **Sequential Mode**: Processes one day at a time (more stable)
- **Parallel Mode**: Processes multiple days simultaneously (faster)
- **Multi-GPU**: Distributes work across all available GPUs

### Output Files

Training produces three CSV files:

- `*hash.csv`: Prediction results (0=wrong, 2=correct SHORT, 3=correct LONG)
- `*hash1.csv`: Confidence scores for predictions
- `*hash2.csv`: Hyperparameter configurations used

## Performance

### Speed

- **CPU**: ~30-60 seconds per day (sequential)
- **Single GPU**: ~10-20 seconds per day (sequential)
- **Multi-GPU Parallel**: ~1-5 seconds per day (6-28 workers)

### Accuracy

- Validation accuracy typically ranges from 50-75%
- High-performing configurations marked with `****` in output
- Best configurations saved for production use

## Development Mode vs Production Mode

### Development Mode (`prod=0`)

- Uses row 0 (or specified row) from hyperparameter file
- Good for testing and experimentation
- Faster iteration

### Production Mode (`prod=1`)

- Uses asset-specific optimized hyperparameter row
- Leverages previously discovered best configurations
- Recommended for actual trading decisions

## Troubleshooting

### Out of Memory Errors

Reduce the number of GPU workers:

```python
max_gpu_workers_override = 2  # Reduce from 6 to 2
```

### Workers Getting Stuck

Switch to sequential processing:

```python
use_parallel = False
```

### CUDA Errors

1. Update PyTorch: `pip install torch --upgrade`
2. Verify CUDA installation
3. Reduce batch size in hyperparameters

### Multi-GPU Errors on Windows

If you see errors like:
- "module must have its parameters and buffers on device cuda:0 but found one of them on device: cuda:1"
- "PyTorch is not compiled with NCCL support"

**Solution**: The code has been fixed to handle Windows multi-GPU properly:

- All workers now use GPU 0 as their primary device
- PyTorch's `DataParallel` automatically distributes training across all GPUs within each worker
- This avoids device mismatch errors while still utilizing all available GPUs

**How it works**:
1. Each parallel worker process uses `cuda:0` (GPU 0) as the primary device
2. Within each worker, the `train_lstm_model` function uses `nn.DataParallel` to distribute the model across all available GPUs
3. This approach is Windows-compatible and doesn't require NCCL

**Performance**: You still get the benefit of multiple GPUs through DataParallel, though the scaling may be slightly less efficient than Linux NCCL-based distribution.

### Multiprocessing Issues (Windows)

The script includes aggressive worker cleanup. If issues persist:

1. Set `use_parallel = False` for sequential processing
2. Check that data can be pickled (automatic test on startup)
3. Ensure no antivirus blocking multiprocessing

## Technical Details

### Random Seed Control

The system uses deterministic algorithms for reproducibility:

- PyTorch: `torch.manual_seed(333)`
- NumPy: `np.random.seed(333)`
- CUDA: `torch.cuda.manual_seed_all(333)`
- cuDNN: Deterministic mode enabled

### Weight Initialization

Matches MATLAB's defaults:

- **LSTM**: Xavier (Glorot) uniform initialization
- **Forget Gate Bias**: 1.0 (prevents vanishing gradients)
- **Linear Layers**: Xavier uniform initialization

### Precision

Uses `float64` (double precision) throughout to match MATLAB's numerical precision.

## File Formats

### Input Data (Excel)

Excel file with 26 sheets, each containing:

| Column | Description |
|--------|-------------|
| 0      | Date        |
| 1      | Open        |
| 2      | Close       |
| 3      | High        |
| 4      | Low         |

### Output CSV Files

**Predictions** (`*hash.csv`):
- Rows: Days (1-based indexing)
- Columns: Hyperparameter configurations
- Values: 0=incorrect, 2=correct SHORT, 3=correct LONG

**Confidences** (`*hash1.csv`):
- Same structure as predictions
- Values: Confidence scores (0.0-1.0)

**Hyperparameters** (`*hash2.csv`):
- Rows: Configuration index
- Columns: 10 hyperparameter values

## Credits

**SAINT1 v3.28**  
6-1-2024 aii llc

Python/PyTorch implementation aligned with MATLAB reference code.

## License

Proprietary - All rights reserved

---

## Quick Start Example

```python
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training (development mode)
python saint1v328.py
```

## Support

For issues or questions:

1. Check error.log for detailed error messages
2. Try sequential processing if parallel mode fails

---

**Note**: This implementation maintains numerical equivalence with the MATLAB version while providing significant performance improvements through GPU acceleration and parallel processing.

