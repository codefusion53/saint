# SAINT1 v3.28 - Financial Market Prediction System

**Stochastic Artificial Intelligence Numerical Transform**

A sophisticated LSTM-based neural network system for financial time series prediction with parallel processing and GPU acceleration support.

## Overview

SAINT1 v3.28 is a Python implementation of a MATLAB-based financial prediction system that uses Long Short-Term Memory (LSTM) neural networks to predict market movements. The system processes data from 26 different financial markets and generates LONG/SHORT trading signals with confidence scores.

## Features

- **Multi-Market Analysis**: Processes data from 26 different financial markets simultaneously
- **LSTM Neural Networks**: Uses stacked LSTM layers for time series prediction
- **GPU Acceleration**: Supports CUDA-enabled GPUs with multi-GPU parallel processing
- **Feature Engineering**: 104 input features including prices, slopes, and delta calculations
- **Parallel Processing**: Multi-process training for faster execution
- **Memory Optimized**: Uses float16 precision for maximum memory efficiency
- **MATLAB Compatibility**: Maintains compatibility with original MATLAB implementation

## System Architecture

### Core Components

1. **`loaddata28.py`** - Data loading and preprocessing
2. **`loaddata228.py`** - Training/validation/test data preparation
3. **`lstm_model.py`** - LSTM neural network implementation
4. **`saint1v328.py`** - Main execution and orchestration

### Data Flow

```
Excel Data → loaddata28 → Feature Engineering → loaddata228 → LSTM Training → Predictions
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Windows 10/11 (tested platform)

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **PyTorch 2.0+** - Neural network framework
- **NumPy 1.24+** - Numerical computations
- **Pandas 2.0+** - Data manipulation
- **OpenPyXL 3.1+** - Excel file reading
- **psutil 5.9+** - Process management

## Usage

### Basic Usage

```python
python saint1v328.py
```

### Configuration

The system supports multiple asset configurations:

- **Asset 1**: GC (Gold)
- **Asset 2**: PE (Crude Oil)
- **Asset 3**: RX (Russell)
- **Asset 4**: ZB (Treasury Bonds)
- **Asset 5**: GC (Gold Extended) - Default
- **Asset 6**: CL (Crude Oil)
- **Asset 7**: YM (Mini Dow)
- **Asset 8**: ZC (Corn)
- **Asset 9**: ZS (Soybeans)
- **Asset 10**: ZW (Wheat)
- **Asset 11**: BTC (Bitcoin)

### Data Format

The system expects Excel files with:
- Multiple sheets (one per market)
- OHLC data (Open, High, Low, Close)
- Date column in first column
- Price data in columns 2-5

## Technical Details

### Feature Engineering

The system creates 104 input features:

- **Features 1-26**: Current close prices from 26 markets
- **Features 27-52**: 1-day slope calculations with shorts multiplier
- **Features 53-78**: 2-day slope calculations with shorts multiplier  
- **Features 79-104**: 3-day slope calculations with shorts multiplier

### LSTM Architecture

- **Input Layer**: 104 features
- **LSTM Layer 1**: Hidden size (configurable)
- **Dropout Layer 1**: Configurable dropout rate
- **LSTM Layer 2**: Hidden size (configurable)
- **Dropout Layer 2**: Configurable dropout rate
- **Fully Connected Layer**: 2 classes (LONG/SHORT)
- **Output**: Classification probabilities

### Training Process

1. **Data Loading**: Load and preprocess market data from Excel
2. **Feature Engineering**: Calculate technical indicators and slopes
3. **Sequence Creation**: Create 20-day sequences for LSTM training
4. **Model Training**: Train LSTM with validation-based early stopping
5. **Prediction**: Generate predictions for next-day market movements

### Performance Optimization

- **Float16 Precision**: Reduces memory usage by 50%
- **GPU Acceleration**: Utilizes CUDA for faster training
- **Parallel Processing**: Multi-process training across available cores
- **Memory Management**: Aggressive cleanup to prevent memory leaks

## Output

The system generates three main output files:

1. **`*hash.csv`** - Prediction results (0=wrong, 2=correct SHORT, 3=correct LONG)
2. **`*hash1.csv`** - Confidence scores for predictions
3. **`*hash2.csv`** - Hyperparameter configurations used

## Configuration Options

### Production vs Development Mode

- **Development Mode** (`prod=0`): Uses row 50 for testing
- **Production Mode** (`prod=1`): Uses asset-specific optimized hyperparameters

### Parallel Processing

- **Parallel Mode**: Uses all available CPU cores and GPUs
- **Sequential Mode**: Single-threaded processing (more reliable on Windows)

### GPU Configuration

- **Multi-GPU Support**: Automatically distributes workers across available GPUs
- **Memory Management**: Configurable worker limits to prevent OOM errors

## File Structure

```
├── loaddata28.py          # Data loading and preprocessing
├── loaddata228.py         # Training data preparation
├── lstm_model.py          # LSTM neural network implementation
├── saint1v328.py          # Main execution script
├── requirements.txt       # Python dependencies
├── Results/               # Output directory
│   ├── Matlab/           # MATLAB comparison results
│   └── Python/           # Python results
└── README.md             # This file
```

## Performance Metrics

The system tracks several performance metrics:

- **Validation Accuracy**: Model performance on validation data
- **Training Time**: Time per epoch and total training time
- **Memory Usage**: GPU and CPU memory utilization
- **Prediction Accuracy**: Historical prediction accuracy

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `max_gpu_workers_override` or use sequential mode
2. **Windows Multiprocessing Issues**: Set `use_parallel = False`
3. **Data Loading Errors**: Ensure Excel files have correct format
4. **Memory Leaks**: System includes aggressive cleanup mechanisms

### Debug Mode

Enable debug output by setting appropriate flags in the configuration section.

## License

Copyright (c) 2024 aii llc. All rights reserved.

## Version History

- **v3.28**: Current version with PyTorch implementation
- **v3.27**: Previous MATLAB version
- **v3.26**: Earlier versions with different architectures

## Contributing

This is a proprietary system. For questions or issues, contact the development team.

## Disclaimer

This software is for research and educational purposes only. Past performance does not guarantee future results. Trading financial instruments involves risk of loss.
