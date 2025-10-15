"""
SAINT1 v3.28 - Python Version
=============================================================================
Stochastic Artificial Intelligence Numerical Transform

Description:
    LSTM-based neural network for financial time series prediction
    with parallel processing and GPU acceleration support

6-1-2024 aii llc
=============================================================================
"""

import random
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
from typing import Tuple
from multiprocessing import Pool, cpu_count, freeze_support, TimeoutError as MPTimeoutError
import os
import signal
import sys
import atexit
import psutil  # For aggressive process cleanup

from loaddata28 import loaddata28
from loaddata228 import loaddata228
from lstm_model import LSTMClassifier, LSTMMarketDataset, create_dataloader, init_weights_matlab_style, train_lstm_model, classify_sequences

# ASCII Art Logo
LOGO = """
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
"""


# Global pool reference for cleanup
_global_pool = None
_worker_pids = []  # Track worker PIDs for aggressive cleanup


def kill_process_tree(pid):
    """Forcefully kill a process and all its children"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # Kill all children first
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        
        # Kill parent
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
            
    except psutil.NoSuchProcess:
        pass  # Process already dead
    except Exception:
        pass  # Ignore other errors


# Signal handler for graceful exit
def signal_handler(sig, frame):
    print('\n\nInterrupt signal received! Exiting...')
    sys.exit(0)


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    sequences, labels = zip(*batch)
    sequences = torch.stack(sequences)  # (batch, features, time_steps)
    labels = torch.stack(labels)  # (batch, time_steps)
    return sequences, labels


def optoai(
    k: int,
    fname: str,
    ain: np.ndarray,
    dout: np.ndarray,
    ifile: int,
    t_len: int,
    hash2: np.ndarray,
    r: int,
    snfai: np.ndarray,
    ssfin: int,
    sst: int,
    device: str = 'cpu',
    gpu_id: int = 0
) -> Tuple[int, int, float]:
    """
    AI calculation and prediction for a single day (individual process)

    MATLAB-to-Python Index Mapping:
    - MATLAB uses 1-based indexing: day k=670 means row 670
    - Python uses 0-based indexing: day k=670 means row k-1=669
    - However, we keep k as the MATLAB day number for consistency
    - When accessing arrays, we use k-1 for 0-based indexing

    Each call to this function runs independently in its own process.
    Elapsed time is measured ONLY for this specific day's training.

    Args:
        k: Current day index (MATLAB-style 1-based day number)
        fname: Data file name
        ain: Input features
        dout: Output labels
        ifile: Target asset index
        t_len: Length of DataFrame (number of days)
        hash2: Hyperparameter matrix
        r: Row index in hash2 (0-based)
        snfai: 3D market data array
        ssfin: Final day
        sst: Start day
        device: Device to train on
        gpu_id: GPU ID to use (for multi-GPU)

    Returns:
        k: Day index (for mapping results)
        ht: Prediction result (0=wrong, 2=correct SHORT, 3=correct LONG)
        h1: Confidence level
    """
    # Limit threads per worker process to avoid oversubscription
    torch.set_num_threads(1)  # Use 2 threads per worker for better CPU utilization
    
    # Initialize GPU in worker if requested (each worker gets its own CUDA context)
    if device == 'cuda':
        try:
            # Re-initialize CUDA in this worker process
            torch.cuda.init()
            # Verify GPU is accessible
            if torch.cuda.is_available():
                # Use assigned GPU (supports multi-GPU)
                num_gpus = torch.cuda.device_count()
                actual_gpu_id = gpu_id % num_gpus  # Wrap around if more workers than GPUs
                torch.cuda.set_device(actual_gpu_id)
                device = f'cuda:{actual_gpu_id}'
                print(f"Day {k}: Using GPU {actual_gpu_id} (of {num_gpus} available)", flush=True)
            else:
                # Fallback to CPU if GPU not available in worker
                device = 'cpu'
                print(f"Day {k}: GPU not available in worker, using CPU", flush=True)
        except Exception as e:
            # If GPU initialization fails, use CPU
            device = 'cpu'
            print(f"Day {k}: GPU init failed ({e}), using CPU", flush=True)
    
    # Create a minimal DataFrame-like object for loaddata228
    # It only needs len() to work
    class TProxy:
        def __init__(self, length):
            self._length = length
        def __len__(self):
            return self._length
    
    t = TProxy(t_len)

    _work_seed = 333

    # Load and prepare data
    # MATLAB: loaddata228(snfai, k-2, t, ifile) where k is 1-based day number
    # In Python, we keep k as the 1-based day number for consistency with MATLAB
    # loaddata228 expects a 0-based index internally, so we pass k-2 (which was k-2 in MATLAB too)
    anin, anout1, avnin, avnout1, pdata = loaddata228(snfai, k - 2, t, ifile)
    afile = ifile
    ht = 0
    h1 = 0.0

    control_number = int(hash2[r, 0])
    hidden_size = int(hash2[r, 1])
    initial_lr = hash2[r, 2] / 10000
    dropout_rate = hash2[r, 3] / 100
    max_epochs = int(hash2[r, 4])
    batch_size = int(hash2[r, 5])
    lr_drop_factor = hash2[r, 6] / 100
    lr_drop_period = int(hash2[r, 7])
    sequence_length = int(hash2[r, 8])
    l2_reg = hash2[r, 9] / 100
    
    # Create datasets and data loaders (BEFORE setting seed for training)
    train_dataset = LSTMMarketDataset(anin, anout1)
    val_dataset = LSTMMarketDataset(avnin, avnout1)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # MATLAB: Shuffle = 'never'
        num_workers=0
    )    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model (BEFORE setting final seed)
    model = LSTMClassifier(
        input_size=104,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        num_classes=2
    )
    
    # CRITICAL: Set random seed IMMEDIATELY before weight initialization and training
    # This matches MATLAB's gpurng(333); rng(333, "threefry"); sequence
    torch.manual_seed(_work_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_work_seed)  # Seed ALL GPUs
    np.random.seed(_work_seed)
    random.seed(_work_seed)
    
    # Enable cudnn deterministic mode for reproducibility
    if device.startswith('cuda'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
    
    # Initialize weights AFTER setting seed (matches MATLAB timing)
    model.apply(init_weights_matlab_style)
    model = model.to(device)
    
    hyperparams = {
        'initial_lr': initial_lr,
        'max_epochs': max_epochs,
        'l2_regularization': l2_reg,
        'lr_drop_factor': lr_drop_factor,
        'lr_drop_period': lr_drop_period,
        'validation_patience': 5,
        'validation_frequency': 7
    }
    
    # START TIMING: Measure only the actual training time for THIS process
    process_start_time = time.time()    

    trained_model, info = train_lstm_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        hyperparams=hyperparams,
        avnin=avnin,
        avnout1=avnout1,
        device=device
    )
    
    # END TIMING: Calculate elapsed time for THIS individual process only
    elapsed_time = time.time() - process_start_time
    
    # Display training results for this individual process
    val_acc = info['final_val_acc']

    # elapsed_time is the training time for THIS day/process only (not cumulative)
    # MATLAB: fprintf('%3.0f %3.0f md %1.0f ...', k, r, 99, ...)
    # k and r are already 1-based in MATLAB, so we print them as-is
    # But in Python, k is 1-based (MATLAB style), r is 0-based (Python style)
    # So we print k as-is, and r+1 to match MATLAB output
    if val_acc > 59:
        fmt = f"{k:3.0f} {r + 1:3.0f} md 99 **** {val_acc:5.1f} {elapsed_time:2.4f} {ifile + 1:1.0f}"
    else:
        fmt = f"{k:3.0f} {r + 1:3.0f} md 99      {val_acc:5.1f} {elapsed_time:2.4f} {ifile + 1:1.0f}"

    print(fmt, flush=True)  # Force output to display immediately
    
    # Display hyperparameters on final day or first day
    # MATLAB: if k == ssfin (1-based), so we compare k with ssfin directly
    show_hash = (k == ssfin) or (k == sst)  # Show on first and last day

    if show_hash:
        a1 = hash2[r, 0]
        a2 = hash2[r, 1]
        a3 = hash2[r, 2]
        a4 = hash2[r, 3]
        a5 = hash2[r, 4]
        a6 = hash2[r, 5]
        a7 = hash2[r, 6]
        a8 = hash2[r, 7]
        a9 = hash2[r, 8]
        a10 = hash2[r, 9]
        print(f" hash {a1:3.0f} {a2:3.3f} {a3:3.3f} {a4:3.3f} {a5:3.3f}", flush=True)
        print(f" hash {a6:3.3f} {a7:3.3f} {a8:3.3f} {a9:3.3f} {a10:3.3f}", flush=True)
    
    # Make predictions
    predictions, confidences = classify_sequences(trained_model, pdata, device)
    
    # Get prediction for last sequence, last time step
    pred_class = predictions[26]  # Index 26 (27th sequence)
    conf = confidences[26]  # Confidence scores
    
    h1 = float(conf[pred_class])  # Confidence level
    
    # Evaluate prediction for betting
    # MATLAB: if k == height(t) + 1 (future prediction, no actual data)
    # Python: k is 1-based day number, t_len = height(t) in MATLAB
    # So if k == t_len + 1, it's the future prediction (day 1001 when len(t)=1000)
    if k == t_len + 1:  # Future prediction (no actual data available at this k)
        if pred_class == 0:  # SHORT
            ht = 2
            h1 = float(conf[pred_class])
        if pred_class == 1:  # LONG
            ht = 3
            h1 = float(conf[pred_class])
    else:  # Historical prediction (can compare with actual)
        # MATLAB: snfai(k, 3, afile) and snfai(k, 4, afile)
        # k is 1-based in MATLAB, so snfai(k, ...) means row k
        # In Python 0-based: snfai[k-1, ..., afile]
        # MATLAB field 3 = Python index 2 (Open), MATLAB field 4 = Python index 3 (Close)

        # Safety check: ensure we're not accessing beyond snfai bounds
        if k - 1 >= snfai.shape[0]:
            print(f"WARNING: k={k} exceeds snfai bounds (shape={snfai.shape}), treating as future prediction", flush=True)
            # Treat as future prediction since we don't have actual data
            if pred_class == 0:  # SHORT
                ht = 2
                h1 = float(conf[pred_class])
            if pred_class == 1:  # LONG
                ht = 3
                h1 = float(conf[pred_class])
        else:
            open_price = snfai[k - 1, 2, afile]   # Open price (MATLAB field 3)
            close_price = snfai[k - 1, 3, afile]  # Close price (MATLAB field 4)

            if pred_class == 0:  # Predicted SHORT
                if open_price > close_price:
                    ht = 2  # Correct SHORT prediction
                else:
                    ht = 0  # Incorrect SHORT prediction
            elif pred_class == 1:  # Predicted LONG
                if open_price < close_price:
                    ht = 3  # Correct LONG prediction
                else:
                    ht = 1  # Incorrect LONG prediction
    
    # Free GPU memory if using CUDA
    if device == 'cuda':
        try:
            torch.cuda.empty_cache()
            # Also delete model from memory
            del model
        except:
            pass
    
    return k, ht, h1


def process_single_day(args):
    """Wrapper function for parallel processing"""
    try:
        k, ht, h1 = optoai(*args)
        return (k, ht, h1)
    except KeyboardInterrupt:
        # Return the day index from args with default values
        return (args[0], 0, 0.0)
    except RuntimeError as e:
        # CUDA out of memory or other GPU errors
        if 'CUDA' in str(e) or 'out of memory' in str(e):
            print(f"\nDay {args[0]}: GPU error ({str(e)[:50]}...), skipping", flush=True)
            return (args[0], 0, 0.0)
        else:
            raise  # Re-raise non-GPU errors
    except Exception as e:
        import traceback
        print(f"\n{'='*70}", flush=True)
        print(f"ERROR processing day {args[0]}:", flush=True)
        print(f"Exception type: {type(e).__name__}", flush=True)
        print(f"Exception message: {str(e)}", flush=True)
        print("Traceback:", flush=True)
        traceback.print_exc()
        print(f"{'='*70}\n", flush=True)
        return (args[0], 0, 0.0)


def cleanup_global_pool():
    """Cleanup function to ensure all workers are terminated on script exit"""
    global _global_pool, _worker_pids
    
    if _global_pool is not None:
        try:
            print("\n[atexit] Terminating remaining worker processes...", flush=True)
            _global_pool.terminate()
            _global_pool.join()  # Wait for termination (no timeout parameter)
            print("[atexit] Workers terminated via pool.", flush=True)
        except:
            pass
        finally:
            _global_pool = None
    
    # Aggressive cleanup: Kill any remaining workers by PID
    if _worker_pids:
        print(f"[atexit] Checking {len(_worker_pids)} tracked worker PIDs...", flush=True)
        killed = 0
        for pid in _worker_pids:
            try:
                if psutil.pid_exists(pid):
                    kill_process_tree(pid)
                    killed += 1
            except:
                pass
        if killed > 0:
            print(f"[atexit] Forcefully killed {killed} remaining processes.", flush=True)
        _worker_pids.clear()


def main():
    """Main execution function"""
    global _global_pool, _worker_pids, _work_seed

    _work_seed = 333
    
    # Register cleanup function to run on script exit
    atexit.register(cleanup_global_pool)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print(LOGO)
    print("="*70)
    print("SAINT1 v3.28 - Python/PyTorch Implementation")
    print("="*70)
    print(f"Start time: {datetime.now()}")
    
    # Set random seeds globally (matches MATLAB's initial rng(333, "threefry"))
    torch.manual_seed(_work_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_work_seed)  # Seed all GPUs
    np.random.seed(_work_seed)
    random.seed(_work_seed)
    
    # Enable deterministic algorithms globally
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variable for CUBLAS determinism (GPU operations)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Check for GPU(s)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = 'cuda'
        print(f"GPU(s) detected: {num_gpus} available")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"Using device: {device} (will distribute across all {num_gpus} GPU(s))")
    else:
        device = 'cpu'
        num_gpus = 0
        print(f"Using device: {device} (no GPU detected)")
    
    # Clean up temporary files
    gtot = Path('D:/testg.csv')
    if gtot.exists():
        gtot.unlink()
    
    # Configuration
    sst = 670   # Start day
    sfin = 0    # Finish day (0 = use all available data)
    
    # Data path configuration (most recent path is used)
    # mypath = "D:/capr4.xls"
    # mypath = "D:/cdec8.xls"
    # mypath = "D:/cdec26zb.xls"
    # mypath = "D:/cjan7.xls"
    # mypath = "D:/cjan28.xls"    # use with 1 2 4
    # mypath = "D:/feb3.xls"      # use with 3 5 6
    # mypath = "D:/cfeb26.xls"
    # mypath = "D:/cmar6.xls"
    # mypath = "D:/cmar21.xls"
    # mypath = "D:/cmar22.xls"
    # mypath = "D:/cmar28e.xls"
    # mypath = "D:/capr7.xls"
    # mypath = "D:/capr8a.xls"
    # mypath = "D:/capr23.xls"
    # mypath = "D:/capr29.xls"
    # mypath = "D:/capr30.xls"
    # mypath = "D:/c.xlsx"
    # mypath = "D:/cmar28e.xls"
    # mypath = "D:/crx.xlsx"
    mypath = "D:/cgc26.xlsx"
    
    fname = mypath
    tdate = 'D:/pate'
    
    # Load hyperparameter data
    hpdataname = 'D:/prodpy/hpdata3.csv'
    if Path(hpdataname).exists():
        hpdata = pd.read_csv(hpdataname, header=None).values.astype(np.float64)
    else:
        hpdata = np.zeros((14, 25), dtype=np.float64)
    
    # Mode configuration
    prod = 0  # Production mode: 0 = development (use r1=0), 1 = production (use asset-specific r1)
    # When prod=1, uses optimized hyperparameter row specific to each asset
    # When prod=0, uses row 0 (first configuration) for testing
    
    # Asset selection (5 = Gold Extended)
    asset_configs = {
        1: {  # GC (Gold)
            'snamet': 'D:/prodpy/gchash.csv',
            'sname1': 'D:/prodpy/gchash1.csv',
            'sname2': 'D:/prodpy/gchash2.csv',
            'sname2r': 'D:/ba/gc40hash2.csv',
            'r1': 0,
            'finf': 1,
        },
        2: {  # PE (Crude Oil)
            'snamet': 'D:/prodpy/2hash.csv',
            'sname1': 'D:/prodpy/2hash1.csv',
            'sname2': 'D:/prodpy/2hash2.csv',
            'sname2r': 'D:/ba/r100pechash2.csv',
            'r1': 128,
            'finf': 1,
        },
        3: {  # RX (Russell)
            'snamet': 'D:/prodpy/3hash.csv',
            'sname1': 'D:/prodpy/3hash1.csv',
            'sname2': 'D:/prodpy/3hash2.csv',
            'sname2r': 'D:/ba/rx30hash2.csv',
            'r1': 80,
            'finf': 1,
        },
        4: {  # ZB (Treasury Bonds)
            'snamet': 'D:/prodpy/4shash.csv',
            'sname1': 'D:/prodpy/4shash1.csv',
            'sname2': 'D:/prodpy/4shash2.csv',
            'sname2r': 'D:/ba/zb25hash2.csv',
            'r1': 301,
            'finf': 1,
        },
        5: {  # GC (Gold Extended)
            'snamet': 'D:/prodpy/5shash.csv',
            'sname1': 'D:/prodpy/5shash1.csv',
            'sname2': 'D:/prodpy/5shash2.csv',
            'sname2r': 'D:/prod/5shash2.csv',  # Use existing MATLAB hash2 file with 730 rows
            'r1': 654,
            'finf': 1,  # Target asset file index
        },
        6: {  # CL (Crude Oil)
            'snamet': 'D:/prodpy/6shash.csv',
            'sname1': 'D:/prodpy/6shash1.csv',
            'sname2': 'D:/prodpy/6shash2.csv',
            'sname2r': 'D:/ba/clhash2.csv',
            'r1': 193,
            'finf': 1,
        },
        7: {  # YM (Mini Dow)
            'snamet': 'D:/prodpy/7shash.csv',
            'sname1': 'D:/prodpy/7shash1.csv',
            'sname2': 'D:/prodpy/7shash2.csv',
            'sname2r': 'D:/ba/ymhash2.csv',
            'r1': 296,
            'finf': 1,
        },
        8: {  # ZC (Corn)
            'snamet': 'D:/prodpy/8shash.csv',
            'sname1': 'D:/prodpy/8shash1.csv',
            'sname2': 'D:/prodpy/8shash2.csv',
            'sname2r': 'D:/ba/zchash2.csv',
            'r1': 0,
            'finf': 1,
        },
        9: {  # ZS (Soybeans)
            'snamet': 'D:/prodpy/9shash.csv',
            'sname1': 'D:/prodpy/9shash1.csv',
            'sname2': 'D:/prodpy/9shash2.csv',
            'sname2r': 'D:/ba/zshash2.csv',
            'r1': 45,
            'finf': 1,
        },
        10: {  # ZW (Wheat)
            'snamet': 'D:/prodpy/10shash.csv',
            'sname1': 'D:/prodpy/10shash1.csv',
            'sname2': 'D:/prodpy/10shash2.csv',
            'sname2r': 'D:/ba/zwhash2.csv',
            'r1': 0,
            'finf': 1,
        },
        11: {  # BTC (Bitcoin)
            'snamet': 'D:/prodpy/11shash.csv',
            'sname1': 'D:/prodpy/11shash1.csv',
            'sname2': 'D:/prodpy/11shash2.csv',
            'sname2r': 'D:/ba/btchash2.csv',
            'r1': 0,
            'finf': 1,
        },
    }
    
    # Select asset
    uu = 5  # Gold Extended
    config = asset_configs[uu]
    
    # Get asset-specific finf.
    finf = config.get('finf', 1)
    finf = max(1, finf)
    
    # Set hyperparameter row based on production mode  
    if prod == 1:
        # In production mode
        # r=N-1 (0-based)
        r1 = config['r1']
        r1 = max(0, r1)  # Subtract 1, but don't go negative
        mode_str = "PRODUCTION"
    else:
        # In development mode
        r1 = 50
        mode_str = "DEVELOPMENT"
    
    r2 = r1
    
    print(f"\nAsset: {uu}")
    print(f"Mode: {mode_str} (prod={prod})")
    print(f"Data file: {fname}")
    print(f"Target asset index (finf): {finf}")
    print(f"Hyperparameter row: r={r2}")
    
    # Load data
    print("\nLoading data...")
    ain, dout, t, snfai = loaddata28(fname, ifile1a=finf - 1)
    
    # Load hash2 (hyperparameters)
    hash2_path = config['sname2r']
    if Path(hash2_path).exists():
        hash2 = pd.read_csv(hash2_path, header=None).values.astype(np.float64)
    else:
        # Default hyperparameters
        hash2 = np.array([[
            0,    # Hash control number
            128,  # Number of cells
            10,   # Learning rate (will be divided by 10000)
            50,   # Dropout rate (will be divided by 100)
            100,  # Max epochs
            16,   # Mini-batch size
            10,   # Learn rate drop factor (will be divided by 100)
            10,   # Learn rate drop period
            20,   # Sequence length
            1     # L2 regularization (will be divided by 100)
        ]], dtype=np.float64)

    print(f"Data shape: {snfai.shape}")
    print(f"Data type: {snfai.dtype}")
    print(f"Total days (len(t)): {len(t)}")
    print(f"DEBUG: t.shape[0] = {t.shape[0]}, snfai.shape[0] = {snfai.shape[0]}")
    print(f"Hyperparameter configurations: {len(hash2)} (hash2 shape: {hash2.shape})")
    print(f"DEBUG: sst={sst}, sfin={sfin}")

    # IMPORTANT: Validate r2 is within hash2 bounds
    # If hash2 only has 1 row, we must use row 0
    if r2 >= len(hash2):
        print(f"WARNING: r2={r2} is out of bounds for hash2 with {len(hash2)} rows")
        print(f"         Using row 0 instead (the only available configuration)")
        r2 = 0

    # Determine final day
    # 
    # Key insight: Day numbers are used directly as array indices    # 
    #   - Days 670-1000: historical days with actual data at snfai(day, :, :)
    #   - Day 1001 (height(t)+1): future prediction day (no data at this index)
    # Python needs range(670, 1002) to process days 670-1001 inclusive
    if sfin == 0:
        ssfin = len(t)
    else:
        ssfin = sfin
    
    print(f"Training from day {sst} to day {ssfin} (will process to {ssfin+1})")

    # Initialize result matrices
    # MATLAB: hasht = zeros(height(t) + 1, length(hash2))
    # where height(t) = len(t) in Python, length(hash2) = number of hyperparameter configurations
    #
    # IMPORTANT: We use day numbers as direct indices (k=1003 → hasht[1003, r])
    # So we need len(t) + 2 rows to accommodate day 1003 (indices 0-1003)
    # But we only SAVE the first len(t) + 1 rows to match MATLAB output (indices 0-1002)
    # When len(t)=1002:
    #   - Create 1004 rows (indices 0-1003) for storage
    #   - Save first 1003 rows (indices 0-1002) matching MATLAB's 1003×730 output
    hasht = np.zeros((len(t) + 2, len(hash2)), dtype=np.float64)
    hash1 = np.zeros((len(t) + 2, len(hash2)), dtype=np.float64)

    # Main training loop
    # MATLAB uses 1-based indexing: r=50 means row 50 (the 50th row)
    # Python uses 0-based indexing: r=49 means row 49 (the 50th row)
    # Since r2 is already 0-based after our validation, use it directly
    r = r2  # Use hyperparameter row from asset config (already 0-based)

    print(f"\nStarting training with hyperparameter set {r} (0-based index)...")
    print("="*70)
    
    # Parallel processing configuration
    # IMPORTANT: If workers get stuck and don't return results, set this to False
    # For SPEED: Set to True (uses all GPUs if available)
    # For RELIABILITY: Set to False (sequential, more stable on Windows)
    use_parallel = True  # Set to True for parallel speed, False for sequential reliability
    
    # GPU worker configuration (only if device == 'cuda')
    # If you get out-of-memory errors, reduce this number:
    max_workers = 28
    max_gpu_workers_override = None  # Set to integer (e.g., 2) to manually limit GPU workers
    
    if use_parallel:
        # Parallel processing
        # GPU parallel processing: distribute workers across all GPUs
        if device == 'cuda':
            # Multi-GPU support
            num_gpus = torch.cuda.device_count()
            
            if max_gpu_workers_override is not None:
                # Use manual override
                num_workers = max_gpu_workers_override
                print(f"\n{num_gpus} GPU(s) detected")
                print(f"Using {num_workers} parallel workers (MANUAL OVERRIDE)")
            else:
                # Automatic calculation based on total GPU memory across all GPUs
                total_gpu_memory = sum(
                    torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    for i in range(num_gpus)
                )
                
                # Each worker needs ~1-2GB, distribute across all GPUs
                max_workers_per_gpu = 16  # NUCLEAR: 16 workers per GPU for absolute maximum speed
                max_gpu_workers = num_gpus * max_workers_per_gpu
                num_workers = min(cpu_count(), max_gpu_workers, 128)  # Cap at 128 total for max throughput
                
                print(f"\n{num_gpus} GPU(s) detected (Total memory: {total_gpu_memory:.1f} GB)")
                print(f"Using {num_workers} parallel workers across {num_gpus} GPU(s)")
                print(f"  ~{num_workers // num_gpus} workers per GPU")
            
            print(f"Note: Workers will be distributed across all {num_gpus} GPU(s)")
            print(f"      Set max_gpu_workers_override if you get OOM errors")
        else:
            # CPU parallel processing: can use more workers
            num_workers = min(cpu_count(), max_workers)
            print(f"Using {num_workers} parallel workers (CPU cores available: {cpu_count()})")
            print("Using CPU for parallel processing")
        
        # Set environment variables to limit thread usage per worker
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # Also limit PyTorch threads
        torch.set_num_threads(1)
        print(f"Limited each worker to 1 threads (total cores used: ~{num_workers})")
        
        # Convert DataFrame to dict for pickling (Windows multiprocessing issue)
        t_len = len(t)
        
        # Prepare arguments for each day, distributing across GPUs if available
        # MATLAB: for iday = sst:ssfin+1 where sst=670, ssfin=height(t)=len(t)
        # If len(t)=1000, MATLAB processes days 670 to 1001 (inclusive)
        # Python: range(sst, ssfin + 1 + 1) = range(670, 1002) gives 670-1001
        print(f"DEBUG: Creating args_list for range({sst}, {ssfin + 1 + 1}) = days {sst} to {ssfin+1}", flush=True)
        args_list = []
        for idx, iday in enumerate(range(sst, ssfin + 1 + 1)):
            # Assign GPU in round-robin fashion across all available GPUs
            gpu_id = idx % num_gpus if device == 'cuda' and num_gpus > 0 else 0

            args_list.append((
                iday, fname, ain, dout, finf - 1, t_len, hash2, r, snfai, ssfin, sst, device, gpu_id
            ))
        
        print(f"Processing {len(args_list)} days in parallel...")
        print("Note: Press Ctrl+C to stop (may take a moment to respond)\n")
        
        # Test pickling (Windows multiprocessing requirement)
        pickle_success = True
        try:
            import pickle
            print("Testing data serialization for multiprocessing...", flush=True)
            test_args = args_list[0]
            pickle.dumps(test_args)
            print("Serialization test passed!", flush=True)
        except Exception as e:
            print(f"ERROR: Cannot serialize data for multiprocessing: {e}")
            print("This is a Windows multiprocessing issue. Falling back to sequential processing...")
            pickle_success = False
        
        if pickle_success:  # Only use parallel if pickle test passed
            global _global_pool, _worker_pids  # Declare globals at start of scope
            
            pool = None
            completed = 0
            interrupted = False
            results_dict = {}  # Store results as they come in
            
            try:
                print("Creating process pool...", flush=True)
                pool = Pool(num_workers)
                _global_pool = pool  # Store in module-level global for atexit cleanup
                
                # Track worker PIDs for aggressive cleanup
                _worker_pids = [p.pid for p in pool._pool] if hasattr(pool, '_pool') and pool._pool else []
                print(f"Process pool created successfully! Tracking {len(_worker_pids)} worker PIDs", flush=True)
                
                # Submit all tasks and get AsyncResult objects
                print("Submitting parallel tasks...", flush=True)
                total_tasks = len(args_list)
                async_results = []
                
                for args in args_list:
                    async_result = pool.apply_async(process_single_day, (args,))
                    async_results.append((args[0], async_result))  # (day_index, AsyncResult)
                
                print(f"Submitted {total_tasks} tasks to {num_workers} workers", flush=True)
                
                # Close pool to prevent new tasks
                pool.close()
                
                # Wait for results with timeout and progress tracking
                print(f"\nProcessing {total_tasks} days...")
                print("Progress updates every 10 seconds...", flush=True)
                
                timeout_per_task = 300  # 5 minutes max per task
                global_timeout = 900  # 15 minutes total (reduced monitoring overhead)
                start_time = time.time()
                last_progress_time = start_time
                progress_interval = 20  # Show progress every 20 seconds (reduced overhead)
                
                stuck_check_interval = 30  # Check for stuck tasks every 30 seconds
                last_stuck_check = start_time
                
                while completed < total_tasks:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Check global timeout
                    if elapsed > global_timeout:
                        print(f"\n{'='*70}")
                        print(f"GLOBAL TIMEOUT: {elapsed:.0f}s exceeded limit of {global_timeout:.0f}s")
                        print(f"Completed: {completed}/{total_tasks} tasks")
                        print("Terminating all workers...")
                        print(f"{'='*70}")
                        pool.terminate()
                        pool.join()
                        break
                    
                    # Periodic progress update (reduced frequency for speed)
                    if current_time - last_progress_time >= progress_interval:
                        print(f"Progress: {completed}/{total_tasks} completed ({100*completed/total_tasks:.1f}%) - Elapsed: {elapsed:.0f}s", flush=True)
                        last_progress_time = current_time
                        progress_interval = 30  # Increase interval to 30s after first update
                    
                    # Check for stuck workers periodically
                    if current_time - last_stuck_check >= stuck_check_interval:
                        # Count how many tasks are still pending
                        pending = sum(1 for _, ar in async_results if not ar.ready())
                        if pending > 0 and completed > 0:  # Only show if we have some progress
                            print(f"Status: {completed} done, {pending} pending, {total_tasks - completed - pending} waiting", flush=True)
                        last_stuck_check = current_time
                    
                    # Try to collect completed results
                    any_collected = False
                    for day_idx, async_result in async_results:
                        if day_idx in results_dict:
                            continue  # Already collected
                        
                        if async_result.ready():
                            try:
                                result = async_result.get(timeout=1.0)
                                day_idx, ht, h1 = result
                                hasht[day_idx, r] = ht
                                hash1[day_idx, r] = h1
                                results_dict[day_idx] = True
                                completed += 1
                                any_collected = True
                            except Exception as e:
                                print(f"\nError collecting result for day {day_idx}: {e}", flush=True)
                                results_dict[day_idx] = True  # Mark as processed to avoid retry
                                completed += 1
                    
                    # If nothing was collected this iteration, sleep briefly
                    if not any_collected:
                        time.sleep(0.05)  # Reduced sleep for faster polling
                    
                    # Check if any tasks have been running too long
                    if current_time - start_time > timeout_per_task and completed == 0:
                        print(f"\n{'='*70}")
                        print(f"WARNING: No tasks completed after {timeout_per_task}s")
                        print("This suggests workers are stuck. Terminating pool...")
                        print(f"{'='*70}")
                        pool.terminate()
                        pool.join()
                        print("Pool terminated. Consider using use_parallel = False")
                        break
                
                # Wait for remaining workers to finish
                print("\nWaiting for pool to close...", flush=True)
                pool.close()
                pool.join()  # Wait for all workers to finish (blocks until done)
                
                # Check if pool is still alive (workers didn't finish)
                if pool._pool and any(p.is_alive() for p in pool._pool):
                    print("\nWarning: Some workers are still alive.", flush=True)
                    print("Forcefully terminating remaining workers...", flush=True)
                    pool.terminate()
                    pool.join()  # Clean up terminated workers
                
                if completed == total_tasks:
                    print(f"\n✓ Successfully completed all {completed} days!")
                else:
                    print(f"\n⚠ Completed {completed}/{total_tasks} days ({100*completed/total_tasks:.1f}%)")
                    print(f"  {total_tasks - completed} tasks did not complete")
                
            except KeyboardInterrupt:
                interrupted = True
                print("\n\n" + "="*70)
                print("KEYBOARD INTERRUPT DETECTED!")
                print("="*70)
                print(f"Processed {completed} out of {total_tasks} days")
                if pool is not None:
                    print("Terminating worker processes (please wait)...")
                    pool.terminate()
                    pool.join()  # Wait for termination (no timeout parameter)
                    print("Workers terminated.")
                print("Saving partial results...")
            
            except Exception as e:
                import traceback
                print(f"\n\n{'='*70}")
                print("FATAL ERROR during parallel processing:")
                print(f"{'='*70}")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {str(e)}")
                print(f"\nCompleted {completed} out of {total_tasks} days before error")
                print("\nFull traceback:")
                traceback.print_exc()
                print(f"{'='*70}")
                if pool is not None:
                    print("Forcefully terminating all workers...")
                    pool.terminate()
                    pool.join()  # Wait for termination (no timeout parameter)
                    print("Workers terminated.")
                print("\nSuggestion: Try setting use_parallel = False in the code")
                print("for sequential processing, which is more reliable on Windows.")
                print(f"{'='*70}")
                print("Saving partial results...")
            
            finally:
                # Ensure pool is always cleaned up
                print("\nCleaning up worker pool...", flush=True)
                if pool is not None:
                    try:
                        print("Terminating worker processes via pool...", flush=True)
                        pool.terminate()
                        pool.join()  # Wait for termination (no timeout parameter)
                        print("Pool terminated.", flush=True)
                        
                        # Aggressive cleanup: Check if any workers are still alive
                        alive_workers = []
                        if hasattr(pool, '_pool') and pool._pool:
                            alive_workers = [p for p in pool._pool if p.is_alive()]
                        
                        if alive_workers:
                            print(f"Warning: {len(alive_workers)} workers still alive after terminate()", flush=True)
                            print("Forcefully killing remaining workers by PID...", flush=True)
                            for p in alive_workers:
                                try:
                                    kill_process_tree(p.pid)
                                except:
                                    pass
                            print("Forced kill complete.", flush=True)
                        
                        # Additional safety: Kill by tracked PIDs
                        if _worker_pids:
                            remaining = 0
                            for pid in _worker_pids:
                                try:
                                    if psutil.pid_exists(pid):
                                        kill_process_tree(pid)
                                        remaining += 1
                                except:
                                    pass
                            if remaining > 0:
                                print(f"Killed {remaining} processes by tracked PID.", flush=True)
                        
                        print("All workers terminated successfully.", flush=True)
                        
                    except Exception as cleanup_error:
                        print(f"Warning during cleanup: {cleanup_error}", flush=True)
                        # Even if cleanup fails, try to kill by PID
                        if _worker_pids:
                            print("Attempting PID-based cleanup...", flush=True)
                            for pid in _worker_pids:
                                try:
                                    kill_process_tree(pid)
                                except:
                                    pass
                    finally:
                        # Clear both local and global references
                        _global_pool = None
                        pool = None
                        _worker_pids.clear()
        else:
            # Pickle failed - fallback to sequential processing
            print("\nFalling back to sequential processing...")
            t_len = len(t)
            try:
                # MATLAB: for iday = sst:ssfin+1 where ssfin=height(t)=len(t)
                # Python: range(sst, ssfin + 1 + 1) = range(670, 1002) gives 670-1001
                for iday in range(sst, ssfin + 1 + 1):
                    gpu_id = 0  # Sequential mode uses first GPU only
                    k, ht, h1 = optoai(
                        iday, fname, ain, dout, finf - 1, t_len, hash2, r, snfai, ssfin, sst, device, gpu_id
                    )
                    hasht[k, r] = ht
                    hash1[k, r] = h1
            except KeyboardInterrupt:
                print("\n\nKeyboard interrupt detected! Stopping...")
                print("Saving partial results...")
    else:
        # Sequential processing (easier for debugging)
        print("\n" + "="*70)
        print("RUNNING SEQUENTIAL PROCESSING")
        print("="*70)
        print(f"Processing days {sst} to {ssfin} sequentially...")
        print(f"Total days to process: {ssfin - sst}")
        print(f"Device: {device}")
        print(f"Hyperparameter row: r={r}")
        print(f"Target asset (finf): {finf}")
        print()
        
        try:
            t_len = len(t)
            processed_count = 0
            # MATLAB: for iday = sst:ssfin+1 where ssfin=height(t)=len(t)
            # Python: range(sst, ssfin + 1 + 1) = range(670, 1002) gives 670-1001
            for iday in range(sst, ssfin + 1 + 1):
                gpu_id = 0  # Sequential mode uses first GPU only
                k, ht, h1 = optoai(
                    iday, fname, ain, dout, finf - 1, t_len, hash2, r, snfai, ssfin, sst, device, gpu_id
                )
                hasht[k, r] = ht
                hash1[k, r] = h1
                processed_count += 1
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected! Stopping...")
            print("Saving partial results...")
    
    # Save results
    print("\nSaving results...")

    # Debug: Check if any predictions were made
    non_zero_count = np.count_nonzero(hasht)
    print(f"Debug: Total non-zero predictions: {non_zero_count}")
    print(f"Debug: Result matrix shape: {hasht.shape}")
    print(f"Debug: Using hyperparameter column r={r}")
    print(f"Debug: Non-zero in column {r}: {np.count_nonzero(hasht[:, r])}")

    # IMPORTANT: Only save first len(t)+1 rows to match MATLAB output
    # MATLAB uses 1-based indexing so hasht has rows 1 to height(t)+1 = 1 to 1003
    # Python stores days using day numbers as indices, so we have indices 0-1003 (1004 rows)
    # But we only save indices 0 to len(t) = 0 to 1002 (first 1003 rows) to match MATLAB
    num_rows_to_save = len(t) + 1  # 1003 rows when len(t)=1002
    print(f"Debug: Saving first {num_rows_to_save} rows (out of {hasht.shape[0]} total) to match MATLAB format")

    # Use numpy savetxt for better control over number formatting
    # Save only the first len(t)+1 rows to match MATLAB's output format
    np.savetxt(config['snamet'], hasht[:num_rows_to_save, :], delimiter=',', fmt='%.0f')  # Integers
    np.savetxt(config['sname1'], hash1[:num_rows_to_save, :], delimiter=',', fmt='%.15g')  # Floats with precision
    np.savetxt(config['sname2'], hash2, delimiter=',', fmt='%.15g')  # Mixed int/float
    
    print(f"\nResults saved to:")
    print(f"  - {config['snamet']}")
    print(f"  - {config['sname1']}")
    print(f"  - {config['sname2']}")
    
    print(f"\nEnd time: {datetime.now()}")
    print("="*70)


if __name__ == '__main__':
    # Required for Windows multiprocessing support
    freeze_support()
    main()

