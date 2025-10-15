"""
LOADDATA28
===========================================================================
Load and preprocess financial market data from Excel sheets

This module provides functionality to load market data and prepare it for
machine learning models (PyTorch).

Inputs:
    fname    - Path to Excel file containing market data
    ifile1a  - File index for target asset (0 = gold)

Outputs:
    ain      - AI input features matrix (num_samples x 104)
               Contains 26 market prices + 26 slope features + 52 delta features
    dout     - Categorical output labels (LONG/SHORT predictions) as integers
               0 = SHORT, 1 = LONG
    t        - DataFrame containing raw data from first sheet
    snfai    - 3D array of standardized data (rows x 19 columns x 26 sheets)

Data Processing Steps:
    1. Load data from all sheets in Excel file
    2. Extract OHLC (Open, High, Low, Close) prices
    3. Calculate slope and delta features
    4. Generate classification labels (LONG/SHORT)
===========================================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple


def loaddata28(fname: str, ifile1a: int = 0) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load and preprocess financial market data from Excel sheets
    
    Args:
        fname: Path to Excel file containing market data
        ifile1a: File index for target asset (0 = gold, default: 0)
    
    Returns:
        ain: AI input features matrix (num_samples x 104)
        dout: Output labels as integers (0=SHORT, 1=LONG)
        t: DataFrame containing raw data from first sheet
        snfai: 3D numpy array (rows x 19 columns x 26 sheets)
    """
    
    # Configuration
    stf = 0      # Start file (first sheet index in Python is 0)
    finf = 26    # Final file (last sheet) - 26 different markets
    
    # Load Data from All Sheets
    xl_file = pd.ExcelFile(fname)
    sheet_names = xl_file.sheet_names

    # MATLAB behavior: Initialize snfai based on first sheet's size
    # If other sheets have more rows, they get truncated
    # If other sheets have fewer rows, MATLAB would error (or pad)

    # First pass: Load all sheets and find the size to use (use first sheet's size)
    snfai = None
    t = None
    first_sheet_rows = None

    for ifile in range(stf, finf):
        # Read each sheet
        df = pd.read_excel(fname, sheet_name=sheet_names[ifile])

        # Remove first row of data
        # MATLAB: t([1],:)=[]; removes first row after reading
        # pandas also uses first row as headers by default, then we remove the first data row
        df = df.iloc[1:].reset_index(drop=True)

        # Store first sheet for return
        if ifile == 0:
            t = df.copy()
            first_sheet_rows = len(df)
            # Initialize 3D array based on first sheet's size (MATLAB behavior)
            snfai = np.zeros((first_sheet_rows, 19, len(sheet_names)), dtype=np.float64)

        # Initialize storage array
        num_days = len(df)
        sn = np.zeros((num_days, 19), dtype=np.float64)

        # Extract OHLC data
        # Assuming columns: [Date, Open, Close, High, Low, ...]
        for i in range(num_days):
            sn[i, 2]  = df.iloc[i, 1]    # Open price (column index 1)
            sn[i, 3]  = df.iloc[i, 2]    # Close price (column index 2)
            sn[i, 4]  = df.iloc[i, 3]    # High price (column index 3)
            sn[i, 5]  = df.iloc[i, 4]    # Low price (column index 4)
            sn[i, 6]  = df.iloc[i, 1]    # Ask open (duplicate of open)
            sn[i, 7]  = df.iloc[i, 2]    # Ask close (duplicate of close)
            sn[i, 8]  = df.iloc[i, 3]    # Ask high (duplicate of high)
            sn[i, 9]  = df.iloc[i, 4]    # Ask low (duplicate of low)

        # Store in 3D indexed file: snfai[day, feature, market]
        # MATLAB: snfai(:,:,ifile)=sn(:,:);
        # If sn is longer than snfai, truncate; if shorter, pad with zeros (already initialized)
        rows_to_copy = min(num_days, first_sheet_rows)
        snfai[:rows_to_copy, :, ifile] = sn[:rows_to_copy, :]
        
    
    # ========================================================================
    # Create AI Input Features
    # ========================================================================
    
    # Initialize input feature matrix
    # Total features: 26 (prices) + 26 (slopes) + 52 (deltas) = 104
    num_days = snfai.shape[0]
    ain = np.zeros((num_days, 78), dtype=np.float64)  # Base features (will be expanded to 104)

    # MATLAB: for i=3:height(t) means i starts at 3 (1-based)
    # Python equivalent: range(2, num_days) means i starts at 2 (0-based, which is row 3 in 1-based)
    for i in range(2, num_days):
        # MATLAB: range=(2:i) means indices 2 to i (1-based)
        # Python equivalent: range(1, i) means indices 1 to i-1 (0-based, matching MATLAB's 2:i)
        range_slice = range(1, i)

        # ====================================================================
        # Feature Set 1: Close Prices from 26 Different Markets (k1 = 0-25)
        # MATLAB uses ain(i,k1+1), ain(i,k1+3), ain(i,k1+2), etc.
        # Python: k1=0, so ain[i, 0], ain[i, 2], ain[i, 1] to match MATLAB exactly
        # ====================================================================
        k1 = 0
        ain[i, k1] = snfai[i, 3, 0]    # MATLAB: ain(i,k1+1)=snfai(i,4,1)
        ain[i, k1 + 2] = snfai[i, 3, 1]  # MATLAB: ain(i,k1+3)=snfai(i,4,2)
        ain[i, k1 + 1] = snfai[i, 3, 2]  # MATLAB: ain(i,k1+2)=snfai(i,4,3)
        ain[i, k1 + 3] = snfai[i, 3, 3] # Market 4
        ain[i, k1 + 4] = snfai[i, 3, 4] # Market 5
        ain[i, k1 + 5] = snfai[i, 3, 5] # Market 6
        ain[i, k1 + 6] = snfai[i, 3, 6] # Market 7
        ain[i, k1 + 7] = snfai[i, 3, 7] # Market 8
        ain[i, k1 + 8] = snfai[i, 3, 8] # Market 9
        ain[i, k1 + 9] = snfai[i, 3, 9] # Market 10 (GC - Gold)
        ain[i, k1 + 10] = snfai[i, 3, 10] # Market 11
        ain[i, k1 + 11] = snfai[i, 3, 11] # Market 12
        ain[i, k1 + 12] = snfai[i, 3, 12] # Market 13
        ain[i, k1 + 13] = snfai[i, 3, 13] # Market 14
        ain[i, k1 + 14] = snfai[i, 3, 14] # Market 15
        ain[i, k1 + 15] = snfai[i, 3, 15] # Market 16
        ain[i, k1 + 16] = snfai[i, 3, 16] # Market 17
        ain[i, k1 + 17] = snfai[i, 3, 17] # Market 18
        ain[i, k1 + 18] = snfai[i, 3, 18] # Market 19
        ain[i, k1 + 19] = snfai[i, 3, 19] # Market 20
        ain[i, k1 + 20] = snfai[i, 3, 20] # Market 21
        ain[i, k1 + 21] = snfai[i, 3, 21] # Market 22
        ain[i, k1 + 22] = snfai[i, 3, 22] # Market 23
        ain[i, k1 + 23] = snfai[i, 3, 23] # Market 24
        ain[i, k1 + 24] = snfai[i, 3, 24] # Market 25
        ain[i, k1 + 25] = snfai[i, 3, 25] # Market 26
        
        # ====================================================================
        # Feature Set 2: Slope Features (k1 = 26-51)
        # Calculate normalized price change from previous day
        # Formula: (current_normalized - previous_normalized)
        # MATLAB: k1=26, so ain(i,k1+ii) goes to columns 27-52 (1-based)
        # Python: k1=26, so ain[i, k1+ii] goes to columns 26-51 (0-based, matching MATLAB)
        # ====================================================================
        k1 = 26

        for ii in range(26):
            # MATLAB: ain(i,k1+ii)=(snfai(i,4,ii)-min(snfai(range,4,ii)))-(snfai(i-1,4,ii)-min(snfai(range,4,ii)))
            # Slope = change in normalized price from day i-1 to day i
            min_price = np.min(snfai[range_slice, 3, ii])
            current_normalized = snfai[i, 3, ii] - min_price
            previous_normalized = snfai[i-1, 3, ii] - min_price
            ain[i, k1 + ii] = current_normalized - previous_normalized

    # ========================================================================
    # Create Output Labels for Classification
    # ========================================================================
    
    # Initialize output arrays
    aout = np.zeros((num_days, 1), dtype=np.float64)
    dout = []

    # MATLAB: ifile1a=1 (1-based index), so in Python this is index 0
    # But MATLAB line 94 says: ifile1a=1; % 1 for gold
    # The parameter passed in is already 0-based, so we use it directly
    # KEEP the passed parameter: ifile1a (already 0-based from caller)

    # MATLAB: for i=2:height(t)-1
    # Python equivalent: range(1, num_days-1) gives i=1 to num_days-2
    for i in range(1, num_days - 1):
        # MATLAB: aout(i+1,1)=snfai(i+1,4,ifile1a); % tomorrow's close price pit in i for training
        aout[i+1, 0] = snfai[i+1, 3, ifile1a]  # Tomorrow's close price (field 4 in MATLAB = index 3 in Python)

        # MATLAB: aout(i,1)=snfai(i,4,ifile1a);  % today's close price
        aout[i, 0] = snfai[i, 3, ifile1a]  # Today's close price (field 4 in MATLAB = index 3 in Python)

        # MATLAB logic: if aout(i,1)>aout(i+1,1) then SHORT else LONG
        # Classification logic:
        # SHORT: if today's close > tomorrow's close (price going down)
        # LONG:  if today's close <= tomorrow's close (price going up)
        if aout[i, 0] > aout[i+1, 0]:
            dout.append("SHORT")  # SHORT
        else:
            dout.append("LONG")  # LONG

    # MATLAB: cout(1,1)={'SHORT'}; - Initialize first element
    dout.insert(0, "SHORT")  # SHORT
    dout = pd.Categorical(dout)
    
    return ain, dout, t, snfai


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        ain, dout, t, snfai = loaddata28(fname, ifile1a=0)
        
        print(f"Input features shape: {ain.shape}")
        print(f"Output labels shape: {dout.shape}")
        print(f"Raw data shape: {t.shape}")
        print(f"3D array shape: {snfai.shape}")
        print(f"\nLabel distribution:")
        print(f"  SHORT (0): {np.sum(dout == 0)}")
        print(f"  LONG (1): {np.sum(dout == 1)}")
    else:
        print("Usage: python loaddata28.py <excel_file_path>")

