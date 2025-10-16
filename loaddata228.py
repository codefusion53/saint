# =========================================================================
# LOADDATA228
# =========================================================================
# Prepare training/validation/test data slices for LSTM neural network
#
# Inputs:
#   snfai - 3D array (rows x fields x sheets) created by loaddata28
#           Contains market data: rows=days, fields=19, sheets=26 markets
#   k     - Current day index (1-based) for creating time slices
#   t     - Table of rows (used for height validation)
#   ifile - Target sheet index used for output labeling (target asset)
#
# Outputs:
#   anin    - Training input sequences list (27 sequences)
#             Each sequence: 20 time steps x 104 features
#   anout1  - Training output labels list (27 sequences)
#             Each sequence: 20 labels (LONG/SHORT encoded as 0/1)
#   avnin   - Validation input sequences list (2 sequences)
#             Same structure as anin
#   avnout1 - Validation output labels list (2 sequences)
#             Same structure as anout1
#   pdata   - Prediction/test input sequences list (27 sequences)
#             Same structure as anin (used for next-day prediction)
#
# Feature Engineering:
#   Total: 104 features per time step
#   - Features 1-26:   Current close prices from 26 markets
#   - Features 27-52:  1-day slope (current - previous day)
#   - Features 53-78:  2-day slope (current - avg of last 2 days)
#   - Features 79-104: 3-day slope (current - avg of last 3 days)
#
# Data Organization:
#   - Training:   27 sequences of 20 time steps each
#   - Validation: 2 sequences of 20 time steps each
#   - Prediction: 27 sequences of 20 time steps each (offset by +1 day)
#
# =========================================================================

import numpy as np
import pandas as pd

def loaddata228(snfai, k, t, ifile):
    nMin = len(t)
    inputSize = 104
    ain = np.zeros((nMin, 52), dtype=np.float64)

    # DEBUG: Check bounds for this day
    # if k > nMin - 50:  # Near the end of data
    #     print(f"DEBUG loaddata228: k={k}, nMin={nMin}, processing near end of data", flush=True)

    # --- Build input matrix ain ---
    for i in range(3, nMin):  # MATLAB: for i=4:height(t)
        # Base features
        k1 = 0
        
        ain[i, k1] = snfai[i, 3, 0]
        ain[i, k1 + 2] = snfai[i, 3, 1]
        ain[i, k1 + 1] = snfai[i, 3, 2]
        ain[i, k1 + 3] = snfai[i, 3, 3]
        ain[i, k1 + 4] = snfai[i, 3, 4]
        ain[i, k1 + 5] = snfai[i, 3, 5]
        ain[i, k1 + 6] = snfai[i, 3, 6]
        ain[i, k1 + 7] = snfai[i, 3, 7]
        ain[i, k1 + 8] = snfai[i, 3, 8]
        ain[i, k1 + 9] = snfai[i, 3, 9]
        ain[i, k1 + 10] = snfai[i, 3, 10]
        ain[i, k1 + 11] = snfai[i, 3, 11]
        ain[i, k1 + 12] = snfai[i, 3, 12]
        ain[i, k1 + 13] = snfai[i, 3, 13]
        ain[i, k1 + 14] = snfai[i, 3, 14]
        ain[i, k1 + 15] = snfai[i, 3, 15]
        ain[i, k1 + 16] = snfai[i, 3, 16]
        ain[i, k1 + 17] = snfai[i, 3, 17]
        ain[i, k1 + 18] = snfai[i, 3, 18]
        ain[i, k1 + 19] = snfai[i, 3, 19]
        ain[i, k1 + 20] = snfai[i, 3, 20]
        ain[i, k1 + 21] = snfai[i, 3, 21]
        ain[i, k1 + 22] = snfai[i, 3, 22]
        ain[i, k1 + 23] = snfai[i, 3, 23]
        ain[i, k1 + 24] = snfai[i, 3, 24]
        ain[i, k1 + 25] = snfai[i, 3, 25]

        # --- Slope and difference features ---
        k1, k2, k3 = 26, 52, 78
        for ii in range(26):
            temp1 = snfai[i, 3, ii] - snfai[i - 1, 3, ii]

            shorts = 3.0
            ain[i, k1 + ii] = temp1 * shorts if temp1 < 0 else temp1

            # i-2
            temp1 = ((snfai[i, 3, ii] + snfai[i - 1, 3, ii]) / 2) - snfai[i - 2, 3, ii]
            temp1 = snfai[i, 3, ii] + (snfai[i - 2, 3, ii] + snfai[i - 1, 3, ii]) / 2

            ain[i, k2 + ii] = temp1 * shorts if temp1 < 0 else temp1

            # i-3
            temp1 = snfai[i, 3, ii] - (snfai[i - 3, 3, ii] + snfai[i - 2, 3, ii] + snfai[i - 1, 3, ii]) / 3
            ain[i, k3 + ii] = temp1 * shorts if temp1 < 0 else temp1

    # --- Build output vectors ---
    aout = np.zeros((nMin, 1), dtype=np.float64)
    # MATLAB creates output with same size as input
    # Initialize with nMin elements
    cout = ["SHORT"] * nMin

    # MATLAB: for i=2:height(t)-1
    # Python: range(1, nMin - 1) gives i=1 to nMin-2 (matching MATLAB's i=2 to height(t)-1)
    for i in range(1, nMin - 1):
        # MATLAB: aout(i+1,1)=snfai(i+1,4,ifile); % tomorrow's close price
        aout[i + 1, 0] = snfai[i + 1, 3, ifile]   # tomorrow close (field 4 = index 3)

        # MATLAB: aout(i,1)=snfai(i+1,3,ifile); % change here on 1-2-25 for open price
        aout[i, 0] = snfai[i + 1, 2, ifile]       # today's open (field 3 = index 2)

        # MATLAB logic: if aout(i,1)>aout(i+1,1) then SHORT else LONG
        cout[i] = "SHORT" if aout[i, 0] > aout[i + 1, 0] else "LONG"

    cout[0] = "SHORT"
    dout = pd.Categorical(cout).codes.astype(np.int64)

    # MATLAB: ain(k-580:k+1,j) where k is 1-based
    # Python: ain[k-580-1:k+1] but must not exceed nMin
    # Cap the upper bound to nMin to prevent index out of bounds
    k_upper = min(k + 1, nMin)  # k+2 for inclusive end, but cap at nMin

    muX = np.mean(ain[k - 580 - 1:k_upper, 0:inputSize], axis=0, dtype=np.float64)
    sigmaX = np.std(ain[k - 580 - 1:k_upper, 0:inputSize], axis=0, ddof=1, dtype=np.float64)

    for i in range(k - 580 - 1, k_upper):
        for j in range(inputSize):
            ain[i, j] = (ain[i, j] - muX[j]) / sigmaX[j]

    ain1 = ain.copy()

    # --- Training input data (anin) ---
    anin = []
    ic = 520
    for i in range(27):
        # MATLAB: tempi=ain1(k-ic-19:k-ic,:).'
        start_idx = k - ic - 19 - 1
        end_idx = k - ic
        # Ensure we don't go beyond array bounds
        if start_idx < 0 or end_idx > nMin:
            # If out of bounds, create a zero-filled sequence
            temp = np.zeros((20, inputSize), dtype=np.float64)
            print(f"WARNING: Training input {i} out of bounds (k={k}, ic={ic}, start={start_idx}, end={end_idx}, nMin={nMin})", flush=True)
        else:
            temp = ain1[start_idx:end_idx, :]     # slice 20 rows, shape: (20, 104)

        # Validate shape
        if temp.shape[0] != 20:
            print(f"ERROR: Training input {i} has wrong shape {temp.shape}, expected (20, 104). k={k}, ic={ic}, indices=[{start_idx}:{end_idx}]", flush=True)

        anin.append(temp)
        ic -= 20

    # --- Training output data (anout1) ---
    anout1 = []
    ic = 520
    for i in range(27):
        # MATLAB: tempnin=dout(k-ic-19:k-ic).'; anout1(i,1)={tempnin};
        start_idx = k - ic - 19 - 1
        end_idx = k - ic
        # Ensure we don't go beyond array bounds
        if start_idx < 0 or end_idx > len(dout):
            # If out of bounds, create a zero-filled sequence
            temp = np.zeros(20, dtype=np.int64)
            print(f"WARNING: Training output {i} out of bounds (k={k}, ic={ic}, start={start_idx}, end={end_idx}, len(dout)={len(dout)})", flush=True)
        else:
            temp = dout[start_idx:end_idx]

        # Validate shape
        if len(temp) != 20:
            print(f"ERROR: Training output {i} has wrong length {len(temp)}, expected 20. k={k}, ic={ic}, indices=[{start_idx}:{end_idx}], len(dout)={len(dout)}", flush=True)

        anout1.append(temp)
        ic -= 20

    # --- Validation input (avnin) ---
    avnin = []
    ic = 560
    for i in range(2):
        # MATLAB: tempvin=ain1(k-19-ic:k-ic,:).'
        start_idx = k - 19 - ic - 1
        end_idx = k - ic
        if start_idx < 0 or end_idx > nMin:
            temp = np.zeros((20, inputSize), dtype=np.float64)
            print(f"WARNING: Validation input {i} out of bounds (k={k}, ic={ic}, start={start_idx}, end={end_idx}, nMin={nMin})", flush=True)
        else:
            temp = ain1[start_idx:end_idx, :]  # shape: (20, 104)

        if temp.shape[0] != 20:
            print(f"ERROR: Validation input {i} has wrong shape {temp.shape}, expected (20, 104). k={k}, ic={ic}", flush=True)

        avnin.append(temp)
        ic -= 20

    # --- Validation output (avnout1) ---
    avnout1 = []
    ic = 560
    for i in range(2):
        # MATLAB: tempv=dout(k-19-ic:k-ic).'
        start_idx = k - 19 - ic - 1
        end_idx = k - ic
        if start_idx < 0 or end_idx > len(dout):
            temp = np.zeros(20, dtype=np.int64)
            print(f"WARNING: Validation output {i} out of bounds (k={k}, ic={ic}, start={start_idx}, end={end_idx}, len(dout)={len(dout)})", flush=True)
        else:
            temp = dout[start_idx:end_idx]

        if len(temp) != 20:
            print(f"ERROR: Validation output {i} has wrong length {len(temp)}, expected 20. k={k}, ic={ic}", flush=True)

        avnout1.append(temp)
        ic -= 20

    # --- Test input (pdata) ---
    pdata = []
    ic = 520
    for i in range(27):
        # MATLAB: tempi=ain1(k-ic-19+1:k-ic+1,:).'
        start_idx = k - ic - 19 + 1 - 1
        end_idx = k - ic + 1
        if start_idx < 0 or end_idx > nMin:
            tempi = np.zeros((20, inputSize), dtype=np.float64)
            print(f"WARNING: Test input {i} out of bounds (k={k}, ic={ic}, start={start_idx}, end={end_idx}, nMin={nMin})", flush=True)
        else:
            tempi = ain1[start_idx:end_idx, :]  # shape: (20, 104)

        if tempi.shape[0] != 20:
            print(f"ERROR: Test input {i} has wrong shape {tempi.shape}, expected (20, 104). k={k}, ic={ic}", flush=True)

        pdata.append(tempi)
        ic -= 20
    
    return anin, anout1, avnin, avnout1, pdata


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        n_samples = sys.argv[1]
        k = sys.argv[2]
        t = sys.argv[3]
        ifile = sys.argv[4]
        snfai = np.random.rand(n_samples, 29, 26)
        t = pd.DataFrame(np.arange(n_samples))
        anin, anout1, avnin, avnout1, pdata = loaddata228(snfai, k-2, t, ifile)
        
        print(f"\nLabel distribution:")
        print(pd.Series(np.concatenate(anout1 + avnout1)).value_counts())
        print(f"Input features shape: {anin[0].shape}")
        print(f"Output labels shape: {anout1[0].shape}")
        print(f"Validation features shape: {avnin[0].shape}")
        print(f"Validation labels shape: {avnout1[0].shape}")
        print(f"Prediction features shape: {pdata[0].shape}")
    
    else:
        print("Usage: python loaddata228.py <n_samples> <k> <t> <ifile>")

