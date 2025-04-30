"""
printing.py
====================================================
A module for pretty printing functions for matricies
(usually numpy arrays)and other commonly used data
structures in qforte.
"""

import numpy as np

def matprint(arr, fmt="g"):
    """
    Print a 1D or 2D numpy array in an intelligible, column-aligned fashion.

    Parameters
    ----------
    arr : array-like
        A real or complex 1D (vector) or 2D (matrix) array.
    fmt : str, optional
        A numpy format specifier (default "g", e.g. ".2f", "e").
    """
    a = np.array(arr)
    if a.ndim == 1:
        # treat as single-row matrix
        a = a[np.newaxis, :]
    elif a.ndim != 2:
        raise ValueError(f"matprint: expected 1D or 2D array, got {a.ndim}D")

    # find max width per column
    col_maxes = []
    for col in a.T:
        # format each entry without padding, measure its length
        lengths = [len(("{:"+fmt+"}").format(x)) for x in col]
        col_maxes.append(max(lengths))

    # print each row with padding
    for row in a:
        for i, x in enumerate(row):
            width = col_maxes[i]
            print(("{:"+str(width)+fmt+"}").format(x), end="  ")
        print()
