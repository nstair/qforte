"""
printing.py
====================================================
A module for pretty printing functions for matricies
(usually numpy arrays)and other commonly used data
structures in qforte.
"""

import numpy as np
import re


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


# --- Helper: convert C printf-style integer formats (e.g., %zu, %12ld) to Python-compatible %d ---
_INT_FMT_RE = re.compile(
    r'%(?P<flags>[#0\- +]*)'   # flags
    r'(?P<width>\d*)'          # width
    r'(?P<lenmod>hh|h|ll|l|z)?' # length modifier (to drop)
    r'(?P<type>[diu])'         # integer types
)

def _sanitize_int_printf(fmt: str) -> str:
    """
    Convert C printf int formats like %zu, %12ld, %05u to Python %-format: %d
    Keeps flags/width, strips length modifier, normalizes type to 'd'.
    """
    def _sub(m):
        flags = m.group('flags') or ''
        width = m.group('width') or ''
        return f'%{flags}{width}d'
    return _INT_FMT_RE.sub(_sub, fmt)


def tensor_str(
    name: str,
    arr: np.ndarray,
    print_data: bool = True,
    print_complex: bool = False,
    maxcols: int = 8,
    data_format: str = "% .6e",
    header_format: str = "%12zu",
) -> str:
    """
    Python clone of your C++ Tensor::str(...) for a NumPy array.

    Matches:
      - Header lines (Tensor name, Ndim, Size, Shape)
      - Data pagination over leading axes (ndim-2), with "Page (...*,*):"
      - Row/column headers and blocks of up to maxcols columns
      - Complex formatting as "%f%+fi" when print_complex=True, otherwise data_format on real part
    """
    a = np.asarray(arr)
    order = int(a.ndim)
    shape = tuple(int(d) for d in a.shape)
    nelem = int(a.size)

    # Sanitize any C-style integer formats to Python's %-format
    header_format_py = _sanitize_int_printf(header_format)

    # Build header
    out = []
    out.append("Tensor: %s\n" % name)
    out.append("  Ndim  = %d\n" % order)
    out.append("  Size  = %d\n" % nelem)
    out.append("  Shape = (")
    if order > 0:
        for dim in range(order):
            out.append("%d" % shape[dim])
            if dim < order - 1:
                out.append(",")
    out.append(")\n")

    if not print_data:
        return "".join(out)

    # Element formatter
    if print_complex:
        data_fmt = "%f%+fi"
        def fmt_val(v):
            v = complex(v)
            return data_fmt % (v.real, v.imag)
    else:
        data_fmt = data_format
        def fmt_val(v):
            if np.iscomplexobj(v):
                v = v.real
            return data_fmt % v

    # Fixed fragments (Python-compatible)
    order0_prefix   = "  "
    order1_indexfmt = "  %5d "
    head_pad        = "  %5s" % ""
    header_fmt_full = " " + header_format_py
    data_prefix     = " "

    out.append("\n")
    out.append("  Data:\n\n")

    if nelem == 0:
        return "".join(out)

    # Determine rows, cols, page_size like C++
    page_size = 1
    rows = 1
    cols = 1
    if order >= 1:
        page_size *= shape[-1]
        rows = shape[-1]
    if order >= 2:
        page_size *= shape[-2]
        rows = shape[-2]
        cols = shape[-1]

    # Number of pages across leading axes (order-2)
    pages = (nelem // page_size) if page_size > 0 else 0
    pages = max(pages, 1)  # ensure at least one iteration for order <= 2

    for page in range(pages):
        # Page header for order > 2
        if order > 2:
            out.append("  Page (")
            num = page
            den = pages
            for k in range(order - 2):
                den //= shape[k]
                val = num // max(den, 1)
                num -= val * max(den, 1)
                out.append("%d," % int(val))
            out.append("*,*):\n\n")

        if order == 0:
            # Scalar
            v = a.item()
            out.append(order0_prefix + fmt_val(v) + "\n")
            continue

        if order == 1:
            # Vector
            for i in range(rows):
                out.append(order1_indexfmt % int(i))
                out.append(fmt_val(a[i]) + "\n")
            continue

        # order >= 2
        # Determine leading index tuple for this page
        if order > 2:
            num = page
            den = pages
            leading_idx = []
            for k in range(order - 2):
                den //= shape[k]
                val = num // max(den, 1)
                num -= val * max(den, 1)
                leading_idx.append(int(val))
            page_view = a[tuple(leading_idx)]
        else:
            page_view = a

        page_mat = np.reshape(page_view, (int(rows), int(cols)))

        j = 0
        while j < cols:
            ncols = int(min(maxcols, cols - j))

            # Column header
            out.append(head_pad)
            for jj in range(j, j + ncols):
                out.append(header_fmt_full % int(jj))
            out.append("\n")

            # Rows
            for i in range(int(rows)):
                out.append("  %5d" % int(i))
                for jj in range(j, j + ncols):
                    out.append(data_prefix + fmt_val(page_mat[i, jj]))
                out.append("\n")

            # Block separator (match C++: blank line if more pages or more column blocks)
            if page < pages - 1 or (j + maxcols) < (cols - 1):
                out.append("\n")

            j += ncols

    return "".join(out)