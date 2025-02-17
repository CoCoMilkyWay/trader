import torch
from torch import Tensor
from typing import Tuple, Optional

# Mask NaN values in tensors and calculate valid data points per row


def _mask_either_nan(x: Tensor, y: Tensor, fill_with: float = torch.nan) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Masks locations where either x or y contain NaNs, replacing NaNs with a specified value.
    Args:
        x: [timestamps, codes], First input tensor.
        y: [timestamps, codes], Second input tensor.
        fill_with: Value to use for replacing NaN entries (default: torch.nan).
    Returns:
        Tuple containing:
          - x (masked tensor)
          - y (masked tensor)
          - n (valid data count per row ignoring NaNs)
          - nan_mask (boolean mask indicating where NaNs exist)
    """
    x = x.clone()  # Clone to avoid modifying the original tensor
    y = y.clone()  # Clone to avoid modifying the original tensor
    nan_mask = x.isnan() | y.isnan()  # Identify where either tensor has NaN values
    x[nan_mask] = fill_with  # Replace NaNs with the specified value in x
    y[nan_mask] = fill_with  # Replace NaNs with the specified value in y
    # Compute the count of valid (non-NaN) values per row
    n = (~nan_mask).sum(dim=1)
    return x, y, n, nan_mask

# Rank the elements in a 1D tensor


def _rank_data_1d(x: Tensor) -> Tensor:
    """
    Ranks data in a 1D tensor, using the average rank for ties.
    Args:
        x: 1D input tensor for ranking.
    Returns:
        Tensor containing ranks for each element of the input tensor.
    """
    _, inv, counts = x.unique(return_inverse=True, return_counts=True)
    cs = counts.cumsum(dim=0)  # Cumulative sum of counts
    # Add 0 at the beginning
    cs = torch.cat((torch.zeros(1, dtype=x.dtype, device=x.device), cs))
    rmin = cs[:-1]  # Minimum rank for each unique value
    rmax = cs[1:] - 1  # Maximum rank for each unique value
    ranks = (rmin + rmax) / 2  # Compute average rank for ties
    return ranks[inv]  # Map ranks back to the original order

# Rank each row of a 2D tensor, accounting for NaNs


def _rank_data(x: Tensor, nan_mask: Tensor) -> Tensor:
    """
    Ranks data row-wise in a 2D tensor, treating NaNs as 0.
    Args:
        x: [timestamps, codes] Input tensor for ranking.
        nan_mask: [timestamps, codes] Boolean mask of NaN values in x.
    Returns:
        [timestamps, codes] Tensor of ranks, with NaNs set to 0.
    """
    rank = torch.stack([_rank_data_1d(row)
                       for row in x])  # Rank rows individually
    rank[nan_mask] = 0  # Replace NaNs with a rank of 0
    return rank

# Calculate Pearson correlation row-wise with masking


def _batch_pearsonr_given_mask(x: Tensor, y: Tensor, n: Tensor, mask: Tensor) -> Tensor:
    """
    Computes row-wise Pearson correlation between x and y with a mask indicating valid data points.
    Args:
        x: [timestamps, codes], First input tensor.
        y: [timestamps, codes], Second input tensor.
        n: [timestamps], Count of valid data points (non-NaN) per row.
        mask: [timestamps, codes], Boolean mask where True indicates invalid data (e.g., NaNs).
    Returns:
        [timestamps] Tensor of Pearson correlation coefficients for each row.
    """
    # Calculate mean and standard deviation for x and y row-wise
    x_mean, x_std = masked_mean_std(x, n, mask)
    y_mean, y_std = masked_mean_std(y, n, mask)
    # Compute covariance between x and y
    cov = (x * y).sum(dim=1) / n - x_mean * y_mean
    # Handle degenerate cases: rows with insufficient data or constant values
    invalid_mask = (n <= 1) | (x_std < 1e-3) | (y_std < 1e-3)
    # Compute correlation, set invalid rows to NaN or 0 as per needs
    corrs = cov / (x_std * y_std)
    # Set correlation explicitly as 0 for degenerate rows
    corrs[invalid_mask] = 0
    return corrs

# Row-wise Spearman correlation


def batch_spearmanr(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes row-wise Spearman correlation between x and y.
    Args:
        x: [timestamps, codes], First input tensor.
        y: [timestamps, codes], Second input tensor.
    Returns:
        [timestamps] Tensor of Spearman correlation coefficients for each row.
    """
    # Mask NaN values and replace them with torch.nan
    x, y, n, nan_mask = _mask_either_nan(x, y)
    # Compute ranks for both x and y, treating NaNs as 0
    rx = _rank_data(x, nan_mask)
    ry = _rank_data(y, nan_mask)
    # Compute Pearson correlation of the ranks
    return _batch_pearsonr_given_mask(rx, ry, n, nan_mask)

# Row-wise Pearson correlation


def batch_pearsonr(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes row-wise Pearson correlation between x and y.
    Args:
        x: [timestamps, codes], First input tensor.
        y: [timestamps, codes], Second input tensor.
    Returns:
        [timestamps] Tensor of Pearson correlation coefficients for each row.
    """
    # Mask NaN values and replace them with 0
    return _batch_pearsonr_given_mask(*_mask_either_nan(x, y, fill_with=0.))

# Compute masked mean and standard deviation for a 2D tensor


def masked_mean_std(
    x: Tensor,
    n: Optional[Tensor] = None,
    mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Computes mean and standard deviation along rows, ignoring masked (e.g., NaN) values.
    Args:
        x: [timestamps, codes], Input data tensor.
        n: [timestamps], Optional, count of valid data points per row.
        mask: [timestamps, codes], Optional, Boolean mask where True indicates invalid data.
    Returns:
        Tuple containing:
          - mean: [timestamps] Row-wise mean ignoring masked values.
          - std: [timestamps] Row-wise standard deviation ignoring masked values.
    """
    if mask is None:  # If no mask is provided, use NaN mask
        mask = torch.isnan(x)
    if n is None:  # If valid data counts aren't provided, compute them
        n = (~mask).sum(dim=1)
    x = x.clone()
    x[mask] = 0.  # Replace masked values with 0
    mean = x.sum(dim=1) / n  # Calculate row-wise mean
    # Calculate row-wise std
    std = ((((x - mean[:, None]) * ~mask) ** 2).sum(dim=1) / n).sqrt()
    return mean, std

# Normalize a 2D tensor row-wise by mean and standard deviation


def normalize_by_timestamp(value: Tensor) -> Tensor:
    """
    Normalizes data row-wise by subtracting mean and dividing by standard deviation.
    Args:
        value: [timestamps, codes], Input data tensor.
    Returns:
        [timestamps, codes] Normalized tensor.
    """
    mean, std = masked_mean_std(value)  # Compute row-wise mean and std
    value = (value - mean[:, None]) / std[:, None]  # Normalize data
    nan_mask = torch.isnan(value)  # Identify NaN locations
    value[nan_mask] = 0.  # Replace NaNs with 0
    return value
