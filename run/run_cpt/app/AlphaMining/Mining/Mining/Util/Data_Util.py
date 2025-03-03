import torch
from torch import Tensor
from typing import List, Dict, Union, Tuple
from datetime import datetime, timedelta


def list_timestamps(start: int, end: int, use_datetime: bool = True):
    def parse_time(time: int) -> datetime:
        time_str = str(time)
        # Expecting a 12-digit integer: YYYYMMDDHHMM
        year = int(time_str[-12:-8])
        month = int(time_str[-8:-6])
        day = int(time_str[-6:-4])
        hour = int(time_str[-4:-2])
        minute = int(time_str[-2:])
        return datetime(year, month, day, hour, minute)

    def format_time(dt: datetime) -> int:
        # Format the datetime back into the 12-digit integer form
        # [202502130000, 202502130001, ...]
        return int(dt.strftime('%Y%m%d%H%M'))

    # Parse the input integers into datetime objects
    start_time = parse_time(start)
    end_time = parse_time(end)

    # Generate timestamps for every minute from start_time to end_time (inclusive)
    current_time = start_time
    timestamps: list = []
    while current_time < end_time:
        if use_datetime:
            timestamps.append(current_time)
        else:
            timestamps.append(format_time(current_time))
        current_time += timedelta(minutes=1)

    return timestamps


def count_nan_and_inf(tensor: torch.Tensor, check: bool = False) -> Tuple[int, int]:
    # Check for NaN values
    num_nan = int(torch.sum(torch.isnan(tensor)).item())

    # Check for Inf values (both positive and negative infinity)
    num_pos_inf = torch.sum(tensor == float('inf')).item()
    num_neg_inf = torch.sum(tensor == float('-inf')).item()

    num_inf = int(num_pos_inf + num_neg_inf)
    if check:
        assert num_inf == 0 and num_inf == 0, f"nan:{num_nan}, inf:{num_inf}"
    return num_nan, num_inf


def tensor_probe(tensor: Tensor):
    # Calculate statistics
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    mean_value = torch.mean(tensor)
    std_value = torch.std(tensor)
    sum_value = torch.sum(tensor)
    num_elements = tensor.numel()

    # Print statistics
    print(f'Tensor:\n{tensor}')
    print(f'Min: {min_value}')
    print(f'Max: {max_value}')
    print(f'Mean: {mean_value}')
    print(f'Standard Deviation: {std_value}')
    print(f'Sum: {sum_value}')
    print(f'Number of elements: {num_elements}')
