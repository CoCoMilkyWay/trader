import hashlib
import pickle
from collections import OrderedDict
from functools import wraps
import torch

class TensorCache:
    def __init__(self, max_depth=100):
        self.max_depth = max_depth
        self.cache = OrderedDict()

    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        tensor_bytes = pickle.dumps(tensor)
        return hashlib.sha256(tensor_bytes).hexdigest()

    def cache_decorator(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a unique cache key based on the function and its arguments
            key = (func.__name__,) + tuple(self._hash_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)

            # Check if result is already cached
            if key in self.cache:
                return self.cache[key]
            
            # Call the actual function
            result = func(*args, **kwargs)
            
            # Store the result in cache
            self.cache[key] = result
            
            # Maintain limit of cache depth
            if len(self.cache) > self.max_depth:
                self.cache.popitem(last=False)  # Remove the oldest item

            return result
        return wrapper

# # Example usage
# cache = Cache(max_depth=5)
# 
# @cache.cache_decorator
# def _calc_IC(value1: torch.Tensor, value2: torch.Tensor) -> float:
#     # Replace this with your actual batch_pearsonr calculation
#     return batch_pearsonr(value1, value2).mean().item()
# 
# @cache.cache_decorator
# def another_function(data: torch.Tensor) -> float:
#     # Some other computation
#     return data.sum().item()
# 
# def batch_pearsonr(value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
#     # Dummy implementation for the sake of example
#     return torch.tensor(0.5)  # Replace with your actual implementation
# 
# # Example use case
# value1 = torch.tensor([1.0, 2.0, 3.0])
# value2 = torch.tensor([4.0, 5.0, 6.0])
# 
# # Calculate and cache
# result1 = _calc_IC(value1, value2)
# print("Result:", result1)
# 
# # Calculate again with the same inputs (should hit the cache)
# result2 = _calc_IC(value1, value2)
# print("Cached Result:", result2)
# 
# # Another function usage
# data = torch.tensor([1, 2, 3])
# result3 = another_function(data)
# print("Another Function Result:", result3)
# 
# # Cached result
# result4 = another_function(data)
# print("Cached Another Function Result:", result4)