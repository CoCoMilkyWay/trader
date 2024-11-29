import inspect
import types

'''
1. Memoization:
   - Caches the result of a method for an instance to avoid redundant computations.
   - Useful for expensive or time-consuming computations based on instance-specific data.

2. Descriptor Protocol:
   - Implements `__get__` to ensure the decorator behaves like an instance method.
   - Maintains the connection to the instance (`self`).

3. Instance-Specific Behavior:
   - Each instance has its own cache (`_memoize_cache`).
   - Allows different instances to maintain separate memoized results.

4. Error Handling:
   - Validates that the decorated method has the expected signature `(self)`.
   - Ensures the decorator is applied only to bound methods.

5. Why Use This Approach:
   - Fine-grained control through the descriptor protocol.
   - Efficient caching tailored to each instance.
   - Prevents misuse by enforcing method signature validation.
'''

class make_cache:
    def __init__(self, func):
        self.func = func

        fargspec = inspect.getfullargspec(func)
        if len(fargspec.args) != 1 or fargspec.args[0] != "self":
            raise Exception("@memoize must be `(self)`")

        # set key for this function
        self.func_key = str(func)

    def __get__(self, instance, cls):
        if instance is None:
            raise Exception("@memoize's must be bound")

        if not hasattr(instance, "_memoize_cache"):
            setattr(instance, "_memoize_cache", {})

        return types.MethodType(self, instance)

    def __call__(self, *args, **kwargs):
        instance = args[0]
        cache = instance._memoize_cache

        if self.func_key in cache:
            return cache[self.func_key]

        result = self.func(*args, **kwargs)
        cache[self.func_key] = result
        return result
