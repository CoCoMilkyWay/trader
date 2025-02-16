from torch import Tensor
from enum import IntEnum
from typing import List, Dict, Tuple, Union


class DimensionType(IntEnum):
    """
    Note that we do not consider num of axis of data here, merely the property of data value itself
    """
    price = 0
    volume = 1
    ratio = 2
    timedelta = 3
    oscillator = 4
    condition = 5
    misc = 6


class Dimension():
    def __init__(self, args: List[DimensionType]|List[str]):
        self._dimension_list:List[DimensionType] = []
        for arg in args:
            if isinstance(arg, str):
                self._dimension_list.append(Dimension_Map(arg)._dimension_list[0])
            elif isinstance(arg, DimensionType):
                self._dimension_list.append(arg)

    def __contains__(self, dimension: DimensionType|str|List[DimensionType]|List[str]) -> bool:
        """Check if a DimensionType or a list of strings is in the Dimension instance."""
        if isinstance(dimension, DimensionType):
            return dimension in self._dimension_list
        elif isinstance(dimension, str):
            return Dimension_Map(dimension)._dimension_list[0] in self._dimension_list
        elif isinstance(dimension, list):
            contain = True
            for item in dimension:
                if isinstance(item, DimensionType):
                    contain = contain and item in self._dimension_list
                elif isinstance(item, str):
                    contain = contain and Dimension_Map(item)._dimension_list[0] in self._dimension_list
            return contain

    def add(self, dimension: Union[DimensionType, 'Dimension', List[str]]):
        # Add dimension_type, all elements from another Dimension instance, or a list of strings
        if isinstance(dimension, DimensionType):
            if dimension not in self._dimension_list:
                self._dimension_list.append(dimension)
        elif isinstance(dimension, Dimension):
            for item in dimension._dimension_list:
                if item not in self._dimension_list:
                    self._dimension_list.append(item)
        elif isinstance(dimension, list):  # Handle list of strings
            for c in dimension:
                item = Dimension_Map(c)._dimension_list[0]
                if item not in self._dimension_list:
                    self._dimension_list.append(item)
        else:
            raise ValueError("Input must be either DimensionType, Dimension instance, or a list of strings.")
        
        return Dimension(self._dimension_list)  # Return a new Dimension instance with the updated list

    def remove(self, dimension: Union[DimensionType, 'Dimension', List[str]]):
        # Remove dimension_type, all elements from another Dimension instance, or a list of strings
        if isinstance(dimension, DimensionType):
            if dimension in self._dimension_list:
                self._dimension_list.remove(dimension)
        elif isinstance(dimension, Dimension):
            for c in dimension._dimension_list:
                if c in self._dimension_list:
                    self._dimension_list.remove(c)
        elif isinstance(dimension, list):  # Handle list of strings
            for c in dimension:
                dimension_type = Dimension_Map(c)._dimension_list[0]
                if dimension_type in self._dimension_list:
                    self._dimension_list.remove(dimension_type)
        else:
            raise ValueError("Input must be either DimensionType, Dimension instance, or a list of strings.")
        
        return Dimension(self._dimension_list)  # Return a new Dimension instance with the updated list

    def are_in(self, other_dimension: 'Dimension') -> bool:
        """
        Check if all types in the current Dimension instance are also in the other Dimension instance.
        Returns True if all types in self are present in other_dimension.
        """
        if not isinstance(other_dimension, Dimension):
            raise ValueError("Input must be a Dimension instance.")
        
        return all(type_ in other_dimension for type_ in self._dimension_list)

    def are_not_in(self, other_dimension: 'Dimension') -> bool:
        if not isinstance(other_dimension, Dimension):
            raise ValueError("Input must be a Dimension instance.")
        
        return all(type_ not in other_dimension for type_ in self._dimension_list)
    
    def identical(self, other_dimension: 'Dimension') -> bool:
        def are_lists_identical(list1, list2) -> bool:
            if len(list1) != len(list2):return False
            list2_copy = list2[:]
            for item in list1:
                if item in list2_copy:list2_copy.remove(item)
                else:return False
            return len(list2_copy) == 0
        if not isinstance(other_dimension, Dimension):
            raise ValueError("Input must be a Dimension instance.")
        return are_lists_identical(self._dimension_list, other_dimension._dimension_list)



def Dimension_Map(dim: Union[str, List[str]]) -> Dimension:
    """
    Maps a string or a list of strings to corresponding DimensionType values.
    Accepts a single string or a list of strings.

    :param dim: A string or list of strings representing the dimension types.
    :return: A Dimension instance with the associated DimensionType(s).
    """
    if isinstance(dim, str):
        dim = [dim]  # Convert to list if a single string is provided

    dimension_types = []
    for d in dim:
        if d == 'price':
            dimension_types.append(DimensionType.price)
        elif d == 'volume':
            dimension_types.append(DimensionType.volume)
        elif d == 'oscillator':
            dimension_types.append(DimensionType.oscillator)
        elif d == 'ratio':
            dimension_types.append(DimensionType.ratio)
        elif d == 'condition':
            dimension_types.append(DimensionType.condition)
        elif d == 'misc':
            dimension_types.append(DimensionType.misc)
        elif d == 'timedelta':
            dimension_types.append(DimensionType.timedelta)
        else:
            raise RuntimeError(f"No matching operand dimension found: {d}")

    return Dimension(dimension_types)  # Return a Dimension instance with the collected dimension types
