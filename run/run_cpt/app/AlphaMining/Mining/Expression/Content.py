from torch import Tensor
from enum import IntEnum
from typing import List, Dict, Tuple, Union


class ContentType(IntEnum):
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


class Content():
    def __init__(self, args: List[ContentType]|List[str]):
        self._content_list:List[ContentType] = []
        for arg in args:
            if isinstance(arg, str):
                self._content_list.append(Content_Map(arg)._content_list[0])
            elif isinstance(arg, ContentType):
                self._content_list.append(arg)

    def __contains__(self, content: ContentType|str|List[ContentType]|List[str]) -> bool:
        """Check if a ContentType or a list of strings is in the Content instance."""
        if isinstance(content, ContentType):
            return content in self._content_list
        elif isinstance(content, str):
            return Content_Map(content)._content_list[0] in self._content_list
        elif isinstance(content, list):
            contain = True
            for item in content:
                if isinstance(item, ContentType):
                    contain = contain and item in self._content_list
                elif isinstance(item, str):
                    contain = contain and Content_Map(item)._content_list[0] in self._content_list
            return contain

    def add(self, content: Union[ContentType, 'Content', List[str]]):
        # Add content_type, all elements from another Content instance, or a list of strings
        if isinstance(content, ContentType):
            if content not in self._content_list:
                self._content_list.append(content)
        elif isinstance(content, Content):
            for item in content._content_list:
                if item not in self._content_list:
                    self._content_list.append(item)
        elif isinstance(content, list):  # Handle list of strings
            for c in content:
                item = Content_Map(c)._content_list[0]
                if item not in self._content_list:
                    self._content_list.append(item)
        else:
            raise ValueError("Input must be either ContentType, Content instance, or a list of strings.")
        
        return Content(self._content_list)  # Return a new Content instance with the updated list

    def remove(self, content: Union[ContentType, 'Content', List[str]]):
        # Remove content_type, all elements from another Content instance, or a list of strings
        if isinstance(content, ContentType):
            if content in self._content_list:
                self._content_list.remove(content)
        elif isinstance(content, Content):
            for c in content._content_list:
                if c in self._content_list:
                    self._content_list.remove(c)
        elif isinstance(content, list):  # Handle list of strings
            for c in content:
                content_type = Content_Map(c)._content_list[0]
                if content_type in self._content_list:
                    self._content_list.remove(content_type)
        else:
            raise ValueError("Input must be either ContentType, Content instance, or a list of strings.")
        
        return Content(self._content_list)  # Return a new Content instance with the updated list

    def are_in(self, other_content: 'Content') -> bool:
        """
        Check if all types in the current Content instance are also in the other Content instance.
        Returns True if all types in self are present in other_content.
        """
        if not isinstance(other_content, Content):
            raise ValueError("Input must be a Content instance.")
        
        return all(type_ in other_content for type_ in self._content_list)

    def are_not_in(self, other_content: 'Content') -> bool:
        if not isinstance(other_content, Content):
            raise ValueError("Input must be a Content instance.")
        
        return all(type_ not in other_content for type_ in self._content_list)
    
    def identical(self, other_content: 'Content') -> bool:
        def are_lists_identical(list1, list2) -> bool:
            if len(list1) != len(list2):return False
            list2_copy = list2[:]
            for item in list1:
                if item in list2_copy:list2_copy.remove(item)
                else:return False
            return len(list2_copy) == 0
        if not isinstance(other_content, Content):
            raise ValueError("Input must be a Content instance.")
        return are_lists_identical(self._content_list, other_content._content_list)



def Content_Map(dim: Union[str, List[str]]) -> Content:
    """
    Maps a string or a list of strings to corresponding ContentType values.
    Accepts a single string or a list of strings.

    :param dim: A string or list of strings representing the content types.
    :return: A Content instance with the associated ContentType(s).
    """
    if isinstance(dim, str):
        dim = [dim]  # Convert to list if a single string is provided

    content_types = []
    for d in dim:
        if d == 'price':
            content_types.append(ContentType.price)
        elif d == 'volume':
            content_types.append(ContentType.volume)
        elif d == 'oscillator':
            content_types.append(ContentType.oscillator)
        elif d == 'ratio':
            content_types.append(ContentType.ratio)
        elif d == 'condition':
            content_types.append(ContentType.condition)
        elif d == 'misc':
            content_types.append(ContentType.misc)
        elif d == 'timedelta':
            content_types.append(ContentType.timedelta)
        else:
            raise RuntimeError(f"No matching operand content found: {d}")

    return Content(*content_types)  # Return a Content instance with the collected content types
