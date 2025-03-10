import sys
import inspect
from colorama import Fore, Back, Style, init
from typing import Any, Dict, Set
from collections.abc import Iterable

class MemoryAnalyzer:
    def __init__(self):
        init(autoreset=True)
        self.seen_ids: Set[int] = set()
        self.recursion_depth = 0
        self.MAX_DEPTH = 20
        
    def _is_basic_type(self, obj: Any) -> bool:
        return isinstance(obj, (int, float, str, bool, bytes, type(None)))
    
    def get_deep_size(self, obj: Any, depth: int = 0) -> int:
        if depth > self.MAX_DEPTH:
            return 0
            
        obj_id = id(obj)
        if obj_id in self.seen_ids:
            return 0
            
        try:
            if self._is_basic_type(obj):
                return sys.getsizeof(obj)
                
            self.seen_ids.add(obj_id)
            size = sys.getsizeof(obj)
            
            if isinstance(obj, dict):
                try:
                    size += sum(self.get_deep_size(k, depth + 1) + 
                              self.get_deep_size(v, depth + 1) 
                              for k, v in obj.items())
                except (RuntimeError, RecursionError):
                    pass
                    
            elif isinstance(obj, (list, tuple, set, frozenset)):
                try:
                    size += sum(self.get_deep_size(i, depth + 1) for i in obj)
                except (RuntimeError, RecursionError):
                    pass
                    
            elif hasattr(obj, '__dict__'):
                try:
                    size += self.get_deep_size(obj.__dict__, depth + 1)
                except (RuntimeError, RecursionError):
                    pass
                    
            return size
            
        except Exception as e:
            print(f"{Fore.RED}Warning: Error measuring size of {type(obj)}: {str(e)}{Style.RESET_ALL}")
            return sys.getsizeof(obj)
        finally:
            if depth == 0:
                self.seen_ids.clear()

    def _format_value(self, obj: Any) -> str:
        try:
            if isinstance(obj, (list, tuple, set)):
                if len(obj) > 0:
                    first_items = ', '.join(repr(x)[:20] for x in list(obj)[:3])
                    return f"{type(obj).__name__}[{len(obj)} items]: [{first_items}{'...' if len(obj) > 3 else ''}]"
                return f"{type(obj).__name__}[empty]"
            elif isinstance(obj, dict):
                if len(obj) > 0:
                    first_items = ', '.join(f"{k}: {repr(v)[:10]}" for k, v in list(obj.items())[:2])
                    return f"dict[{len(obj)} keys]: {{{first_items}{'...' if len(obj) > 2 else ''}}}"
                return "dict[empty]"
            elif isinstance(obj, str):
                if len(obj) > 50:
                    return f"'{obj[:47]}...'"
                return repr(obj)
            else:
                return repr(obj)[:50]
        except:
            return f"[Error formatting {type(obj).__name__}]"

    def analyze_object(self, obj: Any) -> None:
        class_name = obj.__class__.__name__
        print(f"\n{Back.BLUE}{Fore.WHITE} Memory Analysis: {class_name} {Style.RESET_ALL}")
        
        self.seen_ids.clear()
        try:
            instance_size = self.get_deep_size(obj)
            print(f"\n{Fore.CYAN}Overall Memory Usage:{Style.RESET_ALL}")
            print(f"└── Total Instance Size: {instance_size:,} bytes")
            print(f"└── Base Object Size: {sys.getsizeof(obj):,} bytes")
            if hasattr(obj, '__dict__'):
                print(f"└── Dict Size: {sys.getsizeof(obj.__dict__):,} bytes\n")
            
            # Collect attribute data
            attr_data = []
            for name in dir(obj):
                if name.startswith('__'):
                    continue
                    
                try:
                    attr = getattr(obj, name)
                    self.seen_ids.clear()
                    
                    if callable(attr):
                        continue
                    
                    size = self.get_deep_size(attr)
                    shallow_size = sys.getsizeof(attr)
                    type_name = type(attr).__name__
                    value = self._format_value(attr)
                    
                    if size > 0:
                        percentage = (size / instance_size) * 100
                    else:
                        percentage = 0
                    
                    try:
                        source = "instance"
                        if inspect.getattr_static(obj.__class__, name, None) is not None:
                            source = "class"
                    except:
                        source = "unknown"
                    
                    attr_data.append({
                        'name': name,
                        'deep_size': size,
                        'shallow_size': shallow_size,
                        'type': type_name,
                        'percentage': percentage,
                        'source': source,
                        'value': value
                    })
                        
                except Exception as e:
                    attr_data.append({
                        'name': name,
                        'deep_size': 0,
                        'shallow_size': 0,
                        'type': 'error',
                        'percentage': 0,
                        'source': 'error',
                        'value': str(e)[:50]
                    })
            
            # Sort by size
            attr_data.sort(key=lambda x: x['deep_size'], reverse=True)
            
            # Display detailed attributes
            print(f"{Fore.CYAN}Detailed Attribute Analysis:{Style.RESET_ALL}")
            print("=" * 100)
            
            format_str = "{:<30} | {:>12} | {:>12} | {:>10} | {:>8} | {:<8} | {}"
            print(format_str.format(
                "Attribute Name",
                "Deep Size",
                "Shallow Size",
                "Type",
                "% Total",
                "Source",
                "Value Preview"
            ))
            print("-" * 100)
            
            for attr in attr_data:
                print(format_str.format(
                    attr['name'][:30],
                    f"{attr['deep_size']:,}",
                    f"{attr['shallow_size']:,}",
                    attr['type'][:10],
                    f"{attr['percentage']:6.1f}%",
                    attr['source'],
                    attr['value']
                ))
            
            # Memory distribution visualization
            print(f"\n{Fore.CYAN}Memory Distribution (Top 10):{Style.RESET_ALL}")
            print("-" * 80)
            for attr in attr_data[:10]:
                bars = int(attr['percentage'] / 2)
                print(f"{attr['name']:20} [{Fore.BLUE}{'█' * bars}{Style.RESET_ALL}{' ' * (50-bars)}] {attr['percentage']:5.1f}%")
                    
        except Exception as e:
            print(f"{Fore.RED}Error during analysis: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    # Test code
    class TestClass:
        def __init__(self):
            self.data = [1, 2, 3]
            self.name = "test"
            
    analyzer = MemoryAnalyzer()
    test_obj = TestClass()
    analyzer.analyze_object(test_obj)