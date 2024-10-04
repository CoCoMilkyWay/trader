from dataclasses import dataclass
@dataclass
class vertex:
    idx: int
    value: float

@dataclass
class zone:
    idx_start: int # starting time
    idx_end: int # ending time
    top: float
    bottom: float