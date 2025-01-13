import random
import operator
import numpy as np
from deap import gp
from .Types import *
from .Primitives import Primitives_CPU

W = 1

def register_primitives(pset: gp.PrimitiveSetTyped) -> None:
    """Register all primitives with appropriate types"""
    # dummy primitive to generate valid tree structure
    pset.addPrimitive(Primitives_CPU.f, [Integer], Integer)
    pset.addPrimitive(Primitives_CPU.f, [Period], Period)
    pset.addPrimitive(Primitives_CPU.f, [Float], Float)
    
    # Basic arithmetic
    pset.addPrimitive(operator.and_, [bool, bool], bool)
    pset.addPrimitive(operator.or_, [bool, bool], bool)
    pset.addPrimitive(operator.add, [Integer, Integer], Integer)
    pset.addPrimitive(operator.add, [Float, Float], Float)
    pset.addPrimitive(operator.sub, [Integer, Integer], Integer)
    pset.addPrimitive(operator.sub, [Float, Float], Float)
    pset.addPrimitive(operator.mul, [Integer, Integer], Integer)
    pset.addPrimitive(operator.mul, [Float, Float], Float)
    pset.addPrimitive(Primitives_CPU.ge, [Integer, Integer], Bool)
    pset.addPrimitive(Primitives_CPU.ge, [Float, Float], Bool)
    pset.addPrimitive(Primitives_CPU.se, [Integer, Integer], Bool)
    pset.addPrimitive(Primitives_CPU.se, [Float, Float], Bool)
    pset.addPrimitive(operator.truediv, [Float, Float], Float)
    
    # Elementary functions (Float → Float)
    pset.addPrimitive(np.log, [Float], Float)
    pset.addPrimitive(np.sqrt, [Float], Float)
    pset.addPrimitive(operator.neg, [Float], Float)
    pset.addPrimitive(Primitives_CPU.sigmoid, [Float], Float)
    pset.addPrimitive(Primitives_CPU.sign, [Float], Bool)
    
    # Time series operations
    # Delay and differences
    # pset.addPrimitive(Primitives_CPU.delay, [TS_Basic, Period], TS_Basic)
    pset.addPrimitive(Primitives_CPU.delta, [TS_Basic, Period], TS_Basic)
    # pset.addPrimitive(Primitives_CPU.delay_pct, [TS_Basic, Period], TS_Basic)
    
    # Correlation and covariance
    # pset.addPrimitive(Primitives_CPU.correlation, [TS_Basic, TS_Basic, Period], TS_Basic)
    # pset.addPrimitive(Primitives_CPU.covariance, [TS_Basic, TS_Basic, Period], TS_Basic)
    
    # Min/Max positions
    # pset.addPrimitive(Primitives_CPU.argmin, [TS_Basic, Period], Integer)
    # pset.addPrimitive(Primitives_CPU.argmax, [TS_Basic, Period], Integer)
    # pset.addPrimitive(Primitives_CPU.rank, [TS_Basic, Period], TS_Basic)
    
    # Decay and statistical
    # pset.addPrimitive(Primitives_CPU.decay_linear, [TS_Basic, Period], TS_Basic)
    pset.addPrimitive(Primitives_CPU.min, [TS_Basic, Period], TS_Basic)
    pset.addPrimitive(Primitives_CPU.max, [TS_Basic, Period], TS_Basic)
    # pset.addPrimitive(Primitives_CPU.stddev, [TS_Basic, Period], TS_Basic)
    pset.addPrimitive(Primitives_CPU.sum, [TS_Basic, Period], TS_Basic)
    pset.addPrimitive(Primitives_CPU.meandev, [TS_Basic, Period], TS_Basic)
    
    # Grouping operations
    # pset.addPrimitive(Primitives_CPU.groupby_ascend, [TS_Basic, TS_Basic, Period, Integer], TS_Basic)
    # pset.addPrimitive(Primitives_CPU.groupby_descend, [TS_Basic, TS_Basic, Period, Integer], TS_Basic)
    # pset.addPrimitive(Primitives_CPU.groupby_diff, [TS_Basic, TS_Basic, Period, Integer], TS_Basic)
    
    # Rank operations
    # pset.addPrimitive(Primitives_CPU.rank_pct, [TS_Basic], TS_Basic)
    # pset.addPrimitive(Primitives_CPU.rank_add, [TS_Basic, TS_Basic], TS_Basic)
    # pset.addPrimitive(Primitives_CPU.rank_sub, [TS_Basic, TS_Basic], TS_Basic)
    # pset.addPrimitive(Primitives_CPU.rank_div, [TS_Basic, TS_Basic], TS_Basic)
    
    # Constants and Terminals
    
    # Integer lookback periods
    standard_windows = [2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 80]
    for d in standard_windows:
        pset.addTerminal(d, Period, name=f"d{d}")
    
    # Group sizes for group operations
    for n in range(1, 11):  # 1-10 groups
        pset.addTerminal(n, Integer, name=f"n{n}")
        
    # Common float constants
    pset.addTerminal(0.0, Float, name="zero")
    pset.addTerminal(1.0, Float, name="one")
    pset.addTerminal(-1.0, Float, name="neg_one")
    pset.addTerminal(0.5, Float, name="half")
    
    # Random float constants
    def rand_const():
        return random.uniform(-1, 1)
    def pi_const():
        return np.pi
    def e_const():
        return np.e
    
    pset.addEphemeralConstant("rand_const", rand_const, Float)
    
    # Boolean constants
    # pset.addTerminal(True, Bool, name="true")
    # pset.addTerminal(False, Bool, name="false")

def setup_primitives(feature_names) -> gp.PrimitiveSetTyped:
    """Initialize primitive set with correct inputs"""
    # Create primitive set with proper input types
    pset = gp.PrimitiveSetTyped("MAIN", 
                               [TS_Basic] * len(feature_names),  # Input time series
                               TS_Basic)  # Output should be time series
    
    # Rename inputs to meaningful names if provided
    if feature_names:
        for i, name in enumerate(feature_names):
            pset.renameArguments(**{f"ARG{i}": name})
    
    # Register all primitives
    register_primitives(pset)
    
    return pset