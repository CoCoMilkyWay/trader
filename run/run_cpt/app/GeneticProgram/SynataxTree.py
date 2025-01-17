# def genCustomTreeWithConstraints(pset, min_, max_, type_):
#     def createSubtree(template):
#         """Creates a subtree based on a template pattern"""
#         if isinstance(template, tuple):
#             prim, args = template
#             return [prim] + [createSubtree(arg) for arg in args]
#         return template
# 
#     # Define templates of common patterns you want to appear
#     common_patterns = [
#         # Example: (operator, [(subtree1), (subtree2)])
#         ({
#             'pattern': (pset.primitives[type_][0],  # add
#                        [(pset.terminals[type_][0],),  # x
#                         (pset.primitives[type_][1],  # mul
#                          [pset.terminals[type_][0],
#                           pset.terminals[type_][1]])]),
#             'weight': 0.4
#         }),
#         # Add more patterns...
#     ]
# 
#     def generate(height, depth, type_):
#         if depth < max_ and random.random() < 0.6:  # 60% chance to use pattern
#             pattern = random.choices(
#                 [p['pattern'] for p in common_patterns],
#                 weights=[p['weight'] for p in common_patterns]
#             )[0]
#             return createSubtree(pattern)
#         return gp.genHalfAndHalf(pset, min_, max_, type_)
# 
#     return generate(random.randint(min_, max_), 0, type_)

import random
import sys
import warnings
from deap.gp import MetaEphemeral

def genFull(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A full tree with all leaves at the same depth.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    return generate(pset, min_, max_, condition, type_)


def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth == height or \
            (depth >= min_ and random.random() < pset.terminalRatio)

    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


def genRamped(pset, min_, max_, type_=None):
    """
    .. deprecated:: 1.0
        The function has been renamed. Use :func:`~deap.gp.genHalfAndHalf` instead.
    """
    warnings.warn("gp.genRamped has been renamed. Use genHalfAndHalf instead.",
                  FutureWarning)
    return genHalfAndHalf(pset, min_, max_, type_)


def generate(pset, min_, max_, condition, type_=None):
    """Generate a tree as a list of primitives and terminals in a depth-first
    order. The tree is built from the root to the leaves, and it stops growing
    the current branch when the *condition* is fulfilled: in which case, it
    back-tracks, then tries to grow another branch until the *condition* is
    fulfilled again, and so on. The returned list can then be passed to the
    constructor of the class *PrimitiveTree* to build an actual tree object.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a terminal of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            if type(term) is MetaEphemeral:
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a primitive of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr

"""
PROMPT:

objective: alpha mining for timeseries/crosssection(multi-timeseries) data
algorithm: genetic progrmas (tree-based individual)
input: momentum/volatility/... based indicator timeseries of default setting, number > 100
primitive(operators): bool/integer/float/timeseries-based-operators/rank-based-operators(for crossection) ....
fitness function: max IC from mined alpha to labels(near term sharpe as label (not simple returns)), monotone, quantile stability(alpha should be continuous(and have high near-term sharpe))/correlations to exisiting alpha...
motivition: genetic program has vast explore space for this many inputs and primitives, need additional constraints

require:
give a dozen tree structures(e.g.  (indicator set A * primitives set A) and ...) that are likely to generate high fitness alpha for:
    1. trend strategies
    2. mean-reversion strategies
"""
# ============================================================================================

"""
Let me analyze the code and explain how this tree generation differs from typical genetic programming tree generation methods.
The key differences I notice in this implementation are:

Time Series Constants Handling:

pythonCopy# first, check if need to add time series constant
ts_const_num = parent_stack[-1].data.is_ts
while ts_const_num:
    data = random.randint(*self.ts_const_range)
    node = Node(data, is_ts=True)
    parent_stack[-1].add_child(node)
    ts_const_num -= 1
This implementation has special handling for time series constants through the is_ts flag. Before adding regular nodes, it first checks if the current function requires time series constants and adds them with priority. This is not typically present in standard GP tree generation methods.

Build Preference Probabilities:

pythonCopyif depth < max_depth and (
    method == "full" or random.random() < self.build_preference[0]
):
    # add function
    ...
else:
    # add terminal, and determine to add variable or constant
    if random.random() < self.build_preference[1]:
        data = random.choice(self.variable_set)
        ...
    else:
        data = random.randint(*self.const_range)
While standard implementations usually make binary choices between functions and terminals, this implementation uses build_preference to control:

The probability of choosing functions vs terminals
The probability of choosing variables vs constants when selecting terminals


Fixed Parameters Support:

pythonCopychildren_stack = [data.arity - len(data.fixed_params)]
The implementation accounts for functions that have fixed parameters through fixed_params. This allows for functions that take both dynamic (tree-generated) and fixed (pre-defined) parameters, which isn't common in standard GP tree generation.

Parent-Child Relationship Tracking:

pythonCopydef add_child(self, child: "Node"):
    self.children.insert(0, child)
    child.parent = self
The implementation maintains explicit bidirectional parent-child relationships, which while not unique, is more comprehensive than some basic implementations that only track children.
In standard GP tree generation:

genFull forces all branches to reach maximum depth
genGrow allows branches to terminate at any depth
genHalfAndHalf randomly uses either full or grow for each tree

This implementation maintains these basic approaches but adds the specialized features above to handle time series data and provide more fine-grained control over the tree structure through build preferences and fixed parameters.
Would you like me to elaborate on any of these differences or explain how they might impact the generated trees?
"""

"""
1. Basic Components (All output 0~1)
Price-based Normalized Components (PN)
CopyPN1 = (P - MinP(window)) / (MaxP(window) - MinP(window))
PN2 = Percentile(P, window) / 100
PN3 = 1 / (1 + exp(-ZScore(P)))  # Sigmoid of z-score
PN4 = RollingRank(P, window) / window_size
Volume-based Normalized Components (VN)
CopyVN1 = Volume / MaxVolume(window)
VN2 = Percentile(Volume, window) / 100
VN3 = RollingRank(Volume, window) / window_size
VN4 = CurrentVolume / MovingAvgVolume(window)
Volatility-based Normalized Components (σN)
CopyσN1 = 1 - (CurrentVol / MaxVol(window))
σN2 = Percentile(1/Volatility, window) / 100
σN3 = 1 / (1 + Volatility/MovingAvgVol(window))
σN4 = MinVol(window) / CurrentVol
2. Tree Structure Templates
Base Trees (Direct 0~1 output)

Momentum Strength

CopyMS = PN2(Returns(window))

Volume Intensity

CopyVI = VN1 * VN2

Volatility Regime

CopyVR = σN1 * σN3

Trend Persistence

CopyTP = MovingAvg(PN1, fast) / MovingAvg(PN1, slow)

Range Position

CopyRP = (Close - MinPrice(window)) / (MaxPrice(window) - MinPrice(window))
Selector Trees

Momentum-Volume Selector

CopyIF VN1 > threshold
    THEN MS
    ELSE VI

Volatility Regime Selector

CopyIF σN1 > threshold
    THEN TP
    ELSE RP
3. Complex Tree Structures (All outputs guaranteed 0~1)

Momentum-Volume Consensus

CopyMVC = MS * VI * VR

Adaptive Range

CopyAR = RP * σN2 * VN2

Multi-timeframe Momentum

CopyMTM = (MS(fast) * 0.4 + MS(medium) * 0.3 + MS(slow) * 0.3)

Volume-Adjusted Position

CopyVAP = RP * VN1 * σN3

Trend Strength

CopyTS = MovingCorr(PN1, VN1, window) * 0.5 + 0.5  # Maps [-1,1] to [0,1]

Liquidity Score

CopyLS = (VN1 * VN2 * σN2)

Cross-sectional Position

CopyCSP = GroupRank(PN1) / GroupSize

Adaptive Momentum

CopyAM = IF σN1 > 0.7
        THEN MS * VN1
        ELSE RP * VN2

Regime-Based Position

CopyRBP = IF VR > 0.8
        THEN TP * σN2
        ELSE RP * σN3

Volume-Price Agreement

CopyVPA = (PN2 * VN2 + (1-abs(MovingCorr(PN1, VN1, window)-0.5)))
4. Composition Rules

Multiplicative

CopyFinal = Tree1 * Tree2 * Tree3

Weighted Average

CopyFinal = w1*Tree1 + w2*Tree2 + (1-w1-w2)*Tree3
where w1, w2 >= 0 and w1 + w2 <= 1

Conditional Selection

CopyFinal = IF Condition_Tree > threshold
           THEN Tree1
           ELSE Tree2

Smooth Transition

Copyweight = Tree1
Final = weight * Tree2 + (1-weight) * Tree3
Implementation Notes

Every component and tree must mathematically guarantee output in [0,1]
Selector trees use normalized components for conditions
All weights in weighted averages must sum to 1
Moving windows should be parameterized for optimization
Thresholds in selectors should be optimized in [0,1]

Parameter Ranges:

Fast window: [5, 20]
Medium window: [20, 60]
Slow window: [60, 120]
Thresholds: [0.3, 0.7]
Correlation windows: [20, 40]
"""