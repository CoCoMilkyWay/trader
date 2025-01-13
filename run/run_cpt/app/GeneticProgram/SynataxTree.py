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