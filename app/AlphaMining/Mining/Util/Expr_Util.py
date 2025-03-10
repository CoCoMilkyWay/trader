from itertools import product
from typing import List, Type, Union, Tuple, Optional, Callable

from Mining.Expression.Dimension import Dimension, DimensionType, DimensionType_Map
from Mining.Expression.Operand import Operand, into_operand
from Mining.Expression.Operator import Operator
from Mining.Config import ALL_DIMENSION_TYPES


def build_dimension_list(operator: Type[Operator]) -> List[Tuple]:
    """
    Generate a list of valid dimension combinations for operands and outputs based on the operator's dimension rules.

    Args:
        dimension_map (List[List]): A list of dimension rules specified as [[operand1_types], [operand2_types], ..., output_type].
        num_operands (int): Number of operands for the operator (Unary, Binary, or Ternary).

    Returns:
        List[Tuple]: A list of tuples, where each tuple represents valid operand types and an output type.
                     Example for unary: [(price, price), ...]
                     Example for binary: [(price, ratio, price), ...]
    """
    num_operands = operator.n_args()
    dummy_operand = into_operand(0.0, Dimension(['misc']))
    dummy_operands = [dummy_operand] * num_operands
    dimension_map: list[list[list[DimensionType]]] = \
        operator(*dummy_operands).map  # type: ignore
        
    operand_dimensions = [rule[:-1] for rule in dimension_map]
    output_dimensions = [rule[-1] for rule in dimension_map]
    
    # Iterate over all possible combinations of operand dimensions
    all_dimension_types = [DimensionType_Map(t) for t in ALL_DIMENSION_TYPES]
    valid_combinations = []
    for combo in product(all_dimension_types, repeat=num_operands):
        for rule_index, operand_rules in enumerate(operand_dimensions):
            # Check if the combination satisfies the current rule
            if all(combo[operand_idx] in operand_rules[operand_idx] for operand_idx in range(num_operands)):
                # Append the combination along with its corresponding valid output
                valid_combo = [*combo, output_dimensions[rule_index]]
                valid_combo = [c.name for c in valid_combo]
                valid_combinations.append(valid_combo)
    return valid_combinations
