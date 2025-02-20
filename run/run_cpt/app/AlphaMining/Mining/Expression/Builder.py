import numpy as np
from typing import List, Union, Dict

from Mining.Config import *
from Mining.Expression.Expression import Expression
from Mining.Expression.Operator import Operator, UnaryOperator, BinaryOperator, TernaryOperator
from Mining.Expression.Operand import Operand, into_operand
from Mining.Expression.Token import *
from Mining.Util.Misc_Util import find_last_if
from Mining.Util.Expr_Util import build_dimension_list

# Type alias for items that can be on the stack (either an Operand or an Operator)
_StackItem = Union[Operand, Operator]


class ExpressionBuilder:
    def __init__(self):
        """Initialize the ExpressionBuilder with an empty stack."""
        self.stack: List[_StackItem] = []
        self.all_operators_dims: List = []
        for op_type in OPERATORS:
            self.all_operators_dims.append(build_dimension_list(op_type))

    def reset(self):
        """Reset the builder by clearing the stack."""
        self.stack = []

    def is_valid(self) -> bool:
        """
        Check if the current expression is valid.

        An expression is valid if there is exactly one item on the stack and it is
        a featured operator.
        """
        if len(self.stack) == 1:
            final_item = self.stack[0]
            return isinstance(final_item, Operator) and final_item.is_featured
        return False

    def get_built_expression(self) -> Operand:
        """Return the fully built expression as an Operand."""
        assert self.is_valid(), "The current expression is not valid."
        return self.stack[0].output  # type: ignore

    def add_token(self, token: Token):
        """Process a token and update the stack accordingly."""
        if isinstance(token, OperatorToken):
            self._add_operator_token(token)
        elif isinstance(token, ConstantTDToken):
            self._add_constant_token(token.value, Dimension(['timedelta']))
        elif isinstance(token, ConstantRTToken):
            self._add_constant_token(token.value, Dimension(['ratio']))
        elif isinstance(token, ConstantOSToken):
            self._add_constant_token(token.value, Dimension(['oscillator']))
        elif isinstance(token, FeatureToken):
            self._add_feature_token(token.feature)
        else:
            assert False, "Unsupported token type."

    def _add_operator_token(self, token: OperatorToken):
        """Add an operator token to the stack by processing its arguments."""
        n_args = token.operator.n_args()
        operands = [self.stack.pop() for _ in range(n_args)]
        if isinstance(operands[-1], Operator):
            operands[-1] = operands[-1].output
        for operand in operands:
            assert isinstance(operand, Operand)
        operator = token.operator(*reversed(operands))
        assert token.operator.valid
        self.stack.append(operator)

    def _add_constant_token(self, value, dimension):
        """Add a constant value as an Operand to the stack."""
        self.stack.append(into_operand(value, dimension))

    def _add_feature_token(self, feature):
        """Add a feature token as an Operand to the stack."""
        self.stack.append(into_operand(
            feature, DIMENSIONS[FEATURES.index(feature)]))

    def get_action_masks(self) -> Dict:
        """
        Generate action masks based on the current state of the stack.
        Returns a dictionary with validity information and action masks.
        build sequence(bot-up tree, but in order):
        close -> 10 -> STD -> open -> ADD -> ...
        """
        # TODO
        forbidden_output_dim = ['condition']
        
        masks = np.zeros(SIZE_ACTION, dtype=bool)
        is_featured = False
        # Get operands after the last operator in the stack
        op_idx = find_last_if(self.stack, self._is_operator)
        if op_idx == -1:
            operands = self.stack
        else:
            operator = self.stack[op_idx]
            assert isinstance(operator, Operator), str(operator)
            operands = [operator.output] + self.stack[op_idx + 1:]
        
        operands_num = len(operands)
        operands_dim = []

        # Ensure all items after the operator are valid operands
        for operand in operands:
            assert isinstance(operand, Operand)
            is_featured |= operand.is_featured
            operands_dim.append(
                [dim.name for dim in operand.Dimension._dimension_list])

        # Lists for valid unary, binary, and ternary operators
        valid_op = False
        valid_op_type = [[], [], []]
        valid_op_dim = []
        for operator_idx, operator_dims in enumerate(self.all_operators_dims):
            n_args = OPERATORS[operator_idx].n_args()
            if n_args == operands_num:  # check valid operators
                for operator_dim in operator_dims:
                    # operands_dim: [['price'], ['timedelta']]
                    # operator_dim: ['price', 'timedelta', 'price']
                    if all([operator_dim[i] in operands_dim[i] for i in range(operands_num)]):
                        if operator_dim[-1] in forbidden_output_dim: # TODO
                            continue # TODO
                        masks[operator_idx] = True
                        valid_op_type[operands_num -
                                      1].append(str(OPERATORS[operator_idx]))
                        valid_op = True
                        break
            if n_args == operands_num + 1:  # check valid operands
                for operator_dim in operator_dims:
                    # operands_dim: [['price'], ['price']]
                    # operator_dim: ['price', 'price', 'timedelta', 'price']
                    next_operand_dim = operator_dim[-2]
                    if next_operand_dim not in valid_op_dim:
                        if all([operator_dim[i] in operands_dim[i] for i in range(operands_num)]):
                            valid_op_dim.append(next_operand_dim)
        valid_feature = False
        valid_const_dt = False
        valid_const_rt = False
        valid_const_os = False
        valid_stop = self.is_valid()

        offset = SIZE_OP  # Start after operators
        for idx in range(SIZE_FEATURE):
            dim = DIMENSIONS[idx]._dimension_list[0].name
            if dim in valid_op_dim:
                valid_feature = True
                masks[offset + idx] = True
        offset += SIZE_FEATURE
        if is_featured and 'timedelta' in valid_op_dim:
            valid_const_dt = True
            masks[offset: offset + SIZE_CONSTANT_TD] = True
        offset += SIZE_CONSTANT_TD
        if is_featured and 'ratio' in valid_op_dim:
            valid_const_rt = True
            masks[offset: offset + SIZE_CONSTANT_RT] = True
        offset += SIZE_CONSTANT_RT
        if is_featured and 'oscillator' in valid_op_dim:
            valid_const_os = True
            masks[offset: offset + SIZE_CONSTANT_OS] = True
        offset += SIZE_CONSTANT_OS
        if valid_stop:
            masks[offset: offset + SIZE_SEP] = True

        return {
            'valid': [valid_op, valid_feature, valid_const_dt, valid_const_rt, valid_const_os, valid_stop],
            'action_masks': masks,
            'op': {
                1: valid_op_type[0],  # UnaryOperator
                2: valid_op_type[1],  # BinaryOperator
                3: valid_op_type[2],  # TernaryOperator
            }
        }

    def _is_operator(self, item) -> bool:
        """Check if the given item is an Operator."""
        return isinstance(item, Operator)
