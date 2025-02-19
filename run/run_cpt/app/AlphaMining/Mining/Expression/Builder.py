import numpy as np
from typing import List, Union, Dict

from Mining.Config import *
from Mining.Expression.Expression import Expression
from Mining.Expression.Operator import Operator, UnaryOperator, BinaryOperator, TernaryOperator
from Mining.Expression.Operand import Operand, into_operand
from Mining.Expression.Token import *
from Mining.Util.Misc_Util import find_last_if

# Type alias for items that can be on the stack (either an Operand or an Operator)
_StackItem = Union[Operand, Operator]


class ExpressionBuilder:
    def __init__(self):
        """Initialize the ExpressionBuilder with an empty stack."""
        self.stack: List[_StackItem] = []

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
        children = [self.stack.pop() for _ in range(n_args)]
        # for child in children:
        #     assert isinstance(child, Operand)
        operator = token.operator(*reversed(children))
        assert operator.valid, "The operator created is not valid."
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
        """
        masks = np.zeros(SIZE_ACTION, dtype=bool)

        # Get operands after the last operator in the stack
        op_idx = find_last_if(self.stack, self._is_operator)
        operands = self.stack[op_idx + 1:] if op_idx != -1 else self.stack

        # Ensure all items after the operator are valid operands
        for operand in operands:
            assert isinstance(operand, Operand)

        num_operands = len(operands)
        # Lists for valid unary, binary, and ternary operators
        valid_op_array = [[], [], []]
        valid_op = False

        # Determine valid operators based on the number of operands
        for idx, op in enumerate(OPERATORS):
            if op.n_args() == num_operands:
                instance = op(*reversed(operands))
                if instance.valid:
                    masks[idx] = True
                    valid_op_array[num_operands - 1].append(str(op))
                    valid_op = True
                del instance  # Clean up the instance after checking

        valid_feature = num_operands < 3
        valid_const = self.validate_const()
        valid_stop = self.is_valid()

        # Fill action masks based on validity
        self._fill_action_masks(
            masks, valid_feature, valid_const, valid_stop)

        return {
            'valid': [valid_op, valid_feature, valid_const, valid_stop],
            'action_masks': masks,
            'op': {
                UnaryOperator: valid_op_array[0],
                BinaryOperator: valid_op_array[1],
                TernaryOperator: valid_op_array[2],
            }
        }

    def _fill_action_masks(self, masks: np.ndarray, valid_feature: bool, valid_const: bool, valid_stop: bool) -> None:
        """Fill the action masks based on operator, feature, and constant validity."""
        offset = SIZE_OP  # Start after operators

        if valid_feature:
            masks[offset: offset + SIZE_FEATURE] = True
        offset += SIZE_FEATURE

        if valid_const:
            masks[offset: offset + SIZE_CONSTANT_TD] = True
        offset += SIZE_CONSTANT_TD

        if valid_const:
            masks[offset: offset + SIZE_CONSTANT_RT] = True
        offset += SIZE_CONSTANT_RT

        if valid_const:
            masks[offset: offset + SIZE_CONSTANT_OS] = True
        offset += SIZE_CONSTANT_OS

        if valid_stop:
            masks[offset: offset + SIZE_SEP] = True

    def _is_operator(self, item) -> bool:
        """Check if the given item is an Operator."""
        return isinstance(item, Operator)

    def validate_const(self) -> bool:
        """
        Validate if the stack is in a state where a constant can be added.

        The stack is valid for adding constants if it is empty or if the last item
        on the stack is a featured operator.
        """
        if len(self.stack) == 0:
            return True
        last_item = self.stack[-1]
        return isinstance(last_item, Operator) and last_item.is_featured
