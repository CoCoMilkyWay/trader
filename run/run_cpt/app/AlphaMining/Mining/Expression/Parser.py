import re
from typing import Type, List, Dict, Set, Union, Optional, cast, overload, Literal
from Mining.Expression.Dimension import Dimension
from Mining.Expression.Operator import Operator, UnaryOperator, BinaryOperator, TernaryOperator
from Mining.Expression.Operand import Operand, into_operand
from Mining.Util.Misc_Util import find_last_if
from Mining.Config import \
    FEATURES, DIMENSIONS, OPERATORS, OPERAND, CONST_TIMEDELTAS, CONST_RATIOS, CONST_OSCILLATORS
from pprint import pprint

# Compiled regex pattern for general token extraction.
# Matches:
# - A signed numeric value (e.g., +3.14, -2.0)
# - Any non-word character (e.g., punctuation, spaces)
# - Any word character sequence (e.g., variable names, identifiers)
_PATTERN = re.compile(r'([+-]?[\d.]+|\W|\w+)')

# Compiled regex pattern specifically for signed numeric values.
# Matches:
# - A signed numeric value (e.g., +3.14, -2, 5.0)
# Does NOT match non-word characters or word sequences.
_NUMERIC = re.compile(r'[+-]?[\d.]+')

# save instanced operand and un-instanced operator in stack
# {'Abs': [<class 'xxx.Operator.Abs'>],...}
_OpMap = Dict[str, List[Type[Operator]]]
# Operand(), Operator or Operator()
_StackItem = Union[Operand, Operator, Type[Operator]]

class ExpressionParsingError(Exception):
    pass


class ExpressionParser:
    def __init__(
        self,
        operators: List[Type[Operator]] = OPERATORS,
        ignore_case: bool = False,  # do not distinguish upper/lowercases
        suffix_needed: bool = False,
        prefix_needed: bool = False,
        # {"Max": [Greater],...}
        additional_operator_mapping: Optional[_OpMap] = None
    ):
        self._ignore_case = ignore_case
        self._allow_np_dt = False  # allow non-positive timedelta
        self._suffix_needed = suffix_needed  # timedelta has 'd' as suffix
        self._prefix_needed = prefix_needed  # features has '$' as prefix
        self._features: List[str] = FEATURES
        self._dimensions: List[Dimension] = DIMENSIONS
        self._operators: _OpMap = {op.__name__: [op] for op in operators}
        if additional_operator_mapping is not None:
            self._merge_op_mapping(additional_operator_mapping)
        if ignore_case:
            self._operators = {
                k.lower(): v for k, v in self._operators.items()}
        self._stack: List[_StackItem] = []
        self._tokens: List[str] = []

    def parse(self, expr: str) -> Optional[Operand]:
        self._stack = []
        self._tokens = [t for t in _PATTERN.findall(expr) if not t.isspace()]
        self._tokens.reverse()
        while len(self._tokens) > 0:
            item = self._get_next_item()
            self._stack.append(item)
            valid = self._process_punctuation()
            if not valid:
                return None
        if len(self._stack) != 1:
            raise ExpressionParsingError("Multiple items remain in the stack")
        if len(self._stack) == 0:
            raise ExpressionParsingError("Nothing was parsed")
        if isinstance(self._stack[0], Operand):
            return self._stack[0]
        raise ExpressionParsingError(
            f"{self._stack[0]}/{type(self._stack[0])} is not a valid Operator")

    def _merge_op_mapping(self, map: _OpMap) -> None:
        for name, ops in map.items():
            if (old_ops := self._operators.get(name)) is not None:
                self._operators[name] = list(dict.fromkeys(old_ops + ops))
            else:
                self._operators[name] = ops

    def _get_next_item(self) -> _StackItem:
        top = self._pop_token()
        # Operand(Feature)
        if top == '$':
            top = self._pop_token()
            if top not in self._features:
                raise ExpressionParsingError(f"Can't find the feature {top}")
            return into_operand(top, self._dimensions[self._features.index(top)])
        elif top in self._features and not self._prefix_needed:
            return into_operand(top, self._dimensions[self._features.index(top)])
        # Operand(Constant)
        elif _NUMERIC.fullmatch(top) is not None:
            if self._peek_token() == 'd':
                self._pop_token()
                return into_operand(int(top), Dimension(['timedelta']))
            else:
                value = self._to_float(top)
                if value in CONST_RATIOS:
                    return into_operand(value, Dimension(['ratio']))
                elif value in CONST_OSCILLATORS:
                    return into_operand(value, Dimension(['oscillator']))
                else:
                    return into_operand(value, Dimension(['misc']))
        # Operator
        elif (ops := self._operators.get(top)) is not None:
            return ops[0]
        # elif self._tokens_eq(top, "Constant"):
        #     if self._pop_token() != '(':
        #         raise ExpressionParsingError("\"Constant\" should be followed by a left parenthesis")
        #     value = self._to_float(self._pop_token())
        #     if self._pop_token() != ')':
        #         raise ExpressionParsingError("\"Constant\" should be closed by a right parenthesis")
        #     return Constant(value)
        else:
            raise ExpressionParsingError(f"Cannot parse item:'{top}'")

    def _process_punctuation(self) -> bool:
        if len(self._tokens) == 0:
            return True
        top = self._pop_token()
        stack_top_is_ops = len(self._stack) != 0 and \
            self._is_operator(self._stack[-1])
        if (top == '(') and not stack_top_is_ops:
            raise ExpressionParsingError(
                "A left parenthesis should follow an operator name")
        if top == '(' or top == ',':
            return True
        elif top == ')':
            valid = True
            valid &= self._instance_operator_with_operands()  # Pop an operator with its operands
            valid &= self._process_punctuation()  # There might be consecutive right parens
            return valid
        else:
            raise ExpressionParsingError(f"Unexpected token {top}")

    def _instance_operator_with_operands(self) -> bool:
        if (op_idx := find_last_if(self._stack, lambda item: self._is_operator(item))) == -1:
            raise ExpressionParsingError("Unmatched right parenthesis")
        op = cast(Type[Operator], self._stack[op_idx])
        operands = self._stack[op_idx + 1:]
        self._stack = self._stack[:op_idx]
        # print('DEBUG: ', self._stack, op, operands)
        if any(not isinstance(operand, Operand) for operand in operands):
            raise ExpressionParsingError(
                f"operands fetched contains operators:{operands}")
        operands = cast(List[Operand], operands)
        n_operands = len(operands)
        if issubclass(op, UnaryOperator):
            assert n_operands == 1
            operator = op(operands[0])
        if issubclass(op, BinaryOperator):
            assert n_operands == 2
            operator = op(operands[0], operands[1])
        if issubclass(op, TernaryOperator):
            assert n_operands == 3
            operator = op(operands[0], operands[1], operands[2])
        if operator.valid:
            self._stack.append(operator.output)
        else:
            print(f"Not a valid formula: {operator}")
            return False
        return True
            
    def _tokens_eq(self, lhs: str, rhs: str) -> bool:
        if self._ignore_case:
            return lhs.lower() == rhs.lower()
        else:
            return lhs == rhs

    @classmethod
    def _to_float(cls, token: str) -> float:
        try:
            return float(token)
        except:
            raise ExpressionParsingError(
                f"{token} can't be converted to float")

    def _pop_token(self) -> str:
        if len(self._tokens) == 0:
            raise ExpressionParsingError("No more tokens left")
        top = self._tokens.pop()
        return top.lower() if self._ignore_case else top

    def _peek_token(self) -> Optional[str]:
        return self._tokens[-1] if len(self._tokens) != 0 else None

    def _is_operator(self, item) -> bool:
        return isinstance(item, type) and issubclass(item, Operator)


def parse_expression(expr: str) -> Optional[Operand]:
    "Parse an formula to operator using the default parser."
    return ExpressionParser().parse(expr)


"""
[{
    "pool_state": [
        ["Sub(EMA($close,20d),EMA($close,50d))", -0.015287576109810203],
        ["Greater(Delta($low,10d),Delta($low,30d))", -0.03610591847090697],
        ["Div(Max($close,20d),Min($close,20d))", 0.035015690003175975],
        ["Sub(Delta($close,5d),Delta($close,20d))", -0.00890889276138164],
        ["Greater(EMA($close,10d),EMA($close,30d))", 0.21338035711674033],
        ["Sub(Ref($close,1d),$close)", 0.024938661240208257],
        ["Mul(Div(EMA($high,20d),EMA($low,20d)),$close)", -0.23607067191730652],
        ["Div(EMA($volume,20d),EMA($volume,50d))", 0.023835846374445344],
        ["Cov(EMA($low,20d),$close,30d)", 0.018949387850385493],
        ["Sub(Ref($open,1d),$open)", 0.020497391380293373],
        ["Sub(Max($high,10d),Min($low,10d))", 0.07658269026844951],
        ["Mul(Div(Ref($close,5d),$close),$volume)", 0.1467226878454179],
        ["Greater(Mean($volume,5d),Mean($volume,15d))", -0.12168162745698041],
        ["Div(EMA($close,10d),EMA($close,50d))", -0.08405487681107944],
        ["Greater(Max($high,30d),Min($low,30d))", -0.06598538822776981],
        ["Div(EMA($high,10d),EMA($low,20d))", -0.06568499894188438],
        ["Mul(EMA($high,10d),EMA($low,50d))", 0.0210411200962911],
        ["Sub(Ref($low,1d),$low)", 0.004484002269237874],
        ["Cov(EMA($high,50d),$low,30d)", 0.019576081994501074],
        ["Div(Mean($high,20d),Mean($low,20d))", 0.031151629181337657]
        ], "train_ic": 0.203954815864563, "train_ric": 0.091729536652565, "test_ics": [], "test_rics": []
    }]
"""
