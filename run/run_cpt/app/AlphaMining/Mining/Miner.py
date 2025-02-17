import os
import sys
import json
import torch
from torch import Tensor
from typing import List
from pprint import pprint
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Mining.Config import DATA, FEATURES, LABELS
from Mining.Metrics.Calculator import ExpressionCalculator
from Mining.Expression.Parser import ExpressionParser
from Mining.Expression.Operand import Operand

class Miner:
    def __init__(self):
        self.Data = DATA
        self.parser = ExpressionParser()
        self.calculator = ExpressionCalculator(self.Data, 'label_1')
        with open(f"{os.path.join(os.path.dirname(__file__), ".")}/report.json") as f:
            data = json.load(f)

        # alphas with expression as string
        alpha_exprs: List[str] = [expr for expr, _ in data[-1]["alphas"]]

        # alphas with operator/operand linked, but not evaluated
        alpha_built: List[Operand] = [a for expr in alpha_exprs if
                                      (a := self.parser.parse(expr)) is not None]

        # alphas with evaluated Tensor output (timestamps * codes)
        alpha_results: List[Tensor] = [
            self.calculator.evaluate_alpha(alpha) for alpha in alpha_built]
        #
        # pool = MseAlphaPool(
        #     capacity=30,
        #     calculator=calculators[0],
        #     ic_lower_bound=None,
        #     l1_alpha=5e-3,
        #     device=device
        # )
        # if len(exprs) != 0:
        #     pool.force_load_exprs(exprs)
        #
        # AlphaEnvWrapper(AlphaEnvCore(pool=pool, **kwargs), subexprs=subexprs)
        # get_wrapper_attr(EXPECTED_METHOD_NAME)()


if __name__ == '__main__':
    M = Miner()
