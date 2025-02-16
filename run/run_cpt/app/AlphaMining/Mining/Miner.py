import os
import sys
import json
import torch
from torch import Tensor
from typing import List
from pprint import pprint
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Mining.Expression.Parser import ExpressionParser
from Mining.Expression.Operand import Operand
from Mining.Config import DATA, FEATURES

class Miner:
    def __init__(self):
        self.Data = DATA
        self.parser = ExpressionParser()
        with open(f"{os.path.join(os.path.dirname(__file__), ".")}/report.json") as f:
            data = json.load(f)

        # alphas with formulas string
        alpha_formula: List[str] = [expr for expr, _ in data[-1]["alphas"]]
        # alphas with operator/operand linked, but not evaluated
        alpha_built: List[Operand] = [a for expr in alpha_formula if
                                      (a := self.parser.parse(expr)) is not None]
        alpha_results: List[Tensor] = [alpha.final_evaluate(self.Data) for alpha in alpha_built]
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
