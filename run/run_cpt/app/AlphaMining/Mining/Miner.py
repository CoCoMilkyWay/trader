import os
import sys
import json
from time import time
from typing import List
from torch import Tensor
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Mining.Config import DATA, CAPACITY
from Mining.Expression.Operand import Operand
from Mining.Expression.Parser import ExpressionParser
from Mining.Metrics.Calculator import ExpressionCalculator
from Mining.AlphaPool.Linear.LinearAlphaPool import MseAlphaPool


class Miner:
    def __init__(self):
        self.Data = DATA
        self.parser = ExpressionParser()
        self.calculator = ExpressionCalculator(self.Data, 'label_1')
        self.pool = MseAlphaPool(CAPACITY, self.calculator)
        with open(f"{os.path.join(os.path.dirname(__file__), ".")}/report.json") as f:
            data = json.load(f)

        # alphas with expression as string
        alpha_exprs: List[str] = [expr for expr, _ in data[-1]["alphas"]]

        # alphas with operator/operand linked, but not evaluated
        alpha_built: List[Operand] = [a for expr in alpha_exprs if
                                      (a := self.parser.parse(expr)) is not None]
        print(f"Built formulas: {len(alpha_built)}")

        self.start_time = time()
        self.pool.load_formulas(alpha_built[:100])
        self.end_time = time()
        print(f'load_formulas time elapsed: {(time() - self.start_time):2f}s')

if __name__ == '__main__':
    M = Miner()
