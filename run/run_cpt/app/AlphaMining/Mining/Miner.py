import os, sys
import json
import torch
from pprint import pprint
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Mining.Config import DATA

from Mining.Expression.Parser import ExpressionParser

class Miner:
    def __init__(self):
        self.Data = DATA
        parser = ExpressionParser()
        with open(f"{os.path.join(os.path.dirname(__file__), ".")}/report.json") as f:
            data = json.load(f)
            pool_state = data[-1]["pool_state"]
            # pprint(pool_state)
            [parser.parse(expr) for expr, _ in pool_state]
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

