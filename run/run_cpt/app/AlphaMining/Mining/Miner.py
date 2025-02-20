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
from Mining.Expression.Builder import ExpressionBuilder
from Mining.Expression.Parser import ExpressionParser
from Mining.Metrics.Calculator import ExpressionCalculator
from Mining.AlphaPool.Linear.LinearAlphaPool import MseAlphaPool
from Mining.RL.TokenGenEnv import TokenGenEnv

from Mining.Util.RNG_Util import set_seed

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------------------------------------------------------------+
#        RL(baselines3/Keras/...)                                      |
#         ^|                                                           |
#         v|    v----------------------------------------------------+ |
#        Builder(Gym)           Parser             Calculator        | |
# Token(str) ^  v Expression(str) ^ v Operator/Operand ^ v AlphaPool ^ |
#                           Data ---^                                  |
# ---------------------------------------------------------------------+

class Miner:
    def __init__(self):
        # set_seed(1000)
        self.Data = DATA
        self.builder = ExpressionBuilder()
        self.parser = ExpressionParser()
        self.calculator = ExpressionCalculator(self.Data, 'label_1')
        self.pool = MseAlphaPool(CAPACITY, self.calculator)
        
        self.env = TokenGenEnv(self.builder, self.parser, self.pool)
        
        model = PPO("MlpPolicy", self.env, verbose=1)
        model.learn(total_timesteps=10000)
        
        # with open(f"{os.path.join(os.path.dirname(__file__), ".")}/report.json") as f:
        #     report = json.load(f)
        # alpha_exprs: List[str] = [expr for expr, _ in report[-1]["alphas"]]
        # 
        # # alphas with operator/operand linked, but not evaluated
        # alpha_built: List[Operand] = [a for expr in alpha_exprs if
        #                               (a := self.parser.parse(expr)) is not None]
        # print(f"Built formulas: {len(alpha_built)}")
        # 
        # self.start_time = time()
        # self.pool.load_formulas(alpha_built[:100])
        # self.end_time = time()
        # print(f'load_formulas time elapsed: {(time() - self.start_time):2f}s')

if __name__ == '__main__':
    M = Miner()
