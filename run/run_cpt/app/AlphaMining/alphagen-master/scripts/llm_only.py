from typing import Optional, List
from logging import Logger
from datetime import datetime
import json
from itertools import accumulate

import fire
import torch
from openai import OpenAI

from alphagen.data.expression import Expression
from alphagen.data.parser import ExpressionParser
from alphagen.data.expression import *
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, initialize_qlib
from alphagen_generic.features import target
from alphagen_llm.client import OpenAIClient, ChatConfig
from alphagen_llm.prompts.interaction import DefaultInteraction, DefaultReport
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen.utils import get_logger
from alphagen.utils.misc import pprint_arguments

"""
from openai import OpenAI

client = OpenAI(
  api_key=""
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);

"""

"""
https://api.feidaapi.com
https://api.feidaapi.com/v1
https://api.feidaapi.com/v1/chat/completions
https://api.feidaapi.com/query/ # check balance
"""


pool = {
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
        ], 
    "train_ic": 0.203954815864563, 
    "train_ric": 0.091729536652565, 
    "test_ics": [], 
    "test_rics": []
    }


def build_chat(system_prompt: str, logger: Optional[Logger] = None):
    return OpenAIClient(
        OpenAI(
            api_key="sk-1GTDFWtFdYu0DUb35KLHnyRBjjmQX81XyMQ61fHzmuh5eImj",
            base_url="https://api.feidaapi.com/v1",
            ),
        ChatConfig(
            system_prompt=system_prompt,
            logger=logger
        ),
        # model="gpt-4o-mini",
        model="gpt-3.5-turbo-0125",
    )


def build_parser(use_additional_mapping: bool = False) -> ExpressionParser:
    mapping = {
        "Max": [Greater],
        "Min": [Less],
        "Delta": [Sub]
    }
    return ExpressionParser(
        Operators,
        ignore_case=True,
        additional_operator_mapping=mapping if use_additional_mapping else None,
        non_positive_time_deltas_allowed=False
    )


def build_test_data(instruments: str, device: torch.device, n_half_years: int, halves, start_year:int) -> List[Tuple[str, StockData]]:

    def get_dataset(i: int) -> Tuple[str, StockData]:
        year = start_year + i // 2
        start, end = halves[i % 2]
        return (
            f"{year}h{i % 2 + 1}",
            StockData(
                instrument=instruments,
                start_time=f"{year}-{start}",
                end_time=f"{year}-{end}",
                device=device
            )
        )

    return [get_dataset(i) for i in range(n_half_years)]


def run_experiment(
    pool_size: int = 20,
    n_replace: int = 3,
    n_updates: int = 20,
    without_weights: bool = False,
    contextful: bool = False,
    prefix: Optional[str] = None,
    force_remove: bool = False,
    also_report_history: bool = False
):
    """
    :param pool_size: Maximum alpha pool size
    :param n_replace: Replace n alphas on each iteration
    :param n_updates: Run n iterations
    :param without_weights: Do not report the weights of the alphas to the LLM
    :param contextful: Keep context in the conversation
    :param prefix: Output location prefix
    :param force_remove: Force remove worst old alphas
    :param also_report_history: Also report alpha pool update history to the LLM
    """

    args = pprint_arguments()

    initialize_qlib(f"./qlib/qlib_data/cn_data")
    instruments = "csi300"
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    prefix = str(prefix) + "-" if prefix is not None else ""
    out_path = f"./out/llm-tests/interaction/{prefix}{timestamp}"
    logger = get_logger(name="llm", file_path=f"{out_path}/llm.log")

    with open(f"{out_path}/config.json", "w") as f:
        json.dump(args, f)

    data_train = StockData(
        instrument=instruments,
        start_time="2012-01-01",
        end_time="2020-5-1",
        device=device
    )
    data_test = build_test_data(
        instruments,
        device,
        n_half_years=0,
        halves=(("01-01", "06-30"), ("07-01", "12-31")),
        start_year=2019,
        )
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_test = [QLibStockDataCalculator(d, target) for _, d in data_test]

    def make_pool(exprs: List[Expression]) -> MseAlphaPool:
        pool = MseAlphaPool(
            capacity=max(pool_size, len(exprs)),
            calculator=calculator_train,
            device=device
        )
        pool.force_load_exprs(exprs)
        return pool

    def show_iteration(_, iter: int):
        print(f"Iteration {iter} finished...")

    inter = DefaultInteraction(
        parser=build_parser(),
        client=build_chat(EXPLAIN_WITH_TEXT_DESC, logger=logger),
        pool_factory=make_pool,
        calculator_train=calculator_train,
        calculators_test=calculator_test,
        replace_k=n_replace,
        force_remove=force_remove,
        forgetful=not contextful,
        no_actual_weights=without_weights,
        also_report_history=also_report_history,
        on_pool_update=show_iteration
    )
    inter.run(n_updates=n_updates)

    with open(f"{out_path}/report.json", "w") as f:
        json.dump([r.to_dict() for r in inter.reports], f)

    cum_days = list(accumulate(d.n_days for _, d in data_test))
    mean_ic_results = {}
    mean_ics, mean_rics = [], []

    def get_rolling_means(ics: List[float]) -> List[float]:
        cum_ics = accumulate(ic * tup[1].n_days for ic, tup in zip(ics, data_test))
        return [s / n for s, n in zip(cum_ics, cum_days)]

    for report in inter.reports:
        mean_ics.append(get_rolling_means(report.test_ics))
        mean_rics.append(get_rolling_means(report.test_rics))

    for i, (name, _) in enumerate(data_test):
        mean_ic_results[name] = {
            "ics": [step[i] for step in mean_ics],
            "rics": [step[i] for step in mean_rics]
        }
    
    with open(f"{out_path}/rolling_mean_ic.json", "w") as f:
        json.dump(mean_ic_results, f)


if __name__ == "__main__":
    fire.Fire(run_experiment)
