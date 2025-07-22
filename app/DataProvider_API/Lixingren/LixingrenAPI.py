import json
import requests
from typing import List, Dict, Tuple, Any, Optional

from ..License import LIXINGREN_LICENSE


class LixingrenAPI:
    """
    A class to encapsulate Lixingren APIs related to stock trading and data.
    """

    def __init__(self):
        self.LICENSE = LIXINGREN_LICENSE  # Replace with your actual license key
        self.API = CN_C = {}
        # CN_C: China Company

        CN_C['basic_all'] = {
            "link": "https://open.lixinger.com/api/cn/company",
            "description": "获取股票详细信息",
            "payload": {
                "token": f"{self.LICENSE}",
                "includeDelisted": True,  # include delisted stocks
            }
        }

        CN_C['profile'] = {
            "link": "https://open.lixinger.com/api/cn/company/profile",
            "description": "获取公司概况数据",
            "payload": {
                "token": f"{self.LICENSE}",
                "stockCodes": None
            }
        }

        CN_C['industries'] = {
            "link": "https://open.lixinger.com/api/cn/company/industries",
            "description": "获取股票所属行业信息",
            "payload": {
                "token": f"{self.LICENSE}",
                "stockCode": None
            }
        }

        CN_C['dividend'] = {
            "link": "https://open.lixinger.com/api/cn/company/dividend",
            "description": "获取分红信息",
            "payload": {
                "token": f"{self.LICENSE}",
                "startDate": None,  # "xxxx-xx-xx" within 10 years
                "endDate": None,
                "stockCode": None
            },
        }

        CN_C['allotment'] = {
            "link": "https://open.lixinger.com/api/cn/company/allotment",
            "description": "获取配股信息",
            "payload": {
                "token": f"{self.LICENSE}",
                "startDate": None,  # "xxxx-xx-xx" within 10 years
                "endDate": None,
                "stockCode": None
            }
        }

        # Fundamental models:
        set1 = [
            'pe_ttm',       # P/E Ratio TTM: 股票市盈率（过去12个月），衡量当前股价与过去12个月盈利的比例
            'd_pe_ttm',     # Adjusted P/E TTM: 扣除非经常性项目后的市盈率，反映公司正常盈利能力
            'pb',           # P/B Ratio: 市净率，比较股票市场价格与公司账面价值的关系
            'pb_wo_gw',     # P/B excluding Goodwill: 不含商誉的市净率，更准确反映公司有形资产价值
            'ps_ttm',       # P/S Ratio TTM: 市销率（过去12个月），衡量市值与年销售额的比率
            'pcf_ttm',      # P/CF Ratio TTM: 市现率（过去12个月），比较市值与实际现金流量的关系
            'dyr',          # Dividend Yield: 股息率，表示股息收益占当前股价的比例
            'sp',           # Stock Price: 股价，即当前市场上股票的交易价格
            'spc',          # Stock Price Change: 股票价格变化百分比，反映价格的涨跌幅度
            'spa',          # Stock Price Amplitude: 股价振幅，显示一段时间内股票价格波动的范围
            'tv',           # Trading Volume: 成交量，指在特定期间内交易的股票数量
            'ta',           # Trading Amount: 成交金额，指交易股票总价值，反映市场活跃度
            'to_r',         # Turnover Rate: 换手率，衡量股票在市场上被交易的频率
            'shn',          # Number of Shareholders: 总股东人数，表示持有股票的投资者数量
            'mc',           # Market Capitalization: 市值，即公司所有流通股的市场总价值
            'mc_om',        # A-share Market Cap: A股市值，专指国内市场上市股票的总市值
            'cmc',          # Circulating Market Cap: 流通市值，只计算市场上流通交易的股票价值
            'ecmc',         # Free Float Market Cap: 自由流通市值，专注于可自由买卖的股票部分的市值
            'ecmc_psh',     # Free Float Market Cap per Shareholder: 人均自由流通市值，平均分摊到每位股东的自由流通市值
            'fpa',          # Margin Financing Purchase: 融资买入金额，指通过借款购买股票的总金额
            'fra',          # Margin Financing Repayment: 融资偿还金额，指偿还融资买入股票款项的金额
            'fb',           # Margin Financing Balance: 融资余额，剩余未偿还的融资金额
            'ssa',          # Securities Lending Sale: 融券卖出金额，通过借入股票进行卖出的总金额
            'sra',          # Securities Lending Repayment: 融券偿还金额，用于偿还融券买入股票的金额
            'sb',           # Securities Borrowing Balance: 融券余额，当前仍未归还的借入股票金额
            'ha_sh',        # Northbound Held Shares: 陆股通持仓股数，指境外投资者通过陆股通持有的股票数量
            'ha_shm',       # Northbound Holding Amount: 陆股通持仓金额，指境外资金通过陆股通持有股票的总市值
            'mm_nba',       # Northbound Net Buying: 陆股通净买入金额，反映境外资金净流入的金额
        ]
        
        set2 = [
            'ev_ebit_r',    # EV/EBIT: 企业价值与息税前利润比率，衡量公司整体价值相对于经营利润的高低
            'ev_ebitda_r',  # EV/EBITDA: 企业价值与息税折旧摊销前利润比率，剔除折旧和摊销影响评估盈利能力
            'ey',           # Earnings Yield: 公司收益率，收益与市价的比例，是P/E的倒数，反映盈利回报
        ]
        
        set3 = [
            'pev',          # PEV: 市价与嵌入价值比率，专门用于评估保险公司保单未来盈利潜力
        ]

        self.model_fund = {
            'non_financial': set1 + set2,  # 非金融行业：基本指标加上特定盈利指标，用于评估非金融企业的经营和估值情况
            'bank': set1,  # 银行业：只采用基本指标，因为银行业务结构较为特殊，常用基本财务指标评估
            'insurance': set1 + set3,  # 保险业：在基本指标基础上增加特有的评估指标，用于反映保险公司长期价值
            'security': set1,  # 证券业：采用基本指标评估，反映证券公司在市场上的基本表现和估值
            'other_financial': set1,  # 其他金融业：同样只采用基本指标，以评估金融行业其他子领域的基本状况
        }

        self.link_fund = {
            'non_financial': "https://open.lixinger.com/api/cn/company/fundamental/non_financial",
            'bank': "https://open.lixinger.com/api/cn/company/fundamental/bank",
            'security': "https://open.lixinger.com/api/cn/company/fundamental/security",
            'insurance': "https://open.lixinger.com/api/cn/company/fundamental/insurance",
            'other_financial': "https://open.lixinger.com/api/cn/company/fundamental/other_financial",
        }

        self.all_metrics = set1 + set2 + set3

        CN_C['fundamental'] = {
            "link": None,
            "description": "获取基本面数据",
            "payload": {
                "token": f"{self.LICENSE}",
                "startDate": None,  # "xxxx-xx-xx" within 10 years
                "endDate": None,
                "stockCodes": None,  # only 1 allowed with "startDate"
                "metricsList": None,
            }
        }

    def fill_payload(self, payload, *args):
        """
        Fill the payload dictionary with positional values in order.
        The order is based on the keys as defined in the payload.
        """
        new_payload = {}
        fill_index = 0
        for key, value in payload.items():
            if value is not None:
                new_payload[key] = value
            else:
                new_payload[key] = args[fill_index]
                fill_index += 1
        return new_payload

    def query(self, name: str, *args):
        url = self.API[name]["link"]
        payload = self.fill_payload(self.API[name]["payload"], *args)
        headers = {
            'Content-Type': 'application/json'  # Set content type for JSON request
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            decoded_json = response.json()  # This already parses the JSON
            if decoded_json.get('message') == "success":
                return decoded_json.get('data')  # This should be your data

        raise requests.HTTPError(
            f"Request failed with status {response.status_code}: {response.text}")

    def query_fundamental(self, fsTableType: str, *args):
        """
        fsTableType: 财报类型
        """
        url = self.link_fund[fsTableType]
        payload = self.fill_payload(
            self.API['fundamental']["payload"], *args, self.model_fund[fsTableType])
        headers = {
            'Content-Type': 'application/json'  # Set content type for JSON request
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            decoded_json = response.json()  # This already parses the JSON
            if decoded_json.get('message') == "success":
                # [{'date': '2008-12-31T00:00:00+08:00', 'pe_ttm': 6.5687600867120715, 'pb': 1.9689868436975853, 'pb_wo_gw': 1.9689868436975853, 'ps_ttm': 2.254430967075777, 'd_pe_ttm': 6.437766435590457, 'pcf_ttm': 1.164520073342016, 'dyr': 0.009288824376765103, 'sp': 13.25, 'shn': 228113, 'tv': 25907019, 'ecmc_psh': 227365, 'ecmc': 51864904459, 'cmc': 66411309038, 'mc': 75012854507.5, 'to_r': 0.005168818481644693, 'ta': 345815700, 'spc': -0.004508, 'spa': 0.026615969581748944, 'mc_om': 75012854454.5, 'stockCode': '600000'}]
                return decoded_json.get('data')  # This should be your data
        raise requests.HTTPError(
            f"Request failed with status {response.status_code}: {response.text}")
