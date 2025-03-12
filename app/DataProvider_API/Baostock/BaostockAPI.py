import json
import requests
import baostock as bs
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional

from tqdm import tqdm


class BaostockAPI:
    """
    A class to encapsulate Baostock APIs related to stock trading and data.
    """

    def _log_in(self):
        # 登陆系统
        lg = bs.login()
        # 显示登陆返回信息
        assert lg.error_code == '0', f"lg.error_code: {lg.error_code}"
        assert lg.error_msg == 'success', f"lg.error_msg: {lg.error_msg}"

    def _log_out(self):
        # 登出系统
        bs.logout()

    def query_adjust_factor(self, codes, start_date, end_date):
        self._log_in()

        adj_factors = {}
        for code in tqdm(codes):
            rs_list = []
            rs = bs.query_adjust_factor(code, start_date, end_date)
            while (rs.error_code == '0') & rs.next():
                rs_list.append(rs.get_row_data())

            key = code.split('.')[1]
            adj_factors[key] = []
            for r in rs_list:
                adj_factors[key].append({
                    "date": int(r[1].replace('-', '')),
                    "factor": float(r[4]),
                })
                
        self._log_out()
        return adj_factors
