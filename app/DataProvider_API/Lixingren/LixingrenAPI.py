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
                "token": f"{self.LICENSE}"
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
                print(args[fill_index])
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
