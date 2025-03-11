import json
import requests
from typing import List, Dict, Tuple, Optional

from ..License import LIXINGREN_LICENSE


class LixingrenAPI:
    """
    A class to encapsulate Mairui APIs related to stock trading and data.
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

    def query(self, name: str, inputs: List[str] = []):
        url = self.API[name]["link"]
        payload = self.API[name]["payload"]
        headers = {
            'Content-Type': 'application/json'  # Set content type for JSON request
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            decoded_json = response.json()  # This already parses the JSON
            if decoded_json.get('message') == "success":
                return decoded_json.get('data')  # This should be your data

        assert False, f"Error: {response.status_code} - {response.text}"
