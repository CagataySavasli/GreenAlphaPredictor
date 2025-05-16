import requests
import pandas as pd
from tqdm import tqdm
import time

class ESGLoader():
    def __init__(self):
        self.session = requests.Session()

    def load_data(self, symbol: str) -> pd.DataFrame:
        url = "https://query2.finance.yahoo.com/v1/finance/esgChart"
        params = {"symbol": symbol}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"https://finance.yahoo.com/quote/{symbol}/",
            "Connection": "keep-alive",
            "X-Requested-With": "XMLHttpRequest",
        }

        resp = self.session.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        data = data["esgChart"]
        data = data["result"][0]
        try:
            # peer_group = data["peerGroup"]
            symbol_series = data["symbolSeries"]

            result = pd.DataFrame(symbol_series)
            result['symbol'] = symbol
            # result['peerGroup'] = peer_group
            result['timestamp'] = pd.to_datetime(result['timestamp'], unit='s')
            result.rename(columns={"timestamp": "date"}, inplace=True)

            result = result.iloc[:-1]
        except KeyError:
            tqdm.write(f"KeyError: {symbol} için veri bulunamadı.")
            return None
        return result