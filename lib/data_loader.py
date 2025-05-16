from lib.loaders import FinLoader, ESGLoader
import pandas as pd
from tqdm import trange

class DataDownLoader:
    def __init__(self):
        self.esg = ESGLoader()
        self.price = FinLoader()

    def load_data(self, ticker:str) -> pd.DataFrame | None:
        data_esg = self.esg.load_data(ticker)

        if not data_esg is None:
            start_date = data_esg['date'].min()
            end_date = data_esg['date'].max()
            data_price = self.price.load_data(ticker, start_date, end_date)

            data_price = data_price.reset_index(drop=True)
            data_esg = data_esg.reset_index(drop=True)

            data = pd.concat([data_price, data_esg], axis=1)

            data = data[data['date'] < '2019-12-01']

            if data.isnull().sum().sum() == 0:
                return data
        return None

    def load_data_all(self, tickers: list[str]) -> pd.DataFrame:
        data_frames = []
        progress_bar = trange(len(tickers), desc="Loading data", unit="ticker")
        for idx in progress_bar:
            ticker = tickers[idx]
            data = self.load_data(ticker)
            if data is not None:
                data_frames.append(data)

        if len(data_frames) > 0:
            return pd.concat(data_frames, ignore_index=True)
        else:
            return None