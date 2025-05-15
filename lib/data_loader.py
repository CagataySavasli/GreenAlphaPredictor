from lib.loaders import FinLoader, ESGLoader
import pandas as pd


class DataLoader:
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
