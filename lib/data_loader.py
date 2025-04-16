from typing import Any

import yesg as ys
import yfinance as yf
import pandas as pd
from pandas import Series


class DataLoader:
    def load_data(self, ticker:str) -> pd.DataFrame | None:
        data_esg = ys.get_historic_esg(ticker)

        if not data_esg is None:
            data_price = yf.download(ticker, start=data_esg.index[0], end=data_esg.index[-1], interval='1mo')

            data_esg.drop(columns=['Total-Score'], inplace=True)

            data_price = data_price.reset_index()
            data_esg = data_esg.reset_index()

            data_price = data_price.loc[:, ['Close']]
            data_esg = data_esg.iloc[:-1]

            data = pd.concat([data_price, data_esg], axis=1)
            data = data.rename(columns={('Close', ticker): 'Price'})

            data = data[data['Date'] < '2019-12-01']

            if data.isnull().sum().sum() == 0:
                return data
        return None
