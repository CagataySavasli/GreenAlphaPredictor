import yfinance as yf
import pandas as pd


class FinLoader:
    def __init__(self):
        pass

    def load_data(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        data = yf.download(symbol, start=start_date, end=end_date, interval='1mo')
        data = data.reset_index()
        data.columns = data.columns.get_level_values('Price')

        data = data.loc[:, ['Close']]
        data = data.rename(columns={'Close': 'price'})

        return data