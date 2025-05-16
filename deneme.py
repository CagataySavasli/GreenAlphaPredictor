import torch
from torch.utils.data import Dataset, DataLoader

from lib import DataDownLoader, PriceForecastLSTM, TimeSeriesDataset
from lib.utils import (
                        get_sp500_tickers,
                        train_model,
                        evaluate_model
                    )
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Configurations and constants
ticker_list = get_sp500_tickers()

train_threshold = "2018-12-01"
dict_feature_cols = {
    'price': ['price'],
    'esg': ['esgScore', 'governanceScore', 'environmentScore', 'socialScore'],
    'hybrid': ['price', 'esgScore', 'governanceScore', 'environmentScore', 'socialScore']
}
target_col = 'price'
window_size = 11
horizon = 1  # 1 gün sonrası tahmin

data_loader = DataDownLoader()

ticker_list = ticker_list[0:50]

data = data_loader.load_data_all(ticker_list)

data_train = data[data['date'] < train_threshold]
data_test = data[data['date'] >= train_threshold]
data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)

dict_results = {
    'price': {'mse': [], 'r2': []},
    'esg': {'mse': [], 'r2': []},
    'hybrid': {'mse': [], 'r2': []}
}
for feature_set, feature_cols in dict_feature_cols.items():
    print(f"Training with feature set: {feature_set}")
    dataset_train = TimeSeriesDataset(
        df=data_train,
        feature_cols=feature_cols,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon
    )

    dataset_test = TimeSeriesDataset(
        df=data_test,
        feature_cols=feature_cols,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon
    )

    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)

    model = PriceForecastLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model = train_model(model, train_loader, criterion, optimizer, num_epochs=200)
    y_predictions, y_reals = evaluate_model(model, test_loader)

    mse = mean_squared_error(y_reals, y_predictions)
    r2 = r2_score(y_reals, y_predictions)

    dict_results[feature_set]['mse'].append(mse)
    dict_results[feature_set]['r2'].append(r2)

print("Results:")
print(dict_results)


