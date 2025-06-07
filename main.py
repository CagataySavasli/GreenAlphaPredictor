import torch
from torch.utils.data import DataLoader
from lib import DataDownLoader, PriceForecastLSTM, TimeSeriesDataset
from lib.models.price_forecase_transformer import PriceForecastTransformer
from lib.utils import (
    get_sp500_tickers,
    train_model,
    evaluate_model,
    print_results_table
)
from sklearn.metrics import mean_squared_error, r2_score

# Ayarlar
ticker_list = get_sp500_tickers()
train_threshold = "2018-12-01"
dict_feature_cols = {
    'price': ['price'],
    'esg': ['esgScore', 'governanceScore', 'environmentScore', 'socialScore'],
    'hybrid': ['price', 'esgScore', 'governanceScore', 'environmentScore', 'socialScore']
}
target_col = 'price'
window_size = 11
horizon = 1

# Veri indirme
data_loader = DataDownLoader()
data = data_loader.load_data_all(ticker_list)

dict_results = {
    'lstm':   {k: {'mse': [], 'r2': [], 'lr': []} for k in dict_feature_cols},
    'transformer': {k: {'mse': [], 'r2': [], 'lr': []} for k in dict_feature_cols}
}

# Modelleri döngü ile çalıştır
for model_approach in ['lstm', 'transformer']:
    for feature_set, feature_cols in dict_feature_cols.items():
        for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
            print(f"[{model_approach.upper()}] Feature set: {feature_set} | Learning rate: {lr}")

            # Train ve test dataset’leri: pencere seviyesinde tarih eşiğine göre ayrılıyor
            dataset_train = TimeSeriesDataset(
                df=data,
                feature_cols=feature_cols,
                target_col=target_col,
                window_size=window_size,
                horizon=horizon,
                date_split=train_threshold,
                split='train'
            )
            dataset_test = TimeSeriesDataset(
                df=data,
                feature_cols=feature_cols,
                target_col=target_col,
                window_size=window_size,
                horizon=horizon,
                date_split=train_threshold,
                split='test'
            )

            train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
            test_loader  = DataLoader(dataset_test, batch_size=1, shuffle=False)

            # Model seçimi
            if model_approach == 'lstm':
                model = PriceForecastLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2)
            else:
                model = PriceForecastTransformer(input_size=len(feature_cols))

            # Eğitim ve değerlendirme
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model = train_model(model, train_loader, criterion, optimizer, num_epochs=200)
            y_pred, y_true = evaluate_model(model, test_loader)

            # Metriği hesapla
            mse = mean_squared_error(y_true, y_pred)
            r2  = r2_score(y_true, y_pred)
            dict_results[model_approach][feature_set]['mse'].append(mse)
            dict_results[model_approach][feature_set]['r2'].append(r2)
            dict_results[model_approach][feature_set]['lr'].append(lr)

# Sonuçları yazdır
print("Results:", dict_results)
results_table = print_results_table(dict_results)
results_table.to_csv("outputs/results.csv", index=False)
