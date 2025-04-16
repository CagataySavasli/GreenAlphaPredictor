import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lib import DataLoader, PriceForecastLSTM, TimeSeriesDataset


###############################################################################
# ESG Bazlı Forecasting için Veri Hazırlama (Mevcut)
###############################################################################
def load_and_prepare_multi_company_data(ticker_list: list, train_threshold: str, seq_length: int):
    data_loader = DataLoader()
    train_dfs = []
    test_dfs = []
    train_threshold_pd = pd.to_datetime(train_threshold)

    for ticker in ticker_list:
        data = data_loader.load_data(ticker)
        if data is not None:
            data['Date'] = pd.to_datetime(data['Date'])
            data['Ticker'] = ticker

            train_data = data[data['Date'] < train_threshold_pd].copy()
            test_data = data[data['Date'] >= train_threshold_pd].copy()

            if len(train_data) < seq_length + 1 or len(test_data) < seq_length + 1:
                print(f"Not enough data for ticker {ticker}, skipping.")
                continue

            train_dfs.append(train_data)
            test_dfs.append(test_data)

    if not train_dfs or not test_dfs:
        raise ValueError("Not enough data across tickers for the given sequence length and split date.")

    # Tüm eğitim verileri üzerinden scaler fit edilir
    train_all = pd.concat(train_dfs, ignore_index=True)

    # ESG sütunları: 'Date', 'Price' ve 'Ticker' dışındaki sütunlar
    esg_columns = [col for col in train_all.columns if col not in ['Date', 'Price', 'Ticker']]

    scaler_features = StandardScaler()
    scaler_target = StandardScaler()
    scaler_features.fit(train_all[esg_columns])
    scaler_target.fit(train_all[['Price']])

    def create_scaled_sequences(df: pd.DataFrame, scaler_features, scaler_target, seq_length: int, esg_cols: list):
        features_scaled = scaler_features.transform(df[esg_cols])
        target_scaled = scaler_target.transform(df[['Price']]).flatten()
        X, y = [], []
        for i in range(len(features_scaled) - seq_length):
            X.append(features_scaled[i: i + seq_length])
            y.append(target_scaled[i + seq_length])
        return np.array(X), np.array(y)

    X_train_list, y_train_list = [], []
    for df in train_dfs:
        if len(df) >= seq_length + 1:
            X_seq, y_seq = create_scaled_sequences(df, scaler_features, scaler_target, seq_length, esg_columns)
            X_train_list.append(X_seq)
            y_train_list.append(y_seq)

    X_test_list, y_test_list = [], []
    for df in test_dfs:
        if len(df) >= seq_length + 1:
            X_seq, y_seq = create_scaled_sequences(df, scaler_features, scaler_target, seq_length, esg_columns)
            X_test_list.append(X_seq)
            y_test_list.append(y_seq)

    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else None
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else None
    X_test = np.concatenate(X_test_list, axis=0) if X_test_list else None
    y_test = np.concatenate(y_test_list, axis=0) if y_test_list else None

    return X_train, y_train, X_test, y_test, scaler_target, esg_columns


###############################################################################
# Fiyat Bazlı (Genel) Forecasting için Veri Hazırlama
###############################################################################
def load_and_prepare_price_data(ticker_list: list, train_threshold: str, seq_length: int):
    data_loader = DataLoader()
    train_dfs = []
    test_dfs = []
    train_threshold_pd = pd.to_datetime(train_threshold)

    for ticker in ticker_list:
        data = data_loader.load_data(ticker)
        if data is not None:
            data['Date'] = pd.to_datetime(data['Date'])
            data['Ticker'] = ticker

            train_data = data[data['Date'] < train_threshold_pd].copy()
            test_data = data[data['Date'] >= train_threshold_pd].copy()

            if len(train_data) < seq_length + 1 or len(test_data) < seq_length + 1:
                print(f"Not enough data for ticker {ticker}, skipping.")
                continue

            train_dfs.append(train_data)
            test_dfs.append(test_data)

    if not train_dfs or not test_dfs:
        raise ValueError("Not enough data across tickers for Price-based forecasting.")

    train_all = pd.concat(train_dfs, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(train_all[['Price']])

    def create_price_sequences(df, scaler, seq_length):
        prices_scaled = scaler.transform(df[['Price']]).flatten()
        X, y = [], []
        for i in range(len(prices_scaled) - seq_length):
            X.append(prices_scaled[i: i + seq_length])
            y.append(prices_scaled[i + seq_length])
        return np.array(X), np.array(y)

    X_train_list, y_train_list = [], []
    for df in train_dfs:
        if len(df) >= seq_length + 1:
            X_seq, y_seq = create_price_sequences(df, scaler, seq_length)
            X_seq = X_seq.reshape(-1, seq_length, 1)
            X_train_list.append(X_seq)
            y_train_list.append(y_seq)

    X_test_list, y_test_list = [], []
    for df in test_dfs:
        if len(df) >= seq_length + 1:
            X_seq, y_seq = create_price_sequences(df, scaler, seq_length)
            X_seq = X_seq.reshape(-1, seq_length, 1)
            X_test_list.append(X_seq)
            y_test_list.append(y_seq)

    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else None
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else None
    X_test = np.concatenate(X_test_list, axis=0) if X_test_list else None
    y_test = np.concatenate(y_test_list, axis=0) if y_test_list else None

    return X_train, y_train, X_test, y_test, scaler


###############################################################################
# Hibrit Model için Veri Hazırlama: Fiyat ve ESG özelliklerinin birleşimi
###############################################################################
def load_and_prepare_hybrid_data(ticker_list: list, train_threshold: str, seq_length: int):
    """
    Hibrit veri hazırlama; girdiler olarak hem Price hem de ESG özelliklerini içerir.
    Hybrid features: ['Price'] + [diğer ESG sütunları]
    Hedef, ölçeklendirilmiş Price değeridir.
    """
    data_loader = DataLoader()
    train_dfs = []
    test_dfs = []
    train_threshold_pd = pd.to_datetime(train_threshold)

    for ticker in ticker_list:
        data = data_loader.load_data(ticker)
        if data is not None:
            data['Date'] = pd.to_datetime(data['Date'])
            data['Ticker'] = ticker

            train_data = data[data['Date'] < train_threshold_pd].copy()
            test_data = data[data['Date'] >= train_threshold_pd].copy()

            if len(train_data) < seq_length + 1 or len(test_data) < seq_length + 1:
                print(f"Not enough data for ticker {ticker}, skipping.")
                continue

            train_dfs.append(train_data)
            test_dfs.append(test_data)

    if not train_dfs or not test_dfs:
        raise ValueError("Not enough data across tickers for hybrid forecasting.")

    train_all = pd.concat(train_dfs, ignore_index=True)
    # Hibrit girdi olarak: Price sütunu + ESG sütunları (Date, Price, Ticker hariç)
    hybrid_esg = [col for col in train_all.columns if col not in ['Date', 'Ticker', 'Price']]
    hybrid_columns = ['Price'] + hybrid_esg

    scaler_hybrid = StandardScaler()
    scaler_target = StandardScaler()
    scaler_hybrid.fit(train_all[hybrid_columns])
    scaler_target.fit(train_all[['Price']])

    def create_hybrid_sequences(df: pd.DataFrame, scaler_hybrid, scaler_target, seq_length: int, hybrid_cols: list):
        features_scaled = scaler_hybrid.transform(df[hybrid_cols])
        target_scaled = scaler_target.transform(df[['Price']]).flatten()
        X, y = [], []
        for i in range(len(features_scaled) - seq_length):
            X.append(features_scaled[i: i + seq_length])
            y.append(target_scaled[i + seq_length])
        return np.array(X), np.array(y)

    X_train_list, y_train_list = [], []
    for df in train_dfs:
        if len(df) >= seq_length + 1:
            X_seq, y_seq = create_hybrid_sequences(df, scaler_hybrid, scaler_target, seq_length, hybrid_columns)
            X_train_list.append(X_seq)
            y_train_list.append(y_seq)

    X_test_list, y_test_list = [], []
    for df in test_dfs:
        if len(df) >= seq_length + 1:
            X_seq, y_seq = create_hybrid_sequences(df, scaler_hybrid, scaler_target, seq_length, hybrid_columns)
            X_test_list.append(X_seq)
            y_test_list.append(y_seq)

    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else None
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else None
    X_test = np.concatenate(X_test_list, axis=0) if X_test_list else None
    y_test = np.concatenate(y_test_list, axis=0) if y_test_list else None

    return X_train, y_train, X_test, y_test, scaler_target, hybrid_columns


###############################################################################
# Model Eğitimi ve Değerlendirme Fonksiyonları (Orijinal)
###############################################################################
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100, scheduler=None):
    model.to(device)
    training_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * sequences.size(0)
        epoch_loss /= len(train_loader.dataset)
        training_losses.append(epoch_loss)
        if scheduler is not None:
            scheduler.step(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")
    return model, training_losses


def evaluate_model(model, test_loader, criterion, device, scaler_target):
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            targets = targets.to(device).view(-1, 1)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * sequences.size(0)
            preds_inv = scaler_target.inverse_transform(outputs.cpu().numpy())
            acts_inv = scaler_target.inverse_transform(targets.cpu().numpy())
            predictions.extend(preds_inv.flatten())
            actuals.extend(acts_inv.flatten())
    test_loss /= len(test_loader.dataset)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)
    print("\n📊 Evaluation Metrics:")
    print(f"➡️ MSE: {test_loss:.4f}")
    print(f"➡️ MAE: {mae:.4f}")
    print(f"➡️ RMSE: {rmse:.4f}")
    print(f"➡️ MAPE: {mape:.2f}%")
    print(f"➡️ R²: {r2:.4f}")
    return test_loss, predictions, actuals


###############################################################################
# Main Fonksiyonu: Üç Yaklaşımın (ESG, Fiyat, Hibrit) Karşılaştırılması
###############################################################################
def main():
    # -------------------------- Konfigürasyon -------------------------------
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    ticker_list = df["Symbol"].tolist()[:10]

    train_threshold = "2018-12-01"
    seq_length = 5

    num_epochs = 100
    batch_size = 16
    learning_rate = 0.001
    hidden_size = 100
    num_layers = 2
    dropout_rate = 0.2
    # ------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    ############################################################################
    # 1. ESG Bazlı Forecasting Modeli (Sadece ESG datası kullanılarak)
    ############################################################################
    print("ESG bazlı forecasting modeli eğitiliyor...")
    X_train_esg, y_train_esg, X_test_esg, y_test_esg, scaler_target_esg, esg_columns = load_and_prepare_multi_company_data(
        ticker_list=ticker_list,
        train_threshold=train_threshold,
        seq_length=seq_length
    )
    if X_train_esg is None or X_test_esg is None:
        raise ValueError("ESG bazlı model için yeterli veri bulunamadı.")

    train_dataset_esg = TimeSeriesDataset(X_train_esg, y_train_esg)
    test_dataset_esg = TimeSeriesDataset(X_test_esg, y_test_esg)
    train_loader_esg = TorchDataLoader(train_dataset_esg, batch_size=batch_size, shuffle=True)
    test_loader_esg = TorchDataLoader(test_dataset_esg, batch_size=batch_size, shuffle=False)

    input_size_esg = X_train_esg.shape[2]
    model_esg = PriceForecastLSTM(input_size_esg, hidden_size, num_layers, dropout=dropout_rate)
    criterion = nn.MSELoss()
    optimizer_esg = optim.Adam(model_esg.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler_esg = optim.lr_scheduler.ReduceLROnPlateau(optimizer_esg, mode='min', factor=0.5, patience=10, verbose=True)

    model_esg, training_losses_esg = train_model(
        model=model_esg,
        train_loader=train_loader_esg,
        criterion=criterion,
        optimizer=optimizer_esg,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler_esg
    )

    print("ESG bazlı model değerlendirmesi:")
    test_loss_esg, predictions_esg, actuals_esg = evaluate_model(
        model=model_esg,
        test_loader=test_loader_esg,
        criterion=criterion,
        device=device,
        scaler_target=scaler_target_esg
    )


    ############################################################################
    # 2. Fiyat Bazlı (Genel) Forecasting Modeli (Sadece Price datası kullanılarak)
    ############################################################################
    print("\nFiyat bazlı (genel) forecasting modeli eğitiliyor...")
    X_train_price, y_train_price, X_test_price, y_test_price, scaler_price = load_and_prepare_price_data(
        ticker_list=ticker_list,
        train_threshold=train_threshold,
        seq_length=seq_length
    )
    if X_train_price is None or X_test_price is None:
        raise ValueError("Fiyat bazlı model için yeterli veri bulunamadı.")

    train_dataset_price = TimeSeriesDataset(X_train_price, y_train_price)
    test_dataset_price = TimeSeriesDataset(X_test_price, y_test_price)
    train_loader_price = TorchDataLoader(train_dataset_price, batch_size=batch_size, shuffle=True)
    test_loader_price = TorchDataLoader(test_dataset_price, batch_size=batch_size, shuffle=False)

    input_size_price = X_train_price.shape[2]  # Bu durumda 1
    model_price = PriceForecastLSTM(input_size_price, hidden_size, num_layers, dropout=dropout_rate)
    optimizer_price = optim.Adam(model_price.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler_price = optim.lr_scheduler.ReduceLROnPlateau(optimizer_price, mode='min', factor=0.5, patience=10, verbose=True)

    model_price, training_losses_price = train_model(
        model=model_price,
        train_loader=train_loader_price,
        criterion=criterion,
        optimizer=optimizer_price,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler_price
    )

    print("Fiyat bazlı model değerlendirmesi:")
    test_loss_price, predictions_price, actuals_price = evaluate_model(
        model=model_price,
        test_loader=test_loader_price,
        criterion=criterion,
        device=device,
        scaler_target=scaler_price
    )


    ############################################################################
    # 3. Hibrit Model: Hem Price hem de ESG verilerinin birleşimi
    ############################################################################
    print("\nHibrit forecasting modeli (Price + ESG) eğitiliyor...")
    X_train_hyb, y_train_hyb, X_test_hyb, y_test_hyb, scaler_target_hyb, hybrid_columns = load_and_prepare_hybrid_data(
        ticker_list=ticker_list,
        train_threshold=train_threshold,
        seq_length=seq_length
    )
    if X_train_hyb is None or X_test_hyb is None:
        raise ValueError("Hibrit model için yeterli veri bulunamadı.")

    train_dataset_hyb = TimeSeriesDataset(X_train_hyb, y_train_hyb)
    test_dataset_hyb = TimeSeriesDataset(X_test_hyb, y_test_hyb)
    train_loader_hyb = TorchDataLoader(train_dataset_hyb, batch_size=batch_size, shuffle=True)
    test_loader_hyb = TorchDataLoader(test_dataset_hyb, batch_size=batch_size, shuffle=False)

    input_size_hyb = X_train_hyb.shape[2]
    model_hyb = PriceForecastLSTM(input_size_hyb, hidden_size, num_layers, dropout=dropout_rate)
    optimizer_hyb = optim.Adam(model_hyb.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler_hyb = optim.lr_scheduler.ReduceLROnPlateau(optimizer_hyb, mode='min', factor=0.5, patience=10, verbose=True)

    model_hyb, training_losses_hyb = train_model(
        model=model_hyb,
        train_loader=train_loader_hyb,
        criterion=criterion,
        optimizer=optimizer_hyb,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler_hyb
    )

    print("Hibrit model değerlendirmesi:")
    test_loss_hyb, predictions_hyb, actuals_hyb = evaluate_model(
        model=model_hyb,
        test_loader=test_loader_hyb,
        criterion=criterion,
        device=device,
        scaler_target=scaler_target_hyb
    )


    ############################################################################
    # Performans Karşılaştırma ve Grafikler
    ############################################################################
    # Eğitim kayıplarını karşılaştırma grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses_esg, label="ESG Bazlı Model")
    plt.plot(training_losses_price, label="Fiyat Bazlı Model")
    plt.plot(training_losses_hyb, label="Hibrit Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Eğitim Kaybı Karşılaştırması")
    plt.legend()
    plt.savefig("plots/training_loss_comparison.png")
    plt.show()

    # ESG bazlı model sonuçları
    plt.figure(figsize=(10, 5))
    plt.plot(predictions_esg, label="Tahmin (ESG Bazlı)")
    plt.plot(actuals_esg, label="Gerçek (ESG Bazlı)")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.title("ESG Bazlı Forecasting")
    plt.legend()
    plt.savefig("plots/esg_predictions_vs_actuals.png")
    plt.show()

    # Fiyat bazlı model sonuçları
    plt.figure(figsize=(10, 5))
    plt.plot(predictions_price, label="Tahmin (Fiyat Bazlı)")
    plt.plot(actuals_price, label="Gerçek (Fiyat Bazlı)")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.title("Fiyat Bazlı Forecasting")
    plt.legend()
    plt.savefig("plots/price_predictions_vs_actuals.png")
    plt.show()

    # Hibrit model sonuçları
    plt.figure(figsize=(10, 5))
    plt.plot(predictions_hyb, label="Tahmin (Hibrit)")
    plt.plot(actuals_hyb, label="Gerçek (Hibrit)")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.title("Hibrit Forecasting (Price + ESG)")
    plt.legend()
    plt.savefig("plots/hybrid_predictions_vs_actuals.png")
    plt.show()

    print("\n------------------------------")
    print("Modellerin Karşılaştırılması:")
    print(f"ESG Bazlı Model MSE: {test_loss_esg:.4f}")
    print(f"Fiyat Bazlı Model MSE: {test_loss_price:.4f}")
    print(f"Hibrit Model MSE: {test_loss_hyb:.4f}")
    print("------------------------------")


if __name__ == "__main__":
    main()
