import os
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
# Data Preparation for ESG-Based Forecasting
###############################################################################
def load_and_prepare_multi_company_data(ticker_list: list, train_threshold: str, seq_length: int):
    """
    Prepare time-series data for multiple companies using ESG features.

    For each ticker, data is loaded, split into training and testing sets based on a date threshold,
    and the ESG features (all columns except 'date', 'price', and 'symbol') are scaled.
    Sliding window sequences are generated for training and testing.

    Returns:
        X_train (np.array): Training input sequences with shape (num_train_sequences, seq_length, n_features).
        y_train (np.array): Scaled training targets (price).
        X_test (np.array): Testing input sequences.
        y_test (np.array): Testing targets.
        scaler_target (StandardScaler): Scaler fitted on the training price values.
        esg_columns (list): List of ESG feature column names.
    """
    data_loader = DataLoader()
    train_dfs = []
    test_dfs = []
    train_threshold_pd = pd.to_datetime(train_threshold)

    for ticker in ticker_list:
        data = data_loader.load_data(ticker)
        if data is not None:
            data['date'] = pd.to_datetime(data['date'])
            data['symbol'] = ticker

            train_data = data[data['date'] < train_threshold_pd].copy()
            test_data = data[data['date'] >= train_threshold_pd].copy()

            if len(train_data) < seq_length + 1 or len(test_data) < seq_length + 1:
                print(f"Not enough data for ticker {ticker}, skipping.")
                continue

            train_dfs.append(train_data)
            test_dfs.append(test_data)

    if not train_dfs or not test_dfs:
        raise ValueError("Not enough data across tickers for the given sequence length and split date.")

    # Fit scaler on all training data
    train_all = pd.concat(train_dfs, ignore_index=True)
    # ESG columns: all columns except 'date', 'price' and 'symbol'
    esg_columns = [col for col in train_all.columns if col not in ['date', 'price', 'symbol']]

    scaler_features = StandardScaler()
    scaler_target = StandardScaler()
    scaler_features.fit(train_all[esg_columns])
    scaler_target.fit(train_all[['price']])

    def create_scaled_sequences(df: pd.DataFrame, scaler_features, scaler_target, seq_length: int, esg_cols: list):
        features_scaled = scaler_features.transform(df[esg_cols])
        target_scaled = scaler_target.transform(df[['price']]).flatten()
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
# Data Preparation for price-Based (General) Forecasting
###############################################################################
def load_and_prepare_price_data(ticker_list: list, train_threshold: str, seq_length: int):
    """
    Prepare time-series data using only the 'price' column.

    For each ticker, the data is loaded and split into training/testing sets based on a date threshold.
    A global scaler is fit on the training 'price' data and sliding window sequences are generated.

    Returns:
        X_train (np.array): Training input sequences with shape (num_train_sequences, seq_length, 1).
        y_train (np.array): Training targets (scaled price).
        X_test (np.array): Testing input sequences.
        y_test (np.array): Testing targets.
        scaler (StandardScaler): Scaler fitted on price data.
    """
    data_loader = DataLoader()
    train_dfs = []
    test_dfs = []
    train_threshold_pd = pd.to_datetime(train_threshold)

    for ticker in ticker_list:
        data = data_loader.load_data(ticker)
        if data is not None:
            data['date'] = pd.to_datetime(data['date'])
            data['symbol'] = ticker

            train_data = data[data['date'] < train_threshold_pd].copy()
            test_data = data[data['date'] >= train_threshold_pd].copy()

            if len(train_data) < seq_length + 1 or len(test_data) < seq_length + 1:
                print(f"Not enough data for ticker {ticker}, skipping.")
                continue

            train_dfs.append(train_data)
            test_dfs.append(test_data)

    if not train_dfs or not test_dfs:
        raise ValueError("Not enough data across tickers for price-based forecasting.")

    train_all = pd.concat(train_dfs, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(train_all[['price']])

    def create_price_sequences(df, scaler, seq_length):
        prices_scaled = scaler.transform(df[['price']]).flatten()
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
# Data Preparation for Hybrid Model (Combination of price and ESG Features)
###############################################################################
def load_and_prepare_hybrid_data(ticker_list: list, train_threshold: str, seq_length: int):
    """
    Prepare hybrid data, which contains both price and ESG features as input.
    The hybrid features include ['price'] + [all other ESG columns].
    The target remains the scaled price value.
    """
    data_loader = DataLoader()
    train_dfs = []
    test_dfs = []
    train_threshold_pd = pd.to_datetime(train_threshold)

    for ticker in ticker_list:
        data = data_loader.load_data(ticker)
        if data is not None:
            data['date'] = pd.to_datetime(data['date'])
            data['symbol'] = ticker

            train_data = data[data['date'] < train_threshold_pd].copy()
            test_data = data[data['date'] >= train_threshold_pd].copy()

            if len(train_data) < seq_length + 1 or len(test_data) < seq_length + 1:
                print(f"Not enough data for ticker {ticker}, skipping.")
                continue

            train_dfs.append(train_data)
            test_dfs.append(test_data)

    if not train_dfs or not test_dfs:
        raise ValueError("Not enough data across tickers for hybrid forecasting.")

    train_all = pd.concat(train_dfs, ignore_index=True)
    # Hybrid input: price + all ESG columns (excluding 'date', 'symbol', and 'price' for ESG features)
    hybrid_esg = [col for col in train_all.columns if col not in ['date', 'symbol', 'price']]
    hybrid_columns = ['price'] + hybrid_esg

    scaler_hybrid = StandardScaler()
    scaler_target = StandardScaler()
    scaler_hybrid.fit(train_all[hybrid_columns])
    scaler_target.fit(train_all[['price']])

    def create_hybrid_sequences(df: pd.DataFrame, scaler_hybrid, scaler_target, seq_length: int, hybrid_cols: list):
        features_scaled = scaler_hybrid.transform(df[hybrid_cols])
        target_scaled = scaler_target.transform(df[['price']]).flatten()
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
# Model Training and Evaluation Functions
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
    """
    Evaluate the model on the test set and calculate performance metrics.

    Returns:
        mse (float): Mean Squared Error (MSE).
        mae (float): Mean Absolute Error (MAE).
        rmse (float): Root Mean Squared Error (RMSE).
        mape (float): Mean Absolute Percentage Error (MAPE).
        r2 (float): R² Score.
        predictions (np.array): Predicted price (inverse scaled).
        actuals (np.array): Actual price (inverse scaled).
    """
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
    mse = test_loss
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)
    print("\n📊 Evaluation Metrics:")
    print(f"-> MSE: {mse:.4f}")
    print(f"-> MAE: {mae:.4f}")
    print(f"-> RMSE: {rmse:.4f}")
    print(f"-> MAPE: {mape:.2f}%")
    print(f"-> R²: {r2:.4f}")
    return mse, mae, rmse, mape, r2, predictions, actuals


###############################################################################
# Main Function: Comparison of Three Approaches (ESG, price, and Hybrid)
###############################################################################
def main():
    # Create necessary directories if they don't exist
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # -------------------------- Configuration -------------------------------
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    ticker_list = df["Symbol"].tolist()[:5]

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
    # 1. ESG-Based Forecasting Model (Using only ESG data)
    ############################################################################
    print("Training ESG-based forecasting model...")
    X_train_esg, y_train_esg, X_test_esg, y_test_esg, scaler_target_esg, esg_columns = load_and_prepare_multi_company_data(
        ticker_list=ticker_list,
        train_threshold=train_threshold,
        seq_length=seq_length
    )
    if X_train_esg is None or X_test_esg is None:
        raise ValueError("Not enough data for ESG-based model.")

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

    print("Evaluating ESG-based model:")
    mse_esg, mae_esg, rmse_esg, mape_esg, r2_esg, predictions_esg, actuals_esg = evaluate_model(
        model=model_esg,
        test_loader=test_loader_esg,
        criterion=criterion,
        device=device,
        scaler_target=scaler_target_esg
    )

    ############################################################################
    # 2. price-Based Forecasting Model (Using only price data)
    ############################################################################
    print("\nTraining price-based forecasting model...")
    X_train_price, y_train_price, X_test_price, y_test_price, scaler_price = load_and_prepare_price_data(
        ticker_list=ticker_list,
        train_threshold=train_threshold,
        seq_length=seq_length
    )
    if X_train_price is None or X_test_price is None:
        raise ValueError("Not enough data for price-based model.")

    train_dataset_price = TimeSeriesDataset(X_train_price, y_train_price)
    test_dataset_price = TimeSeriesDataset(X_test_price, y_test_price)
    train_loader_price = TorchDataLoader(train_dataset_price, batch_size=batch_size, shuffle=True)
    test_loader_price = TorchDataLoader(test_dataset_price, batch_size=batch_size, shuffle=False)

    input_size_price = X_train_price.shape[2]  # In this case, 1
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

    print("Evaluating price-based model:")
    mse_price, mae_price, rmse_price, mape_price, r2_price, predictions_price, actuals_price = evaluate_model(
        model=model_price,
        test_loader=test_loader_price,
        criterion=criterion,
        device=device,
        scaler_target=scaler_price
    )

    ############################################################################
    # 3. Hybrid Model: Combination of price and ESG data
    ############################################################################
    print("\nTraining Hybrid forecasting model (price + ESG)...")
    X_train_hyb, y_train_hyb, X_test_hyb, y_test_hyb, scaler_target_hyb, hybrid_columns = load_and_prepare_hybrid_data(
        ticker_list=ticker_list,
        train_threshold=train_threshold,
        seq_length=seq_length
    )
    if X_train_hyb is None or X_test_hyb is None:
        raise ValueError("Not enough data for Hybrid model.")

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

    print("Evaluating Hybrid model:")
    mse_hyb, mae_hyb, rmse_hyb, mape_hyb, r2_hyb, predictions_hyb, actuals_hyb = evaluate_model(
        model=model_hyb,
        test_loader=test_loader_hyb,
        criterion=criterion,
        device=device,
        scaler_target=scaler_target_hyb
    )

    ############################################################################
    # Performance Comparison, Plotting, and Saving Outputs
    ############################################################################
    # Plot training losses for comparison
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses_esg, label="ESG-Based Model")
    plt.plot(training_losses_price, label="price-Based Model")
    plt.plot(training_losses_hyb, label="Hybrid Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.savefig("outputs/plots/training_loss_comparison.png")
    plt.show()

    # Plot ESG-based model predictions
    plt.figure(figsize=(10, 5))
    plt.plot(predictions_esg, label="Prediction (ESG-Based)")
    plt.plot(actuals_esg, label="Actual (ESG-Based)")
    plt.xlabel("Time Step")
    plt.ylabel("price")
    plt.title("ESG-Based Forecasting")
    plt.legend()
    plt.savefig("outputs/plots/esg_predictions_vs_actuals.png")
    plt.show()

    # Plot price-based model predictions
    plt.figure(figsize=(10, 5))
    plt.plot(predictions_price, label="Prediction (price-Based)")
    plt.plot(actuals_price, label="Actual (price-Based)")
    plt.xlabel("Time Step")
    plt.ylabel("price")
    plt.title("price-Based Forecasting")
    plt.legend()
    plt.savefig("outputs/plots/price_predictions_vs_actuals.png")
    plt.show()

    # Plot Hybrid model predictions
    plt.figure(figsize=(10, 5))
    plt.plot(predictions_hyb, label="Prediction (Hybrid)")
    plt.plot(actuals_hyb, label="Actual (Hybrid)")
    plt.xlabel("Time Step")
    plt.ylabel("price")
    plt.title("Hybrid Forecasting (price + ESG)")
    plt.legend()
    plt.savefig("outputs/plots/hybrid_predictions_vs_actuals.png")
    plt.show()

    # Save the model comparison results to a CSV file (all metrics rounded to 4 decimals)
    results = {
        "Model": ["ESG-Based", "price-Based", "Hybrid"],
        "MSE": [round(mse_esg, 4), round(mse_price, 4), round(mse_hyb, 4)],
        "MAE": [round(mae_esg, 4), round(mae_price, 4), round(mae_hyb, 4)],
        "RMSE": [round(rmse_esg, 4), round(rmse_price, 4), round(rmse_hyb, 4)],
        "MAPE (%)": [round(mape_esg, 4), round(mape_price, 4), round(mape_hyb, 4)],
        "R^2": [round(r2_esg, 4), round(r2_price, 4), round(r2_hyb, 4)]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/result.csv", index=False)
    print("\n------------------------------")
    print("Model Comparison:")
    print(results_df)
    print("------------------------------")


if __name__ == "__main__":
    main()
