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
# Data Loading and Preparation for Multiple Companies
###############################################################################

def load_and_prepare_multi_company_data(ticker_list: list, train_threshold: str, seq_length: int):
    """
    Load and prepare time-series data for multiple companies.

    For each company (ticker) data is loaded using the provided DataLoader, then split
    into train and test parts based on a date threshold. Later, a global scaler is fitted
    on all training data (ESG features and Price) and used to transform the raw data.
    Finally, sliding window sequences are created for each company's train and test data
    and concatenated.

    Args:
        ticker_list (list): List of S&P500 ticker symbols.
        train_threshold (str): Date string (e.g., '2019-12-01') used to separate train/test.
        seq_length (int): Number of time steps per sequence.

    Returns:
        X_train (np.array): 3D array of shape (num_train_sequences, seq_length, n_features).
        y_train (np.array): 1D array of training targets.
        X_test (np.array): 3D array of shape (num_test_sequences, seq_length, n_features).
        y_test (np.array): 1D array of test targets.
        scaler_target (StandardScaler): Scaler fitted on training prices (for inverse transform).
        esg_columns (list): List of ESG feature column names.
    """
    data_loader = DataLoader()
    train_dfs = []
    test_dfs = []

    train_threshold_pd = pd.to_datetime(train_threshold)

    # Loop over each ticker and split data by date threshold
    for ticker in ticker_list:
        data = data_loader.load_data(ticker)
        if not data is None:
            # Ensure "Date" column is datetime
            data['Date'] = pd.to_datetime(data['Date'])
            # Add a column to keep track of which company the row belongs to (optional)
            data['Ticker'] = ticker

            # NOTE: Ensure that the DataLoader does not already filter by date.
            train_data = data[data['Date'] < train_threshold_pd].copy()
            test_data = data[data['Date'] >= train_threshold_pd].copy()

            if len(train_data) < seq_length + 1 or len(test_data) < seq_length + 1:
                print(f"Not enough data for ticker {ticker}, skipping.")
                continue

            train_dfs.append(train_data)
            test_dfs.append(test_data)

    if not train_dfs or not test_dfs:
        raise ValueError("Not enough data across tickers for the given sequence length and split date.")

    # Combine all training data to fit the scalers.
    train_all = pd.concat(train_dfs, ignore_index=True)

    # Identify ESG feature columns
    # (Assuming columns other than 'Date', 'Price' and 'Ticker' are ESG features)
    esg_columns = [col for col in train_all.columns if col not in ['Date', 'Price', 'Ticker']]

    # Fit scalers on training data
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()
    scaler_features.fit(train_all[esg_columns])
    scaler_target.fit(train_all[['Price']])

    # Function to create sequences from a single company DataFrame
    def create_scaled_sequences(df: pd.DataFrame, scaler_features, scaler_target, seq_length: int, esg_cols: list):
        # Scale features and target of the dataframe
        features_scaled = scaler_features.transform(df[esg_cols])
        target_scaled = scaler_target.transform(df[['Price']]).flatten()
        X, y = [], []
        # Create sliding window sequences (do not cross company boundaries)
        for i in range(len(features_scaled) - seq_length):
            X.append(features_scaled[i: i + seq_length])
            y.append(target_scaled[i + seq_length])
        return np.array(X), np.array(y)

    # Process training data for each company
    X_train_list, y_train_list = [], []
    for df in train_dfs:
        if len(df) >= seq_length + 1:
            X_seq, y_seq = create_scaled_sequences(df, scaler_features, scaler_target, seq_length, esg_columns)
            X_train_list.append(X_seq)
            y_train_list.append(y_seq)

    # Process test data for each company
    X_test_list, y_test_list = [], []
    for df in test_dfs:
        if len(df) >= seq_length + 1:
            X_seq, y_seq = create_scaled_sequences(df, scaler_features, scaler_target, seq_length, esg_columns)
            X_test_list.append(X_seq)
            y_test_list.append(y_seq)

    # Concatenate sequences from all companies
    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else None
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else None
    X_test = np.concatenate(X_test_list, axis=0) if X_test_list else None
    y_test = np.concatenate(y_test_list, axis=0) if y_test_list else None

    return X_train, y_train, X_test, y_test, scaler_target, esg_columns


###############################################################################
# Training and Evaluation Functions
###############################################################################

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100, scheduler=None):
    """
    Train the LSTM model.
    """
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
            # Gradient clipping helps stabilize training in RNN/LSTM
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
    Evaluate the trained model on the test set with extended academic metrics.

    Returns:
        test_loss (float): MSE Loss.
        predictions (np.array): Predicted prices.
        actuals (np.array): Ground-truth prices.
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

            # Inverse-transform predictions & targets to the original scale
            preds_inv = scaler_target.inverse_transform(outputs.cpu().numpy())
            acts_inv = scaler_target.inverse_transform(targets.cpu().numpy())

            predictions.extend(preds_inv.flatten())
            actuals.extend(acts_inv.flatten())

    test_loss /= len(test_loader.dataset)
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Additional metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)

    # Print all metrics
    print("\n📊 Evaluation Metrics:")
    print(f"➡️ Mean Squared Error (MSE): {test_loss:.4f}")
    print(f"➡️ Mean Absolute Error (MAE): {mae:.4f}")
    print(f"➡️ Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"➡️ Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"➡️ R² Score: {r2:.4f}")

    return test_loss, predictions, actuals


###############################################################################
# Main Execution Function
###############################################################################

def main():
    # -------------------------- Configuration -------------------------------
    # Define S&P500 tickers to use.
    # Extend this list as needed.
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    ticker_list = df["Symbol"].tolist()

    # ticker_list = ["A", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    train_threshold = "2018-12-01"  # Data before this date for training, afterwards for testing.
    seq_length = 5  # Number of time steps per input sequence.

    num_epochs = 100
    batch_size = 16
    learning_rate = 0.001
    hidden_size = 100
    num_layers = 2
    dropout_rate = 0.2
    # ------------------------------------------------------------------------

    # Device selection: GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # ------------------------ Data Preparation ------------------------------
    X_train, y_train, X_test, y_test, scaler_target, esg_columns = load_and_prepare_multi_company_data(
        ticker_list=ticker_list,
        train_threshold=train_threshold,
        seq_length=seq_length
    )

    if X_train is None or X_test is None:
        raise ValueError("No valid training/test data. Adjust the ticker list or sequence length.")

    # Create PyTorch Datasets and DataLoaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------- Model, Loss Function, Optimizer -------------------
    input_size = X_train.shape[2]  # Number of ESG features
    model = PriceForecastLSTM(input_size, hidden_size, num_layers, dropout=dropout_rate)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler: reduce LR on plateau of training loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # ------------------------- Training Phase --------------------------------
    print("Starting training...")
    model, training_losses = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler
    )

    # ------------------------- Evaluation Phase ------------------------------
    test_loss, predictions, actuals = evaluate_model(model, test_loader, criterion, device, scaler_target)
    print(f"Test Loss: {test_loss:.6f}")

    # ---------------------------- Performance Plots --------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(training_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()
    plt.savefig("plots/train_loss.png")

    plt.figure(figsize=(8, 4))
    plt.plot(predictions, label="Predicted Price")
    plt.plot(actuals, label="Actual Price")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.title("Price Forecasting: Predictions vs. Actual Prices")
    plt.legend()
    plt.show()
    plt.savefig("plots/predicted_price.png")


if __name__ == "__main__":
    main()
