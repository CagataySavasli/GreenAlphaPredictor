import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

def get_sp500_tickers():
    """
    S&P 500 endeksindeki şirketlerin hisse senedi kodlarını döndürür.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    ticker_list = df["Symbol"].tolist()
    return ticker_list


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Modeli eğitmek için kullanılan fonksiyon.

    Args:
        model: Eğitim yapılacak model.
        train_loader: Eğitim verilerini yükleyen DataLoader.
        criterion: Kayıp fonksiyonu.
        optimizer: Optimizasyon algoritması.
        num_epochs: Eğitim döngüsü sayısı.
    """
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            outputs = outputs.squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    return model


def evaluate_model(model, test_loader):
    """
    Modeli test verileri üzerinde değerlendirmek için kullanılan fonksiyon.

    Args:
        model: Değerlendirilecek model.
        test_loader: Test verilerini yükleyen DataLoader.
    """
    model.eval()
    predictions = []
    labels = []
    print("Evaluating model...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            outputs = outputs.squeeze()
            predictions.append(outputs.numpy())
            labels.append(y_batch.numpy())
    return predictions, labels

def print_results_table(dict_results):
    """
    Print the contents of dict_results in a tabular format,
    extracting the single value from mse/r2 lists when they contain only one element.
    """
    rows = []
    for model, approaches in dict_results.items():
        for approach, metrics in approaches.items():
            mse_list = metrics.get('mse', [])
            r2_list  = metrics.get('r2', [])
            # extract the single element if list length == 1
            mse_val = mse_list[0] if isinstance(mse_list, list) and len(mse_list) == 1 else mse_list
            r2_val  = r2_list[0]  if isinstance(r2_list,  list) and len(r2_list)  == 1 else r2_list
            rows.append({
                'Model':    model,
                'Approach': approach,
                'MSE':      mse_val,
                'R2':       r2_val
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df