import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 feature_cols: list,
                 target_col: str,
                 window_size: int,
                 horizon: int = 1,
                 transform=None):
        """
        df: symbol, date sıralı; feature_cols, target_col içeren DataFrame
        feature_cols: model girdi sütunları (ör. ['close', 'ESG_score'])
        target_col: modelin tahmin edeceği sütun (ör. 'close')
        window_size: geçmiş gün sayısı (ör. 30)
        horizon: tahmin uzaklığı (varsayılan 1 gün sonrası)
        transform: opsiyonel, X y üzerinde yapılacak dönüşümler (Tensor vb.)
        """
        super().__init__()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size
        self.horizon = horizon
        self.transform = transform

        # Her symbol için ayrı kaydırmalı pencere indeksleri çıkar
        self.windows = []  # her eleman: (symbol, start_idx)
        self.grouped = df.groupby('symbol', sort=False)
        for symbol, group in self.grouped:
            group = group.sort_values('date').reset_index(drop=True)
            n = len(group)
            # son pencerelerin tamamını alabilmek için limit: n - window_size - horizon +1
            for start in range(n - window_size - horizon + 1):
                self.windows.append((symbol, start))

        # DataFrame’i bir sözlükte tutmak erişimi hızlandırır
        self.data_dict = {
            symbol: group.sort_values('date').reset_index(drop=True)
            for symbol, group in self.grouped
        }

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        symbol, start = self.windows[idx]
        df_symbol = self.data_dict[symbol]

        # X: geçmiş window_size gün, tüm feature sütunları
        X = df_symbol.loc[start:start + self.window_size - 1, self.feature_cols].values
        # y: start + window_size + horizon -1 gününün target değeri
        target_idx = start + self.window_size + self.horizon - 1
        y = df_symbol.loc[target_idx, self.target_col]

        # tensor dönüşümü
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.transform:
            X, y = self.transform(X, y)

        return X, y
