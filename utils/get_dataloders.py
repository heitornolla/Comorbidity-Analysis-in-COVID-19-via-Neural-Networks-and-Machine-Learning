from typing import Tuple

import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset


def prepare_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def df_to_tensor(df):
        X = torch.tensor(df.drop("died", axis=1).values, dtype=torch.float32)
        y = torch.tensor(df["died"].values, dtype=torch.float32).view(-1, 1)
        return TensorDataset(X, y)

    train_loader = DataLoader(
        df_to_tensor(train_df), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(df_to_tensor(val_df), batch_size=batch_size)
    test_loader = DataLoader(df_to_tensor(test_df), batch_size=batch_size)

    return train_loader, val_loader, test_loader
