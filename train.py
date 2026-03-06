import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from utils.eval_utils import evaluate_model, get_pytorch_preds
from utils.prepare_dataframes import main as get_dfs
from utils.get_dataloders import prepare_loaders
from models.mlp_2layer import TwoLayerMLP
from models.mlp_1layer import OneLayerMLP


def train(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
) -> nn.Module:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 200
    patience = 50
    best_val_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        tp, fn = 0, 0  # Recall

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                out = model(batch_X)
                val_loss += criterion(out, batch_y).item()

                preds = (out > 0.5).float()
                correct += (preds == batch_y).sum().item()

                # Recall: REC = TP / (TP + FN)
                tp += ((preds == 1) & (batch_y == 1)).sum().item()
                fn += ((preds == 0) & (batch_y == 1)).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        acc = correct / len(val_loader.dataset)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"ckpt/{str(model)}_best_val.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping at epoch ", epoch)
                break

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f} | Acc = {acc:.4f} | Recall = {recall:.4f}"
            )

    model.load_state_dict(
        torch.load(f"ckpt/{str(model)}_best_val.pth", weights_only=False)
    )
    return model


if __name__ == "__main__":
    train_df, val_df, test_df = get_dfs()
    train_loader, val_loader, test_loader = prepare_loaders(train_df, val_df, test_df)

    models = [OneLayerMLP, TwoLayerMLP]
    for model in models:
        trained_model = train(model(), train_loader, val_loader)

        targets, preds = get_pytorch_preds(trained_model, test_loader)
        evaluate_model(
            targets, preds, model_name=str(model()), log_file=f"logs/{str(model())}.csv"
        )
