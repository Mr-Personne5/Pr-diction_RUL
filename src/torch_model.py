"""Modèle LSTM pour la prédiction du RUL à partir de fenêtres de capteurs, et sa boucle d'entraînement."""

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import rmse


def set_seed(seed: int) -> None:
    """Fixe toutes les sources d'aléatoire (Python, numpy, PyTorch CPU/GPU) pour la reproductibilité.

    cuDNN choisit par défaut ses algorithmes les plus rapides, pas forcément déterministes :
    sans ces deux lignes, deux runs avec la même graine donnent des résultats légèrement
    différents sur GPU (vérifié par tests/test_torch_model.py).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RULRegressor(nn.Module):
    """LSTM simple : une couche récurrente, puis une couche linéaire vers un RUL scalaire.

    On ne garde que le dernier état caché de la séquence (h_n[-1]) : c'est le résumé que
    le LSTM a construit après avoir vu les window_size cycles dans l'ordre, et c'est lui
    qui contient l'information utile pour prédire le RUL au dernier cycle de la fenêtre.
    """

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # état caché de la dernière couche LSTM, shape (batch, hidden_size)
        return self.head(last_hidden).squeeze(-1)


class TransformerRULRegressor(nn.Module):
    """Encodeur Transformer léger : projection des capteurs vers d_model, embedding positionnel
    appris (la fenêtre a une taille fixe, window_size), puis un encodeur à self-attention.

    On lit la représentation du DERNIER pas de temps pour prédire le RUL — même principe que
    RULRegressor (LSTM) qui lit son dernier état caché : les deux modèles sont comparés à
    protocole de lecture égal, seule l'architecture change.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x) + self.pos_embedding
        x = self.encoder(x)
        last_step = x[:, -1, :]  # dernier pas de temps, shape (batch, d_model)
        return self.head(last_step).squeeze(-1)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[nn.Module, list[dict]]:
    """Entraîne model par descente de gradient (Adam, perte MSE), et retourne les poids du
    MEILLEUR epoch sur le val (pas forcément le dernier) : au-delà de ce point, le modèle
    n'améliore plus sa capacité à généraliser, seulement sa performance sur le train.

    Pour une reproductibilité totale (poids initiaux compris), appeler set_seed(seed) AVANT
    de construire model : cette fonction refixe les graines pour l'entraînement (mélange des
    batches...), mais ne peut pas rejouer l'initialisation des poids, déjà faite à la
    construction du modèle.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()
        val_rmse = rmse(y_val, val_pred)

        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_rmse": val_rmse})

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history
