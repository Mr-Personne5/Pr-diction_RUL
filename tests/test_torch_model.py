"""Tests de src/torch_model.py sur des tenseurs synthétiques minuscules (rapide, pas de vraies données)."""

import numpy as np
import torch

from src.torch_model import RULRegressor, TransformerRULRegressor, set_seed, train_model


def make_toy_data():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 5, 3)).astype("float32")  # 20 fenêtres, 5 cycles, 3 capteurs
    y_train = rng.uniform(0, 125, size=20).astype("float32")
    X_val = rng.normal(size=(6, 5, 3)).astype("float32")
    y_val = rng.uniform(0, 125, size=6).astype("float32")
    return X_train, y_train, X_val, y_val


def test_train_model_returns_full_history():
    X_train, y_train, X_val, y_val = make_toy_data()
    model = RULRegressor(n_features=3, hidden_size=4)

    _, history = train_model(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=4, seed=0)

    assert len(history) == 3
    assert {"epoch", "train_loss", "val_rmse"} <= history[0].keys()


def test_train_model_is_reproducible_with_same_seed():
    # Reproductibilité totale (poids initiaux compris) : il faut fixer la graine AVANT de
    # construire le modèle, pas seulement passer seed= à train_model (cf. docstring).
    X_train, y_train, X_val, y_val = make_toy_data()

    set_seed(123)
    model_1 = RULRegressor(n_features=3, hidden_size=4)
    _, history_1 = train_model(model_1, X_train, y_train, X_val, y_val, epochs=3, seed=123)

    set_seed(123)
    model_2 = RULRegressor(n_features=3, hidden_size=4)
    _, history_2 = train_model(model_2, X_train, y_train, X_val, y_val, epochs=3, seed=123)

    val_rmse_1 = [h["val_rmse"] for h in history_1]
    val_rmse_2 = [h["val_rmse"] for h in history_2]
    assert val_rmse_1 == val_rmse_2


def test_train_model_keeps_best_val_epoch_not_last():
    X_train, y_train, X_val, y_val = make_toy_data()
    model = RULRegressor(n_features=3, hidden_size=4)

    trained_model, history = train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=4, seed=0)

    best_epoch = min(history, key=lambda h: h["val_rmse"])
    # Les poids retournés doivent correspondre au meilleur epoch, pas au dernier : on le
    # vérifie en recalculant le val_rmse avec les poids finaux et en le comparant au minimum.
    import torch

    model_device = next(trained_model.parameters()).device
    trained_model.eval()
    with torch.no_grad():
        final_pred = trained_model(torch.tensor(X_val, dtype=torch.float32, device=model_device)).cpu().numpy()
    from src.metrics import rmse

    assert abs(rmse(y_val, final_pred) - best_epoch["val_rmse"]) < 1e-5


def test_transformer_output_shape():
    model = TransformerRULRegressor(n_features=3, window_size=5, d_model=8, nhead=2, num_layers=1)
    x = torch.randn(4, 5, 3)  # 4 fenêtres, 5 cycles, 3 capteurs

    output = model(x)

    assert output.shape == (4,)


def test_transformer_trains_with_train_model():
    # Vérifie que le Transformer s'intègre à train_model exactement comme le LSTM (même
    # fonction d'entraînement, indépendante de l'architecture).
    X_train, y_train, X_val, y_val = make_toy_data()
    model = TransformerRULRegressor(n_features=3, window_size=5, d_model=8, nhead=2, num_layers=1)

    _, history = train_model(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=4, seed=0)

    assert len(history) == 3
    assert all(np.isfinite(h["val_rmse"]) for h in history)
