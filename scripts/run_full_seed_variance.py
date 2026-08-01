"""Complète la variance sur graines manquante (LSTM FD002, Transformer FD001/FD002),
en sauvegardant chaque étape sur disque au fur et à mesure (results/seed_variance/*.csv) :
un plantage ou une lenteur sur une étape ne fait pas perdre les précédentes, contrairement à
l'exécution d'un notebook entier via nbconvert (un timeout de cellule fait perdre tout le
notebook, y compris les cellules déjà terminées).

Usage : .venv/Scripts/python.exe scripts/run_full_seed_variance.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.metrics import asymmetric_score, rmse
from src.preprocessing import (
    add_piecewise_rul,
    assign_regimes,
    compute_norm_stats,
    compute_norm_stats_by_regime,
    fit_regime_clusters,
    make_windows,
    normalize,
    normalize_by_regime,
    select_features,
    select_features_by_regime,
    split_by_unit,
)
from src.torch_model import RULRegressor, TransformerRULRegressor, set_seed, train_model

SPLIT_SEED = 42
MODEL_SEEDS = [0, 1, 2, 3, 4]
WINDOW_SIZE = 30
RUL_CAP = 125

DATA_DIR = Path(__file__).resolve().parent.parent / "DATA" / "CMaps"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "seed_variance"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_NAMES = ["unit_number", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [f"sensor_{i}" for i in range(1, 22)]
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES


def load_raw(subset):
    raw = pd.read_csv(DATA_DIR / f"train_{subset}.txt", sep=r"\s+", header=None)
    raw = raw.iloc[:, : len(COL_NAMES)]
    raw.columns = COL_NAMES
    return raw


def prepare_fd001():
    raw = load_raw("FD001")
    train_raw, val_raw = split_by_unit(raw, val_fraction=0.2, seed=SPLIT_SEED)
    selected_sensors = select_features(train_raw, SENSOR_NAMES, std_threshold=1e-2)
    norm_stats = compute_norm_stats(train_raw, selected_sensors)

    train_norm = normalize(train_raw, selected_sensors, norm_stats)
    val_norm = normalize(val_raw, selected_sensors, norm_stats)
    train_rul = add_piecewise_rul(train_norm, cap=RUL_CAP)
    val_rul = add_piecewise_rul(val_norm, cap=RUL_CAP)

    X_train, y_train, _ = make_windows(train_rul, selected_sensors, window_size=WINDOW_SIZE, target_col="RUL")
    X_val, y_val, _ = make_windows(val_rul, selected_sensors, window_size=WINDOW_SIZE, target_col="RUL")
    return X_train, y_train, X_val, y_val, len(selected_sensors)


def prepare_fd002():
    raw = load_raw("FD002")
    train_raw, val_raw = split_by_unit(raw, val_fraction=0.2, seed=SPLIT_SEED)

    regime_model = fit_regime_clusters(train_raw, SETTING_NAMES, n_regimes=6, seed=SPLIT_SEED)
    train_regimes = assign_regimes(train_raw, SETTING_NAMES, regime_model)
    val_regimes = assign_regimes(val_raw, SETTING_NAMES, regime_model)

    selected_sensors = select_features_by_regime(train_raw, SENSOR_NAMES, train_regimes, std_threshold=1e-2)
    regime_stats = compute_norm_stats_by_regime(train_raw, selected_sensors, train_regimes)
    train_norm = normalize_by_regime(train_raw, selected_sensors, train_regimes, regime_stats)
    val_norm = normalize_by_regime(val_raw, selected_sensors, val_regimes, regime_stats)

    train_rul = add_piecewise_rul(train_norm, cap=RUL_CAP)
    val_rul = add_piecewise_rul(val_norm, cap=RUL_CAP)

    X_train, y_train, _ = make_windows(train_rul, selected_sensors, window_size=WINDOW_SIZE, target_col="RUL")
    X_val, y_val, _ = make_windows(val_rul, selected_sensors, window_size=WINDOW_SIZE, target_col="RUL")
    return X_train, y_train, X_val, y_val, len(selected_sensors)


def run_seed_variance(name, model_cls, model_kwargs, X_train, y_train, X_val, y_val, n_features):
    csv_path = RESULTS_DIR / f"{name}.csv"
    if csv_path.exists():
        print(f"[{name}] déjà calculé, on charge {csv_path}")
        return pd.read_csv(csv_path)

    rows = []
    for seed in MODEL_SEEDS:
        t0 = time.time()
        set_seed(seed)
        model = model_cls(n_features=n_features, **model_kwargs)
        model, _ = train_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=1e-3, batch_size=64, seed=seed)

        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            y_pred = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()

        row = {"graine": seed, "RMSE": rmse(y_val, y_pred), "score_asymétrique": asymmetric_score(y_val, y_pred)}
        rows.append(row)
        print(f"[{name}] graine {seed} : RMSE={row['RMSE']:.3f} ({time.time() - t0:.0f}s)", flush=True)

        # Sauvegarde après CHAQUE graine, pas seulement à la fin : si le script est interrompu,
        # les graines déjà entraînées ne sont pas reperdues au prochain lancement.
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Préparation des pipelines FD001 et FD002...", flush=True)
    X_train_fd001, y_train_fd001, X_val_fd001, y_val_fd001, n_features_fd001 = prepare_fd001()
    X_train_fd002, y_train_fd002, X_val_fd002, y_val_fd002, n_features_fd002 = prepare_fd002()
    print(f"FD001 : {n_features_fd001} capteurs, {X_train_fd001.shape}", flush=True)
    print(f"FD002 : {n_features_fd002} capteurs, {X_train_fd002.shape}", flush=True)

    transformer_kwargs = {"d_model": 64, "nhead": 4, "num_layers": 1, "window_size": WINDOW_SIZE}

    run_seed_variance(
        "lstm_fd002", RULRegressor, {"hidden_size": 64},
        X_train_fd002, y_train_fd002, X_val_fd002, y_val_fd002, n_features_fd002,
    )
    run_seed_variance(
        "transformer_fd001", TransformerRULRegressor, transformer_kwargs,
        X_train_fd001, y_train_fd001, X_val_fd001, y_val_fd001, n_features_fd001,
    )
    run_seed_variance(
        "transformer_fd002", TransformerRULRegressor, transformer_kwargs,
        X_train_fd002, y_train_fd002, X_val_fd002, y_val_fd002, n_features_fd002,
    )

    print("Terminé.", flush=True)
