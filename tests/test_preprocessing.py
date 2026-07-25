"""Tests du pipeline de préparation (src/preprocessing.py).

Chaque test vérifie une garantie méthodologique du protocole : pas de fuite entre
train et val/test, plafonnement correct du RUL, et padding correct pour les
moteurs courts.
"""

import numpy as np
import pandas as pd

from src.preprocessing import (
    add_piecewise_rul,
    aggregate_window_features,
    compute_norm_stats,
    make_windows,
    normalize,
    select_features,
    split_by_unit,
)


def make_engine_df(unit_id: int, n_cycles: int, sensor_value_start: float = 0.0) -> pd.DataFrame:
    """Construit un petit DataFrame d'un seul moteur, pour des tests isolés et rapides."""
    return pd.DataFrame(
        {
            "unit_number": unit_id,
            "time_cycles": np.arange(1, n_cycles + 1),
            "sensor_a": sensor_value_start + np.arange(n_cycles) * 1.0,  # varie clairement
            "sensor_constant": 42.0,  # ne varie jamais
        }
    )


def test_select_features_drops_constant_and_keeps_varying():
    train_df = make_engine_df(unit_id=1, n_cycles=50)
    selected = select_features(train_df, ["sensor_a", "sensor_constant"], std_threshold=1e-2)
    assert selected == ["sensor_a"]


def test_normalize_uses_train_stats_not_val_stats():
    # Le train est centré sur 10, le val sur 1000 : si normalize() utilisait les stats
    # du val au lieu de celles du train, le résultat ci-dessous serait très différent.
    train_df = pd.DataFrame({"sensor_a": [8.0, 9.0, 10.0, 11.0, 12.0]})
    val_df = pd.DataFrame({"sensor_a": [10.0]})  # égal à la moyenne du train

    stats = compute_norm_stats(train_df, ["sensor_a"])
    val_normalized = normalize(val_df, ["sensor_a"], stats)

    assert val_normalized["sensor_a"].iloc[0] == 0.0


def test_add_piecewise_rul_never_exceeds_cap():
    train_df = make_engine_df(unit_id=1, n_cycles=300)
    result = add_piecewise_rul(train_df, cap=125)

    assert result["RUL"].max() == 125
    assert result["RUL"].min() == 0  # dernier cycle du moteur : RUL = 0


def test_make_windows_pads_short_engine():
    short_engine = make_engine_df(unit_id=1, n_cycles=5)
    X, _, units = make_windows(short_engine, ["sensor_a"], window_size=30)

    assert X.shape == (1, 30, 1)
    assert units.tolist() == [1]
    # Padding = répétition de la première ligne : les 25 premières valeurs de la
    # fenêtre doivent toutes être égales à la première valeur réelle du moteur.
    assert np.all(X[0, :25, 0] == short_engine["sensor_a"].iloc[0])


def test_make_windows_sliding_vs_last_only():
    engine = make_engine_df(unit_id=1, n_cycles=40)

    X_all, y_all, _ = make_windows(engine, ["sensor_a"], window_size=30, target_col="time_cycles")
    X_last, y_last, _ = make_windows(
        engine, ["sensor_a"], window_size=30, target_col="time_cycles", last_only=True
    )

    assert X_all.shape == (40 - 30 + 1, 30, 1)  # une fenêtre par position possible
    assert X_last.shape == (1, 30, 1)  # une seule fenêtre : la dernière
    assert y_last[0] == y_all[-1]  # la dernière fenêtre de chaque mode doit coïncider


def test_split_by_unit_no_shared_engine():
    # 10 moteurs, quelques cycles chacun : peu importe le contenu, seul l'identifiant
    # de moteur doit déterminer le camp (train ou val).
    df = pd.concat([make_engine_df(unit_id=u, n_cycles=5) for u in range(10)], ignore_index=True)

    train_df, val_df = split_by_unit(df, val_fraction=0.2, seed=42)
    train_units = set(train_df["unit_number"])
    val_units = set(val_df["unit_number"])

    assert train_units.isdisjoint(val_units)
    assert len(val_units) == 2  # 20% de 10 moteurs
    assert train_units | val_units == set(range(10))


def test_split_by_unit_is_reproducible_with_same_seed():
    df = pd.concat([make_engine_df(unit_id=u, n_cycles=5) for u in range(10)], ignore_index=True)

    _, val_df_1 = split_by_unit(df, val_fraction=0.3, seed=7)
    _, val_df_2 = split_by_unit(df, val_fraction=0.3, seed=7)

    assert set(val_df_1["unit_number"]) == set(val_df_2["unit_number"])


def test_aggregate_window_features_shape_and_values():
    # 2 fenêtres, longueur 4, 3 capteurs : des valeurs simples pour vérifier moyenne,
    # écart-type et dernière valeur à la main.
    X = np.array(
        [
            [[1.0, 10.0, 0.0], [2.0, 10.0, 0.0], [3.0, 10.0, 0.0], [4.0, 10.0, 0.0]],
        ]
    )

    features = aggregate_window_features(X)

    assert features.shape == (1, 3 * 3)  # (n_fenêtres, 3 stats * n_capteurs)
    mean, std, last = features[0, 0:3], features[0, 3:6], features[0, 6:9]
    assert mean[0] == 2.5  # moyenne de [1, 2, 3, 4]
    assert std[1] == 0.0  # capteur constant sur la fenêtre -> écart-type nul
    assert list(last) == [4.0, 10.0, 0.0]  # dernière ligne de la fenêtre
