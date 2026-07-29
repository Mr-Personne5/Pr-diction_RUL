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


def make_two_regime_df() -> pd.DataFrame:
    # Deux régimes bien séparés sur "setting_1" (autour de 0 et autour de 100), avec un
    # capteur dont l'échelle diffère selon le régime (10 vs 1000) : une normalisation
    # globale confondrait ça avec de la dégradation, une normalisation par régime non.
    rng = np.random.default_rng(0)
    regime_a = pd.DataFrame({
        "setting_1": rng.normal(0, 0.1, size=50),
        "sensor_a": rng.normal(10, 1, size=50),
    })
    regime_b = pd.DataFrame({
        "setting_1": rng.normal(100, 0.1, size=50),
        "sensor_a": rng.normal(1000, 1, size=50),
    })
    return pd.concat([regime_a, regime_b], ignore_index=True)


def test_fit_regime_clusters_recovers_two_separated_groups():
    df = make_two_regime_df()
    true_regime = ["A"] * 50 + ["B"] * 50

    model = fit_regime_clusters(df, ["setting_1"], n_regimes=2, seed=42)
    assigned = assign_regimes(df, ["setting_1"], model)

    # Les labels de k-means sont arbitraires (0/1 ne correspond pas forcément à A/B) : on
    # vérifie que CHAQUE vrai groupe est assigné à un seul label, pas que les valeurs collent.
    labels_for_a = set(assigned[:50])
    labels_for_b = set(assigned[50:])
    assert len(labels_for_a) == 1
    assert len(labels_for_b) == 1
    assert labels_for_a != labels_for_b


def test_select_features_by_regime_drops_sensor_constant_within_each_regime():
    # sensor_fake_varying vaut 10 dans le régime A et 1000 dans le régime B : constant DANS
    # chaque régime, mais avec un écart-type global élevé (à cause du changement de régime
    # seul). select_features (global) le garderait à tort ; select_features_by_regime doit
    # l'exclure, sous peine de diviser par ~0 dans normalize_by_regime (cf. bug rencontré
    # dans 11_fd002_ablation_regime.ipynb).
    df = pd.DataFrame({
        "sensor_fake_varying": [10.0] * 50 + [1000.0] * 50,
        "sensor_real": np.linspace(0, 100, 100),
    })
    regimes = np.array(["A"] * 50 + ["B"] * 50)

    global_selection = select_features(df, ["sensor_fake_varying", "sensor_real"], std_threshold=1e-2)
    regime_selection = select_features_by_regime(df, ["sensor_fake_varying", "sensor_real"], regimes, std_threshold=1e-2)

    assert "sensor_fake_varying" in global_selection
    assert regime_selection == ["sensor_real"]


def test_normalize_by_regime_uses_per_regime_stats():
    df = make_two_regime_df()
    model = fit_regime_clusters(df, ["setting_1"], n_regimes=2, seed=42)
    regimes = assign_regimes(df, ["setting_1"], model)

    stats_by_regime = compute_norm_stats_by_regime(df, ["sensor_a"], regimes)
    normalized = normalize_by_regime(df, ["sensor_a"], regimes, stats_by_regime)

    # Une fois normalisé PAR régime, chaque groupe doit être centré sur 0 séparément,
    # même si leurs échelles brutes (10 vs 1000) étaient très différentes.
    for regime in set(regimes):
        mask = regimes == regime
        assert abs(normalized.loc[mask, "sensor_a"].mean()) < 1e-6


def test_normalize_by_regime_handles_integer_sensor_column():
    # Un capteur lu comme int64 (valeurs entières) a fait planter une première version de
    # normalize_by_regime : l'assignation par masque refusait d'y écrire des floats.
    df = make_two_regime_df()
    df["sensor_int"] = df["sensor_a"].round().astype("int64")  # entier, mais pas constant
    model = fit_regime_clusters(df, ["setting_1"], n_regimes=2, seed=42)
    regimes = assign_regimes(df, ["setting_1"], model)

    stats_by_regime = compute_norm_stats_by_regime(df, ["sensor_int"], regimes)
    normalized = normalize_by_regime(df, ["sensor_int"], regimes, stats_by_regime)

    assert normalized["sensor_int"].dtype == float