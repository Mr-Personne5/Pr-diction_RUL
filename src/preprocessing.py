"""Pipeline de préparation des données C-MAPSS, sans fuite train/val/test.

Règle d'or de ce module : toute statistique utilisée pour transformer les données
(quels capteurs garder, moyenne/écart-type de normalisation) doit être calculée
UNIQUEMENT sur le train, jamais sur le val ou le test.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def split_by_unit(
    df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42, id_col: str = "unit_number"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare train/val par moteur ENTIER : un moteur ne peut pas avoir des cycles des
    deux côtés à la fois, sinon le modèle verrait en val des cycles très proches (donc
    quasi identiques) de cycles déjà vus en train, ce qui gonflerait artificiellement
    les performances de validation.
    """
    unit_ids = df[id_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unit_ids)

    n_val = round(len(unit_ids) * val_fraction)
    val_units = set(unit_ids[:n_val])

    is_val = df[id_col].isin(val_units)
    return df[~is_val].copy(), df[is_val].copy()


def select_features(train_df: pd.DataFrame, feature_cols: list[str], std_threshold: float = 1e-2) -> list[str]:
    """Retourne les colonnes de feature_cols dont l'écart-type (sur le train) dépasse std_threshold.

    Sert à exclure les capteurs constants/quasi-constants (cf. notebooks/02_eda_fd001.ipynb) :
    sur FD001, les capteurs constants ont un écart-type ~1e-15 et le capteur quasi-constant
    (sensor_6) ~1.4e-3, alors que les capteurs informatifs ont tous un écart-type >= 0.037.
    Le seuil par défaut (1e-2) se situe confortablement entre les deux groupes.
    """
    train_std = train_df[feature_cols].std()
    return train_std[train_std >= std_threshold].index.tolist()


def select_features_by_regime(
    train_df: pd.DataFrame, feature_cols: list[str], regimes: np.ndarray, std_threshold: float = 1e-2
) -> list[str]:
    """Comme select_features, mais calcule l'écart-type À L'INTÉRIEUR de chaque régime, et
    ne garde une feature que si elle varie dans TOUS les régimes (le minimum des écarts-types
    par régime dépasse std_threshold).

    Sur FD002, un capteur peut être constant dans chaque régime pris isolément, tout en
    semblant varier globalement à cause du changement de régime (ex. sensor_1, sensor_5,
    sensor_18, sensor_19 : constants sur FD001, mais std global non nul sur FD002).
    select_features (basé sur l'écart-type global) les garderait à tort, ce qui produit des
    divisions par ~0 dans normalize_by_regime (NaN pendant l'entraînement). Cette fonction
    doit être utilisée à la place de select_features avant toute normalisation par régime.
    """
    train_df = train_df.assign(_regime=regimes)
    std_by_regime = train_df.groupby("_regime")[feature_cols].std()
    min_std_across_regimes = std_by_regime.min(axis=0)
    return min_std_across_regimes[min_std_across_regimes >= std_threshold].index.tolist()


def compute_norm_stats(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, pd.Series]:
    """Calcule la moyenne et l'écart-type de chaque feature, sur le train uniquement."""
    return {"mean": train_df[feature_cols].mean(), "std": train_df[feature_cols].std()}


def normalize(df: pd.DataFrame, feature_cols: list[str], stats: dict[str, pd.Series]) -> pd.DataFrame:
    """Applique une normalisation z-score avec des stats déjà calculées (cf. compute_norm_stats).

    Les stats viennent toujours du train, même quand on normalise le val ou le test :
    c'est ce qui garantit l'absence de fuite.
    """
    df = df.copy()
    df[feature_cols] = (df[feature_cols] - stats["mean"]) / stats["std"]
    return df


def fit_regime_clusters(train_df: pd.DataFrame, setting_cols: list[str], n_regimes: int = 6, seed: int = 42) -> KMeans:
    """Ajuste un k-means sur les réglages opérationnels du train, pour retrouver les régimes
    de fonctionnement discrets de FD002 (6 régimes fixes, cf. notebooks/09_fd002_regimes.ipynb).

    Ajusté sur le train uniquement : le val/test sont ensuite assignés aux régimes déjà
    trouvés (assign_regimes), jamais utilisés pour redéfinir les centres des clusters.
    """
    return KMeans(n_clusters=n_regimes, random_state=seed, n_init=10).fit(train_df[setting_cols])


def assign_regimes(df: pd.DataFrame, setting_cols: list[str], regime_model: KMeans) -> np.ndarray:
    """Assigne chaque ligne de df au régime le plus proche, selon un k-means déjà ajusté sur le train."""
    return regime_model.predict(df[setting_cols])


def compute_norm_stats_by_regime(train_df: pd.DataFrame, feature_cols: list[str], regimes: np.ndarray) -> dict:
    """Comme compute_norm_stats, mais une moyenne/écart-type par régime : sur FD002, un
    capteur peut avoir une valeur normale très différente selon le point de fonctionnement,
    donc une seule statistique globale (comme sur FD001, un seul régime) confondrait
    "changement de régime" et "dégradation du moteur".
    """
    train_df = train_df.assign(_regime=regimes)
    return {
        regime: {"mean": group[feature_cols].mean(), "std": group[feature_cols].std()}
        for regime, group in train_df.groupby("_regime")
    }


def normalize_by_regime(
    df: pd.DataFrame, feature_cols: list[str], regimes: np.ndarray, stats_by_regime: dict
) -> pd.DataFrame:
    """Applique une normalisation z-score, avec les stats du régime de chaque ligne (cf.
    compute_norm_stats_by_regime). Les stats viennent toujours du train, y compris pour
    normaliser le val/test.
    """
    df = df.copy()
    # Cast en float AVANT l'assignation par masque : une colonne lue en int64 (capteur à
    # valeurs entières) refuse silencieusement des valeurs flottantes ligne par ligne, alors
    # qu'une normalisation produit toujours des floats.
    df[feature_cols] = df[feature_cols].astype(float)
    for regime, stats in stats_by_regime.items():
        mask = regimes == regime
        df.loc[mask, feature_cols] = (df.loc[mask, feature_cols] - stats["mean"]) / stats["std"]
    return df


def add_piecewise_rul(
    train_df: pd.DataFrame, id_col: str = "unit_number", cycle_col: str = "time_cycles", cap: int = 125
) -> pd.DataFrame:
    """Ajoute une colonne RUL au train, par morceaux et plafonnée.

    Le train va jusqu'à la panne, donc le RUL vrai à un cycle donné est simplement
    "dernier cycle du moteur - cycle courant". On le plafonne à `cap` car en début de
    vie, un moteur en bon état n'est pas plus "sain" à RUL=300 qu'à RUL=125 : au-delà
    du plafond, la dégradation n'est pas encore visible dans les capteurs (Heimes 2008).
    """
    df = train_df.copy()
    max_cycle_per_unit = df.groupby(id_col)[cycle_col].transform("max")
    df["RUL"] = (max_cycle_per_unit - df[cycle_col]).clip(upper=cap)
    return df


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 30,
    id_col: str = "unit_number",
    cycle_col: str = "time_cycles",
    target_col: str | None = None,
    last_only: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Découpe chaque moteur de df en fenêtres glissantes de longueur window_size.

    - last_only=False (train) : une fenêtre par position possible (stride 1), pour avoir
      un maximum d'exemples d'entraînement.
    - last_only=True (test) : uniquement la dernière fenêtre de chaque moteur, puisqu'en
      test on ne prédit le RUL qu'au dernier cycle observé.
    - Moteurs plus courts que window_size : on complète en répétant leur première ligne
      (padding en début de séquence), pour ne perdre aucun moteur.

    Retourne (X, y, units) où X a la forme (n_fenêtres, window_size, n_features), y est le
    RUL de la dernière ligne de chaque fenêtre (None si target_col n'est pas fourni), et
    units donne le moteur d'origine de chaque fenêtre (utile pour retracer les prédictions).
    """
    X_list, y_list, unit_list = [], [], []

    for unit_id, unit_df in df.groupby(id_col):
        unit_df = unit_df.sort_values(cycle_col)
        features = unit_df[feature_cols].to_numpy()
        targets = unit_df[target_col].to_numpy() if target_col is not None else None
        n = len(unit_df)

        if n < window_size:
            # Padding : on répète la première ligne autant de fois qu'il manque de cycles.
            pad = np.repeat(features[[0]], window_size - n, axis=0)
            windows = [np.concatenate([pad, features], axis=0)]
            window_targets = [targets[-1]] if targets is not None else [None]
        elif last_only:
            windows = [features[-window_size:]]
            window_targets = [targets[-1]] if targets is not None else [None]
        else:
            windows = [features[start : start + window_size] for start in range(n - window_size + 1)]
            window_targets = (
                [targets[start + window_size - 1] for start in range(n - window_size + 1)]
                if targets is not None
                else [None] * len(windows)
            )

        X_list.extend(windows)
        y_list.extend(window_targets)
        unit_list.extend([unit_id] * len(windows))

    X = np.stack(X_list)
    y = np.array(y_list) if target_col is not None else None
    units = np.array(unit_list)
    return X, y, units


def aggregate_window_features(X: np.ndarray) -> np.ndarray:
    """Résume chaque fenêtre (n_fenêtres, window_size, n_capteurs) en un vecteur tabulaire
    (n_fenêtres, 3 * n_capteurs) : moyenne, écart-type et dernière valeur de chaque capteur
    sur la fenêtre.

    Sert à la baseline (régression linéaire / forêt aléatoire), qui ne peut pas consommer
    une séquence directement. Contrairement au LSTM, elle ne voit donc pas l'ORDRE des 30
    cycles, seulement un résumé statistique — c'est la différence méthodologique qu'on
    cherche justement à mesurer entre baseline et modèle de séquence.
    """
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    last = X[:, -1, :]
    return np.hstack([mean, std, last])