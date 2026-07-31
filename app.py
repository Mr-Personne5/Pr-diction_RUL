"""App Streamlit de présentation du mémoire — tableau de bord de résultats + démo interactive.

Lancer avec : streamlit run app.py (depuis la racine du dépôt).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
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
from src.torch_model import RULRegressor, TransformerRULRegressor

DATA_DIR = Path("DATA/CMaps")
SEED = 42
WINDOW_SIZE = 30
RUL_CAP = 125

# (dataset, architecture) -> (chemin des poids, classe du modèle, kwargs de construction).
# Les 4 checkpoints viennent de 05/11 (LSTM) et 13/14 (Transformer) — mêmes seed/split/capteurs
# que dans les notebooks correspondants, sinon les poids chargés ne correspondraient pas au
# pipeline de préparation reconstruit ici.
MODEL_SPECS = {
    ("FD001", "LSTM"): ("models/lstm_fd001_seed42.pt", RULRegressor, {"hidden_size": 64}),
    ("FD001", "Transformer"): (
        "models/transformer_fd001_seed42.pt",
        TransformerRULRegressor,
        {"d_model": 64, "nhead": 4, "num_layers": 1, "window_size": WINDOW_SIZE},
    ),
    ("FD002", "LSTM"): ("models/lstm_fd002_seed42.pt", RULRegressor, {"hidden_size": 64}),
    ("FD002", "Transformer"): (
        "models/transformer_fd002_seed42.pt",
        TransformerRULRegressor,
        {"d_model": 64, "nhead": 4, "num_layers": 1, "window_size": WINDOW_SIZE},
    ),
}

INDEX_NAMES = ["unit_number", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [f"sensor_{i}" for i in range(1, 22)]
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

# Une couleur par identité de modèle, réutilisée partout dans l'app (jamais recyclée pour
# autre chose) — palette validée CVD-safe (dataviz skill, slots 1/2/3/4).
COLORS = {
    "Régression linéaire": "#eda100",
    "Forêt aléatoire": "#eb6834",
    "LSTM": "#2a78d6",
    "Transformer": "#1baf7a",
}
MUTED = "#898781"

# Palette de statut (réservée, jamais utilisée pour une identité de modèle) : sert à qualifier
# le SENS de l'écart de prédiction, pas à identifier un modèle.
STATUS_COLORS = {
    "good": "#0ca30c",      # sous-estimation : maintenance anticipée, plus prudent
    "critical": "#d03b3b",  # sur-estimation : maintenance retardée, le cas le plus risqué
}

st.set_page_config(page_title="Mémoire RUL — C-MAPSS", page_icon="✈️", layout="wide")


def show_fig(fig) -> None:
    """st.pyplot puis plt.close : sans ça, les figures matplotlib générées à chaque rerun
    s'accumulent en mémoire (et Matplotlib finit par avertir "more than 20 figures opened").
    """
    st.pyplot(fig)
    plt.close(fig)


def bar_chart(labels: list[str], values: list[float], colors: list[str], ylabel: str, value_fmt: str = "{:.2f}"):
    """Barres fines, labels de valeur visibles (mitigation du contraste sur aqua/jaune),
    grille horizontale discrète, pas d'axe double (une seule métrique par figure).
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), value_fmt.format(value),
            ha="center", va="bottom", fontsize=10, color="#0b0b0b",
        )
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    return fig


def interpretation_figure(y_pred: float, y_true: float, tolerance: float = 5.0):
    """Représente le RUL réel et prédit sur un même axe, colorée selon le SENS de l'écart :

    - prédit > réel (écart positif) : le modèle retarderait la maintenance au-delà de la
      panne réelle — c'est le cas que le score asymétrique pénalise le plus (cf. src/metrics.py).
    - prédit < réel (écart négatif) : maintenance anticipée, moins dangereux.

    Une couleur de STATUT (pas d'identité de modèle) porte ce sens, jamais le texte seul.
    """
    diff = y_pred - y_true
    if abs(diff) <= tolerance:
        color, label = MUTED, "Prédiction précise"
    elif diff > 0:
        color, label = STATUS_COLORS["critical"], "Sur-estimation — retard de maintenance, le plus risqué"
    else:
        color, label = STATUS_COLORS["good"], "Sous-estimation — maintenance anticipée, plus prudent"

    axis_max = max(y_pred, y_true, RUL_CAP) * 1.15

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.hlines(0, 0, axis_max, color="#c3c2b7", linewidth=1, zorder=1)
    ax.annotate(
        "", xy=(y_pred, 0), xytext=(y_true, 0),
        arrowprops={"arrowstyle": "->", "color": color, "lw": 2}, zorder=2,
    )

    ax.plot(y_true, 0, marker="|", markersize=28, markeredgewidth=2, color="#0b0b0b", zorder=3)
    ax.text(y_true, 0.28, f"RUL réel\n{y_true:.1f}", ha="center", fontsize=9, color="#52514e")

    ax.plot(y_pred, 0, marker="o", markersize=11, color=color, zorder=4)
    ax.text(y_pred, -0.32, f"RUL prédit\n{y_pred:.1f}", ha="center", va="top", fontsize=9, color=color)

    ax.set_title(label, color=color, fontsize=11, loc="left")
    ax.set_xlim(0, axis_max)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])
    ax.set_xlabel("RUL (cycles)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(colors="#52514e")
    fig.tight_layout()
    return fig


@st.cache_data
def load_fd001_reference():
    """Reconstruit exactement les capteurs sélectionnés et les stats de normalisation utilisés
    pour entraîner le modèle sauvegardé (05_lstm_fd001.ipynb) : même split, même seed, même
    seuil — sans ça, une prédiction en démo ne serait pas cohérente avec le modèle chargé.
    """
    raw = pd.read_csv(DATA_DIR / "train_FD001.txt", sep=r"\s+", header=None)
    raw = raw.iloc[:, : len(COL_NAMES)]
    raw.columns = COL_NAMES

    train_raw, _ = split_by_unit(raw, val_fraction=0.2, seed=SEED)
    selected_sensors = select_features(train_raw, SENSOR_NAMES, std_threshold=1e-2)
    norm_stats = compute_norm_stats(train_raw, selected_sensors)
    return selected_sensors, norm_stats


@st.cache_data
def load_fd001_test():
    test_raw = pd.read_csv(DATA_DIR / "test_FD001.txt", sep=r"\s+", header=None)
    test_raw = test_raw.iloc[:, : len(COL_NAMES)]
    test_raw.columns = COL_NAMES

    rul_true = pd.read_csv(DATA_DIR / "RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])
    rul_true.index = rul_true.index + 1  # unit_number numéroté à partir de 1
    return test_raw, rul_true


@st.cache_resource
def load_fd002_reference():
    """Équivalent de load_fd001_reference, mais régime-aware (cf. 11_fd002_ablation_regime.ipynb,
    variante "par régime") : k-means sur les réglages, sélection et stats de normalisation par
    régime — tout calculé sur le train uniquement.
    """
    raw = pd.read_csv(DATA_DIR / "train_FD002.txt", sep=r"\s+", header=None)
    raw = raw.iloc[:, : len(COL_NAMES)]
    raw.columns = COL_NAMES

    train_raw, _ = split_by_unit(raw, val_fraction=0.2, seed=SEED)
    regime_model = fit_regime_clusters(train_raw, SETTING_NAMES, n_regimes=6, seed=SEED)
    train_regimes = assign_regimes(train_raw, SETTING_NAMES, regime_model)

    selected_sensors = select_features_by_regime(train_raw, SENSOR_NAMES, train_regimes, std_threshold=1e-2)
    regime_stats = compute_norm_stats_by_regime(train_raw, selected_sensors, train_regimes)
    return selected_sensors, regime_model, regime_stats


@st.cache_data
def load_fd002_test():
    test_raw = pd.read_csv(DATA_DIR / "test_FD002.txt", sep=r"\s+", header=None)
    test_raw = test_raw.iloc[:, : len(COL_NAMES)]
    test_raw.columns = COL_NAMES

    rul_true = pd.read_csv(DATA_DIR / "RUL_FD002.txt", sep=r"\s+", header=None, names=["RUL"])
    rul_true.index = rul_true.index + 1
    return test_raw, rul_true


@st.cache_resource
def load_model(dataset: str, architecture: str, n_features: int):
    path, model_cls, kwargs = MODEL_SPECS[(dataset, architecture)]
    model = model_cls(n_features=n_features, **kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def page_contexte():
    st.title("Prédiction de RUL sur C-MAPSS")
    st.caption("Mémoire M1 IA — maintenance prédictive (Prognostics and Health Management)")

    st.markdown(
        """
        ## Problématique

        Prédire le **Remaining Useful Life (RUL)** — le nombre de cycles restants avant la
        panne — de moteurs d'avion simulés (**C-MAPSS**, NASA/PHM08), à partir de mesures de
        capteurs bruitées.

        Deux questions guident le travail :
        1. Un modèle de séquence profond (LSTM, Transformer) apporte-t-il un gain mesurable
           sur une baseline classique (régression linéaire, forêt aléatoire) ?
        2. Ce gain résiste-t-il au passage d'un seul régime opérationnel (**FD001**) à six
           régimes (**FD002**), qui exige une normalisation adaptée ?

        ## Pipeline (résumé)

        1. **Split train/val par moteur entier** — jamais par cycle, pour éviter toute fuite.
        2. **Sélection des capteurs** informatifs, sur le train uniquement (par régime sur FD002).
        3. **Normalisation z-score**, stats calculées sur le train (par régime sur FD002).
        4. **Cible RUL par morceaux**, plafonnée à 125 cycles.
        5. **Fenêtrage glissant** de 30 cycles, avec padding pour les moteurs plus courts.

        Détail complet dans [`README.md`](README.md) et [`plan_execution.md`](plan_execution.md).
        """
    )


def page_resultats_fd001():
    st.title("Résultats — FD001")

    fd001 = pd.DataFrame([
        {"modèle": "Régression linéaire", "RMSE_val": 21.536573, "score_asym_val": 77665.928021, "RMSE_test": 17.767336},
        {"modèle": "Forêt aléatoire", "RMSE_val": 17.899715, "score_asym_val": 29640.942003, "RMSE_test": 17.869847},
        {"modèle": "LSTM", "RMSE_val": 13.088298, "score_asym_val": 12245.140782, "RMSE_test": 12.225107},
        {"modèle": "Transformer", "RMSE_val": 13.167391, "score_asym_val": 11923.526820, "RMSE_test": None},
    ])
    st.dataframe(fd001, hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_fig(bar_chart(fd001["modèle"], fd001["RMSE_val"], [COLORS[m] for m in fd001["modèle"]], "RMSE (val)"))
    with col2:
        test_df = fd001.dropna(subset=["RMSE_test"])
        show_fig(bar_chart(test_df["modèle"], test_df["RMSE_test"], [COLORS[m] for m in test_df["modèle"]], "RMSE (test, ouvert une fois)"))
        st.caption("Le Transformer n'a pas de score test : le test a été ouvert avant qu'il n'existe (règle d'ouverture unique).")

    st.markdown(
        """
        **Conclusions** :
        - LSTM et Transformer battent nettement les baselines (**H1 confirmée**).
        - Variance du LSTM sur 5 graines : RMSE = 13.25 ± 0.42 — gain robuste au hasard de l'initialisation.
        - Sensibilité au plafond RUL (110/125/140) : effet réel mais modéré (**H4**).
        """
    )


def page_resultats_fd002():
    st.title("Résultats — FD002 (6 régimes)")

    ablation = pd.DataFrame([
        {"configuration": "Normalisation globale (naïve)", "RMSE": 18.178780, "score_asym": 81053.938110},
        {"configuration": "Normalisation par régime", "RMSE": 15.074154, "score_asym": 42105.297588},
    ])
    final = pd.DataFrame([
        {"modèle": "LSTM", "RMSE": 15.074154, "score_asym": 42105.297588},
        {"modèle": "Transformer", "RMSE": 15.633626, "score_asym": 50027.646116},
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ablation (test H3)")
        st.dataframe(ablation, hide_index=True, use_container_width=True)
        show_fig(bar_chart(ablation["configuration"], ablation["RMSE"], [MUTED, COLORS["LSTM"]], "RMSE (val)"))
    with col2:
        st.subheader("LSTM vs Transformer (régime-aware)")
        st.dataframe(final, hide_index=True, use_container_width=True)
        show_fig(bar_chart(final["modèle"], final["RMSE"], [COLORS[m] for m in final["modèle"]], "RMSE (val)"))

    st.markdown(
        """
        **Conclusions** :
        - La normalisation par régime réduit le RMSE de ~17% par rapport à une normalisation
          naïve qui ignore les régimes (**H3 confirmée**).
        - FD001 → FD002 (meilleure config de chaque côté) : RMSE 13.09 → 15.07, soit **+15%**
          de dégradation — réelle mais modérée (**H2 partiellement confirmée**).
        - Sur FD002, le LSTM devance légèrement le Transformer — contrairement à FD001, où ils
          sont quasi à égalité.
        """
    )


@st.fragment
def page_demo():
    # @st.fragment : un changement de sélecteur ne relance que cette page, pas tout le
    # script — sans ça, chaque interaction reconstruit d'un coup plusieurs figures
    # matplotlib sur toute la page, et deux reruns qui se chevauchent (clics rapprochés)
    # font perdre à React le fil de son propre arbre DOM (erreur "removeChild").
    st.title("Démo interactive — prédiction en direct")

    col_a, col_b = st.columns(2)
    dataset = col_a.selectbox("Jeu de données", ["FD001", "FD002"], key="demo_dataset")
    architecture = col_b.selectbox("Modèle", ["LSTM", "Transformer"], key="demo_architecture")

    st.markdown(
        f"Choisit un moteur du **vrai jeu de test {dataset}** (jamais utilisé pour "
        f"l'entraînement) : le {architecture} sauvegardé prédit son RUL à partir de ses 30 "
        f"derniers cycles observés, comparé au RUL réel."
    )

    if dataset == "FD001":
        selected_sensors, norm_stats = load_fd001_reference()
        test_raw, rul_true = load_fd001_test()
    else:
        selected_sensors, regime_model, regime_stats = load_fd002_reference()
        test_raw, rul_true = load_fd002_test()

    model = load_model(dataset, architecture, n_features=len(selected_sensors))

    # Clé incluant dataset : la liste de moteurs change entre FD001 et FD002, un widget
    # gardant la même clé d'un jeu à l'autre peut désynchroniser React (cf. erreur
    # "removeChild" rencontrée en pratique).
    engine_id = st.selectbox(
        "Moteur (unit_number)", sorted(test_raw["unit_number"].unique()), key=f"demo_engine_{dataset}"
    )
    engine_data = test_raw[test_raw["unit_number"] == engine_id].sort_values("time_cycles")

    if dataset == "FD001":
        engine_norm = normalize(engine_data, selected_sensors, norm_stats)
    else:
        engine_regimes = assign_regimes(engine_data, SETTING_NAMES, regime_model)
        engine_norm = normalize_by_regime(engine_data, selected_sensors, engine_regimes, regime_stats)

    X, _, _ = make_windows(engine_norm, selected_sensors, window_size=WINDOW_SIZE, last_only=True)

    with torch.no_grad():
        y_pred = model(torch.tensor(X, dtype=torch.float32)).item()
    y_true = min(float(rul_true.loc[engine_id, "RUL"]), RUL_CAP)

    col1, col2, col3 = st.columns(3)
    col1.metric("RUL prédit", f"{y_pred:.1f} cycles")
    col2.metric("RUL réel (plafonné à 125)", f"{y_true:.1f} cycles")
    col3.metric("Écart", f"{y_pred - y_true:+.1f} cycles")

    st.subheader("Guide d'interprétation")
    show_fig(interpretation_figure(y_pred, y_true))
    st.markdown(
        """
        - **Écart positif** (prédit > réel) : le modèle est optimiste — il retarderait la
          maintenance au-delà de la panne réelle. C'est le cas que le score asymétrique
          pénalise le plus (`exp(d/10) - 1`, cf. `src/metrics.py`), car rater une panne
          imminente coûte plus cher en maintenance que la prévoir trop tôt.
        - **Écart négatif** (prédit < réel) : le modèle est pessimiste — maintenance
          anticipée, moins dangereux mais moins économique.
        - **Écart proche de 0** (±5 cycles) : prédiction jugée précise.
        """
    )

    st.subheader(f"Trajectoires de capteurs — moteur {engine_id} ({len(engine_data)} cycles observés)")
    example_sensors = selected_sensors[:3]
    palette = [COLORS["LSTM"], COLORS["Transformer"], COLORS["Forêt aléatoire"]]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for sensor, color in zip(example_sensors, palette):
        ax.plot(engine_data["time_cycles"], engine_data[sensor], label=sensor, color=color, linewidth=1.5)
    ax.set_xlabel("cycle")
    ax.set_ylabel("valeur brute du capteur")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    show_fig(fig)


if __name__ == "__main__":
    PAGES = {
        "Contexte & méthode": page_contexte,
        "Résultats FD001": page_resultats_fd001,
        "Résultats FD002": page_resultats_fd002,
        "Démo interactive": page_demo,
    }

    page = st.sidebar.radio("Navigation", list(PAGES.keys()))
    PAGES[page]()
