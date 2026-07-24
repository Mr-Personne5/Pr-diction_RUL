# Mémoire M1 IA — Prédiction de RUL sur C-MAPSS

## Contexte et problématique

Ce dépôt contient le travail d'un mémoire de M1 IA portant sur la prédiction du **Remaining
Useful Life (RUL)** — le nombre de cycles de fonctionnement restants avant la panne — de moteurs
d'avion (turboréacteurs), dans un cadre de maintenance prédictive (Prognostics and Health
Management, PHM).

Le jeu de données utilisé est **C-MAPSS** (NASA, Saxena et al., PHM08) : des trajectoires
simulées de moteurs, chacune démarrant en fonctionnement normal et se dégradant jusqu'à la
panne, avec des mesures de capteurs bruitées et plusieurs réglages opérationnels. Quatre
sous-ensembles existent (FD001 à FD004), qui diffèrent par le nombre de régimes opérationnels
(1 ou 6) et de modes de défaillance (1 ou 2) ; ce travail se concentre sur **FD001** (1 régime,
1 mode de défaillance) et **FD002** (6 régimes, 1 mode de défaillance).

La question de recherche centrale est double :
1. Un modèle de séquence profond (LSTM/GRU, puis éventuellement Transformer) apporte-t-il un
   gain mesurable sur la prédiction du RUL par rapport à une baseline classique (régression
   linéaire / forêt aléatoire) ?
2. Ce gain — et la qualité de la prédiction en général — résiste-t-il au passage d'un seul
   régime opérationnel (FD001) à plusieurs régimes (FD002), qui exige une normalisation par
   régime pour rester comparable ?

L'évaluation s'appuie sur deux métriques complémentaires : le RMSE et le score asymétrique de
Saxena & Goebel (2008), qui pénalise davantage les prédictions en retard (sous-estimation du
RUL) que les prédictions en avance — pertinent en maintenance, où sous-estimer une panne
imminente coûte plus cher que la surestimer.

Le détail des phases, priorités et critères d'avancement est dans [`plan_execution.md`](plan_execution.md).

## Structure du dépôt

```
DATA/CMaps/       données C-MAPSS (non versionnées, voir "Données" ci-dessous)
notebooks/        notebooks Jupyter (chargement, EDA, modèles)
Requirements.txt  dépendances Python figées
```

## Installation

```bash
git clone <url-du-depot>
cd Djob
python -m venv .venv
.venv\Scripts\activate
pip install -r Requirements.txt
python -m ipykernel install --user --name djob-cmapss --display-name "Python (Djob venv)"
```

Ouvrir les notebooks dans `notebooks/` et sélectionner le kernel **Python (Djob venv)**.

## Données

Le dossier `DATA/` n'est pas versionné (fichiers volumineux). Récupérer C-MAPSS (FD001 à FD004)
depuis Kaggle ("NASA C-MAPSS") ou le portail NASA Open Data / dépôt PCoE, et placer les fichiers
dans `DATA/CMaps/` :

```
DATA/CMaps/
  readme.txt
  train_FD00{1,2,3,4}.txt
  test_FD00{1,2,3,4}.txt
  RUL_FD00{1,2,3,4}.txt
```

## État d'avancement

Voir [`plan_execution.md`](plan_execution.md) pour le suivi détaillé par phase.
