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
1. Un modèle de séquence profond (LSTM, puis Transformer) apporte-t-il un gain mesurable sur la
   prédiction du RUL par rapport à une baseline classique (régression linéaire / forêt
   aléatoire) ?
2. Ce gain — et la qualité de la prédiction en général — résiste-t-il au passage d'un seul
   régime opérationnel (FD001) à plusieurs régimes (FD002), qui exige une normalisation par
   régime pour rester comparable ?

L'évaluation s'appuie sur deux métriques complémentaires : le RMSE et le score asymétrique de
Saxena & Goebel (2008), qui pénalise davantage les prédictions en retard (sous-estimation du
RUL) que les prédictions en avance — pertinent en maintenance, où sous-estimer une panne
imminente coûte plus cher que la surestimer.

Le détail des phases, priorités et critères d'avancement est dans [`plan_execution.md`](plan_execution.md).

## Résultats clés

**FD001** (1 régime, 14 capteurs retenus sur 21) :

| Modèle | RMSE (val, moyenne ± écart-type sur 5 graines) | RMSE (test, ouvert une fois, seed 42) |
|---|---|---|
| Régression linéaire | 21.54 | 17.77 |
| Forêt aléatoire | 17.90 | 17.87 |
| LSTM | 13.25 ± 0.42 | **12.23** |
| Transformer | 12.90 ± 0.27 | — *(voir note ci-dessous)* |

- Le LSTM et le Transformer battent nettement les baselines classiques (**H1 confirmée**),
  robuste sur 5 graines pour chacun des deux.
- **LSTM vs Transformer sur FD001 : pas de différence statistiquement significative** (test t
  de Welch, p = 0.16) — les deux architectures sont indiscernables au regard de la variance
  inter-graines, malgré un écart de moyenne en apparence favorable au Transformer.
- Sensibilité au plafond RUL (110/125/140) : effet réel mais modéré (RMSE relatif 9.8% → 11.1%) — **H4** : la conclusion générale tient quel que soit le plafond choisi dans cette plage.
- La colonne test correspond au modèle d'une seule graine précise (42), seul modèle jamais
  évalué sur le test : le test FD001 a été ouvert une seule fois
  (`07_final_test_fd001.ipynb`), avant que le Transformer n'existe. Le rouvrir pour lui
  aurait violé la règle d'ouverture unique du protocole — c'est une limite assumée, à laquelle
  répondre explicitement en soutenance plutôt qu'à découvrir sur le moment.

**FD002** (6 régimes opérationnels, confirmés par k-means) :

| Configuration | Capteurs | RMSE (val) |
|---|---|---|
| Normalisation globale (naïve, comme FD001, seed 42) | 20 | 18.18 |
| **Normalisation par régime (seed 42)** | 14 | **15.07** |
| LSTM, normalisé par régime (moyenne ± écart-type, 5 graines) | 14 | **15.21 ± 0.13** |
| Transformer, normalisé par régime (moyenne ± écart-type, 5 graines) | 14 | 15.45 ± 0.18 |

- La normalisation par régime réduit le RMSE de ~17% par rapport à une normalisation naïve qui
  ignore les régimes (**H3 confirmée**).
- **FD001 → FD002, avec incertitude des deux côtés (H2)** : LSTM 13.25 ± 0.42 → 15.21 ± 0.13,
  soit **+14.8% de dégradation**, statistiquement très significative (test t de Welch,
  p = 0.0002). La normalisation par régime contient la difficulté supplémentaire sans
  l'annuler — **H2 confirmée**, cette fois avec une bande d'incertitude sur les deux volets.
- **LSTM vs Transformer sur FD002 : pas de différence statistiquement significative** (p = 0.05,
  à la limite du seuil conventionnel) — comme sur FD001, on ne peut pas affirmer qu'une
  architecture domine l'autre sur ce problème à cette échelle de données.
- Le test FD002 n'a délibérément jamais été ouvert (hors périmètre du plan pour cette
  comparaison).

*Ces comparaisons statistiques (5 graines par configuration, test t de Welch) sont dans
`notebooks/16_full_seed_variance.ipynb`, ajouté après une relecture externe du projet qui a
souligné, à juste titre, qu'un classement LSTM/Transformer sur une seule graine chacun n'était
pas défendable — voir aussi `scripts/run_full_seed_variance.py`, qui reproduit les mêmes
entraînements avec sauvegarde incrémentale (utile pour les runs longs sur FD002).*

## Méthodologie (résumé)

Pipeline commun à toutes les expériences, implémenté dans `src/preprocessing.py` :

1. **Split train/val par moteur entier** (jamais par cycle) — un moteur n'apparaît jamais des
   deux côtés, pour éviter toute fuite entre cycles quasi identiques.
2. **Sélection des capteurs** : on écarte les capteurs constants/quasi-constants, sur la base de
   l'écart-type calculé **sur le train uniquement**. Sur FD002 (plusieurs régimes), la variance
   doit être vérifiée *à l'intérieur de chaque régime* (`select_features_by_regime`), sinon des
   capteurs constants par régime mais variables globalement (à cause du seul changement de
   régime) sont gardés à tort.
3. **Normalisation z-score**, stats calculées sur le train uniquement ; par régime sur FD002
   (`normalize_by_regime`, régimes détectés par k-means sur les 3 réglages opérationnels).
4. **Cible RUL par morceaux, plafonnée** (125 par défaut) : RUL réel jusqu'à la panne, mais
   plafonné en début de vie où la dégradation n'est pas encore visible dans les capteurs.
5. **Fenêtrage glissant** (30 cycles) : toutes les fenêtres possibles pour le train/val, la
   dernière fenêtre uniquement pour le test (seul point où le RUL est connu en conditions
   réelles) ; padding par répétition de la première ligne pour les moteurs plus courts que la
   fenêtre.

Modèles (`src/torch_model.py`) : régression linéaire et forêt aléatoire (scikit-learn) sur
features agrégées par fenêtre (moyenne/écart-type/dernière valeur) ; LSTM et Transformer léger
(PyTorch) sur la séquence brute, tous deux lisant leur représentation du **dernier pas de
temps** pour prédire le RUL (protocole de lecture identique, pour une comparaison équitable).

Métriques (`src/metrics.py`) : RMSE (symétrique) et score asymétrique de Saxena & Goebel
(pénalise plus les prédictions en retard que les prédictions en avance).

## Structure du dépôt

```
DATA/CMaps/       données C-MAPSS (non versionnées, voir "Données" ci-dessous)
src/              pipeline de préparation, métriques, modèles (testé, réutilisé par les notebooks)
tests/            tests pytest de src/ (22 tests)
notebooks/        notebooks Jupyter, un par étape du plan (voir "Guide de lecture" ci-dessous)
models/           poids entraînés sauvegardés (ex. lstm_fd001_seed42.pt)
results/          résultats intermédiaires sauvegardés en CSV (ex. variance sur graines)
scripts/          scripts autonomes (ex. run_full_seed_variance.py, avec checkpointing)
Requirements.txt  dépendances Python figées
plan_execution.md plan d'exécution détaillé, suivi phase par phase
```

## Guide de lecture des notebooks

| Notebook | Contenu |
|---|---|
| `01_data_loading_check.ipynb` | Chargement FD001/FD002, confirmation de la structure vs le readme des données |
| `02_eda_fd001.ipynb` | EDA FD001 : variance des capteurs, capteurs constants/quasi-constants |
| `03_pipeline_check.ipynb` | Démonstration du pipeline de préparation (sélection, normalisation, RUL, fenêtrage) |
| `04_baseline_fd001.ipynb` | Baseline : régression linéaire et forêt aléatoire sur FD001 |
| `05_lstm_fd001.ipynb` | LSTM sur FD001, sauvegarde des poids entraînés |
| `06_lstm_seed_variance.ipynb` | Variance du LSTM sur 5 graines |
| `07_final_test_fd001.ipynb` | **Ouverture unique** du test FD001, scores finaux des 3 modèles |
| `08_rul_cap_sensitivity.ipynb` | Sensibilité au plafond RUL (110/125/140), sur le val |
| `09_fd002_regimes.ipynb` | Identification des 6 régimes opérationnels de FD002 (k-means) |
| `10_fd002_pipeline_check.ipynb` | Démonstration du pipeline régime-aware sur FD002 |
| `11_fd002_ablation_regime.ipynb` | Ablation : normalisation par régime vs globale, sur FD002 |
| `12_fd001_vs_fd002_comparison.ipynb` | Comparaison de dégradation FD001 → FD002 |
| `13_transformer_fd001.ipynb` | Transformer léger sur FD001 |
| `14_transformer_fd002.ipynb` | Transformer léger sur FD002 (régime-aware) |
| `15_phase3_summary.ipynb` | Tableaux de synthèse finaux (tous modèles, FD001 et FD002) |
| `16_full_seed_variance.ipynb` | Variance manquante (LSTM FD002, Transformer FD001/FD002, 5 graines) + test t LSTM vs Transformer |

## App de présentation (Streamlit)

`app.py` est un tableau de bord de présentation du travail, à lancer avec :

```bash
streamlit run app.py
```

Quatre pages (navigation dans la barre latérale) :
- **Contexte & méthode** — problématique et résumé du pipeline.
- **Résultats FD001** / **Résultats FD002** — tableaux et figures repris des notebooks.
- **Démo interactive** — choix du jeu de données (FD001/FD002) et de l'architecture
  (LSTM/Transformer), charge le modèle sauvegardé correspondant (`models/*_seed42.pt`) et
  prédit en direct le RUL d'un moteur de test choisi dans une liste, avec ses trajectoires de
  capteurs.

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

Lancer les tests : `pytest` depuis la racine du dépôt (22 tests, ~10s).

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

## Reproductibilité

- Toutes les graines (split, initialisation des modèles, mélange des batches) sont fixées
  explicitement (`src/torch_model.py::set_seed`), y compris le déterminisme cuDNN sur GPU
  (sans quoi deux runs identiques divergent légèrement).
- **`set_seed(seed)` doit être appelé AVANT de construire un modèle** : l'initialisation des
  poids en dépend et ne peut pas être rejouée après coup (voir docstring de `train_model`).
- Chaque jeu de test (FD001, dans `07_final_test_fd001.ipynb`) n'est ouvert qu'une seule fois,
  conformément au protocole ; toute analyse de sensibilité ultérieure (plafond RUL, etc.) se
  fait exclusivement sur le val.

## Limites connues

- La sélection de capteurs (`select_features` / `select_features_by_regime`) repose sur un
  seuil d'écart-type fixe (1e-2), choisi à partir de l'écart observé sur FD001 entre capteurs
  constants et informatifs — pas optimisé formellement.
- Le Transformer (Phase 3) n'a pas de score sur le test FD001 (cf. "Résultats clés") : le
  budget d'ouverture unique du test a été consommé en Phase 1, avant la conception du
  Transformer. Une meilleure planification aurait réservé l'ouverture du test pour le modèle
  final retenu par jeu de données, tout à la fin du projet.
- Le test FD002 n'a jamais été ouvert : la comparaison FD001/FD002 (H2) repose uniquement sur
  le val (mais désormais avec incertitude sur 5 graines des deux côtés, cf. "Résultats clés").
- Les métriques de validation sont calculées **par fenêtre** (plusieurs par moteur), celles du
  test **par moteur** (dernière fenêtre uniquement) : cohérent en interne pour les comparaisons
  menées ici, mais val et test ne sont pas sur la même base d'échantillons.
- Le score asymétrique utilisé pour comparer les graines (par fenêtre) n'est **pas** le score
  canonique de la littérature (somme par moteur sur le test) : à ne jamais comparer directement
  à des valeurs publiées, seulement en interne entre les configurations de ce projet.
- Le RUL réel du test est plafonné à 125 avant le calcul du RMSE, ce qui abaisse mécaniquement
  le RMSE affiché par rapport à un RUL non plafonné (les grosses erreurs de début de vie sont
  coupées) — cohérent avec ce que les modèles ont appris à prédire, mais à mentionner
  explicitement plutôt que de laisser un chiffre nu.

## État d'avancement

Les Phases 0 à 3 du plan sont complètes. Voir [`plan_execution.md`](plan_execution.md) pour le
détail par phase et la partie transversale restante (rédaction du mémoire, reproductibilité).

## Références

- Saxena, Goebel, Simon & Eklund, *« Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation »*, PHM08, 2008 (jeu de données C-MAPSS + score asymétrique).
- Heimes, *« Recurrent Neural Networks for Remaining Useful Life Estimation »*, PHM 2008 (RUL
  par morceaux, plafonné).
