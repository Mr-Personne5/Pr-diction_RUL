# Plan d'exécution — Mémoire M1 IA · Prédiction de RUL sur C-MAPSS

Liste de tâches cochables, ordonnée par priorité. Dérivée du protocole de recherche (v2).

`☐` = à faire · L'ordre des phases est l'ordre de priorité · **Règle d'or** : ne pas entamer une phase tant que la précédente n'est pas « faite ».

**Comment utiliser ce document.** Tu coches au fur et à mesure. La colonne « Critère fait » te dit quand une tâche est terminée — elle existe pour t'empêcher de t'éterniser ou de déborder. Les phases suivent les priorités du protocole : si le temps manque, tu t'arrêtes en fin de phase, jamais au milieu.

**Sur les implémentations existantes que tu trouveras en ligne** : sers-t'en pour t'orienter, jamais pour copier — surtout pas le code d'évaluation. La plupart des notebooks publics contiennent la fuite « découpage par cycle » que le protocole t'interdit. Tu construis ton propre pipeline.

---

## Phase 0 — Mise en place (prérequis)

| ☐ | Tâche | Comment faire | Ressource / où | Critère « fait » |
|---|---|---|---|---|
| ☐ | Créer le dépôt de code et l'environnement | `git init` ; environnement virtuel (venv/conda) ; figer les versions. | pytorch.org · scikit-learn | requirements figés, projet relancé à blanc. |
| ☐ | Télécharger C-MAPSS (FD001 + FD002) | Récupérer les 3 fichiers (train/test/RUL) de chaque sous-ensemble. | Kaggle · NASA | fichiers lus dans un DataFrame. |
| ☐ | Confirmer la structure depuis le readme | Vérifier colonnes (26), tailles, format du fichier RUL. | readme livré avec le jeu | tableau des sous-ensembles confirmé. |
| ☐ | Exploration initiale (EDA) | Tracer quelques trajectoires de capteurs ; calculer la variance de chaque capteur. | pandas + matplotlib | liste des capteurs constants établie sur FD001. |
| ☐ | Rafraîchir LSTM/Transformer séries temporelles (si besoin) | Revoir les bases avant de coder le modèle. | d2l.ai · PyTorch tutorials | — |

---

## Phase 1 (Priorité 1) — Baseline → LSTM/GRU sur FD001

> Fin de cette phase = un mémoire déjà soutenable. C'est ton plancher livrable.

| ☐ | Tâche | Comment faire | Ressource / où | Critère « fait » |
|---|---|---|---|---|
| ☐ | Pipeline de préparation (sans fuite) | Sélection capteurs ; normalisation (stats sur train seul) ; fenêtrage ; cible RUL par morceaux, plafond 125. | Protocole §4–5 | fonction sortant X, y propres, testée. |
| ☐ | Implémenter les deux métriques | RMSE + score asymétrique : d = préd − vrai ; d<0 → exp(−d/13)−1 ; d≥0 → exp(d/10)−1. | Saxena & Goebel 2008 | fonctions vérifiées sur cas jouets. |
| ☐ | Découpage par moteur | Split train/val sur identifiants de moteurs ENTIERS. | Protocole §7.1 | aucun moteur partagé train/val. |
| ☐ | Baseline P1 | Régression linéaire ou forêt aléatoire sur features agrégées par fenêtre. | scikit-learn | scores RMSE + asymétrique sur val. |
| ☐ | Modèle P2 (LSTM ou GRU) | Réseau de séquence sur fenêtres glissantes. | PyTorch / Keras | scores sur val ; comparaison à la baseline (test H1). |
| ☐ | Variance | Ré-entraîner sur 3 à 5 graines. | — | résultats en moyenne ± écart-type. |
| ☐ | Test final FD001 | Ouvrir le jeu de test fourni UNE seule fois. | — | scores test rapportés. |
| ☐ | Sensibilité du plafond RUL | Refaire avec 110 / 125 / 140. | Protocole §4 | effet rapporté (test H4). |

---

## Phase 2 (Priorité 2) — Robustesse FD001 vs FD002

> Fin de cette phase = mémoire complet et distinctif.

| ☐ | Tâche | Comment faire | Ressource / où | Critère « fait » |
|---|---|---|---|---|
| ☐ | Normalisation par régime sur FD002 | Regrouper par les 3 réglages opératoires ; normaliser par groupe. | Protocole §5 | régimes identifiés, normalisation par groupe. |
| ☐ | Ablation normalisation (test H3) | Comparer avec vs sans normalisation par régime. | — | écart mesuré et rapporté. |
| ☐ | Réentraîner le même modèle sur FD002 | Pipeline identique à FD001. | — | scores FD002 obtenus. |
| ☐ | Comparer la dégradation FD001 → FD002 | Tableau comparatif des deux métriques. | Protocole §7.2 | conclusion sur H2 (réfutée ou non). |

---

## Phase 3 (Priorité 3, conditionnée) — Transformer

> À n'entamer que si les phases 1 et 2 sont entièrement évaluées.

| ☐ | Tâche | Comment faire | Ressource / où | Critère « fait » |
|---|---|---|---|---|
| ☐ | Transformer léger | Encodeur à attention sur fenêtres ; même protocole d'évaluation. | PyTorch tutorials | scores sur les deux volets. |
| ☐ | Intégrer aux comparaisons | Ajouter aux tableaux de résultats des phases 1 et 2. | — | tableaux mis à jour. |

---

## Transversal — au fil de l'eau (ne pas reporter à la fin)

| ☐ | Tâche | Comment faire | Ressource / où | Critère « fait » |
|---|---|---|---|---|
| ☐ | Squelette du mémoire | Intro, état de l'art, problématique, méthode, résultats, discussion, limites. | à co-construire | plan validé par le directeur. |
| ☐ | Rédiger la section méthode | Adapter directement depuis le protocole. | le protocole v2 | section méthode rédigée. |
| ☐ | Journal d'expériences | Consigner hyperparamètres et résultats à chaque run. | — | journal tenu à jour. |
| ☐ | Reproductibilité | Graines, versions figées, README de relance. | Protocole §9 | un tiers peut relancer. |

---

## Ressources de référence

**Données.** Kaggle — NASA C-MAPS · Portail NASA Open Data · Dépôt PCoE (NASA)

**Outils.** scikit-learn (baseline) · PyTorch / Keras (deep learning) · Google Colab (calcul d'appoint, en complément de ton GPU).

**Références primaires (à citer).**
- Saxena, Goebel, Simon & Eklund, *« Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation »*, PHM08, 2008 (jeu + score asymétrique).
- Heimes, *« Recurrent Neural Networks for Remaining Useful Life Estimation »*, PHM 2008 (RUL par morceaux).

À récupérer sur IEEE Xplore / Google Scholar ; ne cite pas de mémoire les constantes sans avoir ouvert la source.

**Apprentissage.** d2l.ai (séries temporelles, RNN, attention) · PyTorch tutorials.

---

**Première action concrète :** Phase 0, ligne 1 — créer le dépôt et l'environnement. Tout le reste en dépend.