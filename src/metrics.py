"""Métriques d'évaluation du RUL prédit (Saxena, Goebel, Simon & Eklund, PHM08 2008)."""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error, symétrique : une prédiction en avance ou en retard
    de la même ampleur coûte pareil. Sert de référence simple, à lire toujours à
    côté du score asymétrique ci-dessous, qui lui reflète le vrai coût métier.
    """
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score asymétrique de Saxena et al. (PHM08) : pénalise plus fort les prédictions
    en retard (d >= 0, le modèle annonce plus de vie restante que la réalité) que les
    prédictions en avance (d < 0), car en maintenance, rater une panne imminente coûte
    plus cher que déclencher une maintenance un peu trop tôt.

    d = prédiction - vérité
    d < 0 (prédiction en avance)  -> exp(-d / 13) - 1
    d >= 0 (prédiction en retard) -> exp( d / 10) - 1

    Retourne la somme sur tous les échantillons (convention de l'article original),
    donc ce score n'est comparable qu'entre évaluations faites sur le même nombre
    d'échantillons.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    per_sample = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(per_sample.sum())
