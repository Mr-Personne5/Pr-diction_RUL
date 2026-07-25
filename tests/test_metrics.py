"""Tests des métriques (src/metrics.py) sur des cas jouets, valeurs calculables à la main."""

import numpy as np

from src.metrics import asymmetric_score, rmse


def test_rmse_zero_for_perfect_predictions():
    y_true = np.array([100, 50, 10])
    assert rmse(y_true, y_true) == 0.0


def test_rmse_known_value():
    # Erreur constante de 4 partout -> RMSE = 4.
    y_true = np.array([100, 50, 10])
    y_pred = y_true + 4
    assert rmse(y_true, y_pred) == 4.0


def test_asymmetric_score_zero_for_perfect_predictions():
    y_true = np.array([100, 50, 10])
    assert asymmetric_score(y_true, y_true) == 0.0


def test_asymmetric_score_penalizes_late_more_than_early():
    # Même ampleur d'erreur (10), mais de signe opposé : une prédiction en retard
    # (le modèle promet 10 cycles de vie en trop) doit coûter plus cher qu'une
    # prédiction en avance de 10 cycles, par construction du score PHM08.
    y_true = np.array([100.0])
    late_prediction = y_true + 10  # d = +10
    early_prediction = y_true - 10  # d = -10

    assert asymmetric_score(y_true, late_prediction) > asymmetric_score(y_true, early_prediction)


def test_asymmetric_score_matches_hand_computed_value():
    y_true = np.array([100.0])
    y_pred = np.array([110.0])  # d = +10 -> exp(10/10) - 1
    expected = np.exp(1) - 1
    assert asymmetric_score(y_true, y_pred) == expected
