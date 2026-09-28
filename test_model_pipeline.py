"""Tests unitaires de model_pipeline."""

import os

from model_pipeline import prepare_data, train_model, save_model, load_model

DATA = "Churn_Modelling.csv"


def test_prepare_data_shapes():
    """Les jeux train/test sont non vides et cohérents."""
    X_train, X_test, y_train, y_test = prepare_data(DATA)
    assert len(X_train) == len(y_train) > 0
    assert len(X_test) == len(y_test) > 0


def test_save_and_load_model(tmp_path):
    """Le modèle peut être sauvegardé puis rechargé."""
    X_train, _, y_train, _ = prepare_data(DATA)
    model = train_model(X_train, y_train)
    path = os.path.join(tmp_path, "model.joblib")
    save_model(model, path)
    assert os.path.exists(path)
    assert load_model(path) is not None
