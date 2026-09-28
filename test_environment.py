import numpy as np
import pandas as pd
import sklearn
import mlflow

from model_pipeline import prepare_data


def test_environment():
    """Vérifie que l'environnement Python est correctement configuré."""
    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None
    assert mlflow.__version__ is not None


def test_prepare_data():
    """Vérifie que la préparation des données fonctionne."""
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")

    assert X_train.shape[0] == 8000
    assert X_test.shape[0] == 2000
    assert X_train.shape[1] == 9
    assert X_test.shape[1] == 9

    assert len(y_train) == 8000
    assert len(y_test) == 2000
