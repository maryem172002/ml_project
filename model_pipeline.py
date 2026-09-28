"""
model_pipeline.py
Fonctions modulaires du pipeline ML : prédiction du churn client.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def prepare_data(csv_path, test_size=0.2, random_state=1):
    """
    Charge et prétraite les données.

    Args:
        csv_path (str): chemin du fichier CSV (Churn_Modelling.csv).
        test_size (float): proportion du jeu de test.
        random_state (int): graine pour la reproductibilité.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(csv_path)

    # Vérifications de base (missing values / doublons)
    print(f"Shape du dataset : {df.shape}")
    print(f"Valeurs manquantes : {int(df.isna().sum().sum())}")
    print(f"Doublons : {df.duplicated().any()}")

    # Encodage de la colonne Gender
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Suppression des colonnes inutiles
    df = df.drop(columns=["Surname", "Geography", "RowNumber", "CustomerId"])

    # Séparation features / cible
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Entraîne un RandomForestClassifier.

    Args:
        X_train: features d'entraînement.
        y_train: cible d'entraînement.
        n_estimators (int): nombre d'arbres.
        random_state (int): graine.

    Returns:
        RandomForestClassifier: modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur le jeu de test.

    Args:
        model: modèle entraîné.
        X_test: features de test.
        y_test: cible de test.

    Returns:
        dict: accuracy et matrice de confusion.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy score : {accuracy * 100:.2f}%")
    print("Matrice de confusion :")
    print(matrix)
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))

    return {"accuracy": accuracy, "confusion_matrix": matrix}


def save_model(model, path="classifier.joblib"):
    """
    Sauvegarde le modèle entraîné avec joblib.

    Args:
        model: modèle à sauvegarder.
        path (str): chemin du fichier de sortie.
    """
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans : {path}")


def load_model(path="classifier.joblib"):
    """
    Charge un modèle sauvegardé avec joblib.

    Args:
        path (str): chemin du fichier modèle.

    Returns:
        Le modèle chargé.
    """
    model = joblib.load(path)
    print(f"Modèle chargé depuis : {path}")
    return model
