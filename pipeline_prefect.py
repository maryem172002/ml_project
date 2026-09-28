# pylint: disable=invalid-name
"""Pipeline ML orchestré avec Prefect."""

import argparse
import os
import subprocess  # nosec B404
import sys

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
)

# Fichiers ciblés pour la qualité et la sécurité (uniquement notre code)
CODE_FILES = [
    "model_pipeline.py",
    "main.py",
    "pipeline_prefect.py",
]

# ============================================================
# TASK : RÉCUPÉRATION DU PROJET (étape 06)
# ============================================================


@task(name="clone-repo")
def clone_repo_task(repo_url, dest="projet_clone"):
    """Clone le dépôt, ou le met à jour s'il existe déjà."""
    print("=== Récupération du projet depuis Git ===")

    if os.path.isdir(os.path.join(dest, ".git")):
        subprocess.run(["git", "-C", dest, "pull"], check=True)  # nosec B603 B607
    else:
        subprocess.run(["git", "clone", repo_url, dest], check=True)  # nosec B603 B607


# ============================================================
# TASKS : QUALITÉ ET CONTRÔLE DU CODE
# ============================================================


@task(name="install-dependencies")
def install_dependencies_task():
    """Installe les dépendances du projet."""
    print("=== Installation des dépendances ===")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )  # nosec B603


@task(name="format-code")
def format_code_task():
    """Formate le code avec Black."""
    print("=== Formatage du code avec Black ===")

    subprocess.run(
        [sys.executable, "-m", "black", *CODE_FILES],
        check=True,
    )  # nosec B603


@task(name="quality-code")
def quality_code_task():
    """Analyse la qualité du code avec Pylint."""
    print("=== Analyse de qualité avec Pylint ===")

    result = subprocess.run(
        [sys.executable, "-m", "pylint", *CODE_FILES],
        check=False,
    )  # nosec B603

    print(f"Pylint terminé avec le code de retour : {result.returncode}")


@task(name="security-code")
def security_code_task():
    """Analyse les problèmes de sécurité avec Bandit."""
    print("=== Analyse de sécurité avec Bandit ===")

    result = subprocess.run(
        [sys.executable, "-m", "bandit", *CODE_FILES],
        check=False,
    )  # nosec B603

    print(f"Bandit terminé avec le code de retour : {result.returncode}")


@task(name="run-tests")
def run_tests_task():
    """Exécute les tests unitaires avec Pytest."""
    print("=== Exécution des tests avec Pytest ===")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_environment.py",
            "test_model_pipeline.py",
            "-v",
        ],
        check=True,
    )  # nosec B603


# ============================================================
# TASKS : PIPELINE ML
# ============================================================


@task(name="prepare-data")
def prepare_data_task(data_path="Churn_Modelling.csv"):
    """Prépare les données."""
    print("=== Préparation des données ===")

    X_train, X_test, y_train, y_test = prepare_data(data_path)

    print(f"Train : {X_train.shape}")
    print(f"Test  : {X_test.shape}")

    return X_train, X_test, y_train, y_test


@task(name="train-model")
def train_model_task(X_train, y_train):
    """Entraîne le modèle."""
    print("=== Entraînement du modèle ===")

    return train_model(X_train, y_train)


@task(name="save-model")
def save_model_task(model, model_path="classifier.joblib"):
    """Sauvegarde le modèle."""
    print("=== Sauvegarde du modèle ===")

    save_model(model, model_path)


@task(name="load-model")
def load_model_task(model_path="classifier.joblib"):
    """Charge le modèle."""
    print("=== Chargement du modèle ===")

    return load_model(model_path)


@task(name="evaluate-model")
def evaluate_model_task(model, X_test, y_test):
    """Évalue le modèle."""
    print("=== Évaluation du modèle ===")

    return evaluate_model(model, X_test, y_test)


# ============================================================
# FLOWS
# ============================================================


@flow(name="install")
def flow_install():
    """Installe les dépendances."""
    install_dependencies_task()


@flow(name="code")
def flow_code():
    """Dépendances, formatage, qualité, sécurité et tests."""
    install_dependencies_task()
    format_code_task()
    quality_code_task()
    security_code_task()
    run_tests_task()


@flow(name="train")
def flow_train():
    """Prépare les données, entraîne et sauvegarde le modèle."""
    X_train, _, y_train, _ = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model)
    return model


@flow(name="evaluate")
def flow_evaluate():
    """Charge et évalue le modèle."""
    _, X_test, _, y_test = prepare_data_task()
    model = load_model_task()
    evaluate_model_task(model, X_test, y_test)


@flow(name="all")
def flow_all(repo_url: str = ""):
    """Clone (optionnel), contrôle le code puis exécute le pipeline ML."""
    if repo_url:
        clone_repo_task(repo_url)

    # Contrôle du code (avant la préparation des données)
    flow_code()

    # Pipeline ML
    X_train, X_test, y_train, y_test = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model)
    loaded_model = load_model_task()
    evaluate_model_task(loaded_model, X_test, y_test)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML orchestré avec Prefect")
    parser.add_argument(
        "--flow",
        required=True,
        choices=["install", "code", "all", "train", "entrainement", "evaluate"],
        help="Flow à exécuter",
    )
    args = parser.parse_args()

    flows = {
        "install": flow_install,
        "code": flow_code,
        "all": flow_all,
        "train": flow_train,
        "entrainement": flow_train,
        "evaluate": flow_evaluate,
    }
    flows[args.flow]()
