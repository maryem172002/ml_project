"""
main.py
Point d'entrée : exécute les étapes du pipeline via des arguments CLI.
"""

import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn Modelling")
    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "all"],
        default="all",
        help="Étape à exécuter",
    )
    parser.add_argument("--data", default="Churn_Modelling.csv", help="Chemin du CSV")
    parser.add_argument("--model", default="classifier.joblib", help="Chemin du modèle")
    args = parser.parse_args()

    if args.step == "prepare":
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        print(f"Train : {X_train.shape} | Test : {X_test.shape}")

    elif args.step == "train":
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        model = train_model(X_train, y_train)
        save_model(model, args.model)

    elif args.step == "evaluate":
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        model = load_model(args.model)
        evaluate_model(model, X_test, y_test)

    elif args.step == "all":
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model, args.model)
        # Vérification du cycle save/load
        loaded = load_model(args.model)
        evaluate_model(loaded, X_test, y_test)


if __name__ == "__main__":
    main()
