"""Déploiements Prefect avec planification quotidienne."""

from prefect import serve

from pipeline_prefect import (
    flow_all,
    flow_train,
    flow_evaluate,
    flow_code,
    flow_install,
)

if __name__ == "__main__":
    serve(
        flow_all.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",  # tous les jours à 02:00
            tags=["full-pipeline", "mlops"],
            # parameters={"repo_url": "https://github.com/<user>/<repo>.git"},
        ),
        flow_train.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        flow_evaluate.to_deployment(
            name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
        ),
        flow_code.to_deployment(name="ml-pipeline-code", tags=["quality", "mlops"]),
        flow_install.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
    )
