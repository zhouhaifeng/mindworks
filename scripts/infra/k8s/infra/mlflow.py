import pulumi
from pulumi_kubernetes.apps.v1 import Deployment

mlflow_deployment = Deployment(
    "mlflow-deployment",
    spec={
        "selector": {
            "matchLabels": {
                "app": "mlflow",
            },
        },
        "replicas": 1,
        "template": {
            "metadata": {
                "labels": {
                    "app": "mlflow"
                },
            },
            "spec": {
                "containers": [{
                    "name": "mlflow",
                    "image": "mlflow/mlflow"
                }]
            }
        }
    },
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)

# Export the name of the deployment
pulumi.export('mlflow_deployment_name', mlflow_deployment.metadata["name"])
