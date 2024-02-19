import pulumi
from pulumi_kubernetes.apps.v1 import Deployment

prefect_deployment = Deployment(
    "prefect-deployment",
    spec={
        "selector": {
            "matchLabels": {
                "app": "prefect"
            },
        },
        "replicas": 1,
        "template": {
            "metadata": {
                "labels": {
                    "app": "prefect"
                },
            },
            "spec": {
                "containers": [{
                    "name": "prefect",
                    "image": "prefecthq/server:latest", # Use the appropriate Prefect version
                    # Refer to the prefect documentation for other necessary configurations
                }]
            }
        }
    },
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)
