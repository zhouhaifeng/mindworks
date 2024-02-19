import pulumi
from pulumi_kubernetes.apps.v1 import Deployment

towhee_deployment = Deployment(
    "towhee-deployment",
    spec={
        "selector": {
            "matchLabels": {
                "app": "towhee",
            },
        },
        "replicas": 1,
        "template": {
            "metadata": {
                "labels": {
                    "app": "towhee"
                },
            },
            "spec": {
                "containers": [{
                    "name": "towhee",
                    "image": "towhee/towhee"
                }]
            }
        }
    },
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)
