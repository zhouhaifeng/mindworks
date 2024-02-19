import pulumi
from pulumi_kubernetes import Provider, core
from pulumi_kubernetes.helm.v3 import Helm
from pulumi_kubernetes.apps.v1 import Job

kubeconfig = pulumi.Config().require_secret("kubeconfig")

k8s_provider = Provider("k8s-provider", kubeconfig=kubeconfig)
job = Job(
    "dvc-job", 
    spec={
        "template": {
            "spec": {
                "containers": [{
                    "name": "dvc",
                    "image": "iterativeai/cml:0-dvc2-base1",  # using a container that has DVC pre-installed
                    "command": ["/bin/sh"],
                    "args": ["-c", "dvc --version"],  # just print DVC version as an example
                }],
                "restartPolicy": "Never",
            }
        }
    },
    metadata={
        "labels": {
            "runner": "dvc"
        }
    }, 
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)
