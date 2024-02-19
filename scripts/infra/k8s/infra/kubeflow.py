import pulumi
from pulumi_kubernetes.core.v1 import Namespace, Service
from pulumi_kubernetes.apps.v1 import Deployment

# Define a new Kubernetes Namespace for the Kubeflow deployment
kubeflow_ns = Namespace(
    "kubeflow",
    metadata={"name": "kubeflow"}
)

# Define the Kubeflow Deployment
kubeflow_deployment = Deployment(
    "kubeflow-deployment",
    metadata={
        "namespace": kubeflow_ns.metadata["name"]
    },
    spec={
        "selector": { "matchLabels": { "app": "kubeflow" } },
        "template": {
            "metadata": { "labels": { "app": "kubeflow" } },
            "spec": {
                "containers": [{
                    "name": "kubeflow",
					# Update the image of the deployment to fit your use case
                    "image": "<kubeflow-docker-image>"
                }]
            }
        }
    }
)

# Define a Service that exposes the Kubeflow deployment
kubeflow_service = Service(
    "kubeflow-service",
    metadata={
        "namespace": kubeflow_ns.metadata["name"],
		"labels": kubeflow_deployment.spec["template"]["metadata"]["labels"],
    },
    spec={
        "type": "LoadBalancer",
        "ports": [{"port": 80, "targetPort": 80, "protocol": "TCP"}],
        "selector": kubeflow_deployment.spec["template"]["metadata"]["labels"],
    },
)

# Export the namespace name
pulumi.export('kubeflow_namespace', kubeflow_ns.metadata["name"])
# Export the deployment name
pulumi.export('kubeflow_deployment', kubeflow_deployment.metadata.apply(lambda metadata: metadata["name"]))
# Export the service name
pulumi.export('kubeflow_service', kubeflow_service.metadata.apply(lambda metadata: metadata["name"]))
