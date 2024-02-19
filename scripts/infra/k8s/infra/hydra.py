import pulumi
from pulumi_kubernetes.core.v1 import Namespace
from pulumi_kubernetes.yaml import ConfigFile

# Define a new Kubernetes Namespace for the Hydra deployment
hydra_ns = Namespace(
    "hydra",
    metadata={"name": "hydra"}
)

# Load the Kubernetes Deployment manifest for Hydra
hydra_manifest = ConfigFile(
    "hydra-deployment",
    file="<hydra-deployment-manifest.yaml>",  # Path to your Hydra YAML deployment file
    transformations=[lambda obj: (
        obj['metadata']['namespace'] = hydra_ns.metadata["name"]
    ) if 'metadata' in obj else None]
)

# Export the namespace name
pulumi.export('hydra_namespace', hydra_ns.metadata["name"])
