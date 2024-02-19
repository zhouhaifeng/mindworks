from pulumi_kubernetes.apps.v1 import Deployment
from pulumi_kubernetes.core.v1 import Namespace, Service

# Define a new Kubernetes Namespace for the Traefik deployment
traefik_ns = Namespace(
    "traefik",
    metadata={"name": "traefik"}
)

# Load the Kubernetes Deployment manifest for Traefik
traefik_manifest = ConfigFile(
    "traefik-deployment",
    file="<traefik-deployment-manifest.yaml>",  # Path to your Traefik YAML deployment file
    transformations=[lambda obj: (
        obj['metadata']['namespace'] = traefik_ns.metadata["name"]
    ) if 'metadata' in obj else None]
)

# Export the namespace name
pulumi.export('traefik_namespace', traefik_ns.metadata["name"])
