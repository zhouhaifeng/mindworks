import pulumi
import pulumi_kubernetes as k8s
import pulumi_kubernetes.helm.v3 as helm

# Create a namespace for feast
namespace = k8s.core.v1.Namespace("feast-namespace",
    metadata=pulumi.ResourceArgs(metadata={"name": "feast"}),
    opts=pulumi.ResourceOptions(provider=k8s_provider))

# Fetch the shared chart from the Feast Helm repo
chart_values = {
    "feast-core": {
        "enabled": True,
    },
    "feast-online-serving": {
        "enabled": True,
    },
    "prometheus": {
        "enabled": True,
    },
}

feast_chart = helm.Chart(
    "feast",
    helm.ChartArgs(
        chart="feast",
        version="0.10.0",
        fetch_options=helm.FetchArgs(
            repo="https://helm.feast.dev",
        ),
        namespace=namespace.metadata.name,
        values=chart_values,
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider))

# Export the namespace name
pulumi.export("namespace", namespace.metadata.name)
