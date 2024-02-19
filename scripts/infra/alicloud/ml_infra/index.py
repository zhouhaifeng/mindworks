import pulumi
import pulumi_alicloud as alicloud 
from pulumi_random import RandomPassword

# Generate a new random password for the DB
password = RandomPassword("cloud-ml-db-password", length=16, special=True)

# ACK Cluster
cluster = alicloud.cs.Kubernetes("k8s", 
    version="1.16.9-aliyun.1", 
    vswitch_ids=["vsw-2zex9p8b0sq7o6b26rs5h"], 
    new_nat_gateway=True, 
    worker_numbers=[3],
    worker_data_disks=[{"category": "cloud_efficiency", "size": "200"}],
    worker_instance_types=["ecs.g5.large"],
    slb_internet_enabled=True,
    container_cidr="172.20.0.0/16",
    service_cidr="172.21.0.0/20",
    install_cloud_monitor=True,
    password=password.result)


# OSS Bucket 
dvc_bucket = alicloud.oss.Bucket("dvc-bucket")
mlflow_bucket = alicloud.oss.Bucket("mlflow-bucket")
artifact_bucket = alicloud.oss.Bucket("artifact-bucket")

# Exports
pulumi.export('dbPassword', password.result)
pulumi.export('dvcBucketName', dvc_bucket.bucket_domain_name)
pulumi.export('mlflowBucketName', mlflow_bucket.bucket_domain_name)
pulumi.export('artifactBucketName', artifact_bucket.bucket_domain_name)
