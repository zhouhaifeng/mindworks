import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";
import * as eks from "@pulumi/eks";
import * as random from "@pulumi/random";
import * as crypto from "crypto";

// Generate a new random password for the DB
const password = new random.RandomPassword("cloud-ml-db-password", {
    length: 16,
    special: true,
});

// VPC and Subnet information for EKS cluster
const vpc = aws.ec2.getVpc({ default: true });
const subnets = vpc.then(vpc => aws.ec2.getSubnetIds({ vpcId: vpc.id }));

// Roles for EKS cluster
const instanceRole = new aws.iam.Role("cloud-ml-eks-instanceRole-role", {
    assumeRolePolicy: aws.iam.assumeRolePolicyForPrincipal({ Service: "ec2.amazonaws.com" }),
});

const eksRole = new aws.iam.Role("cloud-ml-eks-eksRole-role", {
    assumeRolePolicy: aws.iam.assumeRolePolicyForPrincipal({ Service: "eks.amazonaws.com" }),
});

// EKS Cluster
const cluster = new eks.Cluster("cloud-ml-eks", {
    vpcId: vpc.then(vpc => vpc.id),
    subnetIds: subnets.then(subnets => subnets.ids),
    instanceRoles: [instanceRole],
    roleMappings: [{
        roleArn: eksRole.arn,
        groups: ["system:masters"],
        username: "pulumi:admin",
    }],
    skipDefaultNodeGroup: true,
});

// EKS Node Group
const nodeGroup = new eks.NodeGroup("nodes", {
    cluster: cluster,
    instanceType: "t2.medium",
    desiredCapacity: 2,
    minSize: 1,
    maxSize: 2,
    labels: { env: "prod" },
});

// S3 Buckets
const dvcBucket = new aws.s3.Bucket("dvc-bucket");
const mlflowBucket = new aws.s3.Bucket("mlflow-bucket");
const artifactBucket = new aws.s3.Bucket("artifact-bucket");

// Exports
//export const eksClusterName = cluster.name;
//export const nodesName = nodeGroup.nodeGroup.name;
export const dbPassword = password.result;
export const dvcBucketName = dvcBucket.id;
export const mlflowBucketName = mlflowBucket.id;
export const artifactBucketName = artifactBucket.id;
