milvus:

helm repo add milvus https://milvus-io.github.io/milvus-helm/
helm repo update
cd milvus-helm 
helm install milvus milvus/milvus --set cluster.enabled=true --set persistence.enabled=true --set mysql.enabled=true

or 

k8s on local:
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
minikube start
helm repo add milvus https://milvus-io.github.io/milvus-helm/
helm repo update
helm install mindworks milvus/milvus