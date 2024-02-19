git clone https://github.com/milvus-io/milvus-helm.git
cd milvus-helm 
helm install --set cluster.enabled=true --set persistence.enabled=true --set mysql.enabled=true my-release  .