FROM python:3.10

# 定义镜像名称和版本
ARG IMAGE_NAME=llmcis/spider
ARG IMAGE_VERSION=1.0.0

# 将工作目录设置为/app
WORKDIR /app
# 将当前目录下的所有文件复制到容器的工作目录中
COPY . /app
# 安装所需的依赖项
RUN pip install --no-cache-dir -r requirements.txt
# 定义容器启动时运行的命令
CMD ["python3 ./steel.py"]
# 定义镜像标签
LABEL org.opencontainers.image.name=$IMAGE_NAME:$IMAGE_VERSION \
      org.opencontainers.image.version=$IMAGE_VERSION \
      org.opencontainers.image.description="llmcis spider" \
      org.opencontainers.image.authors="llmcis" \
      org.opencontainers.image.vendor="llmcis" \
      org.opencontainers.image.url=""