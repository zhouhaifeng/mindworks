FROM python:3.10

ARG IMAGE_NAME=analyzer
ARG IMAGE_VERSION=1.0.0

WORKDIR /app
COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3 ./api.py"]











