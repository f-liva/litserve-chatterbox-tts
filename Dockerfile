# Change CUDA and cuDNN version here
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install software-properties-common python3.11 python3.11-venv python3.11-distutils -y && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Clone the application repository
RUN git clone --depth 1 --branch main https://github.com/f-liva/litserve-chatterbox-tts.git /app

WORKDIR /app

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["python", "/app/server.py"]
