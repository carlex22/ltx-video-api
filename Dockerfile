# Dockerfile - Versão Robusta

# PASSO 1: Use uma imagem oficial da NVIDIA como base.
# Esta imagem é garantida de existir e tem os drivers CUDA corretos.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# PASSO 2: Evite prompts interativos durante a instalação de pacotes
ENV DEBIAN_FRONTEND=noninteractive

# PASSO 3: Instale Python, pip e outras ferramentas essenciais
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Crie um link simbólico para que 'python' aponte para 'python3.10'
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# PASSO 4: Defina o diretório de trabalho
WORKDIR /app

# PASSO 5: Copie o requirements.txt primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# PASSO 6: Instale as dependências Python
# Instale PyTorch separadamente para garantir a versão correta para CUDA 12.1
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# PASSO 7: Copie o resto do seu código
COPY . .

# PASSO 8: Defina o comando para iniciar a aplicação
CMD ["python", "handler.py"]
