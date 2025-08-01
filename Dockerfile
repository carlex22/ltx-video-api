# Dockerfile

# Use uma imagem base oficial do RunPod com PyTorch e CUDA pré-instalados.
# Isso economiza tempo de build e garante compatibilidade.
FROM runpod/pytorch:2.1.2-py3.10-cuda12.1.1-devel-ubuntu22.04

# Define o diretório de trabalho dentro do contêiner.
# Todos os comandos subsequentes serão executados a partir daqui.
WORKDIR /app

# Copia todos os arquivos do seu repositório (handler.py, requirements.txt, pastas, etc.)
# para o diretório de trabalho /app dentro do contêiner.
COPY . /app

# Instala as dependências Python listadas no requirements.txt.
# --no-cache-dir é uma boa prática para manter a imagem do contêiner menor.
RUN pip install --no-cache-dir -r requirements.txt

# Define o comando que será executado quando um worker do RunPod for iniciado.
# Ele simplesmente executa o seu script handler.py com o Python.
CMD ["python", "handler.py"]
