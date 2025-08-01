Gerador de Vídeo com LTX-Video (Destilado) - Endpoint para RunPod

Este repositório contém o código e a configuração para implantar um endpoint de geração de vídeo de alta performance no RunPod. O sistema utiliza uma versão destilada do modelo Lightricks LTX-Video, otimizada para velocidade e eficiência, e expõe a funcionalidade através de uma API serverless.

Visão Geral do Projeto

O objetivo deste projeto é criar um serviço de back-end robusto e escalável para geração de vídeo a partir de texto ou de uma imagem de referência. A arquitetura é composta por três componentes principais:

O Modelo de IA: Uma versão destilada e eficiente do Lightricks/LTX-Video, capaz de criar vídeos curtos de alta qualidade.

O Endpoint (handler.py): O código do lado do servidor que carrega o modelo, recebe requisições via API, executa a inferência e retorna o vídeo gerado.

O Cliente (teste.py): Um script de linha de comando interativo para testar e interagir facilmente com o endpoint.

Componentes
1. O Modelo: LTX-Video (Versão Destilada)

Utilizamos o modelo Lightricks/LTX-Video na sua variante destilada. Esta versão foi escolhida por seu excelente equilíbrio entre qualidade visual e velocidade de inferência, tornando-a ideal para um serviço interativo onde o tempo de resposta é crucial.

Modelo Base: Lightricks/LTX-Video

Configuração: configs/ltxv-13b-0.9.8-distilled.yaml

Funcionalidades: Suporta tanto Texto-para-Vídeo quanto Imagem-para-Vídeo.

2. O Servidor de Endpoint (handler.py)

Este é o cérebro do nosso serviço. O handler.py foi projetado para rodar em um worker serverless do RunPod e é responsável por:

Inicialização (Cold Start): Baixar os modelos do Hugging Face Hub e carregá-los na memória da GPU na primeira inicialização do worker.

Processamento de Requisições: Aceitar um payload JSON contendo os parâmetros de geração (prompt, dimensões, seed, etc.).

Suporte a Imagem de Entrada: Aceitar uma imagem de referência de duas formas:

Base64 (Preferencial): Um campo input_image_base64 com a imagem codificada, ideal para enviar arquivos locais de forma segura.

URL: Um campo input_image_url para baixar a imagem de um link público.

Execução da Inferência: Chamar o pipeline do modelo LTX-Video com os parâmetros fornecidos.

Retorno do Resultado: Codificar o vídeo gerado em Base64 e retorná-lo diretamente na resposta da API, garantindo um fluxo autossuficiente sem dependências de armazenamento externo.

Gerenciamento de Recursos: Limpar arquivos temporários e liberar a memória da GPU após cada execução para garantir a estabilidade do worker.

3. O Cliente de Teste (teste.py)

Para facilitar a interação e os testes com o endpoint, o script teste.py fornece uma interface de linha de comando amigável.

Interatividade:

Escaneia uma pasta local (imagens_para_enviar) e lista as imagens encontradas.

Permite ao usuário escolher uma imagem da lista ou prosseguir no modo Texto-para-Vídeo.

Solicita o prompt de texto.

Fluxo Assíncrono:

Envia a requisição para a rota /run do endpoint para iniciar o job.

Monitora o progresso consultando a rota /status e exibe o status em tempo real (IN_QUEUE, IN_PROGRESS, COMPLETED).

Ao concluir, decodifica o vídeo recebido em Base64 e o salva localmente como um arquivo .mp4.

Como Usar
1. Configurar o Endpoint no RunPod

Faça o upload do conteúdo deste repositório (incluindo handler.py, inference.py, configs/, etc.) para um repositório Git.

Crie um novo Serverless Endpoint no RunPod.

Aponte o endpoint para o seu repositório Git.

Configure as variáveis de ambiente e o hardware:

Container Disk: Certifique-se de alocar espaço suficiente (ex: 80 GB) para os modelos.

GPU: Escolha uma GPU compatível (ex: A100 80GB).

O RunPod irá construir o ambiente e iniciar o serviço. Copie o ID do seu endpoint.

2. Configurar e Executar o Cliente de Teste (teste.py)

Clone este repositório para sua máquina local.

Instale as dependências:

Generated bash
pip install requests


Configure o script: Abra o arquivo teste.py e edite as seguintes constantes no topo do arquivo:

Generated python
API_KEY = "SEU_API_KEY_DO_RUNPOD_AQUI"
ENDPOINT_ID = "SEU_ID_DO_ENDPOINT_AQUI"
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

(Opcional) Adicione imagens: Crie uma pasta chamada imagens_para_enviar no mesmo diretório do script e coloque seus arquivos de imagem (.png, .jpg) dentro dela.

Execute o cliente:

python teste.py

6.  Siga as instruções no terminal para selecionar uma imagem (ou não) e fornecer um prompt. O script cuidará do resto e salvará o vídeo gerado na pasta atual.
