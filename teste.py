# teste.py (Versão que envia a imagem em Base64)

import requests
import time
import os
import base64

# ... (Constantes e outras funções permanecem as mesmas) ...
API_KEY = "SuaAPIkey"
ENDPOINT_ID = "0rz4mond4u2ax2"
IMAGE_FOLDER = 'imagens_para_enviar'
RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/"

def carregar_imagem_base64(caminho_imagem):
    """Lê um arquivo de imagem e o codifica em uma string Base64."""
    try:
        with open(caminho_imagem, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Erro ao carregar a imagem {caminho_imagem}: {e}")
        return None

def submit_job(prompt_text, image_path):
    """Envia a tarefa, incluindo a imagem em Base64 se fornecida."""
    print("\n[PASSO 1] Enviando a tarefa para o RunPod...")
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    
    job_input = {
        "prompt": prompt_text,
        "negative_prompt": "bad quality, blurry, distorted"
    }

    if image_path:
        print(f"Codificando a imagem {image_path} em Base64...")
        base64_image = carregar_imagem_base64(image_path)
        if base64_image:
            # Adiciona o novo campo que o handler irá esperar
            job_input['input_image_base64'] = base64_image
        else:
            print("Falha ao codificar a imagem. Enviando como Texto-para-Vídeo.")
            
    data = {'input': job_input}
    # ... (o resto da função submit_job continua igual) ...
    try:
        response = requests.post(RUN_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        job_id = result.get('id')
        print(f"Tarefa enviada com sucesso! ID do Job: {job_id}")
        return job_id
    except requests.exceptions.RequestException as e:
        print(f"Erro ao enviar a tarefa: {e}")
        if e.response: print("Resposta da API:", e.response.text)
        return None

# As outras funções (poll_job_status, save_video_from_result, etc.) não precisam de alteração.
# O main também não precisa de alteração.
# (Cole aqui o resto do código do teste.py anterior)
def find_image_files(folder_path):
    """
    Busca e retorna uma lista de arquivos de imagem em um diretório.
    """
    supported_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    try:
        files = os.listdir(folder_path)
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in supported_extensions]
        return sorted(image_files)
    except FileNotFoundError:
        return []

def get_user_input(images):
    """
    Gerencia a interação com o usuário para selecionar uma imagem e um prompt.
    """
    selected_image_path = None
    if not images:
        print(f"\nNenhuma imagem encontrada na pasta '{IMAGE_FOLDER}'. Modo Texto-para-Vídeo.")
    else:
        print("\nImagens encontradas na pasta:")
        for i, img_name in enumerate(images):
            print(f"  [{i+1}] - {img_name}")
        print("  [0] - Não usar nenhuma imagem (modo Texto-para-Vídeo)")

        while True:
            try:
                choice = int(input("\nEscolha o número da imagem (ou 0 para nenhuma): "))
                if 0 <= choice <= len(images):
                    if choice > 0:
                        selected_image_path = os.path.join(IMAGE_FOLDER, images[choice - 1])
                        print(f"Imagem selecionada: {selected_image_path}")
                    else:
                        print("Modo Texto-para-Vídeo selecionado.")
                    break
                else:
                    print("Opção inválida.")
            except ValueError:
                print("Entrada inválida. Por favor, digite um número.")

    prompt = input("\nDigite o seu prompt:\n> ")
    return selected_image_path, prompt

def poll_job_status(job_id):
    """
    Verifica o status do job em intervalos regulares até a sua conclusão.
    """
    print("\n[PASSO 2] Verificando o status da tarefa (a cada 5 segundos)...")
    headers = {'Authorization': f'Bearer {API_KEY}'}

    while True:
        try:
            time.sleep(5)
            status_response = requests.get(STATUS_URL + job_id, headers=headers)
            status_response.raise_for_status()
            result = status_response.json()
            status = result.get('status')

            print(f"  -> Status atual: {status}")

            if status in ['COMPLETED', 'FAILED']:
                print(f"Tarefa finalizada com status: {status}")
                return result
        except requests.exceptions.RequestException as e:
            print(f"Erro de rede ao verificar status: {e}")
            return None
        except KeyboardInterrupt:
            print("\nVerificação de status interrompida pelo usuário.")
            return None

def save_video_from_result(result):
    """
    Processa o resultado final, decodifica o vídeo e o salva no disco.
    """
    print("\n[PASSO 3] Processando o resultado final...")
    if not result:
        print("Nenhum resultado recebido para processar.")
        return

    if result.get('status') == 'FAILED':
        print("A tarefa falhou. Detalhes:", result.get('output', 'Nenhum detalhe fornecido.'))
        return

    output_data = result.get('output')
    if not output_data:
        print("Erro: A chave 'output' está ausente na resposta.")
        print("Resultado recebido:", result)
        return

    # O campo 'video_base64' está diretamente dentro de 'output'
    if 'video_base64' in output_data:
        try:
            video_data = base64.b64decode(output_data['video_base64'])
            output_filename = f"video_gerado_{int(time.time())}.mp4"
            with open(output_filename, 'wb') as f:
                f.write(video_data)
            print(f"\nSucesso! Vídeo salvo como: '{output_filename}'")
        except (TypeError, base64.binascii.Error) as e:
            print(f"Erro ao decodificar os dados do vídeo em Base64: {e}")
    else:
        print("Erro: Campo 'video_base64' não encontrado no resultado.")
        print("Resultado recebido:", result)


def main():
    """
    Função principal que orquestra o fluxo de execução do cliente.
    """
    print("--- Cliente Assíncrono para API RunPod ---")

    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"\nPasta '{IMAGE_FOLDER}' criada. Você pode adicionar imagens de entrada nela.")

    images = find_image_files(IMAGE_FOLDER)
    selected_image_path, prompt = get_user_input(images)

    job_id = submit_job(prompt, selected_image_path)
    if job_id:
        final_result = poll_job_status(job_id)
        save_video_from_result(final_result)

if __name__ == '__main__':
    main()
