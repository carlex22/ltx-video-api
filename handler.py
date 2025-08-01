# handler.py

import torch
import numpy as np
import random
import os
import yaml
from pathlib import Path
import imageio
import tempfile
from PIL import Image
from huggingface_hub import hf_hub_download
import shutil
import gc
import runpod
import requests
import traceback

# Importe as funções e classes do seu projeto.
# Certifique-se que os arquivos 'inference.py' e a pasta 'ltx_video' estão no repositório.
from inference import (
    create_ltx_video_pipeline, create_latent_upsampler, load_image_to_tensor_with_resize_and_crop,
    seed_everething, get_device, calculate_padding
)
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem, LTXMultiScalePipeline, LTXVideoPipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

# ----------------- PARTE 1: INICIALIZAÇÃO (roda uma vez no cold start) -----------------
# Este bloco de código é executado apenas uma vez quando um novo worker é iniciado.
# É o lugar perfeito para carregar modelos pesados e preparar tudo.

print("Iniciando o worker... Carregando modelos na GPU.")

TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config_file_path = "configs/ltxv-13b-0.9.8-distilled.yaml"

with open(config_file_path, "r") as file:
    PIPELINE_CONFIG_YAML = yaml.safe_load(file)

LTX_REPO = "Lightricks/LTX-Video"
models_dir = "downloaded_models"
Path(models_dir).mkdir(parents=True, exist_ok=True)

print("Baixando checkpoint principal...")
distilled_model_actual_path = hf_hub_download(repo_id=LTX_REPO, filename=PIPELINE_CONFIG_YAML["checkpoint_path"], local_dir=models_dir, local_dir_use_symlinks=False)
print("Baixando upscaler espacial...")
spatial_upscaler_actual_path = hf_hub_download(repo_id=LTX_REPO, filename=PIPELINE_CONFIG_YAML["spatial_upscaler_model_path"], local_dir=models_dir, local_dir_use_symlinks=False)

print(f"Carregando pipelines na GPU ({TARGET_DEVICE})...")
pipeline_instance = create_ltx_video_pipeline(ckpt_path=distilled_model_actual_path, precision=PIPELINE_CONFIG_YAML["precision"], text_encoder_model_name_or_path=PIPELINE_CONFIG_YAML["text_encoder_model_name_or_path"], sampler=PIPELINE_CONFIG_YAML["sampler"], device=TARGET_DEVICE)
latent_upsampler_instance = create_latent_upsampler(spatial_upscaler_actual_path, device=TARGET_DEVICE)

print("Modelos carregados na GPU. Worker pronto para receber jobs.")

# Função para fazer upload do vídeo gerado.
# ATENÇÃO: file.io é um serviço público e temporário, ideal para testes.
# Para produção, substitua esta função por um upload para um serviço de armazenamento
# persistente como Amazon S3, Google Cloud Storage ou Cloudflare R2 usando 'boto3'.
def upload_file_to_public_url(file_path):
    try:
        with open(file_path, 'rb') as f:
            response = requests.post('https://file.io', files={'file': f}, timeout=60)
        response.raise_for_status()
        return response.json().get('link')
    except Exception as e:
        print(f"Falha no upload do arquivo: {e}")
        return None

# ----------------- PARTE 2: HANDLER (roda a cada requisição) -----------------
def handler(job):
    """
    Processa uma única requisição de geração de vídeo.
    Recebe os parâmetros via 'job["input"]' e retorna um JSON com a URL do vídeo ou um erro.
    """
    job_input = job['input']

    # Extrai os parâmetros do input, com valores padrão para segurança
    prompt = job_input.get('prompt', 'A majestic dragon flying over a medieval castle')
    negative_prompt = job_input.get('negative_prompt', 'bad quality, inconsistent motion, blurry, shaky, distorted')
    input_image_url = job_input.get('input_image_url', None)
    height = job_input.get('height', 512)
    width = job_input.get('width', 704)
    duration_seconds = job_input.get('duration_seconds', 2.0)
    fps = job_input.get('fps', 24)
    seed = job_input.get('seed', None) # Se nulo, será aleatório
    cfg_scale = job_input.get('cfg_scale', 3.0)
    improve_texture = job_input.get('improve_texture', False)

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    internal_mode = "image-to-video" if input_image_url else "text-to-video"
    input_image_filepath = None
    temp_dir = tempfile.mkdtemp()

    try:
        if internal_mode == "image-to-video":
            print(f"Modo Image-to-Video. Baixando imagem de: {input_image_url}")
            response = requests.get(input_image_url, stream=True, timeout=20)
            response.raise_for_status()
            # Salva a imagem em um diretório temporário
            input_image_filepath = os.path.join(temp_dir, "input_image.png")
            with open(input_image_filepath, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            print(f"Imagem salva em: {input_image_filepath}")

        # Lógica de geração (adaptada da sua versão com Gradio)
        seed_everething(int(seed))
        
        MAX_NUM_FRAMES = 257
        target_frames_ideal = duration_seconds * fps
        n_val = round((float(round(target_frames_ideal)) - 1.0) / 8.0)
        actual_num_frames = max(9, min(int(n_val * 8 + 1), MAX_NUM_FRAMES))
        height_padded = ((int(height) - 1) // 32 + 1) * 32
        width_padded = ((int(width) - 1) // 32 + 1) * 32
        num_frames_padded = ((actual_num_frames - 2) // 8 + 1) * 8 + 1
        padding_values = calculate_padding(int(height), int(width), height_padded, width_padded)
        
        call_kwargs = { "prompt": prompt, "negative_prompt": negative_prompt, "height": height_padded, "width": width_padded, "num_frames": num_frames_padded, "frame_rate": int(fps), "generator": torch.Generator(device=TARGET_DEVICE).manual_seed(int(seed)), "output_type": "pt", "conditioning_items": None, "media_items": None, "decode_timestep": PIPELINE_CONFIG_YAML["decode_timestep"], "decode_noise_scale": PIPELINE_CONFIG_YAML["decode_noise_scale"], "stochastic_sampling": PIPELINE_CONFIG_YAML["stochastic_sampling"], "image_cond_noise_scale": 0.15, "is_video": True, "vae_per_channel_normalize": True, "mixed_precision": (PIPELINE_CONFIG_YAML["precision"] == "mixed_precision"), "offload_to_cpu": False, "enhance_prompt": False }
        stg_mode_str = PIPELINE_CONFIG_YAML.get("stg_mode", "attention_values")
        stg_map = { "stg_av": SkipLayerStrategy.AttentionValues, "attention_values": SkipLayerStrategy.AttentionValues, "stg_as": SkipLayerStrategy.AttentionSkip, "attention_skip": SkipLayerStrategy.AttentionSkip, "stg_r": SkipLayerStrategy.Residual, "residual": SkipLayerStrategy.Residual, "stg_t": SkipLayerStrategy.TransformerBlock, "transformer_block": SkipLayerStrategy.TransformerBlock }
        call_kwargs["skip_layer_strategy"] = stg_map.get(stg_mode_str.lower(), SkipLayerStrategy.AttentionValues)

        if internal_mode == "image-to-video":
            media_tensor = load_image_to_tensor_with_resize_and_crop(input_image_filepath, int(height), int(width))
            media_tensor = torch.nn.functional.pad(media_tensor, padding_values)
            call_kwargs["conditioning_items"] = [ConditioningItem(media_tensor.to(TARGET_DEVICE), 0, 1.0)]
        
        print(f"Executando pipeline com seed {seed}. Melhorar Textura: {improve_texture}")
        if improve_texture and latent_upsampler_instance:
            multi_scale_pipeline_obj = LTXMultiScalePipeline(pipeline_instance, latent_upsampler_instance)
            first_pass_args = {**PIPELINE_CONFIG_YAML.get("first_pass", {}), "guidance_scale": float(cfg_scale)}
            second_pass_args = {**PIPELINE_CONFIG_YAML.get("second_pass", {}), "guidance_scale": float(cfg_scale)}
            multi_scale_call_kwargs = {**call_kwargs, "downscale_factor": PIPELINE_CONFIG_YAML["downscale_factor"], "first_pass": first_pass_args, "second_pass": second_pass_args}
            result_images_tensor = multi_scale_pipeline_obj(**multi_scale_call_kwargs).images
        else:
            first_pass_config = PIPELINE_CONFIG_YAML.get("first_pass", {})
            single_pass_kwargs = {**call_kwargs, "guidance_scale": float(cfg_scale), "timesteps": first_pass_config.get("timesteps")}
            result_images_tensor = pipeline_instance(**single_pass_kwargs).images

        pad_left, pad_right, pad_top, pad_bottom = padding_values
        slice_h, slice_w = (-pad_bottom if pad_bottom > 0 else None), (-pad_right if pad_right > 0 else None)
        result_images_tensor_cropped = result_images_tensor[:, :, :actual_num_frames, pad_top:slice_h, pad_left:slice_w]
        video_np = (result_images_tensor_cropped[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
        
        output_video_path = os.path.join(temp_dir, f"output.mp4")
        print(f"Salvando vídeo em: {output_video_path}")
        with imageio.get_writer(output_video_path, fps=call_kwargs["frame_rate"], codec='libx264', quality=8) as video_writer:
            for frame in video_np: video_writer.append_data(frame)

        print("Fazendo upload do vídeo gerado...")
        public_url = upload_file_to_public_url(output_video_path)

        if not public_url:
            return {"error": "Failed to upload the generated video."}

        print("Geração e upload concluídos com sucesso.")
        return {"video_url": public_url, "seed_used": seed}

    except Exception as e:
        # Imprime o erro completo no log do worker para depuração
        traceback.print_exc()
        return {"error": f"An unexpected error occurred: {e}"}
        
    finally:
        # Limpa o diretório temporário para não acumular arquivos
        print(f"Limpando diretório temporário: {temp_dir}")
        shutil.rmtree(temp_dir)
        # Libera memória da GPU para garantir que o worker esteja limpo para o próximo job
        gc.collect()
        torch.cuda.empty_cache()

# Inicia o servidor serverless do RunPod, passando a função handler como o ponto de entrada para os jobs.
runpod.serverless.start({"handler": handler})
