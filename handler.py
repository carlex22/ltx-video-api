# # ------------------------------------------------------
# handler.py

import torch
import numpy as np
import random
import os
import yaml
from pathlib import Path
import imageio
import tempfile
import shutil
import gc
import runpod
import requests
import traceback
import base64

# Módulos locais do projeto
from inference import (
    create_ltx_video_pipeline, create_latent_upsampler, load_image_to_tensor_with_resize_and_crop,
    seed_everething, get_device, calculate_padding
)
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem, LTXMultiScalePipeline, LTXVideoPipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

# --------------------------------------------------------------------------- #
#                      INICIALIZAÇÃO DO WORKER (COLD START)                     #
# --------------------------------------------------------------------------- #
TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE_PATH = "configs/ltxv-13b-0.9.8-distilled.yaml"

with open(CONFIG_FILE_PATH, "r") as file:
    PIPELINE_CONFIG = yaml.safe_load(file)

LTX_REPO = "Lightricks/LTX-Video"
MODELS_DIR = "downloaded_models"
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

# Download dos modelos a partir do Hugging Face Hub
distilled_model_path = hf_hub_download(
    repo_id=LTX_REPO,
    filename=PIPELINE_CONFIG["checkpoint_path"],
    local_dir=MODELS_DIR
)
spatial_upscaler_path = hf_hub_download(
    repo_id=LTX_REPO,
    filename=PIPELINE_CONFIG["spatial_upscaler_model_path"],
    local_dir=MODELS_DIR
)

# Instanciação e carregamento dos pipelines na GPU
pipeline_instance = create_ltx_video_pipeline(
    ckpt_path=distilled_model_path,
    precision=PIPELINE_CONFIG["precision"],
    text_encoder_model_name_or_path=PIPELINE_CONFIG["text_encoder_model_name_or_path"],
    sampler=PIPELINE_CONFIG["sampler"],
    device=TARGET_DEVICE
)
latent_upsampler_instance = create_latent_upsampler(
    spatial_upscaler_path,
    device=TARGET_DEVICE
)
print(f"Worker inicializado com sucesso no dispositivo: {TARGET_DEVICE}")

# --------------------------------------------------------------------------- #
#                          HANDLER DE REQUISIÇÃO                               #
# --------------------------------------------------------------------------- #
def handler(job):
    """
    Processa uma única requisição de inferência para geração de vídeo.
    """
    job_input = job['input']
    temp_dir = tempfile.mkdtemp()

    try:
        # Extração de parâmetros com valores padrão
        prompt = job_input.get('prompt', 'A majestic dragon flying over a medieval castle')
        negative_prompt = job_input.get('negative_prompt', 'bad quality, inconsistent motion, blurry, shaky, distorted')
        input_image_url = job_input.get('input_image_url')
        height = int(job_input.get('height', 512))
        width = int(job_input.get('width', 704))
        duration_seconds = float(job_input.get('duration_seconds', 2.0))
        fps = int(job_input.get('fps', 24))
        seed = job_input.get('seed')
        cfg_scale = float(job_input.get('cfg_scale', 3.0))
        improve_texture = bool(job_input.get('improve_texture', False))

        seed = int(seed) if seed is not None else random.randint(0, 2**32 - 1)
        seed_everething(seed)

        # Preparação do ambiente e parâmetros de geração
        MAX_NUM_FRAMES = 257
        target_frames_ideal = duration_seconds * fps
        n_val = round((float(round(target_frames_ideal)) - 1.0) / 8.0)
        actual_num_frames = max(9, min(int(n_val * 8 + 1), MAX_NUM_FRAMES))
        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        num_frames_padded = ((actual_num_frames - 2) // 8 + 1) * 8 + 1
        padding_values = calculate_padding(height, width, height_padded, width_padded)

        # --- CORREÇÃO PRINCIPAL: CONSTRUÇÃO COMPLETA DO DICIONÁRIO DE ARGUMENTOS ---
        # Este dicionário agora inclui todas as chaves que o pipeline espera,
        # prevenindo o erro 'KeyError'.
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height_padded,
            "width": width_padded,
            "num_frames": num_frames_padded,
            "frame_rate": fps,
            "generator": torch.Generator(device=TARGET_DEVICE).manual_seed(seed),
            "output_type": "pt",
            "conditioning_items": None,
            "media_items": None,
            "decode_timestep": PIPELINE_CONFIG["decode_timestep"],
            "decode_noise_scale": PIPELINE_CONFIG["decode_noise_scale"],
            "stochastic_sampling": PIPELINE_CONFIG["stochastic_sampling"],
            "image_cond_noise_scale": 0.15,
            "is_video": True,
            "vae_per_channel_normalize": True,
            "mixed_precision": (PIPELINE_CONFIG["precision"] == "mixed_precision"),
            "offload_to_cpu": False,
            "enhance_prompt": False
        }
        stg_mode_str = PIPELINE_CONFIG.get("stg_mode", "attention_values")
        stg_map = {"attention_values": SkipLayerStrategy.AttentionValues, "attention_skip": SkipLayerStrategy.AttentionSkip, "residual": SkipLayerStrategy.Residual, "transformer_block": SkipLayerStrategy.TransformerBlock}
        call_kwargs["skip_layer_strategy"] = stg_map.get(stg_mode_str.lower(), SkipLayerStrategy.AttentionValues)
        
        # Lógica de condicionamento para Image-to-Video
        if input_image_url:
            response = requests.get(input_image_url, stream=True, timeout=20)
            response.raise_for_status()
            input_image_filepath = os.path.join(temp_dir, "input_image.png")
            with open(input_image_filepath, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            media_tensor = load_image_to_tensor_with_resize_and_crop(input_image_filepath, height, width)
            media_tensor = torch.nn.functional.pad(media_tensor, padding_values)
            call_kwargs["conditioning_items"] = [ConditioningItem(media_tensor.to(TARGET_DEVICE), 0, 1.0)]

        # Execução do pipeline de inferência
        if improve_texture and latent_upsampler_instance:
            multi_scale_pipeline_obj = LTXMultiScalePipeline(pipeline_instance, latent_upsampler_instance)
            first_pass_args = {**PIPELINE_CONFIG.get("first_pass", {}), "guidance_scale": cfg_scale}
            second_pass_args = {**PIPELINE_CONFIG.get("second_pass", {}), "guidance_scale": cfg_scale}
            multi_scale_call_kwargs = {**call_kwargs, "downscale_factor": PIPELINE_CONFIG["downscale_factor"], "first_pass": first_pass_args, "second_pass": second_pass_args}
            result_images_tensor = multi_scale_pipeline_obj(**multi_scale_call_kwargs).images
        else:
            first_pass_config = PIPELINE_CONFIG.get("first_pass", {})
            single_pass_kwargs = {**call_kwargs, "guidance_scale": cfg_scale, "timesteps": first_pass_config.get("timesteps")}
            result_images_tensor = pipeline_instance(**single_pass_kwargs).images

        # Pós-processamento e salvamento do vídeo
        pad_left, pad_right, pad_top, pad_bottom = padding_values
        slice_h, slice_w = (-pad_bottom if pad_bottom > 0 else None), (-pad_right if pad_right > 0 else None)
        result_images_tensor_cropped = result_images_tensor[:, :, :actual_num_frames, pad_top:slice_h, pad_left:slice_w]
        video_np = (result_images_tensor_cropped[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
        
        output_video_path = os.path.join(temp_dir, "output.mp4")
        with imageio.get_writer(output_video_path, fps=fps, codec='libx264', quality=8) as video_writer:
            for frame in video_np:
                video_writer.append_data(frame)

        # Codificação do vídeo em Base64 para retorno na API
        with open(output_video_path, "rb") as video_file:
            video_bytes = video_file.read()
        base64_encoded_video = base64.b64encode(video_bytes).decode('utf-8')

        # Retorno de sucesso
        return {
            "video_base64": base64_encoded_video,
            "format": "mp4",
            "seed_used": seed
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Ocorreu um erro inesperado: {str(e)}"}
        
    finally:
        # Garantir a limpeza dos recursos ao final de cada execução
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        gc.collect()
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------- #
#                      INICIALIZAÇÃO DO SERVIDOR RUNPOD                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
