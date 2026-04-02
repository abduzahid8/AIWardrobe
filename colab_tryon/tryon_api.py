"""
IDM-VTON Try-On API Server for Google Colab
=============================================
FastAPI server wrapping IDM-VTON virtual try-on.
Exposes a public HTTPS URL via ngrok for the AIWardrobe React Native app.

Usage (in Colab cell):
    !python tryon_api.py --ngrok-token YOUR_NGROK_TOKEN

Or without ngrok (Gradio share URL instead):
    !python tryon_api.py --use-gradio
"""

import sys
import os

# Add IDM-VTON to path
sys.path.insert(0, "/content/IDM-VTON")
sys.path.insert(0, "/content/IDM-VTON/gradio_demo")

import argparse
import base64
import io
import logging
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tryon-api")

# ── Global state ─────────────────────────────────────────
pipe = None
unet_encoder = None
parsing_model = None
openpose_model = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tensor_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


# ── Model Loading ────────────────────────────────────────
def load_models():
    """Load all IDM-VTON models into memory."""
    global pipe, unet_encoder, parsing_model, openpose_model

    log.info("Loading IDM-VTON models... (this takes 2-5 minutes)")
    t0 = time.time()

    # MONKEYPATCH: diffusers==0.25.0 tries to import cached_download from huggingface_hub
    # but it was removed in huggingface_hub v0.23. We must patch it before importing diffusers.
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download

    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel
    from transformers import (
        AutoTokenizer,
        CLIPImageProcessor,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    )
    from diffusers import AutoencoderKL, DDPMScheduler

    model_path = "/content/IDM-VTON-model"

    # If the model wasn't downloaded separately, try HF hub
    if not os.path.exists(model_path):
        model_path = "yisol/IDM-VTON"
        log.info(f"Using HuggingFace model: {model_path}")
    else:
        log.info(f"Using local model: {model_path}")

    # Load all components
    log.info("  Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", torch_dtype=torch.float16
    )
    unet.requires_grad_(False)

    log.info("  Loading tokenizers...")
    tokenizer_one = AutoTokenizer.from_pretrained(
        model_path, subfolder="tokenizer", revision=None, use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        model_path, subfolder="tokenizer_2", revision=None, use_fast=False
    )

    log.info("  Loading noise scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    log.info("  Loading text encoders...")
    text_encoder_one = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=torch.float16
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        model_path, subfolder="text_encoder_2", torch_dtype=torch.float16
    )

    log.info("  Loading image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_path, subfolder="image_encoder", torch_dtype=torch.float16
    )

    log.info("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float16
    )

    log.info("  Loading UNet encoder (garment)...")
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        model_path, subfolder="unet_encoder", torch_dtype=torch.float16
    )

    # Freeze all
    unet_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Build pipeline
    log.info("  Assembling pipeline...")
    pipe = TryonPipeline.from_pretrained(
        model_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe.unet_encoder = unet_encoder

    # Move to GPU with memory optimization
    log.info("  Moving to GPU with CPU offload...")
    pipe.to(device)
    pipe.unet_encoder.to(device)

    # Load preprocessing models
    log.info("  Loading preprocessing models (OpenPose + Human Parsing)...")
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose

    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)

    elapsed = time.time() - t0
    log.info(f"✅ All models loaded in {elapsed:.1f}s")

    # Log GPU memory
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    log.info(f"  GPU memory: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")


# ── Try-On Logic ─────────────────────────────────────────
def pil_to_binary_mask(pil_image, threshold=0):
    """Convert PIL image to binary mask."""
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = (binary_mask.astype(np.uint8) * 255)
    return Image.fromarray(mask)


def run_tryon(
    person_image: Image.Image,
    garment_image: Image.Image,
    garment_description: str = "garment",
    auto_crop: bool = True,
    denoise_steps: int = 30,
    seed: int = 42,
) -> Image.Image:
    """
    Run IDM-VTON virtual try-on.

    Args:
        person_image: PIL Image of the person
        garment_image: PIL Image of the garment
        garment_description: Text description of the garment
        auto_crop: Whether to auto-crop person image to 3:4 ratio
        denoise_steps: Number of diffusion denoising steps (20-40)
        seed: Random seed for reproducibility

    Returns:
        PIL Image with the person wearing the garment
    """
    global pipe, parsing_model, openpose_model

    from typing import List
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    import apply_net
    from utils_mask import get_mask_location

    t0 = time.time()
    log.info(f"Starting try-on: '{garment_description}', steps={denoise_steps}, seed={seed}")

    # Prepare garment image
    garm_img = garment_image.convert("RGB").resize((768, 1024))

    # Prepare person image
    human_img_orig = person_image.convert("RGB")

    if auto_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    # Step 1: OpenPose keypoints
    log.info("  [1/4] Running OpenPose...")
    keypoints = openpose_model(human_img.resize((384, 512)))

    # Step 2: Human parsing
    log.info("  [2/4] Running Human Parsing...")
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    mask, mask_gray = get_mask_location("hd", "upper_body", model_parse, keypoints)
    mask = mask.resize((768, 1024))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    # Step 3: DensePose
    log.info("  [3/4] Running DensePose...")
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        (
            "show",
            "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
            "./ckpt/densepose/model_final_162be9.pkl",
            "dp_segm",
            "-v",
            "--opts",
            "MODEL.DEVICE",
            "cuda",
        )
    )
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    # Step 4: IDM-VTON diffusion
    log.info(f"  [4/4] Running IDM-VTON diffusion ({denoise_steps} steps)...")
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # Encode prompts
            prompt = "model is wearing " + garment_description
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                prompt_c = "a photo of " + garment_description
                if not isinstance(prompt_c, list):
                    prompt_c = [prompt_c]
                neg_c = [negative_prompt]

                (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                    prompt_c,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=neg_c,
                )

            # Prepare tensors
            pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)

            generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

            images = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=1024,
                width=768,
                ip_adapter_image=garm_img.resize((768, 1024)),
                guidance_scale=2.0,
            )[0]

    result = images[0]

    # Paste back into original if cropped
    if auto_crop:
        out_img = result.resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        result = human_img_orig

    elapsed = time.time() - t0
    log.info(f"✅ Try-on complete in {elapsed:.1f}s")

    return result


# ── Helpers ──────────────────────────────────────────────
def b64_to_pil(b64: str) -> Image.Image:
    """Decode base64 to PIL Image, stripping data-URI header if present."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── FastAPI App ──────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="IDM-VTON Try-On API",
    description="Virtual try-on API powered by IDM-VTON on Google Colab",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TryOnRequest(BaseModel):
    person_image: str = Field(..., description="Base64-encoded person photo (JPEG/PNG)")
    garment_image: str = Field(..., description="Base64-encoded garment photo (JPEG/PNG)")
    garment_description: str = Field(
        default="Short sleeve round neck t-shirt",
        description="Text description of the garment",
    )
    auto_crop: bool = Field(default=True, description="Auto-crop person to 3:4 ratio")
    denoise_steps: int = Field(default=30, ge=20, le=40, description="Denoising steps")
    seed: int = Field(default=42, description="Random seed")


class TryOnResponse(BaseModel):
    result_image: str = Field(..., description="Base64-encoded result image (PNG)")
    elapsed_seconds: float = Field(..., description="Processing time in seconds")


@app.get("/health")
async def health():
    """Health check endpoint."""
    gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_memory_used_gb": round(gpu_mem, 2),
        "models_loaded": pipe is not None,
    }


@app.post("/tryon", response_model=TryOnResponse)
async def tryon(req: TryOnRequest):
    """Run virtual try-on."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Wait a moment.")

    try:
        person_img = b64_to_pil(req.person_image)
        garment_img = b64_to_pil(req.garment_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    t0 = time.time()
    try:
        result = run_tryon(
            person_image=person_img,
            garment_image=garment_img,
            garment_description=req.garment_description,
            auto_crop=req.auto_crop,
            denoise_steps=req.denoise_steps,
            seed=req.seed,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=507,
            detail="GPU out of memory. Try reducing denoise_steps or restart the runtime.",
        )
    except Exception as e:
        log.exception("Try-on failed")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.time() - t0
    result_b64 = pil_to_b64(result)

    return TryOnResponse(result_image=result_b64, elapsed_seconds=round(elapsed, 1))


# ── Main ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IDM-VTON Try-On API Server")
    parser.add_argument(
        "--ngrok-token",
        type=str,
        default=None,
        help="ngrok auth token for public HTTPS URL (get from https://dashboard.ngrok.com)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--use-gradio",
        action="store_true",
        help="Use Gradio share URL instead of ngrok (no token needed)",
    )
    args = parser.parse_args()

    # Change working directory to IDM-VTON for relative path references
    os.chdir("/content/IDM-VTON")

    # Load models
    load_models()

    if args.use_gradio:
        # Use Gradio's built-in sharing (simpler, no ngrok needed)
        import gradio as gr

        def gradio_tryon(person_dict, garment_img, description, auto_crop, steps, seed):
            person_img = person_dict["background"] if isinstance(person_dict, dict) else person_dict
            result = run_tryon(
                person_image=person_img,
                garment_image=garment_img,
                garment_description=description,
                auto_crop=auto_crop,
                denoise_steps=int(steps),
                seed=int(seed),
            )
            return result

        with gr.Blocks() as demo:
            gr.Markdown("## 👕 IDM-VTON Try-On (AIWardrobe)")
            with gr.Row():
                with gr.Column():
                    person_input = gr.Image(label="Person Photo", type="pil")
                    auto_crop_cb = gr.Checkbox(label="Auto-crop to 3:4", value=True)
                with gr.Column():
                    garment_input = gr.Image(label="Garment Photo", type="pil")
                    desc_input = gr.Textbox(
                        label="Garment Description",
                        value="Short sleeve round neck t-shirt",
                    )
                with gr.Column():
                    output_img = gr.Image(label="Result")
            with gr.Row():
                steps_input = gr.Slider(20, 40, value=30, step=1, label="Denoise Steps")
                seed_input = gr.Number(label="Seed", value=42)
            btn = gr.Button("Try On!", variant="primary")
            btn.click(
                fn=gradio_tryon,
                inputs=[person_input, garment_input, desc_input, auto_crop_cb, steps_input, seed_input],
                outputs=output_img,
            )

        demo.launch(share=True)
    else:
        # Use FastAPI + ngrok
        import nest_asyncio
        import uvicorn

        nest_asyncio.apply()

        if args.ngrok_token:
            from pyngrok import ngrok

            ngrok.set_auth_token(args.ngrok_token)
            tunnel = ngrok.connect(args.port)
            public_url = tunnel.public_url

            print("\n" + "=" * 60)
            print("  🌐 PUBLIC API URL (use this in your app):")
            print(f"  {public_url}")
            print("=" * 60)
            print(f"\n  Health check:  curl {public_url}/health")
            print(f"  API docs:      {public_url}/docs")
            print(f"  Try-on:        POST {public_url}/tryon")
            print("=" * 60 + "\n")
        else:
            print("\n⚠️  No ngrok token provided. Server will only be accessible locally.")
            print("   Get a free token at: https://dashboard.ngrok.com/get-started/your-authtoken")
            print(f"   Then run: python tryon_api.py --ngrok-token YOUR_TOKEN\n")

        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
