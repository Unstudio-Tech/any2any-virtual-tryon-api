from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from threading import Lock
import base64
from io import BytesIO
import uuid
import cachetools
from PIL import Image
import numpy as np
import torch
import requests

from app.py import load_models, generate_image, resize_by_height

app = FastAPI()
lock = Lock()

# Load pipeline once
pipe = load_models()

# Simple TTL cache for responses
responses = cachetools.TTLCache(maxsize=100, ttl=600)

class TryOnRequest(BaseModel):
    prompt: str
    model_image: str = None  # base64 or URL
    garment_image: str = None  # base64 or URL
    image_type: str = "base64"  # base64 or url
    num_inference_steps: int = 50
    guidance_scale: float = 3.5
    show_type: str = "follow model image"
    height: int = 1024
    width: int = 1024
    seed: int = 0

def load_image(img_data: str, img_type: str = "base64") -> np.ndarray:
    if not img_data:
        return None
    if img_type == "url":
        response = requests.get(img_data)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        decoded = base64.b64decode(img_data)
        image = Image.open(BytesIO(decoded)).convert("RGB")
    return np.array(image)

def process_tryon(request_id: str, req: TryOnRequest):
    global pipe
    with lock:
        try:
            model_img = load_image(req.model_image, req.image_type)
            garment_img = load_image(req.garment_image, req.image_type)

            output_img = generate_image(
                prompt=req.prompt,
                model_image=model_img,
                garment_image=garment_img,
                height=req.height,
                width=req.width,
                seed=req.seed,
                guidance_scale=req.guidance_scale,
                show_type=req.show_type,
                num_inference_steps=req.num_inference_steps
            )

            buffered = BytesIO()
            output_img.save(buffered, format="PNG")
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

            responses[request_id] = {"image": encoded, "encoding": "base64"}
        except Exception as e:
            responses[request_id] = {"error": str(e)}

@app.post("/tryon")
async def tryon(request: TryOnRequest, background_tasks: BackgroundTasks):
    request_id = str(uuid.uuid4())
    background_tasks.add_task(process_tryon, request_id, request)
    return {"request_id": request_id}

@app.get("/get_result/{request_id}")
async def get_result(request_id: str):
    if request_id in responses:
        return responses.pop(request_id)
    return {"status": "processing"}
