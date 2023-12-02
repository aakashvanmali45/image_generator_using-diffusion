from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import mediapy as media
from torch import autocast
from src.model import process_images


app = FastAPI()

# CORS (Cross-Origin Resource Sharing) settings to allow frontend access
origins = [
    "http://localhost",
    "http://localhost:8000",  # Add your frontend URL(s) here
    "https://yourproductiondomain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    prompt: str
    num_images: int

class OutputData(BaseModel):
    image_url: str

@app.post("/generate_image", response_model=OutputData)
async def generate_image(data: InputData):
    # Your existing code for image generation
    prompt = data.prompt
    num_images = data.num_images

    # Call the modified process_images function
    image_data = await process_images(prompt, num_images)

    return OutputData(image_data=image_data)

