from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
import torch
from PIL import Image
import logging
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory configuration
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "midas_v21_small_256.pt")  # Updated model path

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validate model file exists
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please ensure the model is downloaded during build.")

# Hardware configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Global variables for lazy loading
model = None
midas_transforms = None

def load_midas_model():
    global model, midas_transforms
    if model is None:
        try:
            logger.info("Loading MiDaS model...")
            start_time = time.time()

            # Load the MiDaS small model architecture
            from midas.midas_net import MidasNet_small
            model = MidasNet_small()

            # Load pre-trained weights
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE).eval()

            # Load transforms
            from midas.transforms import Resize, NormalizeImage, PrepareForNet
            midas_transforms = torch.nn.Sequential(
                Resize(384, 384),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet()
            )

            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to load MiDaS model: {str(e)}")

    return model, midas_transforms
@app.post("/generate-depth-map")
async def generate_depth_map(file: UploadFile = File(...)):
    try:
        # Ensure model is loaded
        model, transforms = load_midas_model()

        # Save uploaded file
        upload_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Open and preprocess the image
        img = Image.open(upload_path).convert("RGB")
        img_np = np.array(img)

        # Apply transforms
        input_tensor = transforms(torch.from_numpy(img_np).unsqueeze(0)).to(DEVICE)

        # Generate depth prediction
        logger.info("Generating depth map...")
        start_time = time.time()
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()

        # Normalize depth map
        depth_min = prediction.min()
        depth_max = prediction.max()
        depth_normalized = (prediction - depth_min) / (depth_max - depth_min + 1e-8)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)

        # Save depth map
        depth_filename = f"depth_{file.filename}"
        depth_path = os.path.join(OUTPUT_DIR, depth_filename)
        cv2.imwrite(depth_path, depth_normalized)

        # Clean up uploaded file
        os.remove(upload_path)

        logger.info(f"Depth map generated in {time.time() - start_time:.2f} seconds")
        return FileResponse(depth_path, media_type="image/png", filename=depth_filename)

    except Exception as e:
        logger.error(f"Depth map generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
