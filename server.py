from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
import torch
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn")

app = FastAPI()

# Enable CORS for RapidAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for security if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory configuration
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardware configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Global variables for lazy loading
model = None
midas_transforms = None

def load_midas_model():
    """Load MiDaS model from the official repository when needed."""
    try:
        global model, midas_transforms
        if model is None or midas_transforms is None:
            logger.info("Loading MiDaS model from the server...")
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            model.to(DEVICE).eval()

            logger.info("Loading MiDaS transforms from the server...")
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

            logger.info("MiDaS model and transforms loaded successfully.")
        return model, midas_transforms
    except Exception as e:
        logger.error(f"Failed to load MiDaS model: {str(e)}", exc_info=True)
        raise RuntimeError("MiDaS model loading failed. Check internet connection and repository availability.")

@app.post("/generate-depth-map")
async def generate_depth_map(file: UploadFile = File(...)):
    try:
        # Ensure model is loaded
        model, midas_transforms = load_midas_model()

        # Save uploaded file
        upload_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        original_base = os.path.splitext(file.filename)[0]

        # Open image and convert to RGB
        img = Image.open(upload_path).convert("RGB")
        img_np = np.array(img)

        # Apply MiDaS transforms
        input_tensor = midas_transforms(img_np).to(DEVICE)

        # Generate depth prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        # Convert prediction to numpy array
        depth_np = prediction.cpu().numpy()

        # Normalize depth map
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        depth_normalized = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)

        # Save depth map
        depth_filename = f"depth_{original_base}.png"
        depth_path = os.path.join(OUTPUT_DIR, depth_filename)
        cv2.imwrite(depth_path, depth_normalized)

        # Verify depth map creation
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth map not created at {depth_path}")

        return FileResponse(depth_path, media_type="image/png", filename=depth_filename)

    except Exception as e:
        logger.error(f"Depth map generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
