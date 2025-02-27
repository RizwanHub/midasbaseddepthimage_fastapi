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
            logger.info("=== Starting MiDaS model loading ===")
            
            # Debug: Verify midas folder existence
            midas_path = os.path.join(os.getcwd(), "midas")
            logger.info(f"Checking midas folder at: {midas_path}")
            logger.info(f"MIDAS FOLDER CONTENTS: {os.listdir(midas_path)}")
            
            # Debug: Verify midas_net.py existence
            midas_net_path = os.path.join(midas_path, "midas_net.py")
            logger.info(f"Checking midas_net.py at: {midas_net_path}")
            if not os.path.exists(midas_net_path):
                raise FileNotFoundError(f"midas_net.py not found at {midas_net_path}")

            # Debug: Verify class existence
            logger.info("Attempting to import MidasNet_small...")
            from midas.midas_net import MidasNet_small
            logger.info("Successfully imported MidasNet_small!")

            # Load model
            logger.info("Initializing model...")
            model = MidasNet_small()
            
            # Debug: Verify model weights
            logger.info(f"Loading weights from: {MODEL_PATH}")
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
                
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE).eval()

            # Load transforms
            logger.info("Loading transforms...")
            from midas.transforms import Resize, NormalizeImage, PrepareForNet
            midas_transforms = torch.nn.Sequential(
                Resize(384, 384),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet()
            )

            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error("=== CRITICAL ERROR DURING MODEL LOADING ===")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Model loading failed: {str(e)} (Full logs in Render dashboard)"
            )

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
