from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import logging
import time

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
MODEL_DIR = "models"  # Relative path
MODEL_PATH = os.path.join(MODEL_DIR, "dpt_large-midas.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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
            # Verify the model file exists
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure the model is downloaded.")

            # Load the model
            logger.info("Loading MiDaS model...")
            start_time = time.time()
            model = torch.jit.load(MODEL_PATH, map_location=DEVICE)  # Use torch.jit.load for TorchScript models
            model.eval()
            logger.info(f"MiDaS model loaded successfully in {time.time() - start_time:.2f} seconds.")

            # Define the transforms manually
            midas_transforms = transforms.Compose([
                transforms.Resize((384, 384)),  # Resize to the input size expected by MiDaS
                transforms.ToTensor(),           # Convert to tensor
                transforms.Normalize(            # Normalize with mean and std
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to load MiDaS model: {str(e)}")

    return model, midas_transforms

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

        # Resize the image to reduce processing time
        max_size = 1024  # Maximum dimension (width or height)
        img.thumbnail((max_size, max_size))

        # Apply MiDaS transforms
        input_tensor = midas_transforms(img).unsqueeze(0).to(DEVICE)

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

        # Clean up uploaded file
        os.remove(upload_path)

        logger.info(f"Depth map generated successfully in {time.time() - start_time:.2f} seconds.")
        return FileResponse(depth_path, media_type="image/png", filename=depth_filename)

    except Exception as e:
        logger.error(f"Depth map generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
