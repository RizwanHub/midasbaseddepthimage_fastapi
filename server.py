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

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validate model exists
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model missing at {MODEL_PATH}. Download during build!")

# Hardware configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Model loading
model = None
midas_transforms = None

def load_midas_model():
    global model, midas_transforms
    if model is None:
        try:
            logger.info("Loading MiDaS model...")
            start_time = time.time()
            
            # Load model with correct architecture
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", 
                                  source="local", checkpoint=MODEL_PATH)
            model.to(DEVICE).eval()
            
            # Load appropriate transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            
            logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}", exc_info=True)
            raise HTTPException(500, detail=f"Model loading failed: {str(e)}")
    
    return model, midas_transforms

@app.post("/generate-depth-map")
async def generate_depth_map(file: UploadFile = File(...)):
    try:
        model, transforms = load_midas_model()
        
        # Save upload
        upload_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process image
        start_time = time.time()
        img = Image.open(upload_path).convert("RGB")
        img_np = np.array(img)
        
        # Apply model-specific transforms
        input_batch = transforms(img_np).to(DEVICE)
        
        # Generate prediction
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        
        # Process output
        depth = prediction.cpu().numpy()
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)
        
        # Save result
        depth_path = os.path.join(OUTPUT_DIR, f"depth_{file.filename}")
        cv2.imwrite(depth_path, depth_normalized)
        os.remove(upload_path)
        
        logger.info(f"Processed in {time.time() - start_time:.2f}s")
        return FileResponse(depth_path, media_type="image/png")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
