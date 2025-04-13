from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
from io import BytesIO
from typing import Optional
import tensorflow as tf
from skimage import color, restoration
from skimage.restoration import estimate_sigma
from skimage.filters import median
from scipy.signal import convolve2d
import uvicorn

app = FastAPI(title="Image Forgery Detection API", 
              description="API to detect if an image is authentic or forged")

model_path = "casia2_model.h5"
model = None

def load_model():
    global model
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        else:
            print(f"Model not found at {model_path}. Endpoints requiring model will not work.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.on_event("startup")
async def startup_event():
    load_model()

def weiner_noise_reduction(img):
    img = color.rgb2gray(img)
    psf = np.ones((5, 5)) / 25
    img = convolve2d(img, psf, 'same')
    img += 0.1 * img.std() * np.random.standard_normal(img.shape)
    deconvolved_img = restoration.wiener(img, psf, 1100)
    return deconvolved_img

def estimate_noise(img):
    return estimate_sigma(img, multichannel=True, average_sigmas=True)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    enoise = estimate_noise(image)
    noise_free_image = weiner_noise_reduction(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fingerprint = gray - noise_free_image
    fingerprint = fingerprint / 255
    filtered_img = median(fingerprint, selem=None, out=None, mask=None, shift_x=False,
                          shift_y=False, mode='nearest', cval=0.0, behavior='rank')
    colored = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    return colored

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Forgery Detection API"}

@app.post("/detect/")
async def detect_forgery(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure the model file exists.")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        h, w = img.shape[:2]
        if h != 256 or w != 384:
            raise HTTPException(status_code=400, 
                               detail=f"Image dimensions must be 384x256 pixels. Got {w}x{h}.")
        
        processed_img = preprocess_image(img)
        
        input_img = np.expand_dims(processed_img, axis=0)
        
        prediction = model.predict(input_img)
        prediction_class = int(np.round(prediction[0][0]))
        prediction_confidence = float(prediction[0][0])
        
        result = {
            "is_forged": bool(prediction_class),
            "confidence": prediction_confidence,
            "result": "Forged" if prediction_class == 1 else "Authentic"
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")

@app.get("/model-info/")
def model_info():
    if model is None:
        return {"status": "Model not loaded", "info": None}
    
    try:
        return {
            "status": "Model loaded",
            "info": {
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "trainable_params": int(np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])),
                "non_trainable_params": int(np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights]))
            }
        }
    except Exception as e:
        return {"status": "Error retrieving model info", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)