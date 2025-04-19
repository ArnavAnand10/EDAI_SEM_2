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
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import uuid
import time
import io
import json
from typing import Optional
from pydantic import BaseModel

app = FastAPI(
    title="Image Forgery Detection API",
    description="API to detect if an image is authentic or forged",
)

model_path = "casia2_model.h5"
model = None


class VerificationResult(BaseModel):
    original_metadata: dict
    is_intact: bool
    message: str


def embed_metadata_in_image(image_data: bytes) -> tuple[bytes, dict]:
    metadata = {
        "uuid": str(uuid.uuid4()),
        "timestamp": int(time.time()),
        "created_by": "steganography_api",
    }

    metadata_str = json.dumps(metadata)
    metadata_binary = "".join(format(ord(char), "08b") for char in metadata_str)

    img = Image.open(io.BytesIO(image_data))
    img_array = np.array(img)

    total_pixels = img_array.shape[0] * img_array.shape[1]
    if len(metadata_binary) > total_pixels:
        raise ValueError("Image is too small to embed metadata")

    if len(img_array.shape) == 3:
        flat_img = img_array.reshape(-1, img_array.shape[2])
    else:
        flat_img = img_array.flatten()

    metadata_length = len(metadata_binary)
    length_binary = format(metadata_length, "032b")

    for i in range(32):
        if i < len(length_binary):
            if len(img_array.shape) == 3:
                flat_img[i, 2] = (flat_img[i, 2] & ~1) | int(length_binary[i])
            else:
                flat_img[i] = (flat_img[i] & ~1) | int(length_binary[i])

    for i in range(len(metadata_binary)):
        pixel_position = i + 32
        if pixel_position >= len(flat_img):
            break

        if len(img_array.shape) == 3:
            flat_img[pixel_position, 2] = (flat_img[pixel_position, 2] & ~1) | int(
                metadata_binary[i]
            )
        else:
            flat_img[pixel_position] = (flat_img[pixel_position] & ~1) | int(
                metadata_binary[i]
            )

    if len(img_array.shape) == 3:
        new_img_array = flat_img.reshape(img_array.shape)
    else:
        new_img_array = flat_img.reshape(img_array.shape)

    new_img = Image.fromarray(new_img_array)

    img_byte_arr = io.BytesIO()
    new_img.save(img_byte_arr, format=img.format or "PNG")
    img_byte_arr.seek(0)

    return img_byte_arr.getvalue(), metadata


def extract_metadata_from_image(image_data: bytes) -> Optional[dict]:
    try:
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)

        if len(img_array.shape) == 3:
            flat_img = img_array.reshape(-1, img_array.shape[2])
        else:
            flat_img = img_array.flatten()

        length_binary = ""
        for i in range(32):
            if i >= len(flat_img):
                return None

            if len(img_array.shape) == 3:
                length_binary += str(flat_img[i, 2] & 1)
            else:
                length_binary += str(flat_img[i] & 1)

        metadata_length = int(length_binary, 2)
        if metadata_length <= 0 or metadata_length > len(flat_img):
            return None

        metadata_binary = ""
        for i in range(metadata_length):
            pixel_position = i + 32
            if pixel_position >= len(flat_img):
                break

            if len(img_array.shape) == 3:
                metadata_binary += str(flat_img[pixel_position, 2] & 1)
            else:
                metadata_binary += str(flat_img[pixel_position] & 1)

        metadata_str = ""
        for i in range(0, len(metadata_binary), 8):
            if i + 8 <= len(metadata_binary):
                byte = metadata_binary[i : i + 8]
                metadata_str += chr(int(byte, 2))

        metadata = json.loads(metadata_str)
        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None


def load_model():
    global model
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        else:
            print(
                f"Model not found at {model_path}. Endpoints requiring model will not work."
            )
    except Exception as e:
        print(f"Error loading model: {e}")


@app.on_event("startup")
async def startup_event():
    load_model()


def weiner_noise_reduction(img):
    img = color.rgb2gray(img)
    psf = np.ones((5, 5)) / 25
    img = convolve2d(img, psf, "same")
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
    filtered_img = median(
        fingerprint,
        selem=None,
        out=None,
        mask=None,
        shift_x=False,
        shift_y=False,
        mode="nearest",
        cval=0.0,
        behavior="rank",
    )
    colored = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    return colored


@app.post("/detect/")
async def detect_forgery(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists.",
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        if h != 256 or w != 384:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions must be 384x256 pixels. Got {w}x{h}.",
            )

        processed_img = preprocess_image(img)

        input_img = np.expand_dims(processed_img, axis=0)

        prediction = model.predict(input_img)
        prediction_class = int(np.round(prediction[0][0]))
        prediction_confidence = float(prediction[0][0])

        result = {
            "is_forged": bool(prediction_class),
            "confidence": prediction_confidence,
            "result": "Forged" if prediction_class == 1 else "Authentic",
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during processing: {str(e)}"
        )


@app.post("/embed")
async def embed_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await image.read()

        processed_image, metadata = embed_metadata_in_image(image_data)

        return StreamingResponse(
            io.BytesIO(processed_image),
            media_type=image.content_type,
            headers={
                "Content-Disposition": f"attachment; filename=embedded_{image.filename}",
                "X-Metadata-UUID": metadata["uuid"],
                "X-Metadata-Timestamp": str(metadata["timestamp"]),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/verify", response_model=VerificationResult)
async def verify_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await image.read()

        metadata = extract_metadata_from_image(image_data)

        if metadata:
            return VerificationResult(
                original_metadata=metadata,
                is_intact=True,
                message="Image metadata extracted successfully, image appears intact.",
            )
        else:
            return VerificationResult(
                original_metadata={},
                is_intact=False,
                message="Failed to extract metadata. The image may have been tampered with or does not contain embedded metadata.",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verifying image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
