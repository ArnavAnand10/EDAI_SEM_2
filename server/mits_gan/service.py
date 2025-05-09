import os
from typing import Optional
import tensorflow as tf
from skimage import color, restoration
from skimage.restoration import estimate_sigma
from skimage.filters import median
from scipy.signal import convolve2d
from PIL import Image
import numpy as np
import uuid
import time
import io
import json
from typing import Optional
import cv2


model_path = "casia2_model.h5"
model = None


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
