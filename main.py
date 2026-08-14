"""
Potato Disease Classification API
-----------------------------------
FastAPI backend that serves predictions from a trained Keras CNN model
(potato_model.keras) trained on potato leaf images to classify them as
Early Blight, Late Blight, or Healthy.
"""

import io
import logging

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_PATH = "potato_model.keras"
IMAGE_SIZE = (255, 255)  # (width, height) expected by the model
CLASS_NAMES = ["Early_Blight", "Late_Blight", "Healthy"]

# Human-readable guidance returned alongside each prediction. This is general
# agronomic guidance, not a substitute for a local agricultural extension
# officer or plant pathologist for severe or recurring outbreaks.
DISEASE_INFO = {
    "Early_Blight": {
        "severity": "moderate",
        "summary": (
            "Caused by the fungus Alternaria solani. Usually starts on older, "
            "lower leaves as small dark spots with concentric rings, and can "
            "spread upward over time."
        ),
        "recommendations": [
            "Remove and destroy affected leaves to reduce spore spread.",
            "Apply a fungicide labeled for early blight (e.g. chlorothalonil or a copper-based spray), following label instructions.",
            "Avoid overhead watering; water at the base to keep foliage dry.",
            "Space plants for good airflow and avoid working in wet fields.",
            "Rotate crops — avoid planting potatoes/tomatoes in the same soil for 2-3 years.",
            "Ensure balanced fertilization; nitrogen-stressed plants are more susceptible.",
        ],
    },
    "Late_Blight": {
        "severity": "high",
        "summary": (
            "Caused by the oomycete Phytophthora infestans. Spreads very "
            "quickly in cool, wet weather and can destroy a crop within days "
            "if untreated — this is the pathogen behind the Irish potato famine."
        ),
        "recommendations": [
            "Act immediately — late blight can spread to the whole field within days.",
            "Remove and destroy (do not compost) infected plants/leaves right away.",
            "Apply a fungicide effective against late blight (e.g. mancozeb or a copper-based product) as soon as possible.",
            "Improve field drainage and avoid overhead irrigation, especially in humid conditions.",
            "Monitor neighboring plants closely and isolate infected areas if possible.",
            "Consider consulting a local agricultural extension office for regional outbreak alerts and approved treatments.",
        ],
    },
    "Healthy": {
        "severity": "none",
        "summary": "No signs of blight detected — the leaf appears healthy.",
        "recommendations": [
            "Continue routine monitoring for early signs of disease.",
            "Maintain good watering practices — water at the base, avoid wetting foliage.",
            "Keep up preventive crop rotation and field sanitation practices.",
        ],
    },
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("potato-api")

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
app = FastAPI(
    title="Potato Disease Classification API",
    description="Upload a potato leaf image and get a disease prediction.",
    version="1.0.0",
)

# Allow the frontend (served from any origin / local file / dev server) to
# call this API from the browser without being blocked by CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The model is loaded once, at startup, and kept in memory so that each
# request to /predict is fast (no reloading from disk per request).
model: tf.keras.Model | None = None


@app.on_event("startup")
async def load_model() -> None:
    """Load the trained Keras model into memory when the server starts."""
    global model
    logger.info("Loading model from '%s' ...", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    # "Warm up" the model with a dummy forward pass so the very first real
    # request isn't slowed down by lazy graph/kernel initialization.
    dummy_input = np.zeros((1, IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    logger.info("Model loaded and warmed up successfully.")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def read_image_as_array(file_bytes: bytes) -> np.ndarray:
    """
    Convert raw uploaded image bytes into a numpy array ready for the model.

    Steps:
      1. Open the bytes as a PIL image and force RGB (drops alpha channel /
         handles grayscale uploads gracefully).
      2. Resize to the (255, 255) input size the model expects.
      3. Convert to a numpy array via tf.keras.utils.img_to_array.

    NOTE: We intentionally do NOT divide by 255 here. The model itself
    contains an internal Rescaling(1./255) layer, so raw 0-255 pixel
    values must be passed in directly.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file could not be read as an image.",
        ) from exc

    image = image.resize(IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)  # shape: (255, 255, 3)
    return image_array


def predict_from_array(image_array: np.ndarray) -> tuple[str, float]:
    """Run the model on a single image array and return (label, confidence%)."""
    batch = np.expand_dims(image_array, axis=0)  # shape: (1, 255, 255, 3)

    raw_predictions = model.predict(batch, verbose=0)[0]

    # Some models output raw logits instead of probabilities. If the values
    # don't already sum to ~1 (i.e. aren't a valid probability distribution),
    # apply softmax to convert them. Otherwise use them as-is.
    total = float(np.sum(raw_predictions))
    if not np.isclose(total, 1.0, atol=1e-2):
        probabilities = tf.nn.softmax(raw_predictions).numpy()
    else:
        probabilities = raw_predictions

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = round(float(np.max(probabilities)) * 100, 2)

    return predicted_class, confidence


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
async def root():
    """Simple health-check / welcome route."""
    return {
        "message": "Potato Disease Classification API is running.",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/ping")
async def ping():
    """Lightweight health-check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded potato leaf image and return the predicted disease
    class along with the model's confidence percentage.

    Response shape:
        {
            "class": "Early_Blight" | "Late_Blight" | "Healthy",
            "confidence": 94.52
        }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image_array = read_image_as_array(file_bytes)
    predicted_class, confidence = predict_from_array(image_array)
    info = DISEASE_INFO.get(predicted_class, {})

    return {
        "class": predicted_class,
        "confidence": confidence,
        "severity": info.get("severity", "unknown"),
        "summary": info.get("summary", ""),
        "recommendations": info.get("recommendations", []),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)