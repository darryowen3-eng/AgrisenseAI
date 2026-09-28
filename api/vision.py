import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent.parent


VISION_CONFIG = {

    "maize": {
        "model":
            BASE_DIR
            / "models"
            / "agrisense"
            / "maize"
            / "best_model.keras",

        "classes":
            BASE_DIR
            / "models"
            / "agrisense"
            / "maize"
            / "class_names.json"
    },

    "groundnuts": {
        "model":
            BASE_DIR
            / "models"
            / "agrisense"
            / "groundnut"
            / "best_model.keras",

        "classes":
            BASE_DIR
            / "models"
            / "agrisense"
            / "groundnut"
            / "class_names.json"
    }
}


_models = {}
_classes = {}


def load_vision_models():

    for crop, config in VISION_CONFIG.items():

        model_path = config["model"]
        class_path = config["classes"]

        if not model_path.exists():
            print(
                f"Vision model missing: "
                f"{model_path}"
            )
            continue

        if not class_path.exists():
            print(
                f"Class file missing: "
                f"{class_path}"
            )
            continue

        _models[crop] = load_model(
            model_path
        )

        with open(class_path, "r") as f:
            _classes[crop] = json.load(f)


load_vision_models()


def preprocess_image(
    image
):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    array = np.asarray(
        image
    ).astype("float32")

    array = array / 255.0

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


def diagnose(
    image,
    crop
):

    crop = crop.lower().strip()

    if crop not in _models:

        raise ValueError(
            f"No vision model available "
            f"for {crop}"
        )

    model = _models[crop]

    classes = _classes[crop]

    image_array = preprocess_image(
        image
    )

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[index]
    )

    diagnosis = classes[index]

    healthy = (
        "healthy" in diagnosis.lower()
    )

    if healthy:

        status = "Healthy"
        risk = "Low"

        recommendations = [
            "Continue regular field monitoring.",
            "Check plants regularly for new symptoms."
        ]

    else:

        status = "Disease detected"

        if confidence >= 0.80:
            risk = "High"

        elif confidence >= 0.60:
            risk = "Moderate"

        else:
            risk = "Uncertain"

        recommendations = [
            "Inspect affected plants across the field.",
            "Monitor whether symptoms are spreading.",
            "Follow locally recommended crop disease-management practices."
        ]

    return {
        "crop": crop,
        "diagnosis": diagnosis,
        "confidence": round(
            confidence,
            4
        ),
        "health_status": status,
        "risk": risk,
        "recommendations": recommendations
    }
