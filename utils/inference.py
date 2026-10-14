import json
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Lo guarda el notebook de entrenamiento: nombres de las clases y tamaño de imagen de cada modelo
ANIMALS_META_PATH = Path("models/saved/animals_meta.json")


def load_animals_meta():
    if not ANIMALS_META_PATH.exists():
        raise FileNotFoundError(
            f"Falta {ANIMALS_META_PATH}. Ejecuta el notebook de entrenamiento de animales antes de hacer predicciones."
        )
    return json.loads(ANIMALS_META_PATH.read_text(encoding="utf-8"))


def get_image_size(model_type="cnn"):
    """Tamaño (en píxeles) con el que se entrenó cada modelo de imágenes."""
    meta = load_animals_meta()
    return int(meta["cnn_size"] if model_type == "cnn" else meta["transfer_size"])


def format_image_prediction(probs):
    """Convierte las probabilidades en la clase predicha y las 3 más probables."""
    classes = load_animals_meta()["classes"]
    order = np.argsort(probs)[::-1]
    return {
        "label": classes[int(order[0])],
        "class_index": int(order[0]),
        "confidence": float(probs[order[0]]),
        "top3": [{"label": classes[int(i)], "probability": float(probs[i])} for i in order[:3]],
    }


def run_inference(x, framework, data_type, model_type="cnn"):
    if framework == "tensorflow":
        from models.tensorflow_models import get_model
        model = get_model(data_type, model_type)
        if data_type == "image":
            x = np.expand_dims(x, axis=0)  # (H, W, 3) -> (1, H, W, 3)
        preds = model.predict(x)
    else:
        from models.pytorch_models import get_model
        model = get_model(data_type, model_type)
        import torch
        with torch.no_grad():
            if data_type == "image":
                t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (H, W, 3) -> (1, 3, H, W)
                preds = torch.softmax(model(t), dim=1).cpu().numpy()   # el modelo devuelve logits
            else:
                preds = model(x).cpu().numpy()

    # Convertir probabilidades a clases
    if data_type == "image":
        return format_image_prediction(preds[0])
    if data_type == "tabular":
        if preds.shape[1] == 1:
            # Clasificación binaria (sigmoid)
            classes = (preds > 0.5).astype(int).flatten()
        else:
            # Multiclase (softmax) - ej: Iris con 3 clases
            classes = np.argmax(preds, axis=1)
        return classes.tolist()
    return preds.tolist()
