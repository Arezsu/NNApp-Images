import tensorflow as tf
from pathlib import Path

_models = {}

def get_model(data_type, model_type="cnn"):
    # model_type solo se usa con imágenes: "cnn" (CNN propia) o "transfer" (ResNet)
    key = (data_type, model_type) if data_type == "image" else data_type
    if key not in _models:
        if data_type == "tabular":
            model = tf.keras.models.load_model("models/saved/tf_tabular.keras")
        elif data_type == "image":
            path = Path(f"models/saved/tf_image_{model_type}.keras")
            if not path.exists():
                raise FileNotFoundError(
                    f"Falta {path}. Corre notebooks/nn_animals_cnn_transfer.ipynb para entrenar y guardar los pesos."
                )
            model = tf.keras.models.load_model(path)
        else:
            model = tf.keras.models.load_model("models/saved/tf_audio.keras")
        _models[key] = model
    return _models[key]
