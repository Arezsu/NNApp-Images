import torch
from pathlib import Path

_models = {}

def get_model(data_type, model_type="cnn"):
    # model_type solo se usa con imágenes: "cnn" o "transfer" (EfficientNet)
    key = (data_type, model_type) if data_type == "image" else data_type
    if key not in _models:
        # Importar tu clase de modelo según el tipo
        if data_type == "tabular":
            from .pytorch_arch import TabularNet
            model = TabularNet()
            model.load_state_dict(torch.load("saved_models/pt_tabular.pt"))
        elif data_type == "image":
            from .pytorch_arch import AnimalCNN, AnimalEfficientNet
            from utils.inference import load_animals_meta
            path = Path(f"models/saved/pt_image_{model_type}.pt")
            if not path.exists():
                raise FileNotFoundError(
                    f"Falta {path}. Corre notebooks/CNN_Animals_PyTorch.ipynb para entrenar y guardar los pesos."
                )
            n_classes = len(load_animals_meta()["classes"])
            model = AnimalEfficientNet(n_classes) if model_type == "transfer" else AnimalCNN(n_classes)
            model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            from .pytorch_arch import AudioCNN
            model = AudioCNN()
            model.load_state_dict(torch.load("saved_models/pt_audio.pt"))
        model.eval()
        _models[key] = model
    return _models[key]
