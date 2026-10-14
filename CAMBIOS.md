# Cambios hechos a NNApp (clasificación de animales)

Se partió de la plantilla NNApp tal como venía. Los cambios son pocos y la parte de Iris (tabular) y de audio no se tocó.

## Archivos nuevos

| Archivo | Para qué sirve |
|---|---|
| `notebooks/nn_animals_cnn_transfer.ipynb` | Notebook donde se entrenan y comparan los modelos y se guardan los pesos |
| `datasets/animal_data/` | Dataset de Kaggle (15 animales, 1944 imágenes) |
| `CAMBIOS.md` | Este documento |

## Archivos que se modificaron

| Archivo | Qué cambió | Por qué |
|---|---|---|
| `models/pytorch_arch.py` | Se agregaron las clases `AnimalCNN` y `AnimalResNet` al final (y el import de `torchvision.models`). Las clases que ya estaban no se tocaron. | PyTorch necesita la arquitectura para poder cargar los pesos guardados. Son las mismas del notebook. |
| `models/pytorch_models.py` | `get_model` recibe `model_type` (`cnn` o `transfer`). En imágenes carga `models/saved/pt_image_<model_type>.pt` con las clases nuevas. | Antes buscaba `saved_models/pt_image.pt` y usaba `ImageCNN` de 10 clases. |
| `models/tensorflow_models.py` | `get_model` recibe `model_type`. En imágenes carga `models/saved/tf_image_<model_type>.keras`. | Antes cargaba un solo `tf_image.keras`. |
| `utils/inference.py` | `run_inference` recibe `model_type`, le agrega la dimensión de batch a la imagen (y pasa a formato PyTorch cuando hace falta) y devuelve el animal predicho con su confianza y el top 3. Se agregaron `load_animals_meta`, `get_image_size` y `format_image_prediction`. | Antes la imagen se mandaba al modelo sin batch y la respuesta eran probabilidades sin nombres de clase. |
| `app/main.py` | `/predict/` recibe `model_type` (por defecto `cnn`). Para imágenes usa el tamaño con el que se entrenó el modelo (`preprocess_image` ya tenía el parámetro `size`). Devuelve un mensaje de error si falta la imagen o los pesos. | Que la ResNet reciba 128x128 y la CNN 32x32. |
| `ui/app.py` | Solo la pestaña **Imágenes**: se agregó un selector de modelo y ahora se muestra el animal, la confianza y el top 3. | Para elegir entre CNN y ResNet y ver el resultado. |
| `requirements.txt` | Se agregaron `matplotlib` y `jupyter`. | Los usa el notebook. |
| `README.md` | Se agregó una sección con los pasos para animales. | Documentación. |

## Archivos que genera el notebook en `models/saved/`

| Archivo | Contenido |
|---|---|
| `tf_image_cnn.keras` / `tf_image_cnn.weights.h5` | CNN en TensorFlow (modelo completo / solo pesos) |
| `tf_image_transfer.keras` / `tf_image_transfer.weights.h5` | ResNet50V2 en TensorFlow |
| `pt_image_cnn.pt` | Pesos (`state_dict`) de la CNN en PyTorch |
| `pt_image_transfer.pt` | Pesos (`state_dict`) de ResNet18 en PyTorch |
| `animals_meta.json` | Nombres de las 15 clases y tamaño de imagen de cada modelo |
| `comparacion_modelos.csv` | Tabla de comparación de los 4 modelos |

## Decisiones importantes

- **Tamaño de imagen:** la CNN se entrena con 32x32, como pide el taller. Las ResNet usan 128x128 porque vienen entrenadas con imágenes grandes y con 32x32 pierden mucho. Está en la variable `TRANSFER_SIZE` del notebook; si se cambia, la app lo lee del `animals_meta.json` y sigue funcionando.
- **Entrada de los modelos:** la app manda la imagen en rango [0, 1] (como ya hacía `preprocess_image`). La normalización que necesita cada ResNet está dentro del modelo, así que la app no cambia según el modelo.
- **Mismo preprocesamiento:** `load_image` del notebook hace lo mismo que `preprocess_image` de la app, para que una imagen se prepare igual al entrenar y al predecir.
- **División de datos:** 70 % train, 15 % validación y 15 % test, estratificada y con la misma semilla para los 4 modelos.
- **Sin tocar:** los modelos de Iris, `utils/preprocess.py`, `utils/data.py`, `trainer_tf.py` y `trainer_pt.py`.

## Nota sobre la plantilla

`pytorch_models.py` ya buscaba los pesos tabulares y de audio en `saved_models/`, pero `trainer_pt.py` los guarda en `models/saved/`. No se corrigió porque está fuera de este taller; solo se usan las rutas nuevas para imágenes (`models/saved/`).
