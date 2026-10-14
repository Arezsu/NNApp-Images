
# Install the viraual environment
> python -m venv .venv

# Activate the venv in Windows
> .\.venv\Scripts\activate

# Activate the venv in Linux o Mac
> source .venv\bin\activate

# install requirements packages
> pip install -r .\requirements.txt

# Abre un terminal y ejecutas el Backend:
> uvicorn app.main:app --reload --port 8000

# abre otro terminal y ejecutas el frontned
> streamlit run ui/app.py

# with poetry
Pretty soon

# Clasificación de animales (CNN y transfer learning)
1. Ejecuta el notebook correspondiente al framework: `notebooks/CNN_Animals_PyTorch.ipynb` para PyTorch o `notebooks/CNN_Animals_TensorFlow.ipynb` para TensorFlow.
   Cada notebook guarda los pesos compatibles con la app en `models/saved/`.
2. Ejecuta el backend y el frontend como se explicó arriba.
3. En la pestaña **Imágenes** elige el framework y el modelo (CNN o transfer learning), sube una foto de un animal y pulsa **Predecir**.

   NOTA IMPORTANTE USAR EL TRANSFERLEARNING QUE ES MUCHO MAS EFICIENTE

Los cambios hechos a la plantilla están en `CAMBIOS.md`.
