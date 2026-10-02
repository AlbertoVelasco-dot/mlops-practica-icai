import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Cargar el modelo entrenado
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    print(
        "Error: 'model.pkl' no encontrado. Por favor, asegúrate de haber ejecutado el script de entrenamiento."
    )
    model = None

# Inicializar la aplicación FastAPI
app = FastAPI(title="API de Predicción del Modelo Iris")


# Esquema de los datos de entrada
class PredictionRequest(BaseModel):
    features: list[float]


# Esquema de la respuesta
class PredictionResponse(BaseModel):
    prediction: int


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo no cargado. Por favor, entrene el modelo primero.",
        )

    try:
        # Los datos de la petición ya llegan validados por Pydantic
        features = np.array(request.features).reshape(1, -1)

        # Realizar la predicción
        prediction = model.predict(features)

        # Devolver la predicción en formato JSON
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
