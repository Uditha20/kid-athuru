from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from io import BytesIO
from typing import List

# Load the trained model (saved as .h5)
model = tf.keras.models.load_model("dysgraphia_classifier.h5")

# Manually define the class labels
class_labels = ['low_dysgraphia', 'Potential Dysgraphia', 'no_Dysgraphia']

# FastAPI app setup
app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model
class PredictionResponse(BaseModel):
    class_name: str
    probabilities: List[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = image.load_img(BytesIO(img_bytes), target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return PredictionResponse(
        class_name=class_labels[predicted_class],
        probabilities=prediction[0].tolist()
    )

@app.get("/")
async def root():
    return {"message": "Dysgraphia Classifier API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
