from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from io import BytesIO
from typing import List

# Load the trained model (saved as .h5)
model = tf.keras.models.load_model("dysgraphia_classifier.h5")

# Manually define the class labels (based on the order in the dataset)
class_labels = ['low_dysgraphia', 'Potential Dysgraphia', 'no_Dysgraphia']  # Modify if needed

# FastAPI app setup
app = FastAPI()

# Define a response model
class PredictionResponse(BaseModel):
    class_name: str
    probabilities: List[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Read image file
    img_bytes = await file.read()
    
    # Load image using Keras image loader and preprocess it
    img = image.load_img(BytesIO(img_bytes), target_size=(128, 128))  # Resize image
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image

    # Make prediction using the model
    prediction = model.predict(img_array)
    
    # Get the predicted class index and the corresponding class name
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Return the prediction and probabilities
    return PredictionResponse(
        class_name=class_labels[predicted_class],  # Get class label
        probabilities=prediction[0].tolist()  # Return probabilities as list
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
