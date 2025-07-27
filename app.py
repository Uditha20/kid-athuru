# Create a new cell and add this code for FastAPI implementation
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import io
import uvicorn

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = tf.keras.models.load_model('dysgraphia_model')
# Get the serving function
infer = model.signatures["serve"]

def preprocess_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = image.load_img(io.BytesIO(contents), target_size=(224, 224))
    processed_img = preprocess_image(img)  # shape: (1, 224, 224, 3)
    
    # Convert numpy array to tf.Tensor (float32)
    input_tensor = tf.convert_to_tensor(processed_img, dtype=tf.float32)
    
    # Run inference using the signature function
    outputs = infer(input_tensor)
    
    # The output is a dict of tensors, get the prediction tensor by output key (usually 'output_0' or similar)
    prediction_tensor = list(outputs.values())[0]
    
    probability = float(prediction_tensor.numpy()[0][0])
    
    label = "Dysgraphia detected" if probability > 0.5 else "No dysgraphia detected"
    
    return {
        "label": label,
        "probability": probability,
        "confidence": f"{probability*100:.2f}%" if probability > 0.5 else f"{(1-probability)*100:.2f}%"
    }
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)