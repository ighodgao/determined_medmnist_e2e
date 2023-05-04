import torch
from flask import Flask, request, jsonify
import torch.nn.functional as F

from determined.experimental import client
from determined import pytorch
import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator
from PIL import Image
import io

app = Flask(__name__)

# Load the Determined model
checkpoint = client.get_experiment("641").top_checkpoint()
path = checkpoint.download()
trial = pytorch.load_trial_from_checkpoint_path(path)
model = trial.model
model.eval()

def preprocess_data(image):
    # Resize the image
    image = image.resize((28, 28))

    # Convert the image to a NumPy array
    image = np.array(image)

    # Add a channel dimension if it's a grayscale image
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    # Normalize the image by dividing by 255 and subtracting 0.5
    image = (image / 255.0) - 0.5

    # Transpose the image dimensions
    image = np.transpose(image, (2, 0, 1))

    # Convert the image to a torch tensor
    processed_data = torch.tensor(image, dtype=torch.float32)

    # Add a batch dimension
    processed_data = processed_data.unsqueeze(0)

    return processed_data


# Define a Flask route for serving predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input image file from the request
    file = request.files['file']

    # Read the image file into a PIL Image object
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess the input data
    processed_data = preprocess_data(image)

    # Use the Determined model to make a prediction
    output_tensor = model(processed_data)

    # Convert the output tensor to a numpy array
    output_array = output_tensor.detach().numpy()
    print(output_array)
    # Apply softmax to get probabilities for each class
    probabilities = F.softmax(output_tensor, dim=1)

    # Convert the output tensor to a numpy array
    output_array = probabilities.detach().numpy()
    
    # Get the predicted class label
    class_label = np.argmax(output_array)
    class_labels = ["adipose", "background", "debris", "lymphocytes", "mucus","smooth muscle", "normal colon mucosa", "cancer-associated stroma", "colorectal adenocarcinoma epithelium"]

   # Return the predicted class label as a JSON response
    return jsonify({'prediction': class_labels[class_label]})

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
