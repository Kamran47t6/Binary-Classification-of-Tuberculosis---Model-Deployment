from flask import Flask, request, render_template
import pickle
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

# Load the trained ViT model
model_path = 'model_1_Binary_Updated.pkl'

def load_model(path):
    """Load the model from the specified path and map it to the CPU."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        model.to(torch.device('cpu'))  # Move the model to the CPU
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load the model: {str(e)}")

# Initialize the model
model = load_model(model_path)

# Define preprocessing for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 (standard for ViT)
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize with mean and std
])

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make predictions."""
    if 'image' not in request.files:
        return render_template('index.html', prediction_text='No image file uploaded.')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction_text='Please upload an image file.')

    try:
        # Open and preprocess the image
        image = Image.open(file).convert('RGB')  # Ensure the image is in RGB format
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)  # Forward pass
            
            # Handle non-tensor outputs
            if not isinstance(output, torch.Tensor):
                if hasattr(output, "logits"):  # For models like HuggingFace transformers
                    output = output.logits
                else:
                    raise TypeError("Model output is not a tensor and does not have 'logits' attribute.")

            prediction = torch.argmax(output, dim=1).item()  # Get the predicted class

        # Interpret the prediction
        output_text = 'Active TB' if prediction == 1 else 'Not Active TB'
        return render_template('index.html', prediction_text=f'Prediction: {output_text}')

    except UnidentifiedImageError:
        # Handle invalid image files
        return render_template('index.html', prediction_text='Invalid image file. Please upload a valid image.')
    except Exception as e:
        # Catch all other exceptions
        return render_template('index.html', prediction_text=f'Error processing image: {str(e)}')


if __name__ == "__main__":
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
