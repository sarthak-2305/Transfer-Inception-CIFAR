from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

# Check if MPS is available, otherwise fallback to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the model
model = models.inception_v3(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)

# Load model state, mapping to correct device (CPU or MPS)
model.load_state_dict(torch.load('inception_cifar10.pth', map_location=device))

# Move model to the appropriate device (MPS or CPU)
model = model.to(device)


# Define the transformation (same as during training)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Route to serve the basic UI
@app.route('/')
def index():
    return render_template('index.html')  # Serves the HTML file

# Route to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    try:
        # Open the image file
        img = Image.open(io.BytesIO(file.read()))

        # Preprocess the image (resize, normalize, etc.)
        img = transform(img).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(img)
            _, predicted_class = torch.max(outputs, 1)

        # Return the predicted class as a JSON response
        return jsonify({'predicted_class': predicted_class.item()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)  # Changed port to 8080
