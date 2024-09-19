from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image
import os

# Set up the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # You can generate a secret key

# Define where uploaded images will be stored temporarily
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# CIFAR-10 class labels
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load model
model = models.inception_v3(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load('inception_cifar10.pth', map_location=device))
model = model.to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted_class_idx = torch.max(outputs, 1)
    
    predicted_class = cifar10_classes[predicted_class_idx.item()]
    return predicted_class

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        prediction = predict_image(filepath)
        
        # Clean up by removing the uploaded file
        os.remove(filepath)
        
        return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)  # Changed port to 8080
