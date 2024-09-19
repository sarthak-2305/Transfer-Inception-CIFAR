# %%
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# %%
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# %%
# Check if MPS is available (Apple Silicon), otherwise fallback to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the pretrained InceptionV3 model and set to evaluation mode
model = models.inception_v3(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes

# Load the saved model state
model.load_state_dict(torch.load('inception_cifar10.pth', map_location=device))
model = model.to(device)
model.eval()

# %%
print(device)

# %%
# Preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# %%
def process_image(path, transform=transform):
    # Load and preprocess the image
    image_path = path  # Update with your image path
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure it's in RGB mode

    # Apply transformations (resize, normalize)
    input_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted_class_idx = torch.max(outputs, 1)

    # Get the corresponding class label
    predicted_class = cifar10_classes[predicted_class_idx.item()]
    print(f'Predicted Class: {predicted_class}')

    # Convert the image back to a format for display (denormalize)
    image_np = input_image.cpu().squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5  # Reverse normalization

    # Plot the image with the prediction
    plt.imshow(image_np)
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    plt.show()

# %%
process_image('testing/cat.jpg')

# %%
process_image('testing/aeroplane.jpg')

# %%
process_image('testing/bird.jpg')

# %%
process_image('testing/frog.jpeg')
