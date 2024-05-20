import streamlit as st
import time
import torch
import io
import requests
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
from torch import nn


class ChessClassifierWithBatchNorm(nn.Module):
    def __init__(self, num_classes=6):
        super(ChessClassifierWithBatchNorm, self).__init__()

        # Download pre-trained VGG16 model weights (without verification)
        vgg16_weights_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
        response = requests.get(vgg16_weights_url, verify=False)
        weights_data = BytesIO(response.content)

        # Load pre-trained VGG16 model
        self.features = models.vgg16()
        self.features.load_state_dict(torch.load(weights_data))

        # Freeze pre-trained model parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Get final feature output size from pre-trained model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            in_features = self.features(dummy_input).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=4096),  # Input size based on VGG16
            nn.BatchNorm1d(4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_checkpoint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filepath = "strapp/checkpoint-epoch(7)-18:01:34.pth"
    checkpoint = torch.load(filepath, map_location=device)
    model = ChessClassifierWithBatchNorm()
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    model.eval()
    return model

class_names = ['Pawn', 'Rook', 'Knight', 'Bishop', 'Queen', 'King']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_checkpoint()

# Define the data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = data_transforms(image).unsqueeze(0)
    return image

def predict(image_bytes):
    image_tensor = process_image(image_bytes).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probabilities, top_indices = probabilities.topk(6)
        
    top_probabilities = top_probabilities.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    return top_probabilities, top_indices

# Define the Streamlit app
st.title("Chess Image Classification")
st.write("Upload an image to classify it as a chess piece.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Measure latency and throughput
    start_time = time.time()

    # Make a prediction
    top_probabilities, top_indices = predict(image)
    
    # end time
    end_time = time.time()
    
    # Calculate latency
    latency = end_time - start_time
    
    # Map indices to class labels
    top_class = [class_names[idx] for idx in top_indices]

    st.write(f"Predicted Class: {top_class}")
    st.write(f"Latency: {latency:.4f} seconds")