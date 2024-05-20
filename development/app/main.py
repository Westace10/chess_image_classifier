from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
from torch import nn
from torchvision import models
import io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import requests


app = FastAPI()

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
    filepath = "app/checkpoint-epoch(7)-18:01:34.pth"
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

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
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
        top_probabilities, top_indices = probabilities.topk(1)
        
    top_probabilities = top_probabilities.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    return top_probabilities, top_indices

@app.post("/predict")
async def prediction(file: UploadFile = File(...)):
    try:
        # Read the uploaded image data
        contents = await file.read()
        
        # Make a prediction
        top_probabilities, top_indices = predict(contents)
        
        # Map indices to class labels
        top_classes = [class_names[idx] for idx in top_indices]

        # Prepare the response
        response = {
            "predictions": [{"class_name": top_classes[i], "confidence": float(top_probabilities[i])} for i in range(len(top_classes))]
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
