# Chess Image Classification with Pytorch

## Overview
This project involves creating a deep learning model to classify images of chess pieces using PyTorch. The goal is to accurately identify six different types of chess pieces: Pawn, Rook, Knight, Bishop, Queen, and King. The project employs data augmentation, model training in multiple stages, and thorough evaluation using metrics such as precision, recall, F1-score, and a confusion matrix.

## Getting Started
These instructions will guide you through setting up the project on your local machine for development and testing purposes.

### Dependencies
Ensure you have the following dependencies installed:
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- seaborn
- numpy
- scikit-learn
- requests
- tensorboard

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Westace10/chess_image_classifier.git
   cd chess-image-classifier
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv chess_env
   source chess_env/bin/activate  # On Windows use `chess_env\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and prepare the dataset:**
   - Organize your dataset into `train`, `val`, and `test` directories, each containing subdirectories for each class (e.g., `Pawn`, `Rook`, etc.). Ensure the directory structure is as follows:
     ```
     dataset/
       train/
         Pawn/
         Rook/
         Knight/
         Bishop/
         Queen/
         King/
       val/
         Pawn/
         Rook/
         Knight/
         Bishop/
         Queen/
         King/
       test/
         Pawn/
         Rook/
         Knight/
         Bishop/
         Queen/
         King/
     ```

## Project Instructions

### Stage 1: Basic Training with Data Augmentation
1. **Data Augmentation:**
   - Apply random resized crop, horizontal flip, grayscale conversion, and normalization.
   ```python
   data_transforms = {
       'train': transforms.Compose([
           transforms.RandomResizedCrop(target_size),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Grayscale(num_output_channels=3),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ]),
       'val': transforms.Compose([
           transforms.Resize(target_size),
           transforms.CenterCrop(target_size),
           transforms.ToTensor(),
           transforms.Grayscale(num_output_channels=3),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ]),
       'test': transforms.Compose([
           transforms.Resize(target_size),
           transforms.CenterCrop(target_size),
           transforms.ToTensor(),
           transforms.Grayscale(num_output_channels=3),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ]),
   }
   ```

2. **Create Dataset and DataLoader:**
   ```python
   image_datasets = {x: datasets.ImageFolder(os.path.join('path/to/dataset', x), data_transforms[x]) for x in ['train', 'val', 'test']}
   dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}
   ```

3. **Define the Model:**
   - Use a pre-trained VGG16 model with frozen parameters, modifying the classifier for the chess piece classification task.
   ```python
   class ChessClassifier(nn.Module):
       def __init__(self, num_classes=6):
           super(ChessClassifier, self).__init__()

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
               dummy_input = torch.randn(1, 3, 224, 224)  # Assuming 3 channels, 224x224 image
               in_features = self.features(dummy_input).view(1, -1).shape[1]

           self.classifier = nn.Sequential(
               nn.Linear(in_features=in_features, out_features=4096),  # Input size based on VGG16
               nn.ReLU(inplace=True),
               nn.Dropout(p=0.5),
               nn.Linear(4096, num_classes),
               nn.LogSoftmax(dim=1)
           )

       def forward(self, x):
           with torch.no_grad():
               x = self.features(x)
           x = x.view(x.size(0), -1)
           x = self.classifier(x)
           return x

   model = ChessClassifier(num_classes=6)
   ```

4. **Define Loss and Optimizer:**
   ```python
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
   ```

5. **Train the Model:**
   ```python
   trained_model, train_losses, train_accuracies, val_losses, val_accuracies, best_epoch = train_model(model, criterion, optimizer, dataloaders, image_datasets, num_epochs=10, scheduler=scheduler)
   ```

6. **Evaluate the Model:**
   ```python
   report, conf_matrix = evaluate_model(trained_model, dataloaders, class_folders)
   ```

7. **Save the Model:**
   ```python
   checkpoint_path = f'checkpoints/checkpoint-epoch({best_epoch}).pth'
   torch.save({
       'arch': 'vgg16',
       'classifier': trained_model.classifier,
       'state_dict': trained_model.state_dict(),
       'class_to_idx': dataloaders['train'].dataset.class_to_idx,
       'epoch': best_epoch,
       'train_loss': train_losses[-1],
       'val_loss': val_losses[-1],
       'train_acc': train_accuracies[-1],
       'val_acc': val_accuracies[-1],
   }, checkpoint_path)
   ```

8. **Plot Results:**
   ```python
   plt.savefig(f'img/{img_name}.png')
   plt.savefig(f'img/{img_name}.png')
   ```

### Stage 2: Additional Data Augmentation and Batch Normalization
1. **Data Augmentation:**
   - Add random rotation to the data augmentation techniques.
   ```python
   data_transforms['train'] = transforms.Compose([
       transforms.RandomResizedCrop(target_size),
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(30),
       transforms.ToTensor(),
       transforms.Grayscale(num_output_channels=3),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   ```

2. **Modify the Model:**
   - Add batch normalization layers.
   ```python
   self.classifier = nn.Sequential(
       nn.Linear(in_features=in_features, out_features=4096),
       nn.BatchNorm1d(4096),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5),
       nn.Linear(4096, num_classes),
       nn.BatchNorm1d(4096),
       nn.LogSoftmax(dim=1)
   )
   ```

3. **Training and Evaluation:**
   - Follow the same steps as Stage 1 with the updated transformations and classifier.

### Stage 3: Adjust Learning Rate and Batch Size
1. **Adjust Hyperparameters:**
   - Set the learning rates and batch size then loop the training process simultenously.
   ```python
   lr = [0.01, 0.1]
   batch_size = [64, 128]
   dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}
   optimizer = optim.Adam(model.parameters(), lr=lr)
   ```

2. **Model Training and Evaluation:**
   - Train and evaluate the model with the new hyperparameters following the same steps as the previous stages.

### Save and Load Checkpoints
- Save the model checkpoint:
    ```python
      checkpoint_path = f'checkpoints/checkpoint-epoch({best_epoch}).pth'
      torch.save({
         'arch': 'vgg16',
         'classifier': trained_model.classifier,
         'state_dict': trained_model.state_dict(),
         'class_to_idx': dataloaders['train'].dataset.class_to_idx,
         'epoch': best_epoch,
         'train_loss': train_losses[-1],
         'val_loss': val_losses[-1],
         'train_acc': train_accuracies[-1],
         'val_acc': val_accuracies[-1],
      }, checkpoint_path)
   ```

- Load the model checkpoint:
    ```python
    def load_checkpoint(filepath):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=device)
        model = ChessClassifier(num_classes=6)
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        model.to(device)
        model.eval()
        return model
    ```

## API for Model Predictions

### Setup
1. Install FastAPI and uvicorn:
    ```sh
    pip install fastapi uvicorn
    ```

2. Create an API using FastAPI:
    ```python
   from fastapi import FastAPI, UploadFile, File, HTTPException
   from PIL import Image
   from io import BytesIO
   from torch import nn
   from torchvision import models
   import io
   import torch
   import torch.nn.functional as F
   import torchvision.transforms as transforms
   import torch
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
         top_probabilities, top_indices = probabilities.topk(6)
         
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
    ```

### Docker Environment

1. **Ensure Docker is installed and running on your machine.**

2. **Create Dockerfile:**
   ```docker
   # 
   FROM python:3.9

   # 
   WORKDIR /code

   # 
   COPY ./requirements.txt /code/requirements.txt

   # 
   RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

   # 
   COPY ./app /code/app

   # 
   CMD ["fastapi", "run", "app/main.py", "--port", "80"]
   ```

3. **Build the Docker image:**

    ```bash
    docker build -t chess_image_classifier .
    ```

4. **Run the Docker container:**

    ```bash
    docker run -d --name chesscontainer -p 80:80 chessclassifier
    ```

5. **Verify the application is running:**

    Open your web browser and go to `http://0.0.0.0:80`. You should see your FastAPI application running.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.