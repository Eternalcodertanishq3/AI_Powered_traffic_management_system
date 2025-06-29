import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime

# Import SCENE_CLASSES from constants.py
from .constants import SCENE_CLASSES

# Define a transformation pipeline for the images
# These are standard transformations for ImageNet pre-trained models
data_transforms = transforms.Compose([
    transforms.Resize(256),        # Resize the image to 256x256
    transforms.CenterCrop(224),    # Crop the center of the image to 224x224
    transforms.ToTensor(),         # Convert the image to a PyTorch tensor
    transforms.Normalize(          # Normalize the image with ImageNet's mean and std dev
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

class SceneDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        """
        Args:
            annotations_file (string): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations CSV file not found: {annotations_file}")
            
        self.annotations_frame = pd.read_csv(annotations_file)
        self.transform = transform
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(SCENE_CLASSES)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(SCENE_CLASSES)}

        # Filter out any rows with scene_label not in SCENE_CLASSES
        initial_count = len(self.annotations_frame)
        self.annotations_frame = self.annotations_frame[
            self.annotations_frame['scene_label'].isin(SCENE_CLASSES)
        ].reset_index(drop=True)
        if len(self.annotations_frame) < initial_count:
            print(f"[WARNING] Removed {initial_count - len(self.annotations_frame)} entries with unknown scene_labels from annotations CSV.")
        
        # --- Path Normalization: Ensure all paths use forward slashes ---
        self.annotations_frame['image_path'] = self.annotations_frame['image_path'].apply(
            lambda p: p.replace('\\', '/') if isinstance(p, str) else p
        )
        print(f"SceneDataset initialized with {len(self.annotations_frame)} entries.")


    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        # --- Path Normalization: Ensure img_path is normalized before opening ---
        img_path_raw = self.annotations_frame.iloc[idx]['image_path']
        img_path = img_path_raw.replace('\\', '/') # Ensure forward slashes for opening
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[ERROR] Image not found at {img_path}. This will cause a DataLoader worker error if not handled by PyTorch.")
            raise 
        except Exception as e:
            print(f"[ERROR] Could not load image {img_path}: {e}. This will cause a DataLoader worker error if not handled by PyTorch.")
            raise

        label_name = self.annotations_frame.iloc[idx]['scene_label']
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the SceneClassifier model
class SceneClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SceneClassifier, self).__init__()
        # Load a pre-trained ResNet-50 model
        # IMPORTANT: Naming this 'resnet' to match your EXISTING 'scene_classifier.pth' file.
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all parameters in the ResNet backbone
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer to match our number of scene classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

def train_scene_classifier_main():
    # --- Configuration ---
    annotations_csv_path = "combined_annotations.csv" # Path to your master annotations CSV
    model_save_path = "models/scene_classifier.pth"
    batch_size = 64
    num_epochs = 15 # You might need more epochs for a very large dataset
    learning_rate = 0.001
    
    # Create the models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting scene classifier training on {device}...")

    # Load the dataset
    try:
        full_dataset = SceneDataset(annotations_csv_path, transform=data_transforms)
    except FileNotFoundError as e:
        print(f"[CRITICAL ERROR] {e}. Please ensure '{annotations_csv_path}' exists and is correctly formatted.")
        return
    except Exception as e:
        print(f"[CRITICAL ERROR] Error loading dataset: {e}. Exiting.")
        return

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2 or 1)
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

    # Initialize the model, loss function, and optimizer
    model = SceneClassifier(num_classes=len(SCENE_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Use tqdm for a nice progress bar
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            train_loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{num_epochs} [Train] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total_predictions += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
                
                val_loop.set_postfix(loss=loss.item())

        val_loss = val_running_loss / len(val_dataset)
        val_accuracy = val_correct_predictions / val_total_predictions
        print(f"Epoch {epoch+1}/{num_epochs} [Val] - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Save the model if it's the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with validation accuracy: {best_val_accuracy:.4f} to {model_save_path}")

    print("Training finished!")
    print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    train_scene_classifier_main()
