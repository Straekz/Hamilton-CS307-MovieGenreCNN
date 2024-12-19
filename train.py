import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb

from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
from PIL import Image

from torchvision import models, transforms

BATCH_SIZE = 32
LR = 0.003
NUM_EPOCHS = 40
PATIENCE = 8
IMG_RESIZE = (299, 299)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model using Inception V3
class I_V3Movie(nn.Module):
    def __init__(self, num_genres):
        super(I_V3Movie, self).__init__()
        # Inception v3 expects input size (299x299)
        self.base_model = models.inception_v3(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_genres)

    def forward(self, x):
        # Inception v3 has two outputs during training (auxiliary and final output)
        # For inference or multi-label training, we only need the main output
        if self.training:
            x, aux = self.base_model(x)  # Both outputs during training
            return x
        else:
            x = self.base_model(x)  # Only main output during inference
            return x

# Model using GoogLeNet
class GoogLeMovie(nn.Module):
    def __init__(self, num_genres):
        super(GoogLeMovie, self).__init__()
        self.base_model = models.googlenet(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_genres)

    def forward(self, x):
        return self.base_model(x)

# Model based on LeNet architecture
class MovieGenreClassifierCNN(nn.Module):
    def __init__(self, num_genres):
        super(MovieGenreClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, num_genres)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# For turning data into a dataset
class MoviePostersDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_file)
        self.labels_df['encoded_genres'] = self.labels_df['encoded_genres'].apply(ast.literal_eval)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(row['encoded_genres'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# Add one-hot encoded labels to the DataFrame
def one_hot_encode(genres, genre_to_idx):
    encoding = [0] * len(genre_to_idx)
    for genre in genres:
        encoding[genre_to_idx[genre]] = 1
    return encoding

# Ensure the filenames has a file extension and perform one-hot encoding to all genres
def preprocess_data(raw_data_path):
    raw_df = pd.read_csv(raw_data_path)
    raw_df['genre_ids'] = raw_df['genre_ids'].apply(ast.literal_eval)

    all_genres = set(genre for genres in raw_df['genre_ids'] for genre in genres)
    all_genres = sorted(all_genres)

    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
    new_df = pd.DataFrame()
    new_df['filename'] = raw_df['file'].apply(lambda x: str(x) + ".jpg")
    new_df['encoded_genres'] = raw_df['genre_ids'].apply(lambda x: one_hot_encode(x, genre_to_idx))
    new_df[['filename', 'encoded_genres']].to_csv('data\French\preprocessed.csv', index=False)

    return new_df, all_genres

# Calculate the mean and standard deviation of image set to normalize them
def calculate_mean_std(dataset, batch_size=32):
    # Use a simple transform to load raw pixel values as tensors
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    n_pixels = 0
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)

    for images, _ in tqdm(loader, desc="Calculating Mean and Std"):
        # Images are in shape [batch_size, channels, height, width]
        images = images.numpy()  # Convert to numpy for easier calculations
        batch_pixels = images.shape[0] * images.shape[2] * images.shape[3]  # Pixels per batch
        n_pixels += batch_pixels

        # Sum of pixels for each channel
        channel_sum += images.sum(axis=(0, 2, 3))

        # Sum of squared pixels for each channel
        channel_sum_squared += (images ** 2).sum(axis=(0, 2, 3))

    # Calculate mean and std
    mean = channel_sum / n_pixels
    std = np.sqrt((channel_sum_squared / n_pixels) - (mean ** 2))

    return mean, std

# Create dataloarders
def dataload_train(image_dir, labels_file):
    norm_transform = transforms.Compose([
        transforms.Resize(IMG_RESIZE),
        transforms.ToTensor()
    ])
    temp = MoviePostersDataset(image_dir, labels_file, transform=norm_transform)
    mean, std = calculate_mean_std(temp)

    transform = transforms.Compose([
        transforms.Resize(IMG_RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = MoviePostersDataset(image_dir, labels_file, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Print dataset sizes for verification
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler):
    wandb.init(project="movie-genre-classification", name="training")
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_model = None
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        with tqdm(enumerate(train_loader), desc="Training", unit="batch", total=len(train_loader)) as train_bar:
            for batch_idx, (images, labels) in train_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Calculate training accuracy
                predictions = (outputs.sigmoid() > 0.5).float()
                correct_train += (predictions == labels).all(dim=1).sum().item()  # Match all genres
                total_train += labels.size(0)

                # Update progress bar
                train_bar.set_postfix(loss=train_loss / len(train_loader))

                # Scheduler
                scheduler.step(epoch + batch_idx / len(train_loader))

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss / len(train_loader))
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy})

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate validation accuracy
                predictions = (outputs.sigmoid() > 0.5).float()
                correct_val += (predictions == labels).all(dim=1).sum().item()
                total_val += labels.size(0)

        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss / len(val_loader))
        wandb.log({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_accuracy})

        if val_loss < best_val_loss:
            best_model = model.state_dict()
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
        
        if counter >= PATIENCE:
            print(f"Early stock at Epoch {epoch+1}/{num_epochs}")
            break

        print(f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, "
            f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")
    
    wandb.finish()

    model.load_state_dict(best_model) # Load the best model
    torch.save(model.state_dict(), r'MovieGenreCNN.pth') # Save the best model as pth
    print("Best model saved as 'MovieGenreCNN.pth'.")

    return train_losses, val_losses, train_accuracies, val_accuracies

# Function for plotting training vs validation accuracy or loss
def plot_info(legend_1, legend_2, plot_type):
    plt.figure(figsize=(10, 5))
    plt.plot(legend_1, label=f'Training {plot_type}')
    plt.plot(legend_2, label=f'Validation {plot_type}')
    plt.title(f'Training vs Validation {plot_type}')
    plt.xlabel('Epochs')
    plt.ylabel(plot_type)
    plt.legend()
    plt.show()

def main():
    preprocessed_df, all_genres = preprocess_data(r"data\raw.csv")
    print(all_genres)
    image_dir = "data/posters/"
    labels_file = "data/preprocessed.csv"
    train_loader, val_loader = dataload_train(image_dir, labels_file)

    # model = MovieGenreClassifierCNN(len(all_genres)).to(DEVICE)
    # model = GoogLeMovie(len(all_genres)).to(DEVICE)
    model = I_V3Movie(len(all_genres)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr = LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, scheduler)

    plot_info(train_accuracies, val_accuracies, "Accuracy")
    plot_info(train_losses, val_losses, "Loss")

if __name__ == "__main__":
    main()