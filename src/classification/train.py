import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from copy import deepcopy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Caminho para o arquivo CSV com caminhos e classes das imagens.
            image_dir (str): Diretório contendo as imagens.
            transform (callable, optional): Transformações a serem aplicadas nas imagens.
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.class_to_idx = {
                'W Bags': 0,
                'W Shoes': 1,
                'W Accessories': 2,
                'Watches': 3,
                'W SLG': 4,
                'W RTW': 5}
        self.idx_to_class = {
                0: 'W Bags',
                1: 'W Shoes',
                2: 'W Accessories',
                3: 'Watches',
                4: 'W SLG',
                5: 'W RTW'}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.data.iloc[idx, 1]

        # Converter label de string para índice
        label = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)



def get_data_loaders(csv_file, folder_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomImageDataset(csv_file, folder_path, transform=transform)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=10):
    early_stopping = EarlyStopping(patience=5, delta=0.01)
    best_model_wts = deepcopy(model.state_dict())

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Check early stopping
        early_stopping(val_loss, model)
        if val_loss < early_stopping.best_loss:
            best_model_wts = deepcopy(model.state_dict())

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model



if __name__=="__main__":
    csv_file = "path/to/your/dataset.csv"
    folder_path = "path/to/your/images"

    # Get data loaders
    train_loader, val_loader = get_data_loaders(csv_file, folder_path)

    # Load pre-trained model
    print("Loading pre-trained ResNet-18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Modify the final layer for the number of classes in your dataset
    num_classes = len(pd.read_csv(csv_file).iloc[:, 1].unique())
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    model = model.to(device)

    # Fine-tune specific layers
    layers_to_finetune = ['fc.weight', 'fc.bias', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias']
    params_to_update = []
    for name, param in model.named_parameters():
        if name in layers_to_finetune:
            print(f"Fine-tuning: {name}")
            params_to_update.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update, lr=1e-4)

    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=10)