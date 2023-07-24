import torch
import torch.nn as nn
import os
from tqdm import tqdm
from constants import label_mapping, device

class DocumentAlignmentNet(nn.Module):
    def __init__(self):
        super(DocumentAlignmentNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),  # Output layer with a single neuron for regression
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def predict(self, x):
        x = x.unsqueeze(0)
        x = self.forward(x)
        _, preds = torch.max(x, 1)
        return preds


def train_cnn(model, criterion, optimizer, train_loader, epoch, writer=None):

# for epoch in range(num_epochs):
# Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_samples = 0

    for idx, batch in enumerate(tqdm(train_loader)):
        images = batch['pixels']
        labels = batch['label']

        # print(images.shape, labels.shape)

        optimizer.zero_grad()
        outputs = model(images)

        # print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # writer.add_scalar('Training_Loss', train_loss, epoch)

    train_accuracy = train_correct / total_samples
    train_loss /= len(train_loader)

    # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    # Specify the file path to save the model
    save_path = 'cnn_model.pth'

    # Save the model
    torch.save(model.state_dict(), save_path)

    # print("Model saved successfully!")
    # 
    return train_loss, train_accuracy

def evaluate_cnn(model, criterion, test_loader, epoch, writer=None):
    
    model.eval()  # Set the network to evaluation mode
    total_samples = 0
    total_correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            images = batch['pixels']
            labels = batch['label']

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            val_loss += loss.item()
            
            # writer.add_scalar('Validation_Loss', val_loss, epoch)

        accuracy = total_correct / total_samples
        val_loss /= len(test_loader)
    # 
    return val_loss, accuracy

