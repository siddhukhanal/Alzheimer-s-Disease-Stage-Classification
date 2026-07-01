import os


from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlzhemeirDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.class_to_idx = {
            'NonDemented': 0,
            'VeryMildDemented': 1,
            'MildDemented': 2,
            'ModerateDemented': 3
        }

        for label in ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']:
            class_dir = os.path.join(root_dir, label)

            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                numerical_label = self.class_to_idx[label]
                self.labels.append(numerical_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

            return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = AlzhemeirDataset(root_dir='data2/train', transform=transform)
test_dataset = AlzhemeirDataset(root_dir='data2/test', transform=transform)

val_dataset = AlzhemeirDataset(root_dir='data2/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 4)

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epoch = 3

print("Starting training... Please be patient, CPU training takes longer.")
for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss

    print(f"Epoch{epoch +1}/{num_epoch},Loss:{running_loss/len(train_loader)}")
    model.eval()
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    print('Validation accuracy:', val_accuracy)

model.eval()
test_labels = []
test_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
print('Test accuracy:', test_accuracy)


torch.save(model.state_dict(), 'alzhemeir_classifier.pth')
