import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.models as models

# выбор папки с датасетом
data_dir = 'dataset'

# определяем алгоритм преобразования картинок в цифру
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# подгружаем данные
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# разделяем данные на три выборки по необходимым пропорциям
total_size = len(dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
demo_size = total_size - train_size - val_size

train_data, val_data, demo_data = random_split(dataset, [train_size, val_size, demo_size])

# создаем основной класс для работы с датасетом
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if not isinstance(image, Image.Image):
            image = F.to_pil_image(image)

        if self.transform:
            image = self.transform(image)

        return image, label

# разбиваем уже сами картинки датасета по установленным параметрам
train_dataset = CustomDataset(train_data, transform)
val_dataset = CustomDataset(val_data, transform)
demo_dataset = CustomDataset(demo_data, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
demo_loader = DataLoader(demo_dataset, batch_size=32)

# Загружаем предобученную модель ResNet18
model = models.resnet18(pretrained=False)

for param in model.parameters():
    param.requires_grad = False

# заменяем последний слой на наш классификатор
model.fc = nn.Linear(512, 2)  # у нас 2 класса: smile и not_smile

# проводлим процесс обучения
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # тренировочная часть обучения
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}")

        # валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}%")

train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# тестирование обученной модели на демонстрационной выборке
model.eval()
correct_demo = 0
total_demo = 0
with torch.no_grad():
    for images, labels in demo_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_demo += labels.size(0)
        correct_demo += (predicted == labels).sum().item()
demo_accuracy = 100 * correct_demo / total_demo

# задаем вариации гиперпараметров для сравнения
new_learning_rates = [0.01, 0.001, 0.0001]
new_epochs = [5, 10, 15]

results = []

# для каждой пары гиперпараметров проведем заново обучение и посмотрим его результаты
for lr in new_learning_rates:
    for epochs in new_epochs:
        model = models.resnet18(pretrained=False)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512, 2)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        train(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs)
        model.eval()
        correct_demo = 0
        total_demo = 0
        with torch.no_grad():
            for images, labels in demo_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_demo += labels.size(0)
                correct_demo += (predicted == labels).sum().item()
        demo_accuracy = 100 * correct_demo / total_demo

        results.append(f"Learning Rate: {lr}, Epochs: {epochs}, Demo Accuracy: {demo_accuracy}%")

# выводим результаты для каждого случая в отдельный файл
with open('results.txt', 'w') as f:
    f.write("Original Model:\n")
    f.write(f"Demo Accuracy: {demo_accuracy}%\n")
    f.write("\nHyperparameter Tuning Results:\n")
    for result in results:
        f.write(result + "\n")





