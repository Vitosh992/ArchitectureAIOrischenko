## Данную задачу не получилось проверить на практике
## из-за слишком долгой работы программы.
## По логике код должным быть верным.
 

# Импортируем необходимые библиотеки
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка датасет с отзывами
data = pd.read_csv("IMDB Dataset.csv")

# Разделяем загруженные данные на тренировочную и тестовую выборку
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Загружаем предобученную модель Bert
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Создаем основной класс для программы, в котором будет производится обработка данных
class MovieReviewDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['review']
        label = self.data.iloc[idx]['sentiment']
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': 1 if label == 'positive' else 0  # Преобразование меток в 1 (положительный) или 0 (отрицательный)
        }

# Подгружаем в модель данные и токенизатор
train_dataset = MovieReviewDataset(train_data, tokenizer)
test_dataset = MovieReviewDataset(test_data, tokenizer)
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Задаем оптимизатор Adam для нашей сети
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader))

# Обучаем нашу модель
num_epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Рассчитываем эффективность работы нашей модели на наших данных
model.eval()
predictions = []
true_labels = []

for batch in test_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].numpy()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()

    predictions.extend(predicted_labels)
    true_labels.extend(labels)

accuracy = accuracy_score(true_labels, predictions)
print(f"Итоговая точность модели на тестовой выборке: {accuracy:.4f}")


