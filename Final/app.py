from flask import Flask, render_template, request, redirect, url_for
import torch
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import psycopg2

input_size = 11
# Задаем парамертры для подключения к модели
conn = psycopg2.connect(
    dbname='demo',
    user='postgres',
    password='123',
    host='localhost',
    port='5432'
)
# и подключаемся к ней
cursor = conn.cursor()
# Создаем таблицу в Postgres, если ее нет
create_table_query = '''
CREATE TABLE IF NOT EXISTS insurance (
    datetime DATE,
    id SERIAL PRIMARY KEY,
    fio VARCHAR(255),
    passport VARCHAR(255),
    gender VARCHAR(255),
    children VARCHAR(255),
    smokes VARCHAR(255),
    region VARCHAR(255),
    birthdate DATE,
    weight REAL,
    height REAL,
    bmi REAL,
    result REAL
);
'''
cursor.execute(create_table_query)

# Функция загрузки и определения модели
def load_model_and_preprocessor(model_path, preprocessor_path):
    preprocessor = torch.load(preprocessor_path, map_location=torch.device('cpu'))

    # Объявляем такую же архитектуру НС, как и при обучении модели
    class InsuranceModel(nn.Module):
        def __init__(self, input_size):
            super(InsuranceModel, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    model = Pipeline(steps=[
        ('preprocessor', StandardScaler()),
        ('regressor', InsuranceModel(input_size=input_size))
    ])

    model.named_steps['regressor'].load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.named_steps['regressor'].eval()

    return model, preprocessor

# Создадим функцию вычисления суммы за страховку
def predict(model, preprocessor, input_data):
    input_tensor = torch.FloatTensor(np.array(preprocessor.transform(input_data)))
    with torch.no_grad():
        prediction = model.named_steps['regressor'](input_tensor).squeeze().numpy()

    return prediction

model_path = 'insurance_model.pth'
preprocessor_path = 'preprocessor.pth'

# Загрузка обученной в прошлом модели torch и трансформатора
model, preprocessor = load_model_and_preprocessor(model_path, preprocessor_path)

app = Flask(__name__, template_folder=".")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('result'))
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        ages = []
        genders = []
        bmis = []
        children1 = []
        smokes1 = []
        regions = []
        data = {
            'fio': request.form['fio'],
            'passport': request.form['passport'],
            'gender': request.form['gender'],
            'children': request.form['children'],
            'smokes': request.form['smokes'],
            'region': request.form['region'],
            'birthdate': request.form['birthdate'],
            'weight': request.form['weight'],
            'height': request.form['height'],
        }
        today = date.today()
        if data['gender'] == 'Мужской':
            gender = 'male'
        else:
            gender = 'female'
        weight = int(data['weight'])
        height = int(data['height'])
        bmi = weight / ((height / 100)*(height / 100))
        children = int(data['children'])

        smokes = data['smokes']
        if smokes == 'Да':
            smokes = 'yes'
        else:
            smokes = 'no'

        region = data['region']
        if region == 'Северо-Западный':
            region = 'northwest'
        if region == 'Северо-Восточный':
            region = 'northeast'
        if region == 'Юго-Западный':
            region = 'southwest'
        if region == 'Юго-Восточный':
            region = 'southeast'

        age = relativedelta(today, datetime.strptime(data['birthdate'], "%Y-%m-%d")).years
        ages.append(age)
        genders.append(gender)
        bmis.append(bmi)
        children1.append(children)
        smokes1.append(smokes)
        regions.append(region)
        pred_data = pd.DataFrame({
            'age':  ages,
            'sex': genders,
            'bmi': bmis,
            'children': children1,
            'smoker': smokes1,
            'region': regions
        })
        pred_result = predict(model, preprocessor, pred_data)
        data['pred_result'] = "{:.2f}".format(pred_result)
        data['bmi'] = "{:.2f}".format(bmi)

        # Добавляем данные по клиенту в таблицу Postgres
        insert_query = '''
        INSERT INTO insurance (datetime, fio, passport, gender, children, smokes, region, birthdate, weight, height, bmi, result)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        '''
        cursor.execute(insert_query, (
            datetime.now(),data['fio'], data['passport'], data['gender'], data['children'],
            data['smokes'], data['region'], data['birthdate'],
            data['weight'], data['height'], data['bmi'], format(data['pred_result']
        )))
        conn.commit()
        return render_template('result.html', data=data)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False)
    conn.close()








