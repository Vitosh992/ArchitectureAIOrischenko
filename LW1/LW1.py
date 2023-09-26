#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LinearRegression
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier


# In[2]:


# Загрузим файл в df
df = pd.read_csv('diabetes.csv')
df


# In[3]:


# Разделим независимые переменные и целевые на X и y
X = np.array(df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])
y = np.array(df[['Outcome']]).flatten()


# In[4]:


# Разделим выборку на тестовую и тренировочную
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[5]:


# Построим модель KNN 
y_neig = []
n_neighbors = 5 ## Алгоритм с KNN = 5
model = KNeighborsClassifier(n_neighbors = n_neighbors)
model.fit(X_train, y_train)
y1 = model.score(X_test, y_test)
y_neig.append(y1)
print(y1)


# In[6]:


n_neighbors = 10 ## Алгоритм с KNN = 10
model = KNeighborsClassifier(n_neighbors = n_neighbors)
model.fit(X_train, y_train)
y2 = model.score(X_test, y_test)
y_neig.append(y2)
print(y2)


# In[7]:


n_neighbors = 15 ## Алгоритм с KNN = 15
model = KNeighborsClassifier(n_neighbors = n_neighbors)
model.fit(X_train, y_train)
y3 = model.score(X_test, y_test)
y_neig.append(y3)
print(y3)


# In[8]:


n_neighbors = 20 ## Алгоритм с KNN = 20
model = KNeighborsClassifier(n_neighbors = n_neighbors)
model.fit(X_train, y_train)
y4 = model.score(X_test, y_test)
y_neig.append(y4)
print(y4)


# In[9]:


n_neighbors = 25 ## Алгоритм с KNN = 25
model = KNeighborsClassifier(n_neighbors = n_neighbors)
model.fit(X_train, y_train)
y5 = model.score(X_test, y_test)
y_neig.append(y5)
print(y5)


# In[10]:


# Сравним скорринг модели для разных значений k
x_neig = [5, 10, 15, 20, 25]
y_neig
plt.plot(x_neig, y_neig)
plt.xlabel('KNN amount')
plt.ylabel('Score rate')
plt.xticks([5, 10, 15, 20 ,25])
plt.show()


# In[11]:


# Выделим отдельно выборку для регрессионного анализа
X_reg = df[['SkinThickness', 'BMI']]
y_reg = df[['Insulin']]


# In[12]:


Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size = 0.25)


# In[13]:


# Построим модель множественной лин. регрессии
model2 = LinearRegression()
model2.fit(Xr_train, yr_train)
model2.score(Xr_test, yr_test)
yr_pred = model2.predict(Xr_test)
yr_pred = yr_pred.flatten()
yr_test = np.array(yr_test)
yr_test = yr_test.flatten()

# Сравним отклонение от реальных значений
d = {'yr_pred': yr_pred, 'yr_test': yr_test, 'diff': np.abs(yr_test - yr_pred)}
pd.DataFrame(d)


# In[14]:


model2.score(Xr_test, yr_test) ## Качество данной модели


# In[15]:


model2.coef_ = model2.coef_.reshape(2)


# In[16]:


print(f'Формула y = a + a1*x + a2*x. Коэффициент a = {model2.intercept_[0]}, a1 = {model2.coef_[0]}, a2 = {model2.coef_[1]}.')


# In[17]:


# Построим модель классификации наличия диабета по всем признакам, используя метод Decision Tree Gini
def tree_graph_to_png(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names,
    filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(png_file_to_save)

dt_classifier = DecisionTreeClassifier(criterion = 'gini')


# In[18]:


dt_classifier.fit(X, y)


# In[19]:


headers = df.columns[:-1]


# In[22]:


# Выведем обученную модель в виде диаграммы дерева решений в файл .png
tree_graph_to_png(dt_classifier, feature_names=headers, png_file_to_save='gini.png')


# In[36]:


data = np.array([[5, 120, 40, 30, 160, 25, 0.2, 40]]) # Проверим модель для случайных данных
print(dt_classifier.predict(data)) # В результате получили, что по нашим данным нет заболевания.

