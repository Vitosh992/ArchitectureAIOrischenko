import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


nltk.download('stopwords')
nltk.download('punkt')

# Загружаем полученный датасет с сайта. Тематика: Computational Science
# В датасете 4627 вопроса и ответа
df = pd.read_csv('QueryResults.csv')

stop_words = set(stopwords.words('english'))

# Очищаем датасет от верстки HTML
df['Question'] = df['Question'].str.replace('<p>', '').str.replace('</p>', '')
df['Answer'] = df['Answer'].str.replace('<p>', '').str.replace('</p>', '')
df['Answer'] = df['Answer'].str.replace('<li>', '').str.replace('</li>', '')
df['Answer'] = df['Answer'].str.replace('<ol>', '').str.replace('</ol>', '')
df['Question'] = df['Question'].str.replace('<li>', '').str.replace('</li>', '')
df['Question'] = df['Question'].str.replace('<ol>', '').str.replace('</ol>', '')
df['Answer'] = df['Answer'].str.replace('<em>', '').str.replace('</em>', '')
df['Question'] = df['Question'].str.replace('<em>', '').str.replace('</em>', '')
df['Answer'] = df['Answer'].str.replace('<blockquote>', '').str.replace('</blockquote>', '')
df['Question'] = df['Question'].str.replace('<blockquote>', '').str.replace('</blockquote>', '')


def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(words)

df['processed_question'] = df['Question'].apply(preprocess_text)

# Проводим векторизацию текста
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_question'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Напишем функция прямого поиска ответа на вопрос
def get_answer(user_question):
    user_question = preprocess_text(user_question)
    user_vector = vectorizer.transform([user_question])
    sim_scores = list(cosine_similarity(user_vector, tfidf_matrix)[0])
    max_sim_index = sim_scores.index(max(sim_scores))
    return df['Answer'][max_sim_index]

# Пример использования функции:
user_question = "How to train AI?"
answer = get_answer(user_question)
print("Answer:", answer)
