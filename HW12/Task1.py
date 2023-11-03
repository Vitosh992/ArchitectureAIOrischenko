import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2

#Загрузка стоп-слов и инициализация библиотек
nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))
morph = pymorphy2.MorphAnalyzer()

# Функция для приведения слов в один формат
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яё]", " ", text)
    return text

# Лемматизация слов и удаление стоп-слов
def lemmatize_and_remove_stopwords(text):
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words if word not in stop_words]
    return " ".join(lemmatized_words)

# Функция для расчета метрик TF-IDF
def calculate_tfidf(input_text):
    # Очистка и обработка текста
    cleaned_text = preprocess_text(input_text)
    processed_text = lemmatize_and_remove_stopwords(cleaned_text)

    # Расчет TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([processed_text])

    # Получение списка признаков (слов) и их TF-IDF значения
    features = tfidf_vectorizer.get_feature_names_out()
    tfidf_values = tfidf_matrix.toarray()[0]
    return processed_text, dict(zip(features, tfidf_values)), tfidf_matrix


def tf_idf_result(input_text):
        # Вызов функции и вывод результатов
    processed_text, tfidf_metrics, matrix = calculate_tfidf(input_text)
    print("Обработанный текст:", processed_text)
    print("TF-IDF метрики:")
    for feature, value in tfidf_metrics.items():
        print(f"{feature}: {value}")
    print("Матрица TF-IDF:")
    print(matrix.toarray())

def read_and_result(filename):
    file = open(filename, 'r', encoding='utf-8')
    tf_idf_result(file.read())

if __name__ == "__main__":
    read_and_result("text1.txt")
    read_and_result("text2.txt")
    read_and_result("text3.txt")
    read_and_result("text4.txt")
    read_and_result("text5.txt")
    read_and_result("text6.txt")
    read_and_result("text7.txt")
    read_and_result("text8.txt")
    read_and_result("text9.txt")
    read_and_result("text10.txt")
