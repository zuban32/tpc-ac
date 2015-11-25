import json
import nltk
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

from nltk.stem import SnowballStemmer

# def myTokenizer(text):
#     tokens = []
#     for word in nltk.WordPunctTokenizer().tokenize(text):
#         if word.isalpha() or len(word) > 1:
#             tokens.append(SnowballStemmer('russian').stem(word))
#     return tokens

class AgeDetector:

    def __init__(self):
        self.vec = CountVectorizer(ngram_range=(1, 2), tokenizer=nltk.word_tokenize, binary=True)
        self.vec1 = TfidfVectorizer(ngram_range=(1, 2), tokenizer=nltk.word_tokenize, binary=True)
        # self.model = LogisticRegression(class_weight='auto')
        self.model = SGDClassifier(loss='modified_huber', n_iter=100, class_weight='auto')
        # self.model = LinearSVC(C=2.0, class_weight='auto')

    def extract(self, data):
        texts = list(map(lambda x: reduce(lambda res, y: res + y, x, ''), data))
        # print(texts)
        # print(len(texts))
        return texts

    def extract1(self, data, labels):
        texts = []
        results = []
        for user_texts, label in zip(data, labels):
            for text in user_texts:
                texts.append(text)
            for i in range(len(user_texts)):
                results.append(label)
        print(len(texts), len(results))
        return (texts, results)

    def train(self, instances, labels):
        info = self.extract1(instances, labels)
        X = self.vec.fit_transform(info[0])
        self.model.fit(X, info[1])

    def classify(self, instances):
        X = self.vec.transform(self.extract(instances))
        result = self.model.predict(X)
        return result

if __name__ == '__main__':
    with open('Train.txt.json', encoding='utf-8') as data_file:
        data = json.load(data_file)
    with open('Train.lab.json', encoding='utf-8') as res_file:
        results = json.load(res_file)
    dec = AgeDetector()

    dec.train(data, results)
    print(results == dec.classify(data))

#     X = dec.vec.fit_transform(dec.extract(data))
#     print(cross_val_score(dec.model, X, results, cv=3, n_jobs=-1).mean())
