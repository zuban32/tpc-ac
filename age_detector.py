import json
import nltk
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score


class AgeDetector:

    def __init__(self):
        self.vec = CountVectorizer(ngram_range=(1, 2), tokenizer=nltk.word_tokenize, binary=True)
        # self.model = LogisticRegression(class_weight='auto', dual=True, solver='lbfgs', multi_class='multinomial')
        self.model = LinearSVC(class_weight='auto')

    def extract(self, data):
        texts = list(map(lambda x: reduce(lambda res, y: res + y, x, ''), data))
        # print(texts)
        # print(len(texts))
        return texts

    def train(self, instances, labels):
        X = self.vec.fit_transform(self.extract(instances))
        self.model.fit(X, labels)

    def classify(self, instances):
        X = self.vec.transform(self.extract(instances))
        result = self.model.predict(X)
        return result

if __name__ == '__main__':
    with open('Train.txt.json', encoding="utf-8") as data_file:
        data = json.load(data_file)
    dec = AgeDetector()
    with open('Train.lab.json', encoding="utf-8") as res_file:
        results = json.load(res_file)
    # print(len(data))
    # print(len(results))
    texts = dec.extract(data)
    # print(texts)
    X = dec.vec.fit_transform(texts)
    # dec.model.fit(X, ['1', '2', '3'])

    scores = cross_val_score(dec.model, X, results, cv=3, n_jobs=2)
    print(scores.mean())
