import json
import nltk
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score


class AgeDetector:

    def __init__(self):
        # self.vec = CountVectorizer(ngram_range=(1, 2), tokenizer=nltk.word_tokenize)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=nltk.word_tokenize, binary=True)
        # self.model = LogisticRegression(class_weight='auto', dual=True, solver='lbfgs', multi_class='multinomial')
        self.model = SGDClassifier(loss='modified_huber', n_iter=100, class_weight='auto')
        # self.model = LinearSVC(class_weight='auto')

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
