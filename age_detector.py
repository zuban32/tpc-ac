import nltk
from functools import reduce
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

class AgeDetector:

    def __init__(self):
        # toker = nltk.tokenize.WordPunctTokenizer()
        self.vec0 = TfidfVectorizer(ngram_range=(1, 1), strip_accents='unicode',
                                   tokenizer=nltk.word_tokenize, binary=True)
        self.vec1 = TfidfVectorizer(ngram_range=(1, 1), strip_accents='unicode',
                                   tokenizer=nltk.word_tokenize, binary=True)
        self.vec2 = TfidfVectorizer(ngram_range=(1, 1), strip_accents='unicode',
                                   tokenizer=nltk.word_tokenize, binary=True)
        # self.clf0 = LinearSVC(class_weight='auto')
        # self.clf1 = LinearSVC(C=2.0, class_weight='auto')
        self.clf0 = SGDClassifier(loss='modified_huber', n_iter=100, class_weight='auto')
        self.clf1 = SGDClassifier(loss='modified_huber', n_iter=100, class_weight='auto')
        self.clf2 = SGDClassifier(loss='modified_huber', n_iter=100, class_weight='auto')

    def extract(self, data):
        texts = list(map(lambda x: reduce(lambda res, y: res + y, x, ''), data))
        return texts

    def train(self, instances, labels):
        groups = [['00-18', '19-25'], ['26-35', '36-45', '46-99']]
        new_labels = list(map(lambda x: '0' if x in groups[0] else '1', labels))
        data = self.extract(instances)
        X = self.vec0.fit_transform(data)
        self.clf0.fit(X, new_labels)

        l1 = list(filter(lambda x: x[1] == '0', zip(data, new_labels, labels)))
        l2 = list(filter(lambda x: x[1] == '1', zip(data, new_labels, labels)))

        data1 = list(x[0] for x in l1)
        data2 = list(x[0] for x in l2)
        lbl1 = list(x[2] for x in l1)
        lbl2 = list(x[2] for x in l2)

        X = self.vec1.fit_transform(data1)
        self.clf1.fit(X, lbl1)
        X = self.vec2.fit_transform(data2)
        self.clf2.fit(X, lbl2)

    def classify(self, instances):
        data = self.extract(instances)
        X = self.vec0.transform(data)
        result = self.clf0.predict(X)

        l1 = []
        l2 = []

        for x, y, i in zip(data, result, range(len(data))):
            if y == '0':
                l1.append((x, i))
            elif y == '1':
                l2.append((x, i))

        data1 = list(x[0] for x in l1)
        inds1 = list(x[1] for x in l1)
        data2 = list(x[0] for x in l2)
        inds2 = list(x[1] for x in l2)

        X1 = self.vec1.transform(data1)
        res1 = self.clf1.predict(X1)
        X2 = self.vec2.transform(data2)
        res2 = self.clf2.predict(X2)

        res = list(' ' * len(instances))
        for x, i in zip(res1, inds1):
            res[i] = x
        for x, i in zip(res2, inds2):
            res[i] = x

        return res
