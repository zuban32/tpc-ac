import json
import numpy
import nltk
import re
from functools import reduce
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import accuracy_score

url = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def prepare(text):
    text = text.lower()
    # str1 = ''
    # for i in range(len(text)):
    #     if text[i] not in ['а', 'е', 'я', 'и', 'ю', 'у', 'о', 'ы', 'э', 'ё']:
    #          str1 += text[i]
    # text = re.sub(r"[\)]+", ')', text)  # for :))))) smiles, lots of them
    # text = re.sub(r"[\(]+", '(', text)
    return url.sub("", text)

def myToke(text):
    text = nltk.WordPunctTokenizer().tokenize(text)
    res = []
    for token in text:
        if not token.isspace() and len(token) > 1:
            res.append(token.lower())
    return res

class AgeDetector:

    def __init__(self):
        # toker = nltk.tokenize.WordPunctTokenizer()
        self.vec0 = TfidfVectorizer(ngram_range=(1, 1), strip_accents='unicode',
                                   tokenizer=nltk.word_tokenize, binary=True)
        self.vec1 = TfidfVectorizer(ngram_range=(1, 1), strip_accents='unicode',
                                   tokenizer=nltk.word_tokenize, binary=True)
        self.vec2 = TfidfVectorizer(ngram_range=(1, 1), strip_accents='unicode',
                                   tokenizer=nltk.word_tokenize, binary=True)
        # self.est3 = LogisticRegression(class_weight='balanced')
        # self.model = sklearn.tree.DecisionTreeClassifier()
        # self.model = LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs')
        self.clf0 = LinearSVC(class_weight='auto')
        self.clf1 = LinearSVC(C=2.0, class_weight='auto')
        self.clf2 = SGDClassifier(loss='modified_huber', n_iter=100, class_weight='auto')
        # self.model = LinearSVC(C=500, class_weight='balanced')
        # self.model = NuSVC(nu=0.65, kernel='linear', class_weight='balanced')

    def extract(self, data):
        # print(sum(len(x) for x in data))
        texts = list(map(lambda x: reduce(lambda res, y: res + y, x, ''), data))
        # texts += list(map(lambda x: x[:int(len(x)/2)+1], texts))
        # print(texts)
        # print(len(texts))
        return texts

    def extract2(self, data):
        # print(sum(len(x) for x in data))
        texts = list(map(lambda x: reduce(lambda res, y: res + y, x, ''), data))
        texts += list(map(lambda x: x[:int(len(x)/2)+1], texts))
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
        # print(len(texts), len(results))
        return (texts, results)

    def train(self, instances, labels):
        # print(len(instances), len(labels))
        groups = [['00-18', '19-25'], ['26-35', '36-45', '46-99']]
        new_labels = list(map(lambda x: '0' if x in groups[0] else '1', labels))
        data = self.extract(instances)
        # print(len(data), len(new_labels))
        X = self.vec0.fit_transform(data)
        self.clf0.fit(X, new_labels)

        l1 = list(filter(lambda x: x[1] == '0', zip(data, new_labels, labels)))
        l2 = list(filter(lambda x: x[1] == '1', zip(data, new_labels, labels)))

        # print(len(l1), len(l2))

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

        # l1 = list(filter(lambda x: x[1] == '0', zip(data, result)))
        # l2 = list(filter(lambda x: x[1] == '1', zip(data, result)))

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
        # print(len(res1))
        # print(res1)
        X2 = self.vec2.transform(data2)
        res2 = self.clf2.predict(X2)
        # print(len(res2))
        # print(res2)

        res = list(' ' * len(instances))
        # print(len(res))
        for x, i in zip(res1, inds1):
            # print(i)
            res[i] = x
        for x, i in zip(res2, inds2):
            # print(i)
            res[i] = x
        # numpy.append(res1, res2)

        return res

# if __name__ == '__main__':
#     with open('Train.txt.json', encoding='utf-8') as data_file:
#         data = json.load(data_file)
#     with open('Train.lab.json', encoding='utf-8') as res_file:
#         results = json.load(res_file)
#
#     # for user_texts, label in zip(data, results):
#     #     print(user_texts[0] + ' --> ' + label)
#     dec = AgeDetector()
#
#     # l = list(filter(lambda x: x[1] in ['00-18'], zip(data, results)))
#     # for i in range(3):
#     #     for t in l[i][0]:
#     #         print(myToke(t))
#
#     dec.train(data, results)
#     data[0], data[2000] = data[2000], data[0]
#     results[0], results[2000] = results[2000], results[0]
#     count = 0
#     for x, y in zip(results, dec.classify(data)):
#         if x != y:
#             # print('Not equal\n')
#             count += 1
#     print('Count = ' + str(count) + '\n')

    # lens = list(len(x) for x in data)

    # texts = numpy.array(dec.extract(data))
#     res = numpy.array(results)
#     kf = KFold(len(data))
#     scores = []
#     for tr_ind, test_ind in kf:
#         X_tr, X_test = texts[tr_ind], texts[test_ind]
#         Y_tr, Y_test = res[tr_ind], res[test_ind]
#         dec.model = SGDClassifier(loss='modified_huber', n_iter=100, class_weight='auto')
#         dec.vec = TfidfVectorizer(ngram_range=(1, 1), tokenizer=nltk.word_tokenize, binary=True, strip_accents='unicode')
#         X = dec.vec.fit_transform(X_tr)
#         dec.model.fit(X, Y_tr)
#         X = dec.vec.transform(X_test)
#         res = dec.model.predict(X)
#         scores.append(accuracy_score(Y_test, res))
#     print(sum(scores) / len(scores))
#
    # X = dec.vec.fit_transform(dec.extract(data))
    # print(cross_val_score(dec.model, X, results, cv=3, n_jobs=-1).mean())

    # info = dec.extract1(data, results)
    # X = dec.vec.fit_transform(info[0])
    # print(cross_val_score(dec.model, X, info[1], cv=3, n_jobs=-1).mean())
