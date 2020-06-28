import numpy as np
from scipy.optimize import least_squares
import pandas


class AnswerRecommender(object):

    def __init__(self):
        self.word_feature_size= 10
        self.sentence_size = 10
        self.dict = {'': np.zeros(self.word_feature_size)}
        self.answers = ['']
        self.training_data_filename = '../data/dr_freud_training_data.xlsx'
        self.load_training_data()
        self.sent2vec = self.set2vec_mean

    def word2vec(self, word):
        if word not in self.dict.keys():
            # scaled uniform distribution such that E[word1 dot word2] = .5
            self.dict[word] = np.random.rand(self.word_feature_size) * np.sqrt(2 / self.word_feature_size)
        return self.dict[word]

    def get_state(self):
        return np.asarray(list(self.dict.values())).flatten()

    def set_state(self, new_state):
        i = 0
        for key in self.dict.keys():
            self.dict[key] = new_state[i * self.word_feature_size: (i+1) * self.word_feature_size]
            i += 1

    def sent2vec_append(self, sentence):
        sentence = sentence.split(' ')
        sentence = sentence[:self.sentence_size]
        for _ in range(len(sentence), self.sentence_size):
            sentence.append('')
        sent_list = list(map(self.word2vec, sentence[:self.sentence_size]))
        return np.asarray(sent_list).flatten()

    def set2vec_mean(self, sentence):
        print(sentence)
        word_list = sentence.split(' ')
        vec_list = map(self.word2vec, word_list)
        res = np.zeros(self.word_feature_size)
        for v in vec_list:
            res += v / len(word_list)
        return res

    def get_error(self, training_data):
        e = []
        for context, answer, label in training_data:
            c = self.sent2vec(context)
            a = self.sent2vec(answer)
            y_hat = c.dot(a)
            e.append(label - y_hat)
        return np.asarray(e)

    def train(self):
        training_data = self.load_training_data()

        def r(x):
            self.set_state(x)
            return self.get_error(training_data)
        r0 = r(self.get_state())
        print(np.isfinite(r0))
        res = least_squares(r, self.get_state(), jac='2-point', method='trf', max_nfev=int(10000))
        self.set_state(res.x)

    def load_training_data(self):
        data = []
        df = pandas.read_excel(self.training_data_filename, index_col=False)
        contexts = list(df.index)
        self.answers = list(df.columns)
        for i in range(len(contexts)):
            for j in range(len(self.answers)):
                data.append((contexts[i], self.answers[j], df.iat[i, j]))
        return data

    def get_scores(self, context):
        c = self.sent2vec(context)
        return list(map(lambda a: c.dot(self.sent2vec(a)), self.answers))

    def predict_answer(self, context):
        scores = self.get_scores(context)
        max_id = scores.index(max(scores))
        return self.answers[max_id]









