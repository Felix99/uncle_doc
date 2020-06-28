from chatbot import AnswerRecommender
import unittest
import pandas

rec = AnswerRecommender()
rec.train()


class MyTestCase(unittest.TestCase):

    def test_fixed_set(self):
        test1 = ['Mir geht es schlecht.', 'Mir geht es gut.', 'Mein Tag war schön.', 'Mein Tag war mies.']
        print("test 1")
        for c in test1:
            print(c + " -> ")
            print(rec.sent2vec(c))

    def test_excel_set(self):
        test_set = pandas.read_excel('dr_freud_training_data.xlsx', sheet_name=True, index_col=False)
        for c in rec.answers:
            print(c + " -> ")
            print(rec.sent2vec(c))


if __name__ == '__main__':
    unittest.main()
