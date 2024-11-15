import unittest
from src.models.bert_model import BertModelWrapper
from src.models.fasttext_model import FastTextModelWrapper
import pandas as pd

class TestModels(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'Sarcasmo': ['Si', 'No', 'Si', 'No'],
            'processed_text': [
                'esto es sarcasmo',
                'esto es serio',
                'me encanta programar',
                'odio los errores'
            ]
        })
        self.features = pd.DataFrame({'nouns': [0.5, 0.5, 0.5, 0.5], 'verbs': [0.5, 0.5, 0.5, 0.5]})

    def test_bert_model(self):
        model = BertModelWrapper(use_features=False)
        model.train(self.sample_data, self.features)
        metrics = model.evaluate(self.sample_data, self.features)
        self.assertIsNotNone(metrics['roc_auc'])

    def test_fasttext_model(self):
        model = FastTextModelWrapper(use_features=False)
        model.train(self.sample_data, self.features)
        metrics = model.evaluate(self.sample_data, self.features)
        self.assertIsNotNone(metrics['roc_auc'])

if __name__ == '__main__':
    unittest.main()
