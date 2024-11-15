import unittest
from src.feature_engineering import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.engineer = FeatureEngineer()

    def test_extract_pos_counts(self):
        text = "El gato come pescado."
        features = self.engineer.extract_pos_counts(text)
        self.assertTrue('nouns' in features)
        self.assertTrue('verbs' in features)

if __name__ == '__main__':
    unittest.main()