import unittest
from src.preprocessing import Preprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = Preprocessor()

    def test_correct_text(self):
        text = "Hola, este es un texto con errrores."
        corrected = self.preprocessor.correct_text(text)
        self.assertNotEqual(text, corrected)

    def test_preprocess_text(self):
        text = "¡Hola! ¿Cómo estás?"
        processed = self.preprocessor.preprocess_text(text)
        expected = 'hola'
        self.assertEqual(processed, expected)

if __name__ == '__main__':
    unittest.main()