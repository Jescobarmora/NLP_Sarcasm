import unittest
from src.pipeline import SarcasmDetectionPipeline
import os

class TestPipelineIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        pipeline = SarcasmDetectionPipeline()
        pipeline.run_pipeline()
        self.assertTrue(os.path.exists('resultados_modelos.csv'))
        self.assertFalse(pipeline.results.empty)
        
        if os.path.exists('resultados_modelos.csv'):
            os.remove('resultados_modelos.csv')

if __name__ == '__main__':
    unittest.main()