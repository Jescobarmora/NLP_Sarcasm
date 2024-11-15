import pandas as pd
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.models.bert_model import BertModelWrapper
from src.models.fasttext_model import FastTextModelWrapper
from src.models.evaluator import Evaluator

class SarcasmDetectionPipeline:
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = Evaluator()
        # Almacenes para datos y resultados
        self.train_df = None
        self.test_df = None
        self.results = None

    def run_pipeline(self):
        # Cargar datos
        print()
        print("Iniciando pipeline...")
        self.train_df, self.test_df = self.data_loader.load_data()
        
        # Preprocesamiento
        print("1. Preprocesando datos...")
        self.train_df = self.preprocessor.process(self.train_df)
        self.test_df = self.preprocessor.process(self.test_df)
        print("   - Datos preprocesados correctamente.")

        # Ingeniería de características
        print("2. Construyendo Ingeniería de características...")
        self.train_features = self.feature_engineer.transform(self.train_df)
        self.test_features = self.feature_engineer.transform(self.test_df)
        print("   - Ingeniería de características aplicadas correctamente.")

        # Modelado y evaluación
        print("3. Entrenando modelos...")
        models = [
            BertModelWrapper(use_features=False),
            BertModelWrapper(use_features=True),
            FastTextModelWrapper(use_features=False),
            FastTextModelWrapper(use_features=True)
        ]

        self.results = []

        for model in models:
            model_name = model.get_name()
            print(f"   - Entrenando y evaluando {model_name}...")
            model.train(self.train_df, self.train_features)
            metrics = model.evaluate(self.test_df, self.test_features)
            self.results.append({
                'Modelo': model_name,
                'ROC AUC': metrics['roc_auc'],
                'Accuracy': metrics['accuracy']
            })

        # Compilar y guardar resultados
        print("4. Compilando resultados finales...")
        self.results = pd.DataFrame(self.results).sort_values(by='ROC AUC', ascending=False)
        self.results.to_csv('resultados_modelos.csv', index=False)
        print("   - Resultados compilados con éxito.")
        
        print("Pipeline ejecutado exitosamente. Los resultados se han guardado en 'resultados_modelos.csv'.")
