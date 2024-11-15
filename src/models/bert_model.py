from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from src.models.evaluator import Evaluator
import numpy as np

class BertModelWrapper:
    def __init__(self, use_features=False):
        self.use_features = use_features
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased') 

        self.classifier = LogisticRegression(max_iter=1000)
        self.scaler = None

    def get_name(self):
        return 'BERT con ingeniería de características' if self.use_features else 'BERT sin ingeniería de características'

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return cls_embedding.flatten()

    def train(self, df, features):
        df['embedding'] = df['processed_text'].apply(self.get_embedding)
        X = np.stack(df['embedding'].values)
        if self.use_features:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X = np.hstack([X, features.values])
            X = self.scaler.fit_transform(X)
        y = df['Sarcasmo'].map({'Si': 1, 'No': 0})
        self.classifier.fit(X, y)

    def evaluate(self, df, features):
        df['embedding'] = df['processed_text'].apply(self.get_embedding)
        X = np.stack(df['embedding'].values)
        if self.use_features:
            X = np.hstack([X, features.values])
            X = self.scaler.transform(X)
        y_true = df['Sarcasmo'].map({'Si': 1, 'No': 0})
        y_pred = self.classifier.predict(X)
        y_probs = self.classifier.predict_proba(X)[:, 1]
        evaluator = Evaluator()
        roc_auc, accuracy = evaluator.evaluate(y_true, y_pred, y_probs)
        return {'roc_auc': roc_auc, 'accuracy': accuracy}