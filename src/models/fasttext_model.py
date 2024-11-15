import fasttext
from sklearn.linear_model import LogisticRegression
from src.models.evaluator import Evaluator
import numpy as np
import os

class FastTextModelWrapper:
    def __init__(self, use_features=False):
        self.use_features = use_features
        self.ft_model = None
        self.classifier = LogisticRegression(max_iter=1000)
        self.scaler = None

    def get_name(self):
        return 'FastText con ingeniería de características' if self.use_features else 'FastText sin ingeniería de características'
        
    def train_fasttext(self, df):
        df['processed_text'].to_csv('fasttext_train.txt', index=False, header=False)
        self.ft_model = fasttext.train_unsupervised('fasttext_train.txt', model='skipgram', minCount=1, verbose=0)
        os.remove('fasttext_train.txt')


    def get_embedding(self, text):
        words = text.split()
        word_embeddings = [self.ft_model.get_word_vector(word) for word in words if word in self.ft_model.words]
        if len(word_embeddings) == 0:
            return np.zeros(self.ft_model.get_dimension())
        else:
            return np.mean(word_embeddings, axis=0)

    def train(self, df, features):
        self.train_fasttext(df)
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