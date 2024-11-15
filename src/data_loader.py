import pandas as pd

class DataLoader:
    def __init__(self, train_path='data/Sarcasmo_train.csv', test_path='data/Sarcasmo_test.csv'):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        train_df = pd.read_csv(self.train_path, sep=';', encoding='utf-8')
        test_df = pd.read_csv(self.test_path, sep=';', encoding='utf-8')
        return train_df, test_df
